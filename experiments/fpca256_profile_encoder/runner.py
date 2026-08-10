"""CLI and evaluation orchestration for the research-only FPCA256 experiment."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aramina.patient_features import TARGET_CASE_ID
from aramina.training_evaluation import _patient_split_pairs

from .config import load_experiment_config, resolve_path
from .lineage import (
    create_research_npt256_artifact,
    file_sha256,
    load_cohort_datasets as _load_cohort_datasets,
    require_matching_case_order as _require_matching_case_order,
    validate_artifact_lineage,
    validate_common_cohort,
    validate_pyfai_runtime,
)
from .model import (
    ProfileSpec,
    build_dataset_context,
    fit_split,
    fit_train_all,
    metric_row,
)
from .result_outputs import (
    build_fold_manifest,
    dataset_summary,
    paired_fold_deltas,
    record_pca,
    summarize_results,
    write_outputs,
)


LOGGER = logging.getLogger(__name__)
_validate_artifact_lineage = validate_artifact_lineage


def run_experiment_from_config(
    config_path: str | Path,
    *,
    cohort: str = "all",
    input_h5_path: str | Path | None = None,
    output_folder: str | Path | None = None,
) -> dict[str, Any]:
    """Run configured common/full experiments from artifacts or one raw H5."""
    config, source = load_experiment_config(config_path)
    validate_pyfai_runtime(config["lineage"])
    if cohort not in {"all", "common", "full_npt256"}:
        raise ValueError("cohort must be all, common, or full_npt256.")
    if input_h5_path is not None and cohort != "full_npt256":
        raise ValueError("Raw H5 input supports only cohort='full_npt256'.")

    output_root = (
        Path(output_folder).expanduser().resolve()
        if output_folder is not None
        else resolve_path(config["output"]["folder"], source)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    selected = ["common", "full_npt256"] if cohort == "all" else [cohort]
    if input_h5_path is not None:
        generated_path = resolve_path(
            config["preprocessing"]["generated_npt256_artifact_path"],
            source,
        )
        create_research_npt256_artifact(
            input_h5_path,
            base_config_path=resolve_path(
                config["preprocessing"]["base_config_path"], source
            ),
            output_artifact_path=generated_path,
            lineage=config["lineage"],
        )
        config = copy.deepcopy(config)
        generated_pin = config["cohorts"]["full_npt256"]["npt256_artifact"]
        generated_pin["path"] = str(generated_path)
        generated_pin["sha256"] = file_sha256(generated_path)
        generated_pin["pipeline_fingerprint"] = joblib.load(generated_path).get(
            "pipeline_fingerprint"
        )

    results: dict[str, Any] = {}
    for cohort_name in selected:
        if not config["cohorts"][cohort_name]["enabled"]:
            continue
        datasets, lineage = _load_cohort_datasets(
            config,
            source,
            cohort_name=cohort_name,
            enforce_expected=input_h5_path is None,
            generated_mode=input_h5_path is not None,
        )
        cohort_output = output_root / cohort_name
        results[cohort_name] = run_cohort_experiment(
            datasets,
            config=config,
            cohort_name=cohort_name,
            output_folder=cohort_output,
            lineage=lineage,
        )
    if not results:
        raise ValueError("No enabled cohort mode was selected.")
    return results


def run_cohort_experiment(
    datasets: dict[int, pd.DataFrame],
    *,
    config: dict[str, Any],
    cohort_name: str,
    output_folder: str | Path | None = None,
    lineage: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate all encoders for one cohort on shared patient-safe folds."""
    model = config["model"]
    evaluation = config["evaluation"]
    if 256 not in datasets:
        raise ValueError("Every cohort requires an npt256 dataset.")
    contexts = {
        npt: build_dataset_context(df, model)
        for npt, df in datasets.items()
    }
    if cohort_name == "common":
        if 100 not in datasets:
            raise ValueError("Common cohort requires npt100 and npt256 datasets.")
        validate_common_cohort(datasets[100], datasets[256], contexts)

    specs = _cohort_specs(cohort_name, model["fpca_components"])
    canonical_context = contexts[256].reset_index(drop=True)
    split_pairs = _patient_split_pairs(
        mode="stratified_kfold",
        base_features=canonical_context,
        y_patients=canonical_context["label"].to_numpy(dtype=int),
        n_splits=int(evaluation["folds"]),
        n_repeats=int(evaluation["repeats"]),
        random_state=int(evaluation["random_seed"]),
    )
    split_metrics: list[dict[str, Any]] = []
    split_predictions: list[pd.DataFrame] = []
    pca_variance: list[dict[str, Any]] = []
    fold_basis: dict[str, dict[str, Any]] = {}
    train_all_models: dict[str, Any] = {}
    train_all_basis: list[pd.DataFrame] = []

    for spec in specs:
        df = datasets[spec.npt]
        context = contexts[spec.npt].reset_index(drop=True)
        _require_matching_case_order(canonical_context, context)
        LOGGER.info("%s: evaluating %s", cohort_name, spec.name)
        for split_id, (train_index, test_index) in enumerate(split_pairs):
            train_patients = set(
                canonical_context.iloc[train_index][model["group_column"]].astype(str)
            )
            test_patients = set(
                canonical_context.iloc[test_index][model["group_column"]].astype(str)
            )
            if train_patients.intersection(test_patients):
                raise RuntimeError("Patient leakage detected in outer split.")
            train_df = df[
                df[model["group_column"]].astype(str).isin(train_patients)
            ].copy()
            test_df = df[
                df[model["group_column"]].astype(str).isin(test_patients)
            ].copy()
            train_context = context[
                context[model["group_column"]].astype(str).isin(train_patients)
            ].copy()
            test_context = context[
                context[model["group_column"]].astype(str).isin(test_patients)
            ].copy()
            fitted = fit_split(
                train_df=train_df,
                test_df=test_df,
                train_context=train_context,
                test_context=test_context,
                spec=spec,
                model=model,
                target_sensitivity=float(evaluation["target_sensitivity"]),
                random_state=int(evaluation["random_seed"]) + split_id,
            )
            threshold = float(fitted["thresholds"]["threshold_target"])
            features = fitted["features"]
            score = fitted["score"]
            row = metric_row(
                features["label"].to_numpy(dtype=int),
                score,
                threshold=threshold,
            )
            split_metrics.append(
                {
                    "cohort": cohort_name,
                    "profile_encoder": spec.name,
                    "split_id": split_id,
                    **row,
                }
            )
            prediction = features[
                [
                    TARGET_CASE_ID,
                    model["group_column"],
                    "target_side",
                    "label",
                    "symmetry_available",
                ]
            ].copy()
            prediction.insert(0, "cohort", cohort_name)
            prediction.insert(1, "profile_encoder", spec.name)
            prediction.insert(2, "split_id", split_id)
            prediction["p_cancer"] = score
            prediction["threshold_target"] = threshold
            prediction["suggested_class"] = np.where(
                score >= threshold,
                "CANCER",
                "BENIGN",
            )
            split_predictions.append(prediction)
            record_pca(
                fitted["encoder"],
                scope="outer_fold",
                split_id=split_id,
                variance_rows=pca_variance,
                basis_store=fold_basis,
            )

        fitted_all = fit_train_all(
            df=df,
            context=context,
            spec=spec,
            model=model,
            target_sensitivity=float(evaluation["target_sensitivity"]),
            random_state=int(evaluation["random_seed"]),
        )
        train_all_models[spec.name] = fitted_all
        record_pca(
            fitted_all["profile_encoder"],
            scope="train_all",
            split_id=-1,
            variance_rows=pca_variance,
            basis_frames=train_all_basis,
        )

    metrics = pd.DataFrame(split_metrics)
    predictions = pd.concat(split_predictions, ignore_index=True)
    variance = pd.DataFrame(pca_variance)
    aggregate, repeat_averaged = summarize_results(
        metrics,
        predictions,
        train_all_models,
    )
    fold_manifest = build_fold_manifest(
        canonical_context,
        split_pairs,
        cohort=cohort_name,
        folds=int(evaluation["folds"]),
        group_column=model["group_column"],
    )
    paired_deltas, paired_delta_summary = paired_fold_deltas(
        metrics,
        cohort_name=cohort_name,
        folds=int(evaluation["folds"]),
    )
    result = {
        "contract": "aramina_fpca256_profile_encoder_results_v0_1",
        "clinical_stage": "research_only",
        "cohort": cohort_name,
        "dataset": dataset_summary(datasets, contexts),
        "lineage": lineage or {},
        "controlled_variables": {
            "integration_npt": sorted(datasets),
            "lr1_row_policy": model["lr1_row_policy"],
            "lr1_logreg_c": model["lr1_logreg_c"],
            "lr1_class_weight": model["class_weight"],
            "profile_aggregation": model["profile_aggregation"],
            "lr2_architecture": model["lr2_architecture"],
            "lr2_logreg_c": model["lr2_logreg_c"],
            "target_sensitivity": evaluation["target_sensitivity"],
            "evaluation": {
                "method": evaluation["method"],
                "folds": evaluation["folds"],
                "repeats": evaluation["repeats"],
                "random_seed": evaluation["random_seed"],
            },
        },
        "fpca_definition": (
            "scikit-learn PCA on one shared uniformly spaced q grid; "
            "discrete approximation to functional PCA; fitted inside each outer fold"
        ),
        "aggregate_summary": aggregate,
        "repeat_averaged_cross_fitted_predictions": repeat_averaged,
        "fold_metrics": metrics,
        "fold_predictions": predictions,
        "fold_manifest": fold_manifest,
        "paired_fold_deltas": paired_deltas,
        "paired_delta_summary": paired_delta_summary,
        "pca_explained_variance": variance,
        "train_all_models": train_all_models,
        "fold_pca_basis": fold_basis,
        "train_all_basis": (
            pd.concat(train_all_basis, ignore_index=True)
            if train_all_basis
            else pd.DataFrame()
        ),
    }
    if output_folder is not None:
        write_outputs(result, Path(output_folder), config=config)
    return result


def _cohort_specs(cohort_name: str, components: list[int]) -> list[ProfileSpec]:
    specs = []
    if cohort_name == "common":
        specs.append(ProfileSpec(name="raw100", npt=100, kind="raw"))
    specs.append(ProfileSpec(name="raw256", npt=256, kind="raw"))
    specs.extend(
        ProfileSpec(
            name=f"fpca256_{n_components}",
            npt=256,
            kind="fpca",
            n_components=int(n_components),
        )
        for n_components in components
    )
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aramina-fpca256-profile-encoder")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--cohort",
        choices=("all", "common", "full_npt256"),
        default="all",
    )
    parser.add_argument("--input-h5", type=Path)
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run_experiment_from_config(
        args.config,
        cohort=args.cohort,
        input_h5_path=args.input_h5,
        output_folder=args.output_folder,
    )
    for name, result in results.items():
        print(f"\n{name}")
        print(result["aggregate_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
