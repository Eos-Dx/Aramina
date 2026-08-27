"""Leakage-safe comparison of PCA-denoised and raw Aramina 0.2.14 profiles."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from xrd_preprocessing import (
    SmoothedPCAProfileTransformer,
    SparsePCAProfileTransformer,
)

from ..config_paths import resolve_config_path
from ..model_utils import compute_binary_thresholds, profile_matrix
from ..model_metrics import binary_metric_values
from ..patient_features import (
    build_patient_prediction_feature_row,
    empty_lr1_scores,
    patient_feature_table,
)
from ..target_breast_model import GatedSymmetryLogistic
from ..training_config import PRODUCT_MODEL_NAME, resolve_model_definition
from ..training_evaluation import (
    _default_routes,
    _fit_split_feature_tables,
    _patient_metric_row,
    _patient_prediction_frame,
    _patient_split_pairs,
    _split_assignment_frame,
    _summarize_patient_model_metrics,
    _validate_patient_split_assignments,
)
from ..training_model import _fit_patient_model_input, _fit_target_breast_model


EXPERIMENT_CONTRACT = "aramina_pca_denoising_experiment_v0_1"
SUPPORTED_METHOD_TYPES = {"raw", "smoothed_pca", "sparse_pca"}


def run_pca_denoising_experiment(config_path: str | Path) -> dict[str, Any]:
    """Run fixed-fold comparison without changing the product model contract."""
    source = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    _validate_experiment_config(config)
    input_path = resolve_config_path(
        config["input"]["dataframe_joblib_path"],
        source,
    )
    output_root = resolve_config_path(config["output"]["folder"], source)
    dataframe = _load_dataframe(input_path)
    run_folder = _new_run_folder(output_root)
    run_folder.mkdir(parents=True, exist_ok=False)

    result = evaluate_pca_denoising_methods(
        dataframe,
        methods=config["methods"],
        evaluation=config["evaluation"],
        train_on_all=bool(config["run"]["train_on_all"]),
        artifact_folder=run_folder / "models",
    )
    held_out_path = resolve_config_path(
        config["input"]["held_out_artifact_path"],
        source,
    )
    held_out = score_pca_denoising_models_on_held_out(
        held_out_path,
        methods=config["methods"],
        artifact_folder=run_folder / "models",
    )
    result.update(held_out)
    result["split_metrics"].to_csv(run_folder / "split_metrics.csv", index=False)
    result["predictions"].to_csv(run_folder / "predictions.csv", index=False)
    result["split_assignments"].to_csv(
        run_folder / "patient_split_assignments.csv",
        index=False,
    )
    result["profile_fidelity"].to_csv(
        run_folder / "profile_fidelity.csv",
        index=False,
    )
    result["summary"].to_csv(run_folder / "summary.csv", index=False)
    result["train_on_all_metrics"].to_csv(
        run_folder / "train_on_all_metrics.csv",
        index=False,
    )
    result["held_out_metrics"].to_csv(
        run_folder / "held_out_metrics.csv",
        index=False,
    )
    result["held_out_predictions"].to_csv(
        run_folder / "held_out_predictions.csv",
        index=False,
    )
    (run_folder / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "contract": EXPERIMENT_CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "patient_safe_internal_evaluation",
        "clinical_stage": "research draft",
        "input_path": str(input_path),
        "input_sha256": _file_sha256(input_path),
        "dataset_id": str(config["input"]["dataset_id"]),
        "dataset_fingerprint": str(config["input"]["dataset_fingerprint"]),
        "source_h5_sha256": str(config["input"]["source_h5_sha256"]),
        "held_out_path": str(held_out_path),
        "held_out_sha256": _file_sha256(held_out_path),
        "measurements": int(len(dataframe)),
        "patients": int(dataframe["patientId"].astype(str).nunique()),
        "methods": [method["name"] for method in config["methods"]],
        "folds": int(config["evaluation"]["folds"]),
        "repeats": int(config["evaluation"]["repeats"]),
        "product_model": PRODUCT_MODEL_NAME,
        "product_model_version": "0.2.14-beta",
        "product_contract_changed": False,
        "held_out_evidence_status": (
            "quality_challenging_demonstration_subset_not_formal_validation"
        ),
        "held_out_target_cases": int(len(result["held_out_predictions"])),
    }
    (run_folder / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_folder / "README.md").write_text(
        _render_result_markdown(
            result["summary"],
            result["train_on_all_metrics"],
            result["held_out_metrics"],
        ),
        encoding="utf-8",
    )
    return {"run_folder": run_folder, "manifest": manifest, **result}


def score_pca_denoising_models_on_held_out(
    artifact_path: str | Path,
    *,
    methods: list[dict[str, Any]],
    artifact_folder: Path,
) -> dict[str, pd.DataFrame]:
    """Score the fixed 22-case T130 demonstration subset once per method."""
    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("dataframe"), pd.DataFrame
    ):
        raise TypeError("Held-out artifact must contain a DataFrame.")
    cases = payload.get("case_manifest")
    if not isinstance(cases, list) or not cases:
        raise TypeError("Held-out artifact must contain a nonempty case_manifest.")
    dataframe = payload["dataframe"]
    predictions = []
    metrics = []
    for method in methods:
        model_payload = joblib.load(artifact_folder / f"{method['name']}.joblib")
        transformer = model_payload["denoiser"]
        model_info = model_payload["model"]
        transformed = (
            dataframe.copy()
            if transformer is None
            else transformer.transform(dataframe)
        )
        profile_column = str(model_info["profile_encoder"]["model_profile"])
        method_rows = []
        for case in cases:
            feature = build_patient_prediction_feature_row(
                transformed,
                model_info,
                patient_id=str(case["patient_id"]),
                target_side=str(case["target_side"]),
                profile_column=profile_column,
            )
            score = float(model_info["final_model"].predict_proba(feature)[0, 1])
            threshold = float(model_info["thresholds"]["threshold_target"])
            label = 1 if case["reference_label"] == "CANCER" else 0
            method_rows.append(
                {
                    "model_name": method["name"],
                    "patient_id": str(case["patient_id"]),
                    "target_side": str(case["target_side"]),
                    "reference_label": str(case["reference_label"]),
                    "label": label,
                    "p_cancer": score,
                    "decision_threshold": threshold,
                    "predicted_label": "CANCER" if score >= threshold else "BENIGN",
                }
            )
        method_frame = pd.DataFrame(method_rows)
        predictions.append(method_frame)
        y = method_frame["label"].to_numpy(dtype=int)
        score = method_frame["p_cancer"].to_numpy(dtype=float)
        threshold = method_frame["decision_threshold"].to_numpy(dtype=float)
        pred = (score >= threshold).astype(int)
        values = binary_metric_values(y, score, threshold)
        metrics.append(
            {
                "model_name": method["name"],
                "evidence_status": (
                    "quality_challenging_demonstration_subset_not_formal_validation"
                ),
                "patients": int(method_frame["patient_id"].nunique()),
                "target_cases": int(len(method_frame)),
                "cancer_target_cases": int((y == 1).sum()),
                "benign_target_cases": int((y == 0).sum()),
                **values,
                "true_positives": int(((y == 1) & (pred == 1)).sum()),
                "true_negatives": int(((y == 0) & (pred == 0)).sum()),
                "false_negatives": int(((y == 1) & (pred == 0)).sum()),
                "false_positives": int(((y == 0) & (pred == 1)).sum()),
            }
        )
    return {
        "held_out_metrics": pd.DataFrame(metrics),
        "held_out_predictions": pd.concat(predictions, ignore_index=True),
    }


def evaluate_pca_denoising_methods(
    dataframe: pd.DataFrame,
    *,
    methods: list[dict[str, Any]],
    evaluation: dict[str, Any],
    train_on_all: bool,
    artifact_folder: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate every method on identical patient-safe splits."""
    _validate_method_specs(methods)
    model = resolve_model_definition(PRODUCT_MODEL_NAME)
    model_config = model["model"]
    source_profile = str(model_config["profile_column"])
    label_column = str(model_config["label_column"])
    group_column = str(model_config["group_column"])
    specimen_column = str(model_config["specimen_column"])
    side_column = str(model_config["side_column"])
    q_column = str(model_config["q_column"])
    age_column = str(model_config["age_column"])
    biopsy_column = str(model_config["biopsy_column"])
    lr1_row_policy = str(model_config["lr1_row_policy"])
    lr1_logreg_c = float(model_config["lr1_logreg_c"])
    lr2_logreg_c = float(model_config["lr2_logreg_c"])
    target_sensitivity = float(model["target_sensitivity"])
    random_state = int(evaluation["random_seed"])
    n_splits = int(evaluation["folds"])
    n_repeats = int(evaluation["repeats"])

    base_features = patient_feature_table(
        dataframe,
        empty_lr1_scores(
            dataframe,
            group_column=group_column,
            side_column=side_column,
            label_column=label_column,
            biopsy_column=biopsy_column,
        ),
        profile_column=source_profile,
        label_column=label_column,
        group_column=group_column,
        specimen_column=specimen_column,
        side_column=side_column,
        q_column=q_column,
        age_column=age_column,
        biopsy_column=biopsy_column,
    )
    split_pairs = _patient_split_pairs(
        mode="stratified_kfold",
        base_features=base_features,
        y_patients=base_features["label"].to_numpy(dtype=int),
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    fidelity: list[dict[str, Any]] = []
    assignments: list[pd.DataFrame] = []
    expected_patients = set(base_features["patientId"].astype(str))

    for split_id, (train_idx, test_idx) in enumerate(split_pairs):
        train_patients = set(base_features.iloc[train_idx]["patientId"].astype(str))
        test_patients = set(base_features.iloc[test_idx]["patientId"].astype(str))
        if train_patients.intersection(test_patients):
            raise RuntimeError("Patient leakage detected before denoising.")
        assignments.append(
            _split_assignment_frame(
                split_id=split_id,
                n_splits=n_splits,
                train_patients=train_patients,
                test_patients=test_patients,
            )
        )
        raw_train = dataframe.loc[
            dataframe[group_column].astype(str).isin(train_patients)
        ].copy()
        raw_test = dataframe.loc[
            dataframe[group_column].astype(str).isin(test_patients)
        ].copy()
        for method_index, method in enumerate(methods):
            seed = random_state + split_id
            train_df, test_df, active_profile, _transformer = _transform_split(
                raw_train,
                raw_test,
                method=method,
                source_profile=source_profile,
                q_column=q_column,
                random_state=seed,
            )
            fidelity.append(
                _profile_fidelity_row(
                    method["name"],
                    split_id,
                    raw_test,
                    test_df,
                    source_profile=source_profile,
                    transformed_profile=active_profile,
                )
            )
            train_features, test_features = _fit_split_feature_tables(
                train_df,
                test_df,
                profile_column=active_profile,
                label_column=label_column,
                group_column=group_column,
                specimen_column=specimen_column,
                side_column=side_column,
                q_column=q_column,
                age_column=age_column,
                biopsy_column=biopsy_column,
                lr1_row_policy=lr1_row_policy,
                lr1_logreg_c=lr1_logreg_c,
                random_state=seed,
            )
            final_model = GatedSymmetryLogistic(
                logreg_c=lr2_logreg_c,
                random_state=seed,
            ).fit(train_features, train_features["label"].to_numpy(dtype=int))
            train_score = final_model.predict_proba(train_features)[:, 1]
            test_score = final_model.predict_proba(test_features)[:, 1]
            thresholds = compute_binary_thresholds(
                train_features["label"].to_numpy(dtype=int),
                train_score,
                target_sensitivity=target_sensitivity,
            )
            thresholds["selected_lr1_c"] = lr1_logreg_c
            thresholds["selected_lr2_c"] = lr2_logreg_c
            decision_thresholds = np.full(
                len(test_features),
                thresholds["threshold_target"],
                dtype=float,
            )
            metrics.append(
                _patient_metric_row(
                    str(method["name"]),
                    split_id,
                    train_features,
                    test_features,
                    test_score,
                    thresholds,
                    decision_thresholds,
                    evaluation_mode="stratified_kfold",
                )
            )
            predictions.append(
                _patient_prediction_frame(
                    str(method["name"]),
                    split_id,
                    test_features,
                    test_score,
                    thresholds,
                    _default_routes(test_features),
                    decision_thresholds,
                    evaluation_mode="stratified_kfold",
                )
            )

    assignment_frame = pd.concat(assignments, ignore_index=True)
    _validate_patient_split_assignments(
        assignment_frame,
        expected_patients=expected_patients,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    summary = _summarize_patient_model_metrics(
        metric_frame,
        prediction_frame,
        random_state=random_state,
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
    )
    threshold_summary = (
        metric_frame.groupby("model_name", as_index=False)
        .agg(
            threshold_target_mean=("threshold_target", "mean"),
            threshold_target_std=("threshold_target", lambda values: values.std(ddof=0)),
        )
    )
    summary = summary.merge(threshold_summary, on="model_name", how="left")
    train_on_all_metrics = pd.DataFrame()
    if train_on_all:
        train_on_all_metrics = _fit_all_methods(
            dataframe,
            methods=methods,
            model_config=model_config,
            target_sensitivity=target_sensitivity,
            random_state=random_state,
            artifact_folder=artifact_folder,
        )
    return {
        "split_metrics": metric_frame,
        "predictions": prediction_frame,
        "split_assignments": assignment_frame,
        "profile_fidelity": pd.DataFrame(fidelity),
        "summary": summary,
        "train_on_all_metrics": train_on_all_metrics,
    }


def _transform_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    method: dict[str, Any],
    source_profile: str,
    q_column: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str, Any]:
    transformer = _build_transformer(
        method,
        source_profile=source_profile,
        q_column=q_column,
        random_state=random_state,
    )
    if transformer is None:
        return train_df.copy(), test_df.copy(), source_profile, None
    source_train = profile_matrix(train_df, source_profile).copy()
    source_test = profile_matrix(test_df, source_profile).copy()
    transformed_train = transformer.fit_transform(train_df)
    transformed_test = transformer.transform(test_df)
    np.testing.assert_array_equal(profile_matrix(train_df, source_profile), source_train)
    np.testing.assert_array_equal(profile_matrix(test_df, source_profile), source_test)
    return transformed_train, transformed_test, transformer.output_column, transformer


def _build_transformer(
    method: dict[str, Any],
    *,
    source_profile: str,
    q_column: str,
    random_state: int,
) -> Any:
    method_type = str(method["type"])
    params = dict(method.get("params", {}))
    if method_type == "raw":
        return None
    output_column = f"{source_profile}_{method['name']}"
    shared = {
        "q_column": q_column,
        "profile_column": source_profile,
        "output_column": output_column,
    }
    if method_type == "smoothed_pca":
        return SmoothedPCAProfileTransformer(**params, **shared)
    if method_type == "sparse_pca":
        params.setdefault("random_state", random_state)
        return SparsePCAProfileTransformer(**params, **shared)
    raise ValueError(f"Unsupported denoising method: {method_type!r}")


def _profile_fidelity_row(
    method_name: str,
    split_id: int,
    raw_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    *,
    source_profile: str,
    transformed_profile: str,
) -> dict[str, Any]:
    source = profile_matrix(raw_df, source_profile)
    transformed = profile_matrix(transformed_df, transformed_profile)
    difference = transformed - source
    source_roughness = np.sqrt(np.mean(np.diff(source, n=2, axis=1) ** 2))
    transformed_roughness = np.sqrt(
        np.mean(np.diff(transformed, n=2, axis=1) ** 2)
    )
    source_norm = float(np.linalg.norm(source))
    return {
        "model_name": method_name,
        "split_id": int(split_id),
        "test_measurements": int(len(raw_df)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2_error": float(np.linalg.norm(difference) / source_norm)
        if source_norm
        else float("nan"),
        "roughness_ratio": float(transformed_roughness / source_roughness)
        if source_roughness
        else float("nan"),
        "negative_value_fraction": float(np.mean(transformed < 0.0)),
    }


def _fit_all_methods(
    dataframe: pd.DataFrame,
    *,
    methods: list[dict[str, Any]],
    model_config: dict[str, Any],
    target_sensitivity: float,
    random_state: int,
    artifact_folder: Path | None,
) -> pd.DataFrame:
    source_profile = str(model_config["profile_column"])
    q_column = str(model_config["q_column"])
    if artifact_folder is not None:
        artifact_folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for method_index, method in enumerate(methods):
        transformer = _build_transformer(
            method,
            source_profile=source_profile,
            q_column=q_column,
            random_state=random_state + method_index,
        )
        if transformer is None:
            transformed = dataframe.copy()
            active_profile = source_profile
        else:
            source_values = profile_matrix(dataframe, source_profile).copy()
            transformed = transformer.fit_transform(dataframe)
            np.testing.assert_array_equal(
                profile_matrix(dataframe, source_profile),
                source_values,
            )
            active_profile = transformer.output_column
        feature_table, lr1_rows = _fit_patient_model_input(
            transformed,
            profile_column=active_profile,
            label_column=str(model_config["label_column"]),
            group_column=str(model_config["group_column"]),
            specimen_column=str(model_config["specimen_column"]),
            side_column=str(model_config["side_column"]),
            q_column=q_column,
            age_column=str(model_config["age_column"]),
            biopsy_column=str(model_config["biopsy_column"]),
            lr1_row_policy=str(model_config["lr1_row_policy"]),
            lr1_logreg_c=float(model_config["lr1_logreg_c"]),
            random_state=random_state + method_index,
        )
        fitted = _fit_target_breast_model(
            feature_table,
            lr1_rows,
            profile_column=active_profile,
            label_column=str(model_config["label_column"]),
            lr1_logreg_c=float(model_config["lr1_logreg_c"]),
            lr2_logreg_c=float(model_config["lr2_logreg_c"]),
            random_state=random_state + method_index,
            target_sensitivity=target_sensitivity,
        )
        fitted["profile_encoder"] = {
            "type": str(method["type"]),
            "source_profile": source_profile,
            "model_profile": active_profile,
            "fit_scope": "all_training_patient_measurements",
            "parameters": dict(method.get("params", {})),
        }
        metrics = dict(fitted["final_fit_training_metrics"])
        row = {"model_name": method["name"], **metrics}
        if artifact_folder is not None:
            artifact_path = artifact_folder / f"{method['name']}.joblib"
            joblib.dump(
                {
                    "contract": "aramina_pca_denoising_model_v0_1",
                    "evidence_status": "in_sample_not_independent",
                    "method": method,
                    "denoiser": transformer,
                    "model": fitted,
                },
                artifact_path,
            )
            row["artifact_path"] = str(artifact_path)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_dataframe(path: Path) -> pd.DataFrame:
    payload = joblib.load(path)
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("dataframe"), pd.DataFrame):
        return payload["dataframe"]
    raise TypeError("Experiment input must be a DataFrame or preprocessing artifact.")


def _validate_experiment_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise TypeError("Experiment config must be a mapping.")
    required = {"contract", "model", "run", "input", "output", "evaluation", "methods"}
    if set(config) != required:
        raise ValueError(
            f"Experiment config fields must be exactly {sorted(required)}."
        )
    if config["contract"] != EXPERIMENT_CONTRACT:
        raise ValueError(f"Unsupported experiment contract: {config['contract']!r}")
    if config["model"] != {
        "name": PRODUCT_MODEL_NAME,
        "version": "0.2.14-beta",
    }:
        raise ValueError("Experiment is fixed to Aramina 0.2.14-beta.")
    if config["run"] != {"train_on_all": True}:
        raise ValueError("run.train_on_all must be true for held-out scoring.")
    expected_input = {
        "dataframe_joblib_path",
        "held_out_artifact_path",
        "dataset_id",
        "dataset_fingerprint",
        "source_h5_sha256",
    }
    if set(config["input"]) != expected_input:
        raise ValueError(f"input fields must be exactly {sorted(expected_input)}.")
    for key in expected_input:
        value = config["input"][key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"input.{key} must be a nonempty string.")
    if set(config["output"]) != {"folder"}:
        raise ValueError("output must contain folder only.")
    evaluation = config["evaluation"]
    expected_evaluation = {
        "method",
        "folds",
        "repeats",
        "random_seed",
        "bootstrap_samples",
    }
    if not isinstance(evaluation, dict) or set(evaluation) != expected_evaluation:
        raise ValueError(
            f"evaluation fields must be exactly {sorted(expected_evaluation)}."
        )
    if evaluation["method"] != "repeated_stratified_kfold":
        raise ValueError("evaluation.method must be repeated_stratified_kfold.")
    for key, minimum in (
        ("folds", 2),
        ("repeats", 1),
        ("random_seed", 0),
        ("bootstrap_samples", 0),
    ):
        value = evaluation[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"evaluation.{key} must be an integer >= {minimum}.")
    _validate_method_specs(config["methods"])


def _validate_method_specs(methods: Any) -> None:
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a nonempty list.")
    names = []
    for method in methods:
        if not isinstance(method, dict) or set(method) != {"name", "type", "params"}:
            raise ValueError("Each method requires name, type, and params only.")
        if not isinstance(method["name"], str) or not method["name"]:
            raise ValueError("Method name must be nonempty.")
        if re.fullmatch(r"[a-z0-9_]+", method["name"]) is None:
            raise ValueError("Method name must contain lowercase letters, digits, or underscores.")
        if method["type"] not in SUPPORTED_METHOD_TYPES:
            raise ValueError(f"Unsupported method type: {method['type']!r}")
        if not isinstance(method["params"], dict):
            raise TypeError("Method params must be a mapping.")
        if method["type"] == "raw" and method["params"]:
            raise ValueError("Raw method cannot define parameters.")
        names.append(method["name"])
    if len(set(names)) != len(names):
        raise ValueError("Method names must be unique.")


def _new_run_folder(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / f"pca_denoising_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"pca_denoising_{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_result_markdown(
    summary: pd.DataFrame,
    train_on_all: pd.DataFrame,
    held_out: pd.DataFrame,
) -> str:
    columns = [
        "model_name",
        "roc_auc_mean",
        "sensitivity_target_mean",
        "specificity_target_mean",
        "balanced_accuracy_target_mean",
        "threshold_target_mean",
    ]
    header = (
        "# Aramina 0.2.14 PCA denoising experiment\n\n"
        "Research-draft decision-support experiment. Denoisers were fitted only "
        "on training-patient measurements inside each patient-safe fold. Product "
        "preprocessing, labels, LR1/LR2 regularization, target-sensitivity threshold "
        "selection, and report contracts were not changed.\n\n"
        "## Patient-safe repeated cross-validation\n\n"
    )
    text = header + summary.loc[:, columns].to_markdown(index=False, floatfmt=".5f")
    if not train_on_all.empty:
        text += (
            "\n\n## Train on all\n\n"
            "In-sample values below are descriptive and are not independent "
            "validation.\n\n"
            + train_on_all.loc[
                :,
                [
                    "model_name",
                    "roc_auc",
                    "sensitivity",
                    "specificity",
                    "balanced_accuracy",
                    "decision_threshold",
                ],
            ].to_markdown(index=False, floatfmt=".5f")
        )
    if not held_out.empty:
        text += (
            "\n\n## T130 quality-challenging held-out subset\n\n"
            "This 17-patient, 22-case demonstration subset was not used to fit "
            "the models. Its small size and quality-based selection do not make "
            "it a formal independent validation cohort.\n\n"
            + held_out.loc[
                :,
                [
                    "model_name",
                    "roc_auc",
                    "sensitivity",
                    "specificity",
                    "balanced_accuracy",
                    "false_negatives",
                    "false_positives",
                ],
            ].to_markdown(index=False, floatfmt=".5f")
        )
    return text + "\n"


__all__ = [
    "EXPERIMENT_CONTRACT",
    "evaluate_pca_denoising_methods",
    "run_pca_denoising_experiment",
    "score_pca_denoising_models_on_held_out",
]
