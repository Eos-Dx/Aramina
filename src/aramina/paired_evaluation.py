"""Reproducible paired raw100, FPCA30, and additive evaluation."""

from __future__ import annotations

import argparse
from importlib.metadata import version as distribution_version
import logging
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd
import yaml

from .paired_cohort import (
    construct_common_cohort,
    file_sha256,
    load_dataframe_artifact,
    measurement_manifest,
    model_columns,
    ordered_context,
    require_common_source_h5,
)
from .paired_contract import (
    ADDITIVE_MODEL,
    ADDITIVE_REGULARIZATION,
    ADDITIVE_SOURCE_COMMIT,
    ADDITIVE_SOURCE_RECORD,
    CONTRACT,
    FPCA30_MODEL,
    PAIRED_COMPARISONS,
    RAW100_MODEL,
)
from .paired_metrics import (
    assert_shared_evaluation_cases,
    paired_delta_summary,
    paired_fold_deltas,
)
from .paired_models import (
    ProfileSpec,
    evaluate_additive_comparator,
    evaluate_product_comparator,
    outer_fold_manifest,
    patient_ids,
    strict_split_pairs,
)
from .patient_features import TARGET_CASE_ID
from .training_evaluation import _summarize_patient_model_metrics


LOGGER = logging.getLogger(__name__)


def run_paired_evaluation(
    raw100: pd.DataFrame,
    fpca256: pd.DataFrame,
    output_dir: str | Path,
    *,
    n_splits: int = 5,
    n_repeats: int = 20,
    inner_lr1_splits: int = 5,
    meta_splits: int = 4,
    random_state: int = 42,
    target_sensitivity: float = 0.95,
    bootstrap_samples: int = 2_000,
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all three models on one shared patient-safe outer manifest."""
    if float(target_sensitivity) != 0.95:
        raise ValueError("Paired evaluation target_sensitivity must remain 0.95.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_common, fpca_common, case_manifest = construct_common_cohort(
        raw100, fpca256
    )
    measurement_rows = measurement_manifest(raw100, fpca256)
    raw_context = ordered_context(raw_common, case_manifest)
    fpca_context = ordered_context(fpca_common, case_manifest)
    split_pairs = strict_split_pairs(
        raw_context,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        description="outer",
    )
    LOGGER.info(
        "Paired cohort: %d measurements, %d patients, %d target cases, %d outer splits",
        len(raw_common),
        raw_context["patientId"].nunique(),
        len(case_manifest),
        len(split_pairs),
    )
    outer_manifest = outer_fold_manifest(
        raw_context,
        split_pairs,
        n_splits=n_splits,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    nested_manifest_frames: list[pd.DataFrame] = []
    model = model_columns()
    raw_spec = ProfileSpec(RAW100_MODEL, 100, "raw")
    fpca_spec = ProfileSpec(FPCA30_MODEL, 256, "fpca", 30)

    for split_id, (train_index, test_index) in enumerate(split_pairs):
        LOGGER.info("Outer split %d/%d", split_id + 1, len(split_pairs))
        train_ids = patient_ids(raw_context, train_index)
        test_ids = patient_ids(raw_context, test_index)
        if train_ids.intersection(test_ids):
            raise RuntimeError("Patient leakage detected in shared outer split.")
        split_seed = int(random_state) + split_id * 100_000
        raw_result = evaluate_product_comparator(
            raw_common,
            raw_context,
            train_ids=train_ids,
            test_ids=test_ids,
            spec=raw_spec,
            model=model,
            split_id=split_id,
            n_splits=n_splits,
            random_state=split_seed,
            target_sensitivity=target_sensitivity,
        )
        fpca_result = evaluate_product_comparator(
            fpca_common,
            fpca_context,
            train_ids=train_ids,
            test_ids=test_ids,
            spec=fpca_spec,
            model=model,
            split_id=split_id,
            n_splits=n_splits,
            random_state=split_seed,
            target_sensitivity=target_sensitivity,
        )
        additive_result = evaluate_additive_comparator(
            fpca_common,
            fpca_context,
            train_ids=train_ids,
            test_ids=test_ids,
            test_features=fpca_result["test_features"],
            model=model,
            split_id=split_id,
            n_splits=n_splits,
            inner_lr1_splits=inner_lr1_splits,
            meta_splits=meta_splits,
            random_state=split_seed + 10_000,
            target_sensitivity=target_sensitivity,
        )
        for result in (raw_result, fpca_result, additive_result):
            metric_rows.append(result["metrics"])
            prediction_frames.append(result["predictions"])
            threshold_frames.append(result["threshold_scores"])
        nested_manifest_frames.append(additive_result["nested_manifest"])

    fold_metrics = pd.DataFrame(metric_rows).sort_values(
        ["split_id", "model_name"], kind="stable"
    )
    fold_predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["split_id", "model_name", TARGET_CASE_ID], kind="stable"
    )
    threshold_scores = pd.concat(threshold_frames, ignore_index=True).sort_values(
        ["split_id", "model_name", TARGET_CASE_ID], kind="stable"
    )
    fold_manifest = pd.concat(
        [outer_manifest, *nested_manifest_frames], ignore_index=True
    ).sort_values(
        ["split_id", "level", "parent_fold_id", "fold_id", "role", TARGET_CASE_ID],
        kind="stable",
    )
    assert_shared_evaluation_cases(
        fold_predictions,
        outer_manifest,
        case_manifest,
    )
    summary = _summarize_patient_model_metrics(
        fold_metrics,
        fold_predictions,
        random_state=random_state + 7_000_000,
        bootstrap_samples=bootstrap_samples,
    )
    fold_deltas = paired_fold_deltas(fold_metrics)
    delta_summary = paired_delta_summary(
        fold_deltas,
        fold_predictions,
        random_state=random_state + 8_000_000,
        bootstrap_samples=bootstrap_samples,
    )

    frames = {
        "measurement_manifest.csv": measurement_rows,
        "case_manifest.csv": case_manifest,
        "fold_manifest.csv": fold_manifest,
        "fold_metrics.csv": fold_metrics,
        "fold_predictions.csv": fold_predictions,
        "threshold_scores.csv": threshold_scores,
        "summary.csv": summary,
        "paired_fold_deltas.csv": fold_deltas,
        "paired_delta_summary.csv": delta_summary,
    }
    for filename, frame in frames.items():
        frame.to_csv(output / filename, index=False)

    metadata = _run_metadata(
        raw100=raw100,
        fpca256=fpca256,
        raw_common=raw_common,
        case_manifest=case_manifest,
        controls={
            "outer_splits": int(n_splits),
            "outer_repeats": int(n_repeats),
            "inner_lr1_splits": int(inner_lr1_splits),
            "meta_splits": int(meta_splits),
            "random_state": int(random_state),
            "target_sensitivity": float(target_sensitivity),
            "bootstrap_samples": int(bootstrap_samples),
        },
        input_metadata=input_metadata or {},
        output=output,
    )
    (output / "run_metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    return {
        "metadata": metadata,
        "measurement_manifest": measurement_rows,
        "case_manifest": case_manifest,
        "fold_manifest": fold_manifest,
        "fold_metrics": fold_metrics,
        "fold_predictions": fold_predictions,
        "threshold_scores": threshold_scores,
        "summary": summary,
        "paired_fold_deltas": fold_deltas,
        "paired_delta_summary": delta_summary,
    }


def run_from_artifacts(
    raw100_path: str | Path,
    fpca256_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load both source artifacts and run the paired comparison."""
    raw100, raw_metadata = load_dataframe_artifact(raw100_path)
    fpca256, fpca_metadata = load_dataframe_artifact(fpca256_path)
    require_common_source_h5(raw_metadata, fpca_metadata)
    return run_paired_evaluation(
        raw100,
        fpca256,
        output_dir,
        input_metadata={"raw100": raw_metadata, "fpca256": fpca_metadata},
        **kwargs,
    )


def _run_metadata(
    *,
    raw100: pd.DataFrame,
    fpca256: pd.DataFrame,
    raw_common: pd.DataFrame,
    case_manifest: pd.DataFrame,
    controls: dict[str, Any],
    input_metadata: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    return {
        "contract": CONTRACT,
        "clinical_stage": "research_only",
        "intended_use": (
            "Breast cancer decision support research; requires radiologist review."
        ),
        "validation_status": "paired_internal_patient_safe_evaluation",
        "not_for_autonomous_diagnosis": True,
        "source": {
            "git_sha": _git_output(repository, "rev-parse", "HEAD"),
            "git_worktree_dirty": bool(
                _git_output(repository, "status", "--porcelain")
            ),
            "inputs": input_metadata,
            "runtime_versions": {
                name: distribution_version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
            },
        },
        "cohort": {
            "raw100_input_measurements": int(len(raw100)),
            "fpca256_input_measurements": int(len(fpca256)),
            "common_measurements": int(len(raw_common)),
            "common_patients": int(case_manifest["patientId"].nunique()),
            "common_target_cases": int(len(case_manifest)),
            "common_cancer_target_cases": int((case_manifest["label"] == 1).sum()),
            "common_benign_target_cases": int((case_manifest["label"] == 0).sum()),
        },
        "controls": controls,
        "paired_comparisons": [
            {
                "comparison": name,
                "candidate_model": candidate,
                "reference_model": reference,
            }
            for name, candidate, reference in PAIRED_COMPARISONS
        ],
        "models": {
            RAW100_MODEL: {
                "profile": "100-bin raw profile",
                "fit": "outer-train same-data LR1 to LR2",
                "threshold_scores": "outer-train fitted LR1 and fitted LR2",
            },
            FPCA30_MODEL: {
                "profile": "256-bin profile to fold-local PCA30",
                "fit": "outer-train same-data LR1 to LR2",
                "threshold_scores": "outer-train fitted LR1 and fitted LR2",
            },
            ADDITIVE_MODEL: {
                "architecture": (
                    "fold-local FPCA30 profile logit + age + gated SK Core4 symmetry"
                ),
                "fit": "outer-train patient-safe FPCA30/LR1 OOF additive meta fit",
                "threshold_scores": "outer-train nested full-chain meta OOF",
                "regularization": ADDITIVE_REGULARIZATION,
                "regularization_source_commit": ADDITIVE_SOURCE_COMMIT,
                "regularization_source_record": ADDITIVE_SOURCE_RECORD,
            },
        },
        "manifests": {
            name: {"path": name, "sha256": file_sha256(output / name)}
            for name in (
                "measurement_manifest.csv",
                "case_manifest.csv",
                "fold_manifest.csv",
            )
        },
        "limitations": [
            "Retrospective internal comparison; not independent clinical validation.",
            (
                "Fixed additive regularization was selected for a prior raw100-based "
                "experiment on an overlapping T100 cohort and is transferred to "
                "FPCA30 without retuning."
            ),
            (
                "Repeated folds overlap; bootstrap intervals are descriptive for "
                "repeat-averaged OOF predictions."
            ),
            (
                "Product same-data FPCA30/LR1-to-LR2 fitting and additive "
                "FPCA30/LR1 OOF meta fitting are intentionally different fitted "
                "procedures."
            ),
        ],
    }


def _git_output(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    """Run paired comparison from two preprocessing artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw100-input", type=Path, required=True)
    parser.add_argument("--fpca256-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--inner-lr1-splits", type=int, default=5)
    parser.add_argument("--meta-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_from_artifacts(
        args.raw100_input,
        args.fpca256_input,
        args.output_dir,
        n_splits=args.folds,
        n_repeats=args.repeats,
        inner_lr1_splits=args.inner_lr1_splits,
        meta_splits=args.meta_splits,
        random_state=args.random_state,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(result["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
