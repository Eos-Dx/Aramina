"""Top-level, fail-closed polar-harmonic ablation orchestration.

The core polar-basis experiment evaluates one angular resolution at a time.
This module creates one controlled child configuration per resolution, verifies
that their held-out data are identical, and only then performs paired analysis.
It is research-only and never changes product models, reports, or thresholds.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config_paths import resolve_config_path
from ..mlflow_tracking import MlflowRun
from ..runtime_identity import file_sha256
from .measurement_uncertainty import _tracking_uri
from .polar_basis_compression import run_polar_basis_compression_from_config
from .polar_harmonic_statistics import (
    PolarHarmonicStatisticsConfig,
    analyze_polar_harmonic_runs,
)


CONTRACT = "aramina_polar_harmonic_ablation_v0_1"
CORE_CONTRACT = "aramina_polar_basis_compression_v0_1"
_ALLOWED_N_CHI = (12, 18, 36, 72)
_MODE_SETS = {
    "A0": [0],
    "A0_A2": [0, 2],
    "A0_A2_A4": [0, 2, 4],
}
_CORE_MODE_SETS = {
    "A0": "A0",
    "A0_A2": "A0+A2",
    "A0_A2_A4": "A0+A2+A4",
}
_CORE_REQUIRED_ARTIFACTS = (
    "cohort_manifest.csv",
    "fold_manifest.csv",
    "fold_metrics.csv",
    "lineage.json",
    "polar_cake_manifest.csv",
    "predictions.csv",
    "raw100_fold_metrics.csv",
    "raw100_predictions.csv",
    "run_manifest.json",
)
_PARENT_REQUIRED_ARTIFACTS = (
    "effective_experiment_config.yaml",
    "child_index.csv",
    "combined_fold_manifest.csv",
    "combined_fold_metrics.csv",
    "combined_metrics.csv",
    "combined_predictions.csv",
    "primary_analysis_fold_metrics.csv",
    "primary_analysis_predictions.csv",
    "raw100_reference_fold_metrics.csv",
    "raw100_reference_predictions.csv",
    "lineage_pointers.json",
    "run_manifest.json",
    "statistics_bootstrap_confidence_intervals.csv",
    "statistics_chi_resolution_per_split.csv",
    "statistics_chi_resolution_summary.csv",
    "statistics_direction_consistency.csv",
    "statistics_fingerprints.csv",
    "statistics_holm_correction.csv",
    "statistics_paired_contrasts.csv",
    "statistics_paired_split_deltas.csv",
)


class PolarHarmonicAblationError(ValueError):
    """Raised when a polar-harmonic ablation cannot be reproduced safely."""


def run_polar_harmonic_ablation_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one parent polar-harmonic ablation with paired child experiments."""
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    run_folder = _create_parent_run_folder(config, path)
    child_rows: list[dict[str, Any]] = []

    try:
        for n_chi in config["polar_representation"]["n_chi_values"]:
            child_config = _child_core_config(config, path, run_folder, n_chi=int(n_chi))
            child_config_path = run_folder / "child_configs" / f"n_chi_{n_chi}.yaml"
            child_config_path.parent.mkdir(parents=True, exist_ok=True)
            child_config_path.write_text(
                yaml.safe_dump(child_config, sort_keys=False), encoding="utf-8"
            )
            child_result = run_polar_basis_compression_from_config(
                child_config_path,
                verbose=verbose,
            )
            child_rows.append(
                _collect_child_output(
                    n_chi=int(n_chi),
                    child_config_path=child_config_path,
                    child_result=child_result,
                    config=config,
                )
            )

        mlflow = _finalize_parent_run(
            config=config,
            config_path=path,
            run_folder=run_folder,
            child_rows=child_rows,
        )
    except Exception as exc:
        _write_json(
            run_folder / "run_manifest.json",
            {
                "contract": CONTRACT,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "product_artifact_modified": False,
                "report_modified": False,
            },
        )
        raise

    return {
        "contract": CONTRACT,
        "run_folder": run_folder,
        "child_runs": int(len(child_rows)),
        "target_cases": int(config["cohort"]["target_cases"]),
        "mlflow": mlflow,
    }


def resume_polar_harmonic_ablation_from_config(
    config_path: str | Path,
    run_folder: str | Path,
) -> dict[str, Any]:
    """Finalize a parent run from complete, validated child artifacts."""
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    parent = Path(run_folder).expanduser().resolve()
    output_root = _resolve_path(config["output"]["folder"], path).resolve()
    if not parent.is_relative_to(output_root) or not parent.is_dir():
        raise PolarHarmonicAblationError(
            "Resume run folder must be an existing child of the configured output root."
        )
    child_rows = []
    for n_chi in config["polar_representation"]["n_chi_values"]:
        child_config_path = parent / "child_configs" / f"n_chi_{n_chi}.yaml"
        child_root = parent / "children" / f"n_chi_{n_chi}"
        candidates = sorted(
            candidate
            for candidate in child_root.glob("polar_basis_compression_*")
            if all((candidate / name).is_file() for name in _CORE_REQUIRED_ARTIFACTS)
        )
        if len(candidates) != 1 or not child_config_path.is_file():
            raise PolarHarmonicAblationError(
                f"Resume requires exactly one complete child for n_chi={n_chi}."
            )
        child_rows.append(
            _collect_child_output(
                n_chi=int(n_chi),
                child_config_path=child_config_path,
                child_result={"run_folder": candidates[0], "mlflow": {}},
                config=config,
            )
        )
    mlflow = _finalize_parent_run(
        config=config,
        config_path=path,
        run_folder=parent,
        child_rows=child_rows,
    )
    return {
        "contract": CONTRACT,
        "run_folder": parent,
        "child_runs": len(child_rows),
        "target_cases": int(config["cohort"]["target_cases"]),
        "mlflow": mlflow,
    }


def _finalize_parent_run(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_folder: Path,
    child_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    combined = _validate_and_combine_children(child_rows, config=config)
    statistics = analyze_polar_harmonic_runs(
        combined["primary_analysis_predictions"],
        combined["primary_analysis_fold_metrics"],
        config=PolarHarmonicStatisticsConfig(
            seed=int(config["evaluation"]["seed"]),
            reference_n_chi=int(config["harmonic_ablation"]["primary"]["n_chi"]),
        ),
    )
    _write_parent_artifacts(
        run_folder=run_folder,
        config=config,
        config_path=config_path,
        child_rows=child_rows,
        combined=combined,
        statistics=statistics,
    )
    return _log_parent_mlflow(
        config=config,
        config_path=config_path,
        run_folder=run_folder,
        statistics=statistics,
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the top-level polar-harmonic contract exactly."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Polar harmonic ablation config is unavailable: {source}")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    _validate_config(config)
    return config


def _validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise PolarHarmonicAblationError("Polar harmonic ablation config must be a mapping.")
    _exact_keys(
        config,
        {
            "contract",
            "experiment",
            "input",
            "data_version",
            "cohort",
            "polar_representation",
            "harmonic_ablation",
            "evaluation",
            "controls",
            "mlflow",
            "output",
        },
        "config",
    )
    if config["contract"] != CONTRACT:
        raise PolarHarmonicAblationError(
            f"Unsupported contract: {config['contract']!r}."
        )
    _validate_experiment(config["experiment"])
    _validate_input(config["input"])
    _validate_data_version(config["data_version"])
    _validate_cohort(config["cohort"])
    _validate_polar_representation(config["polar_representation"])
    _validate_harmonic_ablation(config["harmonic_ablation"], config["polar_representation"])
    _validate_evaluation(config["evaluation"])
    _validate_controls(config["controls"])
    _validate_mlflow(config["mlflow"])
    _validate_output(config["output"])


def _validate_experiment(value: Any) -> None:
    _exact_keys(value, {"name", "model_name", "model_version", "purpose"}, "experiment")
    for key in ("name", "model_name", "model_version", "purpose"):
        _nonempty(value.get(key), f"experiment.{key}")
    if value["model_name"] != "aramina_target_breast_risk":
        raise PolarHarmonicAblationError("experiment.model_name must be aramina_target_breast_risk.")
    if value["model_version"] != "0.2.14-beta":
        raise PolarHarmonicAblationError("Experiment must pin Aramina 0.2.14-beta.")


def _validate_input(value: Any) -> None:
    _exact_keys(value, {"input_h5_path", "model_joblib_path"}, "input")
    for key in ("input_h5_path", "model_joblib_path"):
        _nonempty(value.get(key), f"input.{key}")


def _validate_data_version(value: Any) -> None:
    _exact_keys(
        value,
        {"contract", "system", "dataset_id", "dvc_version", "pointer_path"},
        "data_version",
    )
    if value.get("contract") != "aramina_dvc_input_v0_1" or value.get("system") != "dvc":
        raise PolarHarmonicAblationError("data_version must use aramina DVC input lineage.")
    for key in ("dataset_id", "dvc_version", "pointer_path"):
        _nonempty(value.get(key), f"data_version.{key}")


def _validate_cohort(value: Any) -> None:
    _exact_keys(
        value,
        {
            "selection",
            "accepted_target_measurements",
            "target_cases",
            "patient_grouping",
        },
        "cohort",
    )
    if value.get("selection") != "all_accepted_target_measurements_and_cases":
        raise PolarHarmonicAblationError("cohort.selection must retain all accepted target cases.")
    if value.get("patient_grouping") != "patient_safe":
        raise PolarHarmonicAblationError("cohort.patient_grouping must be patient_safe.")
    for key in ("accepted_target_measurements", "target_cases"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise PolarHarmonicAblationError(f"cohort.{key} must be a positive integer.")


def _validate_polar_representation(value: Any) -> None:
    _exact_keys(
        value,
        {
            "n_q",
            "n_chi_values",
            "radial_q_range",
            "azimuthal_range",
            "normalization_q_range",
            "harmonic_q_range",
            "missing_sector_policy",
            "cache_folder",
            "force_rebuild",
        },
        "polar_representation",
    )
    if value.get("n_q") != 256:
        raise PolarHarmonicAblationError("polar_representation.n_q must be 256.")
    n_chi_values = value.get("n_chi_values")
    if (
        not isinstance(n_chi_values, list)
        or not n_chi_values
        or any(isinstance(item, bool) or not isinstance(item, int) for item in n_chi_values)
        or tuple(n_chi_values) != tuple(sorted(set(n_chi_values)))
        or not set(n_chi_values).issubset(_ALLOWED_N_CHI)
    ):
        raise PolarHarmonicAblationError(
            f"polar_representation.n_chi_values must be ordered unique values from {_ALLOWED_N_CHI}."
        )
    if value.get("radial_q_range") != [2.0, 23.0]:
        raise PolarHarmonicAblationError("radial_q_range must remain [2.0, 23.0].")
    if value.get("azimuthal_range") != [-180.0, 180.0]:
        raise PolarHarmonicAblationError("azimuthal_range must remain [-180.0, 180.0].")
    if value.get("normalization_q_range") != [6.7, 7.1]:
        raise PolarHarmonicAblationError("normalization_q_range must remain [6.7, 7.1].")
    if value.get("harmonic_q_range") != [2.1, 12.2]:
        raise PolarHarmonicAblationError("harmonic_q_range must remain [2.1, 12.2].")
    if value.get("missing_sector_policy") != "weighted_fit_with_zero_weight_for_missing_sectors":
        raise PolarHarmonicAblationError("Unsupported missing-sector policy.")
    _nonempty(value.get("cache_folder"), "polar_representation.cache_folder")
    if not isinstance(value.get("force_rebuild"), bool):
        raise PolarHarmonicAblationError("polar_representation.force_rebuild must be boolean.")


def _validate_harmonic_ablation(value: Any, polar: dict[str, Any]) -> None:
    _exact_keys(
        value,
        {"mode_sets", "encoder", "coefficients_per_channel", "primary"},
        "harmonic_ablation",
    )
    if value.get("mode_sets") != _MODE_SETS:
        raise PolarHarmonicAblationError("harmonic_ablation.mode_sets must define nested A0/A2/A4 modes.")
    if value.get("encoder") != "cubic_bspline":
        raise PolarHarmonicAblationError("harmonic_ablation.encoder must be cubic_bspline.")
    if value.get("coefficients_per_channel") != [8, 12, 16]:
        raise PolarHarmonicAblationError("coefficients_per_channel must be [8, 12, 16].")
    primary = value.get("primary")
    _exact_keys(primary, {"n_chi", "coefficients_per_channel"}, "harmonic_ablation.primary")
    if primary["n_chi"] not in polar["n_chi_values"]:
        raise PolarHarmonicAblationError("Primary n_chi must be included in n_chi_values.")
    if primary["coefficients_per_channel"] not in value["coefficients_per_channel"]:
        raise PolarHarmonicAblationError(
            "Primary coefficients_per_channel must be included in the configured values."
        )


def _validate_evaluation(value: Any) -> None:
    _exact_keys(
        value,
        {
            "method",
            "folds",
            "repeats",
            "seed",
            "inner_oof_lr1_to_lr2",
            "threshold_policy",
            "target_sensitivity",
            "compare_on_identical_measurement_and_fold_manifests",
            "metrics",
            "confidence_interval",
        },
        "evaluation",
    )
    if value.get("method") != "repeated_stratified_patient_kfold":
        raise PolarHarmonicAblationError("Evaluation must use repeated patient-safe folds.")
    for key, minimum in (("folds", 2), ("repeats", 1)):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
            raise PolarHarmonicAblationError(f"evaluation.{key} is invalid.")
    if isinstance(value.get("seed"), bool) or not isinstance(value.get("seed"), int) or value["seed"] < 0:
        raise PolarHarmonicAblationError("evaluation.seed must be a non-negative integer.")
    if value.get("inner_oof_lr1_to_lr2") is not True:
        raise PolarHarmonicAblationError("evaluation.inner_oof_lr1_to_lr2 must be true.")
    if value.get("threshold_policy") != "training_fold_target_sensitivity":
        raise PolarHarmonicAblationError(
            "evaluation.threshold_policy must be training_fold_target_sensitivity."
        )
    sensitivity = value.get("target_sensitivity")
    if (
        isinstance(sensitivity, bool)
        or not isinstance(sensitivity, int | float)
        or not 0.0 < float(sensitivity) <= 1.0
    ):
        raise PolarHarmonicAblationError(
            "evaluation.target_sensitivity must be explicitly provided inside (0, 1]."
        )
    if value.get("compare_on_identical_measurement_and_fold_manifests") is not True:
        raise PolarHarmonicAblationError(
            "evaluation.compare_on_identical_measurement_and_fold_manifests must be true."
        )
    if value.get("metrics") != [
        "sensitivity",
        "specificity",
        "roc_auc",
        "balanced_accuracy",
        "ppv",
        "npv",
        "confusion_matrix",
    ]:
        raise PolarHarmonicAblationError("evaluation.metrics must use the fixed metric set.")
    if value.get("confidence_interval") != "paired_patient_cluster_bootstrap_95":
        raise PolarHarmonicAblationError("Unsupported confidence_interval policy.")


def _validate_controls(value: Any) -> None:
    _exact_keys(
        value,
        {
            "radial_baseline",
            "qc_modes",
            "confounder_fields",
            "permutation_control",
            "session_stress_test",
        },
        "controls",
    )
    if value.get("radial_baseline") != "raw100":
        raise PolarHarmonicAblationError("controls.radial_baseline must be raw100.")
    if value.get("qc_modes") != [1, 3]:
        raise PolarHarmonicAblationError("controls.qc_modes must be [1, 3].")
    if value.get("confounder_fields") != [
        "age",
        "thickness",
        "session",
        "date",
        "target_side",
        "measurement_count",
        "snr",
    ]:
        raise PolarHarmonicAblationError("controls.confounder_fields are not the fixed contract values.")
    if value.get("permutation_control") != "not_executed_partial_arc_limitation":
        raise PolarHarmonicAblationError(
            "controls.permutation_control must declare its partial-arc limitation."
        )
    if (
        value.get("session_stress_test")
        != "unavailable_sparse_or_high_cardinality_session_labels"
    ):
        raise PolarHarmonicAblationError(
            "controls.session_stress_test must declare unavailable session labels."
        )


def _validate_mlflow(value: Any) -> None:
    _exact_keys(value, {"enabled", "tracking_uri", "experiment_name"}, "mlflow")
    if value.get("enabled") is not True:
        raise PolarHarmonicAblationError("Parent MLflow tracking is required.")
    _nonempty(value.get("tracking_uri"), "mlflow.tracking_uri")
    _nonempty(value.get("experiment_name"), "mlflow.experiment_name")


def _validate_output(value: Any) -> None:
    _exact_keys(value, {"folder", "product_artifact_changes", "report_changes"}, "output")
    _nonempty(value.get("folder"), "output.folder")
    if value.get("product_artifact_changes") is not False:
        raise PolarHarmonicAblationError("Polar ablation must not change product artifacts.")
    if value.get("report_changes") is not False:
        raise PolarHarmonicAblationError("Polar ablation must not change reports.")


def _child_core_config(
    config: dict[str, Any],
    config_path: Path,
    parent_run_folder: Path,
    *,
    n_chi: int,
) -> dict[str, Any]:
    """Create the exact core contract from one parent angular-resolution arm."""
    polar = config["polar_representation"]
    top_evaluation = config["evaluation"]
    child_root = parent_run_folder / "children" / f"n_chi_{n_chi}"
    return {
        "contract": CORE_CONTRACT,
        "experiment": {
            "name": f"{config['experiment']['name']}_n_chi_{n_chi}",
            "model_name": config["experiment"]["model_name"],
            "model_version": config["experiment"]["model_version"],
        },
        "input": {
            "input_h5_path": str(_resolve_path(config["input"]["input_h5_path"], config_path)),
            "model_joblib_path": str(
                _resolve_path(config["input"]["model_joblib_path"], config_path)
            ),
        },
        # Keep the canonical relative pointer recorded by the frozen model.
        # The core resolver still resolves it from the repository root.
        "data_version": deepcopy(config["data_version"]),
        "cohort": {"max_patients_per_class": None},
        "polar_cakes": {
            "n_q": polar["n_q"],
            "n_chi": n_chi,
            "radial_q_range": deepcopy(polar["radial_q_range"]),
            "azimuthal_range": deepcopy(polar["azimuthal_range"]),
            "normalization_q_range": deepcopy(polar["normalization_q_range"]),
            "harmonic_q_range": deepcopy(polar["harmonic_q_range"]),
            "cache_folder": str(_resolve_path(polar["cache_folder"], config_path)),
            "force_rebuild": polar["force_rebuild"],
        },
        "representations": {
            "families": ["fourier_bspline"],
            "candidate_modes": [0, 2, 4],
            "qc_modes": deepcopy(config["controls"]["qc_modes"]),
            "coefficient_budgets": deepcopy(
                config["harmonic_ablation"]["coefficients_per_channel"]
            ),
        },
        "evaluation": {
            "method": top_evaluation["method"],
            "folds": top_evaluation["folds"],
            "repeats": top_evaluation["repeats"],
            "seed": top_evaluation["seed"],
            "target_sensitivity": top_evaluation["target_sensitivity"],
            "threshold_policy": "training_fold_target_sensitivity",
        },
        "confounders": {"fields": ["age", "thickness", "session", "date"]},
        "runtime": {"reconstruction_examples_per_variant": 3},
        "mlflow": {
            "enabled": True,
            "tracking_uri": _tracking_uri(config["mlflow"]["tracking_uri"], config_path),
            "experiment_name": f"{config['mlflow']['experiment_name']}_child_core",
        },
        "output": {"folder": str(child_root)},
    }


def _collect_child_output(
    *,
    n_chi: int,
    child_config_path: Path,
    child_result: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(child_result, dict) or "run_folder" not in child_result:
        raise PolarHarmonicAblationError(
            f"Child n_chi={n_chi} returned no run_folder; refusing partial parent run."
        )
    run_folder = Path(child_result["run_folder"]).expanduser().resolve()
    expected_root = child_config_path.parent.parent / "children" / f"n_chi_{n_chi}"
    if not run_folder.is_relative_to(expected_root.resolve()):
        raise PolarHarmonicAblationError(
            f"Child n_chi={n_chi} run_folder is outside its isolated child root."
        )
    missing = [name for name in _CORE_REQUIRED_ARTIFACTS if not (run_folder / name).is_file()]
    if missing:
        raise PolarHarmonicAblationError(
            f"Child n_chi={n_chi} output is partial; missing required artifacts: {missing}."
        )

    predictions = pd.read_csv(run_folder / "predictions.csv")
    fold_metrics = pd.read_csv(run_folder / "fold_metrics.csv")
    fold_manifest = pd.read_csv(run_folder / "fold_manifest.csv")
    cohort_manifest = pd.read_csv(run_folder / "cohort_manifest.csv")
    measurement_manifest = pd.read_csv(run_folder / "polar_cake_manifest.csv")
    raw100_predictions = pd.read_csv(run_folder / "raw100_predictions.csv")
    raw100_fold_metrics = pd.read_csv(run_folder / "raw100_fold_metrics.csv")
    _validate_child_tables(
        n_chi=n_chi,
        config=config,
        predictions=predictions,
        fold_metrics=fold_metrics,
        fold_manifest=fold_manifest,
        cohort_manifest=cohort_manifest,
        measurement_manifest=measurement_manifest,
        raw100_predictions=raw100_predictions,
        raw100_fold_metrics=raw100_fold_metrics,
    )
    return {
        "n_chi": n_chi,
        "config_path": child_config_path,
        "run_folder": run_folder,
        "result": child_result,
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "fold_manifest": fold_manifest,
        "cohort_manifest": cohort_manifest,
        "measurement_manifest": measurement_manifest,
        "raw100_predictions": raw100_predictions,
        "raw100_fold_metrics": raw100_fold_metrics,
        "lineage": _read_json(run_folder / "lineage.json"),
        "run_manifest": _read_json(run_folder / "run_manifest.json"),
    }


def _validate_child_tables(
    *,
    n_chi: int,
    config: dict[str, Any],
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    cohort_manifest: pd.DataFrame,
    measurement_manifest: pd.DataFrame,
    raw100_predictions: pd.DataFrame,
    raw100_fold_metrics: pd.DataFrame,
) -> None:
    _require_columns(
        predictions,
        {
            "mode_set",
            "n_chi",
            "coefficients_per_channel",
            "split_id",
            "target_case_id",
            "patientId",
            "label",
            "p_cancer",
            "threshold",
        },
        f"child n_chi={n_chi} predictions",
    )
    _require_columns(
        fold_metrics,
        {
            "mode_set",
            "n_chi",
            "coefficients_per_channel",
            "split_id",
            "sensitivity",
            "specificity",
            "roc_auc",
        },
        f"child n_chi={n_chi} fold_metrics",
    )
    _require_columns(
        fold_manifest,
        {"split_id", "partition", "target_case_id", "patientId", "label"},
        f"child n_chi={n_chi} fold_manifest",
    )
    _require_columns(
        cohort_manifest,
        {"target_case_id", "patientId", "label"},
        f"child n_chi={n_chi} cohort_manifest",
    )
    _require_columns(
        measurement_manifest,
        {"measurement_key", "target_case_id", "patient_id", "label", "n_chi"},
        f"child n_chi={n_chi} polar_cake_manifest",
    )
    _require_columns(
        raw100_predictions,
        {"split_id", "target_case_id", "patientId", "label", "p_cancer", "threshold"},
        f"child n_chi={n_chi} raw100_predictions",
    )
    _require_columns(
        raw100_fold_metrics,
        {"split_id", "sensitivity", "specificity", "roc_auc"},
        f"child n_chi={n_chi} raw100_fold_metrics",
    )
    if set(pd.to_numeric(predictions["n_chi"], errors="raise")) != {n_chi}:
        raise PolarHarmonicAblationError(
            f"Child n_chi={n_chi} predictions contain a different angular resolution."
        )
    if set(pd.to_numeric(fold_metrics["n_chi"], errors="raise")) != {n_chi}:
        raise PolarHarmonicAblationError(
            f"Child n_chi={n_chi} fold_metrics contain a different angular resolution."
        )
    if set(pd.to_numeric(measurement_manifest["n_chi"], errors="raise")) != {n_chi}:
        raise PolarHarmonicAblationError(
            f"Child n_chi={n_chi} polar cache manifest contains a different angular resolution."
        )
    if len(cohort_manifest) != int(config["cohort"]["target_cases"]):
        raise PolarHarmonicAblationError(
            f"Child n_chi={n_chi} target-case count differs from the contract."
        )
    if len(measurement_manifest) != int(config["cohort"]["accepted_target_measurements"]):
        raise PolarHarmonicAblationError(
            f"Child n_chi={n_chi} accepted-measurement count differs from the contract."
        )
    if cohort_manifest["target_case_id"].duplicated().any():
        raise PolarHarmonicAblationError("Child cohort_manifest duplicates target cases.")
    if measurement_manifest["measurement_key"].duplicated().any():
        raise PolarHarmonicAblationError("Child polar_cake_manifest duplicates measurements.")
    if fold_manifest.duplicated(["split_id", "target_case_id"]).any():
        raise PolarHarmonicAblationError("Child fold_manifest duplicates a case in one split.")

    expected_variants = {
        (mode, coefficient)
        for mode in _CORE_MODE_SETS.values()
        for coefficient in config["harmonic_ablation"]["coefficients_per_channel"]
    }
    observed_prediction_variants = {
        (str(row.mode_set), int(row.coefficients_per_channel))
        for row in predictions[["mode_set", "coefficients_per_channel"]].drop_duplicates().itertuples(index=False)
    }
    observed_metric_variants = {
        (str(row.mode_set), int(row.coefficients_per_channel))
        for row in fold_metrics[["mode_set", "coefficients_per_channel"]].drop_duplicates().itertuples(index=False)
    }
    if observed_prediction_variants != expected_variants or observed_metric_variants != expected_variants:
        raise PolarHarmonicAblationError(
            f"Child n_chi={n_chi} did not produce the exact configured mode/K grid."
        )

    expected_test = fold_manifest.loc[
        fold_manifest["partition"] == "test", ["split_id", "target_case_id"]
    ].drop_duplicates()
    if expected_test.empty:
        raise PolarHarmonicAblationError("Child fold_manifest has no held-out target cases.")
    expected_test_keys = _key_set(expected_test, ["split_id", "target_case_id"])
    expected_splits = set(pd.to_numeric(expected_test["split_id"], errors="raise").astype(int))
    for mode_set, coefficient in sorted(expected_variants):
        prediction_variant = predictions[
            (predictions["mode_set"] == mode_set)
            & (pd.to_numeric(predictions["coefficients_per_channel"], errors="raise") == coefficient)
        ]
        metric_variant = fold_metrics[
            (fold_metrics["mode_set"] == mode_set)
            & (pd.to_numeric(fold_metrics["coefficients_per_channel"], errors="raise") == coefficient)
        ]
        if prediction_variant.duplicated(["split_id", "target_case_id"]).any():
            raise PolarHarmonicAblationError("Child predictions duplicate a held-out case.")
        if _key_set(prediction_variant, ["split_id", "target_case_id"]) != expected_test_keys:
            raise PolarHarmonicAblationError(
                f"Child n_chi={n_chi} predictions do not cover the shared held-out cases."
            )
        if set(pd.to_numeric(metric_variant["split_id"], errors="raise").astype(int)) != expected_splits:
            raise PolarHarmonicAblationError(
                f"Child n_chi={n_chi} metrics do not cover the shared fold grid."
            )


def _validate_and_combine_children(
    child_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    if len(child_rows) != len(config["polar_representation"]["n_chi_values"]):
        raise PolarHarmonicAblationError("Parent has incomplete child results.")
    reference = child_rows[0]
    reference_fingerprints = _child_fingerprints(reference)
    for child in child_rows[1:]:
        observed = _child_fingerprints(child)
        mismatched = [
            name
            for name, value in reference_fingerprints.items()
            if observed[name] != value
        ]
        if mismatched:
            raise PolarHarmonicAblationError(
                f"Child n_chi={child['n_chi']} fingerprint mismatch: {mismatched}."
            )
    fold_manifest = reference["fold_manifest"].copy()
    fold_manifest.insert(0, "parent_reference_n_chi", int(reference["n_chi"]))
    combined_predictions = pd.concat(
        [child["predictions"] for child in child_rows], ignore_index=True
    )
    combined_metrics = pd.concat(
        [child["fold_metrics"] for child in child_rows], ignore_index=True
    )
    summary = _summarize_metrics(combined_metrics)
    primary_coefficients = int(
        config["harmonic_ablation"]["primary"]["coefficients_per_channel"]
    )
    analysis_predictions = combined_predictions.loc[
        pd.to_numeric(combined_predictions["coefficients_per_channel"], errors="raise")
        == primary_coefficients
    ].copy()
    analysis_fold_metrics = combined_metrics.loc[
        pd.to_numeric(combined_metrics["coefficients_per_channel"], errors="raise")
        == primary_coefficients
    ].copy()
    _validate_primary_analysis_grid(
        predictions=analysis_predictions,
        fold_metrics=analysis_fold_metrics,
        config=config,
    )
    primary_n_chi = int(config["harmonic_ablation"]["primary"]["n_chi"])
    primary_child = next(child for child in child_rows if child["n_chi"] == primary_n_chi)
    return {
        "predictions": combined_predictions,
        "fold_metrics": combined_metrics,
        "metrics": summary,
        "fold_manifest": fold_manifest,
        "primary_analysis_predictions": analysis_predictions,
        "primary_analysis_fold_metrics": analysis_fold_metrics,
        "raw100_predictions": primary_child["raw100_predictions"],
        "raw100_fold_metrics": primary_child["raw100_fold_metrics"],
        "fingerprints": pd.DataFrame([reference_fingerprints]),
    }


def _child_fingerprints(child: dict[str, Any]) -> dict[str, str]:
    return {
        "cohort_fingerprint": _frame_fingerprint(
            child["cohort_manifest"], ["target_case_id", "patientId", "label"]
        ),
        "measurement_fingerprint": _frame_fingerprint(
            child["measurement_manifest"],
            [
                "measurement_key",
                "dataset_sha256",
                "patient_id",
                "target_case_id",
                "label",
                "calibration_session_uid",
                "poni_sha256",
            ],
        ),
        "fold_fingerprint": _frame_fingerprint(
            child["fold_manifest"],
            ["split_id", "partition", "target_case_id", "patientId", "label"],
        ),
    }


def _write_parent_artifacts(
    *,
    run_folder: Path,
    config: dict[str, Any],
    config_path: Path,
    child_rows: list[dict[str, Any]],
    combined: dict[str, pd.DataFrame],
    statistics: dict[str, pd.DataFrame],
) -> None:
    (run_folder / "effective_experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    combined["predictions"].to_csv(run_folder / "combined_predictions.csv", index=False)
    combined["fold_metrics"].to_csv(run_folder / "combined_fold_metrics.csv", index=False)
    combined["metrics"].to_csv(run_folder / "combined_metrics.csv", index=False)
    combined["fold_manifest"].to_csv(run_folder / "combined_fold_manifest.csv", index=False)
    combined["primary_analysis_predictions"].to_csv(
        run_folder / "primary_analysis_predictions.csv", index=False
    )
    combined["primary_analysis_fold_metrics"].to_csv(
        run_folder / "primary_analysis_fold_metrics.csv", index=False
    )
    combined["raw100_predictions"].to_csv(
        run_folder / "raw100_reference_predictions.csv", index=False
    )
    combined["raw100_fold_metrics"].to_csv(
        run_folder / "raw100_reference_fold_metrics.csv", index=False
    )
    for name, table in statistics.items():
        table.to_csv(run_folder / f"statistics_{name}.csv", index=False)

    index_rows = []
    lineage_children = []
    for child in child_rows:
        fingerprints = _child_fingerprints(child)
        artifact_hashes = {
            name: file_sha256(child["run_folder"] / name)
            for name in _CORE_REQUIRED_ARTIFACTS
        }
        index_rows.append(
            {
                "n_chi": child["n_chi"],
                "child_config_path": str(child["config_path"]),
                "child_run_folder": str(child["run_folder"]),
                "child_mlflow_run_id": child["result"].get("mlflow", {}).get("run_id", ""),
                **fingerprints,
                "predictions_sha256": artifact_hashes["predictions.csv"],
                "fold_metrics_sha256": artifact_hashes["fold_metrics.csv"],
                "fold_manifest_sha256": artifact_hashes["fold_manifest.csv"],
            }
        )
        lineage_children.append(
            {
                "n_chi": child["n_chi"],
                "child_config_path": str(child["config_path"]),
                "child_run_folder": str(child["run_folder"]),
                "child_lineage": child["lineage"],
                "child_run_manifest": child["run_manifest"],
                "artifact_sha256": artifact_hashes,
            }
        )
    pd.DataFrame(index_rows).sort_values("n_chi").to_csv(
        run_folder / "child_index.csv", index=False
    )
    _write_json(
        run_folder / "lineage_pointers.json",
        {
            "contract": CONTRACT,
            "source_config_path": str(config_path),
            "source_config_sha256": file_sha256(config_path),
            "child_core_contract": CORE_CONTRACT,
            "child_cache_isolation": "core_grid_namespace_by_n_q_n_chi_axis_contract",
            "threshold_policy_mapping": {
                "parent": config["evaluation"]["threshold_policy"],
                "core": "training_fold_target_sensitivity",
                "core_threshold_source": "outer_train_inner_oof_lr1_scores",
            },
            "shared_fingerprints": combined["fingerprints"].iloc[0].to_dict(),
            "children": lineage_children,
        },
    )
    _write_json(
        run_folder / "run_manifest.json",
        {
            "contract": CONTRACT,
            "status": "complete",
            "clinical_stage": "research_only",
            "product_artifact_modified": False,
            "report_modified": False,
            "target_cases": int(config["cohort"]["target_cases"]),
            "accepted_target_measurements": int(config["cohort"]["accepted_target_measurements"]),
            "n_chi_values": config["polar_representation"]["n_chi_values"],
            "primary": deepcopy(config["harmonic_ablation"]["primary"]),
            "target_sensitivity": float(config["evaluation"]["target_sensitivity"]),
            "controls": deepcopy(config["controls"]),
            "child_runs": len(child_rows),
            "required_artifacts": list(_PARENT_REQUIRED_ARTIFACTS),
            "limitations": [
                "retrospective research-only cohort",
                "no independent blind validation cohort",
                "polar angular features may encode detector geometry or positioning",
                "chi permutation was not executed for this partial-arc representation",
                "session stress testing was unavailable from recorded session labels",
                "no product artifact or report was changed",
            ],
        },
    )
    missing = [name for name in _PARENT_REQUIRED_ARTIFACTS if not (run_folder / name).is_file()]
    if missing:
        raise PolarHarmonicAblationError(
            f"Parent output is partial; missing required artifacts: {missing}."
        )


def _log_parent_mlflow(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_folder: Path,
    statistics: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    primary_n_chi = int(config["harmonic_ablation"]["primary"]["n_chi"])
    primary = statistics["paired_contrasts"]
    primary = primary.loc[primary["n_chi"] == primary_n_chi].copy()
    if primary.empty:
        raise PolarHarmonicAblationError("Primary paired contrast statistics are absent.")
    metrics: dict[str, float] = {}
    for row in primary.itertuples(index=False):
        contrast = str(row.contrast).replace(" ", "_").replace("+", "plus").replace("-", "minus")
        metrics[f"primary.{contrast}.{row.metric}.delta_mean"] = float(row.delta_mean)
        metrics[f"primary.{contrast}.{row.metric}.ci_low"] = float(row.ci_low)
        metrics[f"primary.{contrast}.{row.metric}.ci_high"] = float(row.ci_high)
    tracking_uri = _tracking_uri(config["mlflow"]["tracking_uri"], config_path)
    params = {
        "contract": CONTRACT,
        "evaluation": {
            "folds": config["evaluation"]["folds"],
            "repeats": config["evaluation"]["repeats"],
            "seed": config["evaluation"]["seed"],
            "target_sensitivity": config["evaluation"]["target_sensitivity"],
            "threshold_policy": config["evaluation"]["threshold_policy"],
        },
        "polar": {
            "n_q": config["polar_representation"]["n_q"],
            "n_chi_values": ",".join(
                str(value) for value in config["polar_representation"]["n_chi_values"]
            ),
            "harmonic_q_range": ",".join(
                str(value) for value in config["polar_representation"]["harmonic_q_range"]
            ),
        },
        "primary": config["harmonic_ablation"]["primary"],
    }
    tags = {
        "product": "aramina",
        "clinical_stage": "research_only",
        "endpoint": "target_breast_BENIGN_vs_CANCER_decision_support",
        "product_artifact_modified": False,
        "report_modified": False,
        "source_config_sha256": file_sha256(config_path),
    }
    run_name = f"{config['experiment']['name']}_{run_folder.name.rsplit('_', 1)[-1]}"
    with MlflowRun(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=config["mlflow"]["experiment_name"],
        run_name=run_name,
        params=params,
        tags=tags,
    ) as run:
        run.log_metrics(metrics)
        run.log_artifact_directory(
            run_folder,
            required_files=_PARENT_REQUIRED_ARTIFACTS,
            artifact_path="polar_harmonic_ablation",
        )
        run_id = run.run_id
    return {
        "enabled": True,
        "run_id": run_id,
        "status": run.status,
        "tracking_uri": tracking_uri,
    }


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "sensitivity",
        "specificity",
        "roc_auc",
        "balanced_accuracy",
        "ppv",
        "npv",
    )
    group_columns = ["n_chi", "mode_set", "coefficients_per_channel", "budget"]
    rows = []
    for keys, group in metrics.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        row["splits"] = int(len(group))
        for name in metric_columns:
            values = pd.to_numeric(group[name], errors="raise")
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_primary_analysis_grid(
    *,
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    expected_mode_sets = set(_CORE_MODE_SETS.values())
    expected_n_chi = set(config["polar_representation"]["n_chi_values"])
    if predictions.duplicated(
        ["n_chi", "mode_set", "split_id", "target_case_id"]
    ).any():
        raise PolarHarmonicAblationError(
            "Primary analysis predictions are not unique by resolution/mode/split/case."
        )
    if fold_metrics.duplicated(["n_chi", "mode_set", "split_id"]).any():
        raise PolarHarmonicAblationError(
            "Primary analysis fold metrics are not unique by resolution/mode/split."
        )
    observed_predictions = set(
        predictions[["n_chi", "mode_set"]].itertuples(index=False, name=None)
    )
    observed_metrics = set(
        fold_metrics[["n_chi", "mode_set"]].itertuples(index=False, name=None)
    )
    expected = {(n_chi, mode_set) for n_chi in expected_n_chi for mode_set in expected_mode_sets}
    if observed_predictions != expected or observed_metrics != expected:
        raise PolarHarmonicAblationError(
            "Primary analysis does not cover the exact configured resolution/mode grid."
        )


def _create_parent_run_folder(config: dict[str, Any], config_path: Path) -> Path:
    root = _resolve_path(config["output"]["folder"], config_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"polar_harmonic_ablation_{timestamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _frame_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    _require_columns(frame, set(columns), "fingerprint source")
    selected = frame.loc[:, columns].copy()
    selected = selected.fillna("<NA>").astype(str).sort_values(columns, kind="stable")
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    return set(map(tuple, frame.loc[:, columns].astype(str).itertuples(index=False, name=None)))


def _resolve_path(value: str, config_path: Path) -> Path:
    return resolve_config_path(_nonempty(value, "path"), config_path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PolarHarmonicAblationError(f"Expected JSON object in {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_columns(frame: pd.DataFrame, columns: set[str], where: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise PolarHarmonicAblationError(f"{where} missing required columns: {missing}.")


def _exact_keys(value: Any, expected: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise PolarHarmonicAblationError(f"{where} must be a mapping.")
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        raise PolarHarmonicAblationError(
            f"{where} fields invalid; missing={missing}, unknown={unknown}."
        )


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolarHarmonicAblationError(f"{where} must be a non-empty string.")
    return value.strip()
