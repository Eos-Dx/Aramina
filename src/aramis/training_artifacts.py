"""Training artifact construction, lineage, and evaluation-output writers."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .model_schema import m2q_feature_schema, m2q_warnings
from .model_description import (
    _aramis_git_sha,
    _aramis_version,
    _evaluation_artifact_paths,
    _file_sha256,
    _jsonable,
    _model_reference,
    _records,
    _write_yaml,
)
from .training_config import PRODUCT_MODEL_NAME
from .training_evaluation import (
    _patient_dataset_summary,
    _summarize_patient_model_metrics,
)
from .tra_policy import derive_tra_policy


PATIENT_BOOTSTRAP_SAMPLES = 2_000


def _patient_training_artifact(
    *,
    df: pd.DataFrame,
    config: dict[str, Any],
    config_text: str,
    input_dataframe_joblib_path: str | Path,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None,
    models: dict[str, Any],
    feature_table: pd.DataFrame,
    lr1_rows: pd.DataFrame,
    split_metrics: pd.DataFrame,
    split_predictions: pd.DataFrame,
    preprocess_train_config_yaml: str | None,
) -> dict[str, Any]:
    """Build the traceable joblib payload for target-breast model training."""
    evaluation_config = config.get("evaluation", {})
    metric_summary = (
        _summarize_patient_model_metrics(
            split_metrics,
            split_predictions,
            random_state=int(evaluation_config.get("random_state", 42)),
            bootstrap_samples=int(
                evaluation_config.get("bootstrap_samples", PATIENT_BOOTSTRAP_SAMPLES)
            ),
        )
        if not split_metrics.empty
        else pd.DataFrame()
    )
    dataset_summary = _patient_dataset_summary(df, feature_table, lr1_rows)
    model_descriptions = {
        PRODUCT_MODEL_NAME: {
            "name": "Aramis target-breast profile, optional SK symmetry refinement, and age",
            "description": (
                "One final model: profile and age are always evaluated; SK Core4 "
                "adds a neutral-gated refinement only when paired symmetry is available."
            ),
        }
    }
    return {
        "kind": "aramis_training_artifact",
        "version": "0.3",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": "m2q_gated_target_case",
        "model_columns": {
            key: config["model"][key]
            for key in (
                "profile_column",
                "group_column",
                "specimen_column",
                "side_column",
                "age_column",
            )
        },
        "models": models,
        "model_descriptions": model_descriptions,
        "feature_schema": m2q_feature_schema(),
        "warnings": m2q_warnings(feature_table),
        "training_config_yaml": config_text,
        "prediction_contract_yaml": yaml.safe_dump(
            config["prediction_contract"], sort_keys=False
        ),
        **_preprocessing_lineage_fields(
            preprocessing_artifact,
            prediction_preprocessing,
        ),
        "input_dataframe_joblib_sha256": _file_sha256(input_dataframe_joblib_path),
        "dataset_summary": dataset_summary,
        "metric_summary": metric_summary,
        "split_metrics": split_metrics,
        "split_predictions": split_predictions,
        "metadata": {
            "aramis_version": _aramis_version(),
            "aramis_git_sha": _aramis_git_sha(),
        },
        "reproducibility": _reproducibility_manifest(
            preprocessing_artifact=preprocessing_artifact,
            training_config_yaml=config_text,
            prediction_preprocessing=prediction_preprocessing,
            preprocess_train_config_yaml=preprocess_train_config_yaml,
        ),
    }


def _prediction_preprocessing_payload(
    config_path: Path | None,
) -> dict[str, Any] | None:
    if config_path is None:
        return None
    from xrd_preprocessing import load_preprocessing_config

    config = load_preprocessing_config(config_path)
    return {
        "path": str(config_path),
        "yaml": yaml.safe_dump(config, sort_keys=False),
    }


def _project_owned_path(value: str, config_path: Path) -> Path:
    """Resolve a fixed model resource relative to the supplied config root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for parent in config_path.parents:
        if parent.name == "config":
            candidate = (parent.parent / path).resolve()
            if candidate.exists():
                return candidate
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _preprocessing_lineage_fields(
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None,
) -> dict[str, Any]:
    training_config_yaml = preprocessing_artifact.get("preprocessing_config_yaml")
    fields = {
        "historical_preprocessing_yaml": training_config_yaml,
        "preprocessing_metadata": preprocessing_artifact.get("metadata", {}),
    }
    if prediction_preprocessing is None:
        fields["prediction_preprocessing_yaml"] = None
        return fields
    fields["prediction_preprocessing_yaml"] = prediction_preprocessing["yaml"]
    return fields


def _reproducibility_manifest(
    *,
    preprocessing_artifact: dict[str, Any],
    training_config_yaml: str,
    prediction_preprocessing: dict[str, Any] | None,
    preprocess_train_config_yaml: str | None,
) -> dict[str, Any]:
    """Record inputs required to repeat this research-draft training run."""
    historical_preprocessing_yaml = str(
        preprocessing_artifact["preprocessing_config_yaml"]
    )
    prediction_preprocessing_yaml = (
        str(prediction_preprocessing["yaml"])
        if prediction_preprocessing is not None
        else None
    )
    source_metadata = dict(preprocessing_artifact.get("metadata", {}))
    historical_preprocessing = yaml.safe_load(historical_preprocessing_yaml)
    source_path = historical_preprocessing.get("io", {}).get("input_h5_path")
    configs = {
        "preprocess_train_yaml": preprocess_train_config_yaml,
        "training_yaml": training_config_yaml,
        "historical_preprocessing_yaml": historical_preprocessing_yaml,
        "prediction_preprocessing_yaml": prediction_preprocessing_yaml,
    }
    return {
        "contract": "aramis_reproducibility_v0_1",
        "reproduction_mode": (
            "raw_h5_preprocess_train"
            if preprocess_train_config_yaml is not None
            else "preprocessed_artifact_train"
        ),
        "source_h5": {
            "filename": Path(str(source_path)).name if source_path else "unknown",
            "sha256": str(source_metadata["input_h5_sha256"]),
        },
        "source_code": {
            "aramis": {
                "version": _aramis_version(),
                "git_sha": _aramis_git_sha(),
            },
            "xrd_preprocessing": _distribution_provenance("xrd-preprocessing"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _installed_version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "joblib",
                    "h5py",
                    "pyFAI",
                    "fabio",
                    "PyYAML",
                )
            },
        },
        "configs": configs,
        "checksums": {
            f"{name}_sha256": _text_sha256(text)
            for name, text in configs.items()
            if text is not None
        },
    }


def _distribution_provenance(distribution_name: str) -> dict[str, Any]:
    """Return installed package identity and pip's VCS provenance when present."""
    result: dict[str, Any] = {"version": _installed_version(distribution_name)}
    try:
        payload = distribution(distribution_name).read_text("direct_url.json")
    except PackageNotFoundError:
        return result
    if payload is None:
        return result
    direct_url = json.loads(payload)
    result["url"] = direct_url.get("url")
    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, dict):
        result["requested_revision"] = vcs_info.get("requested_revision")
        result["git_commit"] = vcs_info.get("commit_id")
    return result


def _installed_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unavailable"


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _evaluation_artifact(
    artifact: dict[str, Any],
    *,
    model_identity: dict[str, Any],
    target_sensitivity: float,
    training_config_yaml: str,
) -> dict[str, Any]:
    return {
        "output_type": "aramis_evaluation_artifact",
        "version": "0.1",
        "created_at": artifact["created_at"],
        "model": _model_reference(model_identity),
        "target_sensitivity": target_sensitivity,
        "training_config_yaml": training_config_yaml,
        "historical_preprocessing_yaml": artifact.get("historical_preprocessing_yaml"),
        "dataset_summary": artifact["dataset_summary"],
        "metric_summary": artifact["metric_summary"],
        "split_metrics": artifact["split_metrics"],
        "split_predictions": artifact["split_predictions"],
        "metadata": artifact["metadata"],
    }


def _write_evaluation_outputs(
    artifact: dict[str, Any],
    folder: Path,
    *,
    model: dict[str, Any] | None = None,
    decision_threshold: dict[str, Any] | None = None,
) -> None:
    artifact["split_metrics"].to_csv(folder / "evaluation_metrics.csv", index=False)
    artifact["split_predictions"].to_csv(
        folder / "evaluation_predictions.csv", index=False
    )
    _write_yaml(
        folder / "evaluation.yaml",
        {
            "output_type": artifact["output_type"],
            "version": artifact["version"],
            "created_at": artifact["created_at"],
            "model": model or artifact["model"],
            "threshold_selection": "train_fold_target_sensitivity",
            "target_sensitivity": artifact["target_sensitivity"],
            "training_config_sha256": _text_sha256(
                str(artifact["training_config_yaml"])
            ),
            "decision_threshold": decision_threshold,
            "dataset_summary": _records(artifact["dataset_summary"]),
            "metric_summary": _records(artifact["metric_summary"]),
            "files": _evaluation_artifact_paths(folder, include_summary=False),
        },
    )


def _final_model_artifact(
    artifact: dict[str, Any],
    *,
    public_config: dict[str, Any],
    model_definition: dict[str, Any],
    training_config_yaml: str,
) -> dict[str, Any]:
    model_name = next(iter(artifact["models"]))
    model_info = artifact["models"][model_name]
    model_info["tissue_risk_assessment"] = derive_tra_policy(
        artifact["split_predictions"],
        decision_threshold=float(model_info["thresholds"]["threshold_target"]),
    )
    return {
        "kind": "aramis_training_artifact",
        "version": "0.3",
        "model_type": artifact["model_type"],
        "model_columns": artifact["model_columns"],
        "model_identity": {
            **dict(public_config["model"]),
        },
        "models": artifact["models"],
        "model_descriptions": artifact["model_descriptions"],
        "feature_schema": artifact["feature_schema"],
        "warnings": artifact["warnings"],
        "dataset_summary": artifact["dataset_summary"],
        "training_config_yaml": training_config_yaml,
        "historical_preprocessing_yaml": artifact.get("historical_preprocessing_yaml"),
        "prediction_preprocessing_yaml": artifact["prediction_preprocessing_yaml"],
        "prediction_contract_yaml": artifact["prediction_contract_yaml"],
        "model_definition_yaml": yaml.safe_dump(model_definition, sort_keys=False),
        "model_performance": _frozen_model_performance(
            artifact,
            evaluation_config=public_config["evaluation"],
            evaluation_requested=bool(public_config["run"]["evaluation"]),
            target_sensitivity=float(model_definition["target_sensitivity"]),
        ),
        "final_fit_training_metrics": _jsonable(
            artifact["models"][model_name]["final_fit_training_metrics"]
        ),
        "evaluation": {
            "protocol": dict(public_config["evaluation"]),
            "requested": bool(public_config["run"]["evaluation"]),
            "summary": _records(artifact["metric_summary"]),
            "artifacts": (
                {
                    "summary": "evaluation.yaml",
                    "metrics": "evaluation_metrics.csv",
                    "predictions": "evaluation_predictions.csv",
                }
                if public_config["run"]["evaluation"]
                else {}
            ),
        },
        "input_dataframe_joblib_sha256": artifact.get("input_dataframe_joblib_sha256"),
        "preprocessing_metadata": artifact.get("preprocessing_metadata", {}),
        "metadata": artifact["metadata"],
        "reproducibility": artifact["reproducibility"],
    }


def _frozen_model_performance(
    artifact: dict[str, Any],
    *,
    evaluation_config: dict[str, Any],
    evaluation_requested: bool,
    target_sensitivity: float,
) -> dict[str, Any]:
    """Create the concise performance record carried by a final model."""
    performance: dict[str, Any] = {
        "evaluation_available": evaluation_requested,
        "evaluation_method": str(evaluation_config["method"]),
        "folds": int(evaluation_config["folds"]),
        "repeats": int(evaluation_config["repeats"]),
        "random_seed": int(evaluation_config["random_seed"]),
        "target_sensitivity": target_sensitivity,
    }
    if not evaluation_requested:
        performance["held_out_metrics"] = {}
        return performance

    summary = artifact["metric_summary"]
    if len(summary) != 1:
        raise ValueError("Expected exactly one product evaluation summary.")
    row = summary.iloc[0]
    performance["held_out_metrics"] = {
        "roc_auc": {
            "mean": float(row["roc_auc_mean"]),
            "std": float(row["roc_auc_std"]),
        },
        "sensitivity": {
            "mean": float(row["sensitivity_target_mean"]),
            "std": float(row["sensitivity_target_std"]),
        },
        "specificity": {
            "mean": float(row["specificity_target_mean"]),
            "std": float(row["specificity_target_std"]),
        },
    }
    return performance
