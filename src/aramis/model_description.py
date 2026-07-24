"""Model description and serialisation helpers for immutable Aramis artifacts."""

from __future__ import annotations

import subprocess
import tomllib
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .m2q_model import GatedSymmetryLogistic, SK_CORE4_FEATURE_COLUMNS


def _safe_artifact_stem(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    ).strip("_")


def _model_artifact_id(model: dict[str, Any], model_sha: str) -> str:
    return _safe_artifact_stem(f"{model['name']}_{model['version']}_{model_sha[:12]}")


def _model_reference(
    model_identity: dict[str, Any],
    *,
    model_id: str | None = None,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    reference = {
        "name": str(model_identity["name"]),
        "version": str(model_identity["version"]),
    }
    if model_id is not None:
        reference["id"] = model_id
    if artifact_sha256 is not None:
        reference["artifact_sha256"] = artifact_sha256
    return reference


def _decision_threshold_record(model_artifact: dict[str, Any]) -> dict[str, Any]:
    model_name = next(iter(model_artifact["models"]))
    thresholds = model_artifact["models"][model_name]["thresholds"]
    return {
        "id": "target_sensitivity_0_95",
        "value": float(thresholds["threshold_target"]),
        "target_sensitivity": float(thresholds["target_sensitivity"]),
    }


def _model_description(
    artifact: dict[str, Any],
    *,
    model_id: str,
    model_sha: str,
    model_path: Path,
) -> dict[str, Any]:
    model_name = next(iter(artifact["models"]))
    model = artifact["models"][model_name]
    return {
        "output_type": "aramis_model_description",
        "version": "0.1",
        "model": _model_reference(
            artifact["model_identity"],
            model_id=model_id,
            artifact_sha256=model_sha,
        ),
        "model_summary": _model_summary(model),
        "model_joblib": model_path.name,
        "model_performance": artifact["model_performance"],
        "final_fit_training_metrics": _jsonable(model["final_fit_training_metrics"]),
        "decision_thresholds": _jsonable(model.get("thresholds", {})),
        "feature_schema": _jsonable(artifact["feature_schema"]),
        "dataset_summary": _records(artifact["dataset_summary"]),
        "evaluation_artifacts": _evaluation_artifact_paths(model_path.parent),
        "reproducibility": _jsonable(artifact["reproducibility"]),
        "clinical_stage": "research draft",
    }


def _write_model_input_snapshots(artifact: dict[str, Any], folder: Path) -> None:
    """Write the frozen YAML inputs required to reproduce a final-fit run."""
    snapshots = {
        "preprocessing_config.yaml": artifact.get("historical_preprocessing_yaml"),
        "prediction_preprocessing_config.yaml": artifact.get(
            "prediction_preprocessing_yaml"
        ),
        "training_config.yaml": artifact.get("training_config_yaml"),
        "preprocess_and_train_config.yaml": artifact.get("reproducibility", {})
        .get("configs", {})
        .get("preprocess_train_yaml"),
    }
    for filename, text in snapshots.items():
        if isinstance(text, str):
            (folder / filename).write_text(text, encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(_round_yaml_values(_jsonable(payload)), sort_keys=False),
        encoding="utf-8",
    )


def _round_yaml_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round_yaml_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_yaml_values(item) for item in value]
    if isinstance(value, float):
        return round(value, 5)
    return value


def _model_summary(model_info: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "architecture": {
            "stage_1": "target_xrd_profile_logistic_regression",
            "stage_2": "age_and_optional_symmetry_refinement",
            "symmetry_behavior": "neutralized_unless_2_valid_measurements_per_breast_and_finite_core4_features",
        },
        "lr1_profile_model": _pipeline_summary(model_info.get("lr1_model")),
        "thresholds": _jsonable(model_info.get("thresholds", {})),
    }
    if "routes" not in model_info:
        summary["feature_columns"] = list(model_info.get("feature_columns", []))
        summary["final_model"] = _pipeline_summary(model_info.get("final_model"))
        if "symmetry_policy" in model_info:
            summary["symmetry_policy"] = model_info["symmetry_policy"]
            summary["symmetry_gate"] = model_info["symmetry_gate"]
            summary["symmetry_feature_contract"] = model_info.get(
                "symmetry_feature_contract", "aramis_sk_symmetry_v0_1"
            )
        if "tissue_risk_assessment" in model_info:
            summary["tissue_risk_assessment"] = _jsonable(
                model_info["tissue_risk_assessment"]
            )
        return summary

    summary["routing_field"] = model_info.get("routing_field")
    summary["routing_policy"] = model_info.get("routing_policy")
    summary["routes"] = {
        route_name: {
            "feature_columns": list(route_info.get("feature_columns", [])),
            "training_patients": route_info.get("training_patients"),
            "thresholds": _jsonable(route_info.get("thresholds", {})),
            "final_model": _pipeline_summary(route_info.get("final_model")),
        }
        for route_name, route_info in model_info["routes"].items()
    }
    return summary


def _pipeline_summary(
    model: Pipeline | GatedSymmetryLogistic | None,
) -> dict[str, Any] | None:
    """Describe architecture and frozen hyperparameters, not learned weights."""
    if model is None:
        return None
    if isinstance(model, GatedSymmetryLogistic):
        return {
            "type": type(model).__name__,
            "base_feature_columns": list(model.base_feature_columns_),
            "symmetry_feature_columns": list(SK_CORE4_FEATURE_COLUMNS),
            "symmetry_gate": "symmetry_available",
            "logreg": {
                "C": float(model.logreg_.C),
                "class_weight": model.logreg_.class_weight,
                "solver": model.logreg_.solver,
                "max_iter": int(model.logreg_.max_iter),
                "random_state": model.logreg_.random_state,
                "classes": _class_labels(model.logreg_.classes_),
                "feature_count": int(model.logreg_.n_features_in_),
            },
        }
    summary: dict[str, Any] = {"type": type(model).__name__, "steps": {}}
    for step_name, step in model.named_steps.items():
        step_summary: dict[str, Any] = {"type": type(step).__name__}
        if isinstance(step, SimpleImputer):
            step_summary["strategy"] = step.strategy
        elif isinstance(step, StandardScaler):
            step_summary["feature_count"] = int(step.n_features_in_)
        elif isinstance(step, LogisticRegression):
            step_summary.update(
                {
                    "C": float(step.C),
                    "class_weight": step.class_weight,
                    "solver": step.solver,
                    "max_iter": int(step.max_iter),
                    "random_state": step.random_state,
                    "classes": _class_labels(step.classes_),
                    "feature_count": int(step.n_features_in_),
                }
            )
        summary["steps"][step_name] = step_summary
    return summary


def _class_labels(classes: Any) -> list[str]:
    labels = {0: "BENIGN", 1: "CANCER"}
    return [labels.get(int(value), str(value)) for value in np.asarray(classes)]


def _evaluation_artifact_paths(
    folder: Path,
    *,
    include_summary: bool = True,
) -> dict[str, str]:
    paths = {
        "summary": "evaluation.yaml",
        "metrics": "evaluation_metrics.csv",
        "predictions": "evaluation_predictions.csv",
    }
    if not include_summary:
        paths.pop("summary")
    return paths


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aramis_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.exists():
        with pyproject_path.open("rb") as handle:
            pyproject = tomllib.load(handle)
        return str(pyproject.get("project", {}).get("version", "unknown"))
    try:
        return version("aramis")
    except PackageNotFoundError:
        return "unknown"


def _aramis_git_sha() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return "unavailable"
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
