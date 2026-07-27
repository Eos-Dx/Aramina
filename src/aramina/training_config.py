"""Strict public contract for the single Aramina product model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


TRAINING_CONTRACT = "aramina_training_config_v0_3"
PRODUCT_MODEL_NAME = "aramina_target_breast_risk"
PRODUCT_MODELS = {
    PRODUCT_MODEL_NAME: {
        "description": "Aramina target-breast risk model with age and optional SK symmetry refinement.",
        "class_definition": {
            "reference_class": "BENIGN",
            "target_class": "CANCER",
        },
        "model": {
            "type": "m2q_gated_target_case",
            "profile_column": "radial_profile_data",
            "label_column": "product_status_group",
            "group_column": "patientId",
            "specimen_column": "specimenId",
            "side_column": "side",
            "q_column": "q_range",
            "age_column": "age",
            "biopsy_column": "biopsy",
            "lr1_row_policy": "biopsy_only",
            "selected_models": [PRODUCT_MODEL_NAME],
            "lr1_logreg_c": 0.1,
            "lr2_logreg_c": 0.3,
        },
        "target_sensitivity": 0.95,
        "prediction_preprocessing_config_path": "config/preprocessing/config_preprocessing_prediction_patient_v0_1.yaml",
        "prediction_contract": {
            "container": {"schema_version": "0.3", "format": "xrd-session"},
            "reporting": {
                "external_report": {
                    "version": "0.6",
                    "reference_doc": "docs/modeling/prediction_pipeline_v0_1.md",
                },
                "internal_report": {
                    "version": "0.9",
                    "reference_doc": "docs/modeling/internal_clinical_report_content_v0_9.md",
                },
            },
            "decision": {"threshold_key": "threshold_target"},
        },
    }
}
PRODUCT_EVALUATION = {
    "method": "repeated_stratified_kfold",
    "folds": 5,
    "repeats": 20,
    "random_seed": 42,
}


def load_training_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and strictly validate one public training YAML."""
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    validate_training_config(config, source)
    return config, text


def validate_training_config(config: Any, source: str | Path) -> None:
    """Reject unknown or missing training contract fields."""
    source = Path(source)
    if not isinstance(config, dict):
        raise TypeError(f"Training config must be a mapping: {source}")
    _exact_keys(
        config,
        required={"contract", "input", "output", "model", "run", "evaluation"},
        allowed={"contract", "input", "output", "model", "run", "evaluation"},
        where="training config",
    )
    if config["contract"] != TRAINING_CONTRACT:
        raise ValueError(f"Unsupported training contract: {config['contract']!r}")
    _exact_keys(
        config["model"],
        required={"name", "version", "model_author", "clinical_stage", "intended_use"},
        allowed={"name", "version", "model_author", "clinical_stage", "intended_use"},
        where="model",
    )
    for key in ("name", "version", "model_author", "clinical_stage", "intended_use"):
        _require_nonempty_string(config["model"], key, f"model.{key}")
    resolve_model_definition(str(config["model"]["name"]))
    _exact_keys(
        config["run"],
        required={"evaluation", "train_on_all"},
        allowed={"evaluation", "train_on_all"},
        where="run",
    )
    if not all(isinstance(config["run"][key], bool) for key in ("evaluation", "train_on_all")):
        raise TypeError("run.evaluation and run.train_on_all must be boolean.")
    if not config["run"]["evaluation"] and not config["run"]["train_on_all"]:
        raise ValueError("At least one of run.evaluation or run.train_on_all must be true.")
    _exact_keys(config["input"], required={"dataframe_joblib_path"}, allowed={"dataframe_joblib_path"}, where="input")
    _exact_keys(config["output"], required={"folder"}, allowed={"folder"}, where="output")
    _require_nonempty_string(
        config["input"], "dataframe_joblib_path", "input.dataframe_joblib_path"
    )
    _require_nonempty_string(config["output"], "folder", "output.folder")
    _validate_evaluation(config["evaluation"])


def resolve_model_definition(model_name: str) -> dict[str, Any]:
    """Return the fixed implementation owned by one product model name."""
    try:
        return deepcopy(PRODUCT_MODELS[model_name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown Aramina model {model_name!r}; available: {sorted(PRODUCT_MODELS)}"
        ) from exc


def available_product_models() -> list[str]:
    """Return model names available in this Aramina version."""
    return sorted(PRODUCT_MODELS)


def describe_product_model(model_name: str) -> str:
    """Render the fixed model definition as readable YAML."""
    return yaml.safe_dump({model_name: resolve_model_definition(model_name)}, sort_keys=False)


def _validate_evaluation(evaluation: Any) -> None:
    if not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a mapping.")
    _exact_keys(
        evaluation,
        required={"method", "folds", "repeats", "random_seed"},
        allowed={"method", "folds", "repeats", "random_seed"},
        where="evaluation",
    )
    if evaluation["method"] != PRODUCT_EVALUATION["method"]:
        raise ValueError("evaluation.method must be 'repeated_stratified_kfold'.")
    _validate_int_at_least(evaluation["folds"], 2, "evaluation.folds")
    _validate_int_at_least(evaluation["repeats"], 1, "evaluation.repeats")
    _validate_int_at_least(evaluation["random_seed"], 0, "evaluation.random_seed")


def _validate_int_at_least(value: Any, minimum: int, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where} must be an integer.")
    if value < minimum:
        raise ValueError(f"{where} must be >= {minimum}.")


def _require_nonempty_string(section: dict[str, Any], key: str, where: str) -> None:
    value = section[key]
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a string.")
    if not value.strip():
        raise ValueError(f"{where} must not be empty.")


def _exact_keys(value: Any, *, required: set[str], allowed: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a mapping.")
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"Missing {where} fields: {missing}")
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {where} fields: {unknown}")
