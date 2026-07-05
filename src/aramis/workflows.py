"""End-to-end Aramis workflow entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from xrd_preprocessing import load_preprocessing_config

from .pipelines import run_preprocessing_artifact_from_config, run_preprocessing_from_config
from .training import run_training_from_config


def run_workflow_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run an Aramis preprocess+train workflow declared by one YAML file.

    The workflow YAML references two sub-YAML files: one preprocessing config and
    one training config. The preprocessing config writes the DataFrame joblib;
    the training config reads that joblib and writes the model artifacts.
    """
    config_path = Path(config_path)
    config = _load_workflow_config(config_path)
    preprocessing_config_path = _workflow_config_path(
        config,
        config_path,
        section="preprocessing",
    )
    training_config_path = _workflow_config_path(
        config,
        config_path,
        section="training",
    )
    if bool(config.get("workflow", {}).get("validate_io_match", True)):
        _validate_preprocess_train_paths(preprocessing_config_path, training_config_path)
    mode = str(config.get("workflow", {}).get("mode", "memory")).lower()
    if mode not in {"memory", "artifact"}:
        raise ValueError(f"Unsupported workflow.mode: {mode!r}")
    preprocessing_artifact = None
    preprocessing_df = None
    if bool(config.get("workflow", {}).get("run_preprocessing", True)):
        if mode == "memory":
            preprocessing_artifact = run_preprocessing_artifact_from_config(
                preprocessing_config_path
            )
            preprocessing_df = preprocessing_artifact["dataframe"]
        else:
            preprocessing_df = run_preprocessing_from_config(preprocessing_config_path)
    training_artifact = None
    if bool(config.get("workflow", {}).get("run_training", True)):
        if mode == "memory" and preprocessing_df is not None:
            training_artifact = run_training_from_config(
                training_config_path,
                dataframe=preprocessing_df,
                preprocessing_artifact=preprocessing_artifact,
            )
        else:
            training_artifact = run_training_from_config(training_config_path)
    return {
        "workflow_config_path": config_path,
        "preprocessing_config_path": preprocessing_config_path,
        "training_config_path": training_config_path,
        "mode": mode,
        "preprocessing_artifact": preprocessing_artifact,
        "preprocessing_dataframe": preprocessing_df,
        "training_artifact": training_artifact,
    }


def _load_workflow_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Workflow config must be a mapping: {config_path}")
    for section in ("workflow", "preprocessing", "training"):
        if section not in config:
            raise ValueError(f"Missing workflow config section: {section}")
    return config


def _workflow_config_path(
    config: dict[str, Any],
    config_path: Path,
    *,
    section: str,
) -> Path:
    value = config.get(section, {}).get("config_path")
    if value in {None, ""}:
        raise ValueError(f"Missing {section}.config_path in {config_path}")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _validate_preprocess_train_paths(
    preprocessing_config_path: Path,
    training_config_path: Path,
) -> None:
    preprocessing_config = load_preprocessing_config(preprocessing_config_path)
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    preprocessing_output = _config_path(
        preprocessing_config,
        preprocessing_config_path,
        section="io",
        key="output_joblib_path",
    )
    training_input = _config_path(
        training_config,
        training_config_path,
        section="io",
        key="input_dataframe_joblib_path",
    )
    if preprocessing_output != training_input:
        raise ValueError(
            "Workflow preprocessing output does not match training input: "
            f"{preprocessing_output} != {training_input}"
        )


def _config_path(
    config: dict[str, Any],
    config_path: Path,
    *,
    section: str,
    key: str,
) -> Path:
    value = config.get(section, {}).get(key)
    if value in {None, ""}:
        raise ValueError(f"Missing {section}.{key} in {config_path}")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()
