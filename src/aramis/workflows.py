"""Combined Aramis preprocessing and training workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from .pipelines import run_preprocessing_artifact_from_config
from .training import run_training_from_config


WORKFLOW_CONTRACT = "aramis_preprocess_train_workflow_v0_1"


def run_preprocess_train_from_config(config_path: str | Path) -> dict[str, Any]:
    """Preprocess once, persist the DataFrame, then pass it directly to training."""
    config_path = Path(config_path).expanduser().resolve()
    config = _load_workflow_config(config_path)
    preprocessing_config_path = _relative_path(
        config["preprocessing_config_path"], config_path
    )
    training_config_path = _relative_path(config["training_config_path"], config_path)
    run_folder = _workflow_run_folder(config, config_path)
    dataframe_path = run_folder / "preprocessing" / "dataframe.joblib"
    dataframe_path.parent.mkdir(parents=True)

    preprocessing_artifact = run_preprocessing_artifact_from_config(
        preprocessing_config_path,
        output_joblib_path=dataframe_path,
    )
    training_artifact = run_training_from_config(
        training_config_path,
        dataframe=preprocessing_artifact["dataframe"],
        preprocessing_artifact=preprocessing_artifact,
        dataframe_joblib_path=dataframe_path,
        output_folder=run_folder / "training",
        workflow_config_yaml=config_path.read_text(encoding="utf-8"),
    )
    return {
        "workflow_config_path": config_path,
        "preprocessing_config_path": preprocessing_config_path,
        "training_config_path": training_config_path,
        "run_folder": run_folder,
        "preprocessing_dataframe": preprocessing_artifact["dataframe"],
        "preprocessing_artifact": preprocessing_artifact,
        "training_artifact": training_artifact,
    }


def _load_workflow_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Workflow config must be a mapping: {config_path}")
    required = {
        "contract",
        "workflow",
        "preprocessing_config_path",
        "training_config_path",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing workflow fields: {missing}")
    unknown = sorted(set(config).difference(required))
    if unknown:
        raise ValueError(f"Unknown workflow fields: {unknown}")
    if config["contract"] != WORKFLOW_CONTRACT:
        raise ValueError(f"Unsupported workflow contract: {config['contract']!r}")
    workflow = config["workflow"]
    workflow_fields = {"name", "created_by", "created_at", "output_folder"}
    if not isinstance(workflow, dict):
        raise TypeError("workflow must be a mapping.")
    missing = sorted(workflow_fields.difference(workflow))
    if missing:
        raise ValueError(f"Missing workflow fields: {missing}")
    unknown = sorted(set(workflow).difference(workflow_fields))
    if unknown:
        raise ValueError(f"Unknown workflow fields: {unknown}")
    return config


def _relative_path(value: Any, config_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _workflow_run_folder(config: dict[str, Any], config_path: Path) -> Path:
    workflow = config["workflow"]
    root = _relative_path(workflow["output_folder"], config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(workflow["name"])
    ).strip("_")
    folder = root / f"{name}_{stamp}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder
