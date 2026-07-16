"""Combined Aramis preprocessing and training entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from .pipelines import run_preprocessing_artifact_from_config
from .training import run_training_from_config


PREPROCESS_TRAIN_CONTRACT = "aramis_preprocess_train_config_v0_1"
logger = logging.getLogger(__name__)


def run_preprocess_train_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Preprocess once, persist the DataFrame, then pass it directly to training."""
    config_path = Path(config_path).expanduser().resolve()
    config = _load_preprocess_train_config(config_path)
    preprocessing_config_path = _project_path(
        config["preprocessing_config_path"], config_path
    )
    training_config_path = _project_path(config["training_config_path"], config_path)
    run_folder = _preprocess_train_run_folder(config, config_path)
    dataframe_path = run_folder / "preprocessing" / "dataframe.joblib"
    dataframe_path.parent.mkdir(parents=True)

    logger.info("Preprocess-train config: %s", config_path)
    logger.info("Preprocess-train output: %s", run_folder)
    logger.info("Stage 1/2: preprocessing")
    preprocessing_kwargs = {"verbose": True} if verbose else {}
    preprocessing_artifact = run_preprocessing_artifact_from_config(
        preprocessing_config_path,
        output_joblib_path=dataframe_path,
        **preprocessing_kwargs,
    )
    cohort_summary = _write_preprocessing_summary(
        preprocessing_artifact,
        dataframe_path.parent / "cohort_summary.json",
    )
    logger.info(
        "Preprocessing cohort: rows=%d patients=%d labels=%s biopsy_labels=%s",
        cohort_summary["rows"],
        cohort_summary["patients"],
        cohort_summary["product_status_group_counts"],
        cohort_summary["biopsy_product_status_group_counts"],
    )
    logger.info("Stage 2/2: training")
    training_artifact = run_training_from_config(
        training_config_path,
        dataframe=preprocessing_artifact["dataframe"],
        preprocessing_artifact=preprocessing_artifact,
        dataframe_joblib_path=dataframe_path,
        output_folder=run_folder / "training",
        preprocess_train_config_yaml=config_path.read_text(encoding="utf-8"),
    )
    logger.info("Preprocess-train complete: %s", run_folder)
    return {
        "preprocess_train_config_path": config_path,
        "preprocessing_config_path": preprocessing_config_path,
        "training_config_path": training_config_path,
        "run_folder": run_folder,
        "preprocessing_dataframe": preprocessing_artifact["dataframe"],
        "preprocessing_artifact": preprocessing_artifact,
        "training_artifact": training_artifact,
    }


def _load_preprocess_train_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Workflow config must be a mapping: {config_path}")
    required = {
        "contract",
        "preprocess_train",
        "preprocessing_config_path",
        "training_config_path",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing preprocess-train fields: {missing}")
    unknown = sorted(set(config).difference(required))
    if unknown:
        raise ValueError(f"Unknown preprocess-train fields: {unknown}")
    if config["contract"] != PREPROCESS_TRAIN_CONTRACT:
        raise ValueError(f"Unsupported preprocess-train contract: {config['contract']!r}")
    preprocess_train = config["preprocess_train"]
    fields = {"name", "created_by", "output_folder"}
    if not isinstance(preprocess_train, dict):
        raise TypeError("preprocess_train must be a mapping.")
    missing = sorted(fields.difference(preprocess_train))
    if missing:
        raise ValueError(f"Missing preprocess-train fields: {missing}")
    unknown = sorted(set(preprocess_train).difference(fields))
    if unknown:
        raise ValueError(f"Unknown preprocess-train fields: {unknown}")
    return config


def _project_path(value: Any, config_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (_project_root(config_path) / path).resolve()


def _project_root(config_path: Path) -> Path:
    """Find the project root for a config stored under ``config/``."""
    for parent in config_path.parents:
        if parent.name == "config":
            return parent.parent
    return config_path.parent


def _preprocess_train_run_folder(config: dict[str, Any], config_path: Path) -> Path:
    preprocess_train = config["preprocess_train"]
    root = _project_path(preprocess_train["output_folder"], config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(preprocess_train["name"])
    ).strip("_")
    folder = root / f"{name}_{stamp}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _write_preprocessing_summary(
    artifact: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Persist label counts before training so a failed run remains inspectable."""
    dataframe = artifact["dataframe"]
    label_column = "product_status_group"
    biopsy_column = "biopsy"
    patient_column = "patientId"
    labels = _value_counts(dataframe, label_column)
    biopsy_labels = _value_counts(
        dataframe.loc[_boolean_values(dataframe, biopsy_column)], label_column
    )
    metadata = artifact.get("metadata", {})
    summary = {
        "contract": "aramis_preprocessing_cohort_summary_v0_1",
        "rows": int(len(dataframe)),
        "patients": int(dataframe[patient_column].astype(str).nunique())
        if patient_column in dataframe
        else 0,
        "product_status_group_counts": labels,
        "biopsy_product_status_group_counts": biopsy_labels,
        "input_h5_sha256": metadata.get("input_h5_sha256"),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _value_counts(dataframe: Any, column: str) -> dict[str, int]:
    if column not in dataframe:
        return {}
    return {
        str(value): int(count)
        for value, count in dataframe[column].fillna("<missing>").value_counts().items()
    }


def _boolean_values(dataframe: Any, column: str) -> Any:
    if column not in dataframe:
        return [False] * len(dataframe)
    return dataframe[column].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
