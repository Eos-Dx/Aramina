"""Combined Aramina preprocessing and training entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from .config_paths import resolve_config_path
from .mlflow_artifacts import (
    MLFLOW_REQUIRED_ARTIFACTS,
    write_mlflow_product_artifacts,
)
from .mlflow_tracking import MlflowRun
from .pipelines import run_preprocessing_artifact_from_config
from .runtime_identity import aramina_git_sha, xrd_preprocessing_git_sha
from .training import run_training_from_config
from .training_config import load_training_config


PREPROCESS_TRAIN_CONTRACT = "aramina_preprocessing_and_training_config_v0_2"
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
    tracking_config = config["mlflow"]
    _require_complete_product_run(training_config_path, tracking_config)
    _require_product_source_provenance(tracking_config)
    run_folder = _preprocess_train_run_folder(config, config_path)
    dataframe_path = run_folder / "preprocessing" / "dataframe.joblib"
    dataframe_path.parent.mkdir(parents=True)

    tracking_uri = _tracking_uri(tracking_config["tracking_uri"], config_path)
    tracker = MlflowRun(
        enabled=tracking_config["enabled"],
        tracking_uri=tracking_uri,
        experiment_name=tracking_config["experiment_name"],
        run_name=run_folder.name,
        params={
            "workflow": {
                "name": config["preprocessing_and_training"]["name"],
                "run_author": config["preprocessing_and_training"]["run_author"],
            }
        },
        tags={
            "product": "aramina",
            "data_contract": PREPROCESS_TRAIN_CONTRACT,
        },
    )
    mlflow_bundle = None
    with tracker:
        logger.info("Preprocessing-and-training config: %s", config_path)
        logger.info("Preprocessing-and-training output: %s", run_folder)
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
        if tracker.enabled:
            mlflow_bundle = write_mlflow_product_artifacts(
                run_folder=run_folder,
                preprocessing_artifact=preprocessing_artifact,
                training_artifact=training_artifact,
                preprocessing_config_path=preprocessing_config_path,
                preprocess_train_contract=PREPROCESS_TRAIN_CONTRACT,
            )
            tracker.set_tags(mlflow_bundle["tags"])
            tracker.log_params(mlflow_bundle["params"])
            tracker.log_metrics(mlflow_bundle["metrics"])
            tracker.log_artifact_directory(
                mlflow_bundle["artifact_directory"],
                required_files=MLFLOW_REQUIRED_ARTIFACTS,
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
        "mlflow": {
            "enabled": tracker.enabled,
            "tracking_uri": tracking_uri,
            "experiment_name": tracking_config["experiment_name"],
            "run_id": tracker.run_id,
            "status": tracker.status,
            "artifact_directory": (
                str(mlflow_bundle["artifact_directory"])
                if mlflow_bundle is not None
                else None
            ),
        },
    }


def _load_preprocess_train_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Workflow config must be a mapping: {config_path}")
    required = {
        "contract",
        "preprocessing_and_training",
        "preprocessing_config_path",
        "training_config_path",
        "mlflow",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing preprocessing-and-training fields: {missing}")
    unknown = sorted(set(config).difference(required))
    if unknown:
        raise ValueError(f"Unknown preprocessing-and-training fields: {unknown}")
    if config["contract"] != PREPROCESS_TRAIN_CONTRACT:
        raise ValueError(
            f"Unsupported preprocessing-and-training contract: {config['contract']!r}"
        )
    preprocess_train = config["preprocessing_and_training"]
    fields = {"name", "run_author", "output_folder"}
    if not isinstance(preprocess_train, dict):
        raise TypeError("preprocessing_and_training must be a mapping.")
    missing = sorted(fields.difference(preprocess_train))
    if missing:
        raise ValueError(f"Missing preprocessing-and-training fields: {missing}")
    unknown = sorted(set(preprocess_train).difference(fields))
    if unknown:
        raise ValueError(f"Unknown preprocessing-and-training fields: {unknown}")
    for key in fields:
        _require_nonempty_string(
            preprocess_train[key], f"preprocessing_and_training.{key}"
        )
    _require_nonempty_string(
        config["preprocessing_config_path"], "preprocessing_config_path"
    )
    _require_nonempty_string(config["training_config_path"], "training_config_path")
    _validate_mlflow_config(config["mlflow"])
    return config


def _validate_mlflow_config(value: Any) -> None:
    if not isinstance(value, dict):
        raise TypeError("mlflow must be a mapping.")
    fields = {"enabled", "tracking_uri", "experiment_name"}
    missing = sorted(fields.difference(value))
    if missing:
        raise ValueError(f"Missing mlflow fields: {missing}")
    unknown = sorted(set(value).difference(fields))
    if unknown:
        raise ValueError(f"Unknown mlflow fields: {unknown}")
    if not isinstance(value["enabled"], bool):
        raise TypeError("mlflow.enabled must be boolean.")
    _require_nonempty_string(value["tracking_uri"], "mlflow.tracking_uri")
    _require_nonempty_string(value["experiment_name"], "mlflow.experiment_name")


def _require_complete_product_run(
    training_config_path: Path,
    tracking_config: dict[str, Any],
) -> None:
    if not tracking_config["enabled"]:
        return
    training_config, _ = load_training_config(training_config_path)
    if (
        not training_config["run"]["evaluation"]
        or not training_config["run"]["train_on_all"]
    ):
        raise ValueError(
            "Enabled product MLflow tracking requires evaluation=true and "
            "train_on_all=true."
        )


def _require_product_source_provenance(tracking_config: dict[str, Any]) -> None:
    if not tracking_config["enabled"]:
        return
    sources = {
        "Aramina": aramina_git_sha(),
        "XRD-preprocessing": xrd_preprocessing_git_sha(),
    }
    unavailable = [
        name for name, git_sha in sources.items() if git_sha == "unavailable"
    ]
    if unavailable:
        raise ValueError(
            "Enabled product MLflow tracking requires source Git SHAs before "
            f"preprocessing starts: {unavailable}"
        )


def _tracking_uri(value: str, config_path: Path) -> str:
    if value.startswith("sqlite:///"):
        return _sqlite_tracking_uri(value, config_path)
    if "://" in value or value.startswith("databricks"):
        return value
    path = _project_path(value, config_path)
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def _sqlite_tracking_uri(value: str, config_path: Path) -> str:
    """Resolve a relative SQLite database path from the configuration root."""
    database_path = value.removeprefix("sqlite:///")
    if database_path == ":memory:" or value.startswith("sqlite:////"):
        return value
    path = _project_path(database_path, config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _project_path(value: Any, config_path: Path) -> Path:
    return resolve_config_path(value, config_path)


def _require_nonempty_string(value: Any, where: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a string.")
    if not value.strip():
        raise ValueError(f"{where} must not be empty.")


def _preprocess_train_run_folder(config: dict[str, Any], config_path: Path) -> Path:
    preprocess_train = config["preprocessing_and_training"]
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
        "contract": "aramina_preprocessing_cohort_summary_v0_1",
        "rows": int(len(dataframe)),
        "patients": int(dataframe[patient_column].astype(str).nunique())
        if patient_column in dataframe
        else 0,
        "product_status_group_counts": labels,
        "biopsy_product_status_group_counts": biopsy_labels,
        "input_h5_sha256": metadata.get("input_h5_sha256"),
        "data_version": metadata.get("data_version"),
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
    return (
        dataframe[column]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
