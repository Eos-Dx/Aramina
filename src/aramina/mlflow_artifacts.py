"""Build the auditable artifact set for one product MLflow run."""

from __future__ import annotations

from datetime import date, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import struct
from typing import Any

import numpy as np
import pandas as pd
import yaml
from xrd_preprocessing import list_h5_measurement_sets

from .config_paths import resolve_config_path


ARTIFACT_CONTRACT = "aramina_mlflow_product_run_v0_1"
INTENDED_USE_ID = "aramina_target_breast_biopsy_decision_support_v0_1"
MLFLOW_REQUIRED_ARTIFACTS = (
    "preprocessing_config.json",
    "product_filter_rules.json",
    "selected_measurement_ids.csv",
    "dropped_measurements.csv",
    "preprocessed_dataset.parquet",
    "feature_schema.json",
    "label_mapping.json",
    "train_test_split.csv",
    "model.joblib",
    "metrics.json",
    "predictions.csv",
)
_IDENTITY_COLUMNS = (
    "patientId",
    "specimenId",
    "side",
    "position",
    "started_at",
)


def write_mlflow_product_artifacts(
    *,
    run_folder: str | Path,
    preprocessing_artifact: dict[str, Any],
    training_artifact: dict[str, Any],
    preprocessing_config_path: str | Path,
    preprocess_train_contract: str,
) -> dict[str, Any]:
    """Write the complete product lineage package and return MLflow records."""
    root = Path(run_folder) / "mlflow_artifacts"
    root.mkdir(parents=True, exist_ok=False)
    dataframe = _require_dataframe(preprocessing_artifact)
    preprocessing_config = _preprocessing_config(preprocessing_artifact)
    training_folder = Path(str(training_artifact["run_folder"]))

    _write_json(root / "preprocessing_config.json", preprocessing_config)
    _write_json(
        root / "product_filter_rules.json",
        _product_filter_rules(preprocessing_config),
    )
    _write_preprocessed_dataset(dataframe, root / "preprocessed_dataset.parquet")
    selected, dropped = _measurement_manifests(
        dataframe=dataframe,
        preprocessing_config=preprocessing_config,
        preprocessing_config_path=Path(preprocessing_config_path),
    )
    selected.to_csv(root / "selected_measurement_ids.csv", index=False)
    dropped.to_csv(root / "dropped_measurements.csv", index=False)
    feature_schema = _feature_schema(training_artifact, dataframe)
    _write_json(root / "feature_schema.json", feature_schema)
    _write_json(
        root / "label_mapping.json",
        _label_mapping(training_artifact, preprocessing_config),
    )
    _write_split_manifest(
        training_folder=training_folder,
        dataframe=dataframe,
        path=root / "train_test_split.csv",
    )
    _copy_required(training_folder / "model.joblib", root / "model.joblib")
    _copy_required(
        training_folder / "evaluation_predictions.csv",
        root / "predictions.csv",
    )
    metrics = _metrics_record(training_artifact, training_folder)
    _write_json(root / "metrics.json", metrics)

    dataset_fingerprint = _dataset_fingerprint(
        dataframe=dataframe,
        selected_measurements=selected,
        feature_schema=feature_schema,
    )
    tags = _mlflow_tags(
        training_artifact=training_artifact,
        preprocessing_config=preprocessing_config,
        preprocess_train_contract=preprocess_train_contract,
        dataset_fingerprint=dataset_fingerprint,
    )
    _require_complete_tags(tags)
    params = _mlflow_params(training_artifact, preprocessing_config)
    scalar_metrics = _scalar_metrics(metrics)
    manifest = {
        "contract": ARTIFACT_CONTRACT,
        "dataset_fingerprint": dataset_fingerprint,
        "selected_measurements": int(len(selected)),
        "dropped_measurements": int(len(dropped)),
        "artifacts": sorted(path.name for path in root.iterdir()),
        "tags": tags,
        "params": params,
        "metrics": scalar_metrics,
    }
    _write_json(root / "mlflow_manifest.json", manifest)
    return {
        "artifact_directory": root,
        "tags": tags,
        "params": params,
        "metrics": scalar_metrics,
        "manifest": manifest,
    }


def _require_dataframe(artifact: dict[str, Any]) -> pd.DataFrame:
    dataframe = artifact.get("dataframe")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("MLflow tracking requires preprocessing_artifact.dataframe.")
    return dataframe


def _preprocessing_config(artifact: dict[str, Any]) -> dict[str, Any]:
    text = artifact.get("preprocessing_config_yaml")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("MLflow tracking requires resolved preprocessing YAML.")
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise TypeError("Resolved preprocessing YAML must be a mapping.")
    return config


def _product_filter_rules(config: dict[str, Any]) -> dict[str, Any]:
    pipeline = config.get("pipeline", {})
    steps = pipeline.get("steps", []) if isinstance(pipeline, dict) else []
    return {
        "contract": "aramina_product_filter_rules_v0_1",
        "aramina_preprocessing": config.get("aramina_preprocessing", {}),
        "raw_data": config.get("raw_data", {}),
        "filters": config.get("filters", {}),
        "product_filter": config.get("product_filter", {}),
        "labels": config.get("labels", {}),
        "integration": config.get("integration", {}),
        "snr": config.get("snr", {}),
        "normalization": config.get("normalization", {}),
        "profile_gate": config.get("profile_gate", {}),
        "pipeline_steps": [
            step.get("name") for step in steps if isinstance(step, dict)
        ],
    }


def _write_preprocessed_dataset(dataframe: pd.DataFrame, path: Path) -> None:
    output = dataframe.copy()
    for column in output.select_dtypes(include="object").columns:
        output[column] = output[column].map(_parquet_value)
    output.to_parquet(path, index=False)


def _parquet_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _measurement_manifests(
    *,
    dataframe: pd.DataFrame,
    preprocessing_config: dict[str, Any],
    preprocessing_config_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [column for column in _IDENTITY_COLUMNS if column not in dataframe]
    if missing:
        raise KeyError(f"Preprocessed data lacks measurement identity columns: {missing}")
    input_h5 = resolve_config_path(
        preprocessing_config.get("io", {}).get("input_h5_path"),
        preprocessing_config_path,
    )
    candidates = list_h5_measurement_sets(
        input_h5,
        session_category="SAMPLE",
        set_category="SAMPLE",
    )
    candidate_missing = [
        column for column in _IDENTITY_COLUMNS if column not in candidates
    ]
    if candidate_missing:
        raise KeyError(f"H5 audit lacks measurement identity columns: {candidate_missing}")

    candidate_records = _identity_records(candidates)
    selected_records = _identity_records(dataframe)
    candidate_by_key = {
        record["identity_key"]: record for record in candidate_records
    }
    selected_keys = {record["identity_key"] for record in selected_records}
    selected = []
    for record in selected_records:
        candidate = candidate_by_key.get(record["identity_key"], {})
        selected.append(
            {
                "measurement_id": candidate.get(
                    "measurement_id", record["measurement_id"]
                ),
                **{key: record[key] for key in _manifest_identity_names()},
                "session_uid": candidate.get("session_uid", "unavailable"),
                "set_path": candidate.get("set_path", "unavailable"),
                "biopsy": record.get("biopsy", "unknown"),
                "product_status_group": record.get(
                    "product_status_group", "unknown"
                ),
                "target_case_eligible": record.get(
                    "target_case_eligible", False
                ),
            }
        )
    dropped = [
        {
            "measurement_id": record["measurement_id"],
            **{key: record[key] for key in _manifest_identity_names()},
            "session_uid": record.get("session_uid", "unavailable"),
            "set_path": record.get("set_path", "unavailable"),
            "biopsy": record.get("biopsy", "unknown"),
            "product_status_group": record.get("product_status_group", "unknown"),
            "drop_stage": "product_preprocessing",
            "drop_reason": "not_selected_by_resolved_product_pipeline",
        }
        for record in candidate_records
        if record["identity_key"] not in selected_keys
    ]
    return pd.DataFrame(selected), pd.DataFrame(dropped)


def _identity_records(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    records = []
    for row in dataframe.to_dict(orient="records"):
        values = {
            "patient_id": _identity_value(row.get("patientId")),
            "specimen_id": _identity_value(row.get("specimenId")),
            "side": _identity_value(row.get("side")).lower(),
            "position": _identity_value(row.get("position")).upper(),
            "started_at": _timestamp_value(row.get("started_at")),
        }
        identity_key = "|".join(values.values())
        records.append(
            {
                **values,
                "identity_key": identity_key,
                "measurement_id": sha256(identity_key.encode("utf-8")).hexdigest(),
                "session_uid": _identity_value(row.get("session_uid")),
                "set_path": _identity_value(row.get("set_path")),
                "biopsy": _boolean_text(row.get("biopsy")),
                "product_status_group": _identity_value(
                    row.get("product_status_group")
                ).upper(),
                "target_case_eligible": _target_case_eligible(row),
            }
        )
    return records


def _manifest_identity_names() -> tuple[str, ...]:
    return ("patient_id", "specimen_id", "side", "position", "started_at")


def _identity_value(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "unknown"
    return str(value).strip()


def _timestamp_value(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "unknown"
    timestamp = pd.Timestamp(value)
    return timestamp.isoformat()


def _boolean_text(value: Any) -> str:
    return "true" if str(value).strip().lower() in {"true", "1", "yes"} else "false"


def _target_case_eligible(row: dict[str, Any]) -> bool:
    return (
        _boolean_text(row.get("biopsy")) == "true"
        and _identity_value(row.get("product_status_group")).upper()
        in {"BENIGN", "CANCER"}
    )


def _feature_schema(
    training_artifact: dict[str, Any],
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    model_name = next(iter(training_artifact["models"]))
    model = training_artifact["models"][model_name]
    return {
        "contract": "aramina_feature_schema_v0_1",
        "model_features": training_artifact.get("feature_schema", {}),
        "model_columns": training_artifact.get("model_columns", {}),
        "profile_encoder": model.get("profile_encoder", {}),
        "preprocessed_columns": [
            {"name": column, "dtype": str(dataframe[column].dtype)}
            for column in dataframe.columns
        ],
    }


def _label_mapping(
    training_artifact: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, Any]:
    model_definition = yaml.safe_load(
        str(training_artifact.get("model_definition_yaml", "{}"))
    )
    return {
        "contract": "aramina_label_mapping_v0_1",
        "class_definition": model_definition.get("class_definition", {}),
        "preprocessing_labels": preprocessing_config.get("labels", {}),
        "numeric_mapping": {"BENIGN": 0, "CANCER": 1},
    }


def _write_split_manifest(
    *,
    training_folder: Path,
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    predictions = pd.read_csv(training_folder / "evaluation_predictions.csv")
    required = {"split_id", "patientId", "evaluation_mode"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"Evaluation predictions lack split columns: {missing}")
    patients = sorted(dataframe["patientId"].astype(str).unique())
    rows = []
    for split_id, split in predictions.groupby("split_id", sort=True):
        test_patients = set(split["patientId"].astype(str))
        if not test_patients.issubset(patients):
            raise RuntimeError("Evaluation split contains an unknown patient.")
        mode = str(split["evaluation_mode"].iloc[0])
        rows.extend(
            {
                "split_id": int(split_id),
                "evaluation_mode": mode,
                "patient_id": patient_id,
                "partition": "test" if patient_id in test_patients else "train",
            }
            for patient_id in patients
        )
    manifest = pd.DataFrame(rows)
    overlap = (
        manifest.groupby(["split_id", "patient_id"])["partition"].nunique() > 1
    ).any()
    if overlap:
        raise RuntimeError("Patient leakage detected in MLflow split manifest.")
    manifest.to_csv(path, index=False)


def _metrics_record(
    training_artifact: dict[str, Any],
    training_folder: Path,
) -> dict[str, Any]:
    evaluation = yaml.safe_load(
        (training_folder / "evaluation.yaml").read_text(encoding="utf-8")
    )
    return {
        "contract": "aramina_metrics_v0_1",
        "evaluation": evaluation,
        "final_fit_training_metrics": training_artifact.get(
            "final_fit_training_metrics", {}
        ),
        "model_performance": training_artifact.get("model_performance", {}),
    }


def _mlflow_tags(
    *,
    training_artifact: dict[str, Any],
    preprocessing_config: dict[str, Any],
    preprocess_train_contract: str,
    dataset_fingerprint: str,
) -> dict[str, str]:
    reproducibility = training_artifact.get("reproducibility", {})
    source_code = reproducibility.get("source_code", {})
    xrd = source_code.get("xrd_preprocessing", {})
    model_identity = training_artifact.get("model_identity", {})
    source_h5 = reproducibility.get("source_h5", {})
    preprocessing = preprocessing_config.get("aramina_preprocessing", {})
    return {
        "product": "aramina",
        "intended_use_id": INTENDED_USE_ID,
        "clinical_stage": str(model_identity.get("clinical_stage", "research draft")),
        "data_contract": preprocess_train_contract,
        "input_h5_id": str(source_h5.get("filename", "unknown")),
        "input_h5_checksum": str(source_h5.get("sha256", "unavailable")),
        "pipeline_version": str(preprocessing.get("version", "unknown")),
        "preprocessing_git_sha": str(
            xrd.get("git_commit") or xrd.get("requested_revision") or "unavailable"
        ),
        "model_git_sha": str(
            source_code.get("aramina", {}).get("git_sha", "unavailable")
        ),
        "dataset_fingerprint": dataset_fingerprint,
        "validation_status": "research_draft_patient_safe_internal_evaluation",
    }


def _require_complete_tags(tags: dict[str, str]) -> None:
    invalid = sorted(
        key
        for key, value in tags.items()
        if not value.strip() or value.lower() in {"unknown", "unavailable", "null"}
    )
    if invalid:
        raise ValueError(f"Required MLflow tags are unavailable: {invalid}")


def _mlflow_params(
    training_artifact: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    model_name = next(iter(training_artifact["models"]))
    model = training_artifact["models"][model_name]
    identity = training_artifact.get("model_identity", {})
    evaluation = training_artifact.get("evaluation", {}).get("protocol", {})
    encoder = model.get("profile_encoder", {})
    integration = preprocessing_config.get("integration", {})
    snr = preprocessing_config.get("snr", {})
    normalization = preprocessing_config.get("normalization", {})
    thresholds = model.get("thresholds", {})
    return {
        "model.name": str(identity.get("name", model_name)),
        "model.version": str(identity.get("version", "unknown")),
        "model.type": str(training_artifact.get("model_type", "unknown")),
        "profile_encoder.type": str(encoder.get("type", "unknown")),
        "profile_encoder.input_q_bins": int(encoder.get("input_q_bins", 0)),
        "profile_encoder.components": int(encoder.get("components", 0)),
        "integration.npt": int(integration.get("npt", 0)),
        "integration.error_model": str(integration.get("error_model", "unknown")),
        "snr.method": str(snr.get("method", "unknown")),
        "snr.min_db": float(snr.get("min_snr_db", float("nan"))),
        "normalization.q_range": json.dumps(
            normalization.get("q_range_nm_inv", [])
        ),
        "evaluation.method": str(evaluation.get("method", "unknown")),
        "evaluation.folds": int(evaluation.get("folds", 0)),
        "evaluation.repeats": int(evaluation.get("repeats", 0)),
        "evaluation.random_seed": int(evaluation.get("random_seed", 0)),
        "decision_threshold": float(thresholds.get("threshold_target", float("nan"))),
    }


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    held_out = metrics.get("model_performance", {}).get("held_out_metrics", {})
    for metric_name, record in held_out.items():
        if not isinstance(record, dict):
            continue
        for statistic in ("mean", "std"):
            value = record.get(statistic)
            if _finite_number(value):
                values[f"held_out.{metric_name}.{statistic}"] = float(value)
    for name, value in metrics.get("final_fit_training_metrics", {}).items():
        if _finite_number(value):
            values[f"final_fit.{name}"] = float(value)
    return values


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float | np.integer | np.floating)
        and np.isfinite(value)
    )


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required MLflow source artifact is missing: {source}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _dataset_fingerprint(
    *,
    dataframe: pd.DataFrame,
    selected_measurements: pd.DataFrame,
    feature_schema: dict[str, Any],
) -> str:
    """Hash accepted values, identities, and schema independent of file encoding."""
    digest = sha256()
    digest.update(
        json.dumps(feature_schema, sort_keys=True, default=_json_default).encode("utf-8")
    )
    selected_ids = sorted(selected_measurements["measurement_id"].astype(str))
    for measurement_id in selected_ids:
        _update_hash(digest, measurement_id)
    sort_columns = [column for column in _IDENTITY_COLUMNS if column in dataframe]
    ordered = dataframe.sort_values(sort_columns, kind="stable")
    for column in ordered.columns:
        digest.update(column.encode("utf-8"))
        digest.update(str(ordered[column].dtype).encode("utf-8"))
    for row in ordered.to_dict(orient="records"):
        for column in ordered.columns:
            digest.update(column.encode("utf-8"))
            _update_hash(digest, row[column])
    return digest.hexdigest()


def _update_hash(digest: Any, value: Any) -> None:
    if isinstance(value, dict):
        digest.update(b"mapping")
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _update_hash(digest, value[key])
        return
    if isinstance(value, np.ndarray | list | tuple):
        array = np.asarray(value)
        digest.update(b"array")
        digest.update(str(array.shape).encode("ascii"))
        if np.issubdtype(array.dtype, np.number):
            numeric = np.asarray(array, dtype=np.float64)
            numeric[np.isnan(numeric)] = np.nan
            digest.update(numeric.astype(">f8", copy=False).tobytes())
        else:
            for item in array.ravel().tolist():
                _update_hash(digest, item)
        return
    if value is None or (
        not isinstance(value, str) and not isinstance(value, bytes) and pd.isna(value)
    ):
        digest.update(b"null")
        return
    if isinstance(value, bool | np.bool_):
        digest.update(b"bool:1" if bool(value) else b"bool:0")
        return
    if isinstance(value, int | np.integer):
        digest.update(f"int:{int(value)}".encode("ascii"))
        return
    if isinstance(value, float | np.floating):
        digest.update(b"float:")
        digest.update(struct.pack(">d", float(value)))
        return
    if isinstance(value, pd.Timestamp | date | datetime):
        digest.update(value.isoformat().encode("utf-8"))
        return
    if isinstance(value, bytes):
        digest.update(value)
        return
    digest.update(str(value).encode("utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp | date | datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")
