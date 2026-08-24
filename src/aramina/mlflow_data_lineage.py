"""DVC lineage records shared by product MLflow runs."""

from __future__ import annotations

from typing import Any


def mlflow_data_version(
    preprocessing_artifact: dict[str, Any],
    training_artifact: dict[str, Any],
) -> dict[str, Any]:
    """Return one matching DVC identity for preprocessing and training."""
    metadata = preprocessing_artifact.get("metadata", {})
    data_version = metadata.get("data_version") if isinstance(metadata, dict) else None
    if not isinstance(data_version, dict):
        raise ValueError("MLflow tracking requires preprocessing data_version metadata.")
    source_h5 = training_artifact.get("reproducibility", {}).get("source_h5", {})
    if source_h5.get("data_version") != data_version:
        raise ValueError("Training and preprocessing DVC data versions do not match.")
    required = {
        "contract",
        "system",
        "dataset_id",
        "dvc_version",
        "pointer_path",
        "hash_algorithm",
        "hash",
        "size_bytes",
        "input_h5_sha256",
    }
    missing = sorted(required.difference(data_version))
    if missing:
        raise ValueError(f"DVC data version metadata is missing: {missing}")
    return dict(data_version)


def mlflow_data_tags(reproducibility: dict[str, Any]) -> dict[str, str]:
    """Build portable MLflow tags for one raw DVC dataset revision."""
    source_h5 = reproducibility.get("source_h5", {})
    data_version = source_h5.get("data_version", {})
    source_code = reproducibility.get("source_code", {})
    input_sha256 = str(source_h5.get("sha256", "unavailable"))
    return {
        "input_h5_id": str(source_h5.get("filename", "unknown")),
        "input_h5_checksum": input_sha256,
        "raw_dataset_id": f"sha256:{input_sha256}",
        "data_version_system": str(data_version.get("system", "unavailable")),
        "dvc_dataset_id": str(data_version.get("dataset_id", "unavailable")),
        "dvc_data_hash": str(data_version.get("hash", "unavailable")),
        "dvc_data_hash_algorithm": str(
            data_version.get("hash_algorithm", "unavailable")
        ),
        "dvc_data_size_bytes": str(data_version.get("size_bytes", "unavailable")),
        "dvc_pointer_path": str(data_version.get("pointer_path", "unavailable")),
        "dvc_pointer_git_sha": str(
            source_code.get("aramina", {}).get("git_sha", "unavailable")
        ),
        "dvc_version": str(data_version.get("dvc_version", "unavailable")),
    }
