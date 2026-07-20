"""Promote one reviewed train-on-all run into an immutable model directory."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from shutil import copy2
from typing import Any

import joblib
import yaml


REQUIRED_FILES = (
    "model.joblib",
    "model_description.yaml",
    "preprocessing_config.yaml",
    "prediction_preprocessing_config.yaml",
    "training_config.yaml",
)
OPTIONAL_FILES = (
    "preprocess_and_train_config.yaml",
    "evaluation.yaml",
    "evaluation_metrics.csv",
    "evaluation_predictions.csv",
)


def promote_model_run(
    run_folder: str | Path,
    *,
    models_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy a reviewed final-fit run without mutating its source files."""
    source = Path(run_folder).expanduser().resolve()
    _validate_source_run(source)
    model_path = source / "model.joblib"
    artifact = joblib.load(model_path)
    identity = artifact.get("model_identity")
    if not isinstance(identity, dict):
        raise ValueError("Training artifact has no model_identity.")
    name = identity.get("name")
    version = identity.get("version")
    if not name or version in {None, ""}:
        raise ValueError("Training artifact model_identity requires name and version.")

    model_sha256 = _file_sha256(model_path)
    model_id = _safe_stem(f"{name}_{version}_{model_sha256[:12]}")
    destination_root = (
        Path(models_root).expanduser().resolve()
        if models_root is not None
        else Path(__file__).resolve().parents[2] / "models"
    )
    destination = destination_root / model_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing promoted model: {destination}")

    destination.mkdir(parents=True)
    for filename in (*REQUIRED_FILES, *OPTIONAL_FILES):
        source_path = source / filename
        if source_path.exists():
            copy2(source_path, destination / filename)
    return {
        "model_id": model_id,
        "artifact_sha256": model_sha256,
        "source_run_folder": source,
        "model_folder": destination,
    }


def _validate_source_run(source: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Training run folder not found: {source}")
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Training run is not promotable; missing files: {missing}")
    description = yaml.safe_load((source / "model_description.yaml").read_text(encoding="utf-8"))
    if not isinstance(description, dict) or description.get("output_type") != "aramis_model_description":
        raise ValueError("model_description.yaml is not an Aramis model description.")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    ).strip("_")
