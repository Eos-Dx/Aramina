"""Aramis preprocessing pipeline entrypoints."""

from __future__ import annotations

import subprocess
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import yaml
from xrd_preprocessing import (
    build_pipeline_from_config,
    load_preprocessing_config,
    save_preprocessing_artifact,
)


class AramisPreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Aramis preprocessing route declared by YAML `pipeline.steps`."""

    def __init__(self, *, config: dict[str, Any] | str | Path) -> None:
        self.config = config

    def fit(self, X: str | Path, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: str | Path) -> pd.DataFrame:
        config = _load_config(self.config)
        _require_output_columns(config)
        self.config_ = config
        self.pipeline_ = build_pipeline_from_config(config)
        return self.pipeline_.fit_transform(X)


def run_preprocessing_pipeline(
    h5_path: str | Path,
    config: dict[str, Any] | str | Path,
    *,
    output_joblib_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the Aramis preprocessing DataFrame declared by YAML."""
    pipeline = AramisPreprocessingPipeline(config=config)
    df = pipeline.fit_transform(h5_path)
    _write_joblib_if_requested(
        df,
        output_joblib_path,
        config_source=config,
        effective_config=pipeline.config_,
        input_h5_path=h5_path,
    )
    return df


def run_preprocessing_artifact_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run preprocessing and return the written artifact without reloading joblib."""
    config_path = Path(config_path)
    config = load_preprocessing_config(config_path)
    branch = config["aramis_preprocessing"]["branch"]
    h5_path = _config_path(config, config_path, "input_h5_path")
    output_joblib_path = _config_path(config, config_path, "output_joblib_path")
    if branch != "one_to_many":
        raise ValueError(f"Unknown Aramis preprocessing branch: {branch}")
    pipeline = AramisPreprocessingPipeline(config=config)
    df = pipeline.fit_transform(h5_path)
    return _write_joblib_if_requested(
        df,
        output_joblib_path,
        config_source=config,
        effective_config=pipeline.config_,
        input_h5_path=h5_path,
    )


def run_preprocessing_from_config(config_path: str | Path) -> pd.DataFrame:
    """Run Aramis preprocessing using only paths stored in YAML."""
    config_path = Path(config_path)
    config = load_preprocessing_config(config_path)
    branch = config["aramis_preprocessing"]["branch"]
    h5_path = _config_path(config, config_path, "input_h5_path")
    output_joblib_path = _config_path(config, config_path, "output_joblib_path")
    if branch == "one_to_many":
        return run_preprocessing_pipeline(
            h5_path,
            config,
            output_joblib_path=output_joblib_path,
        )
    raise ValueError(f"Unknown Aramis preprocessing branch: {branch}")


def _config_path(config: dict[str, Any], config_path: Path, key: str) -> Path:
    value = config.get("io", {}).get(key)
    if value in {None, ""}:
        raise ValueError(f"Missing io.{key} in preprocessing config: {config_path}")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _load_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, str | Path):
        return load_preprocessing_config(config)
    return config


def _require_output_columns(config: dict[str, Any]) -> None:
    if not config.get("metadata", {}).get("output_columns"):
        raise ValueError("Aramis preprocessing config requires metadata.output_columns.")


def _write_joblib_if_requested(
    df: pd.DataFrame,
    output_joblib_path: str | Path | None,
    *,
    config_source: dict[str, Any] | str | Path,
    effective_config: dict[str, Any],
    input_h5_path: str | Path,
) -> dict[str, Any] | None:
    if output_joblib_path is None:
        return None
    output_path = Path(output_joblib_path)
    config_text = _config_source_text(config_source)
    return save_preprocessing_artifact(
        df,
        output_path,
        preprocessing_config=effective_config,
        preprocessing_config_text=config_text,
        metadata={
            "branch": effective_config.get("aramis_preprocessing", {}).get("branch"),
            "input_h5_sha256": _file_sha256(input_h5_path),
            "aramis_version": _aramis_version(),
            "aramis_git_sha": _aramis_git_sha(),
        },
    )


def _config_source_text(config_source: dict[str, Any] | str | Path) -> str:
    if isinstance(config_source, str | Path):
        path = Path(config_source)
        return path.read_text(encoding="utf-8")
    return yaml.safe_dump(config_source, sort_keys=False)


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aramis_version() -> str:
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
