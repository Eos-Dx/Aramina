"""Aramis preprocessing pipeline entrypoints."""

from __future__ import annotations

import logging
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


logger = logging.getLogger(__name__)


class AramisPreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Aramis preprocessing route declared by YAML `pipeline.steps`."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | str | Path,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.verbose = verbose

    def fit(self, X: str | Path, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: str | Path) -> pd.DataFrame:
        config = _load_config(self.config)
        _require_output_columns(config)
        self.config_ = config
        self.pipeline_ = build_pipeline_from_config(config, verbose=self.verbose)
        logger.info("Preprocessing input H5: %s", X)
        logger.info(
            "Preprocessing steps: %s",
            ", ".join(self.pipeline_.named_steps),
        )
        df = self.pipeline_.fit_transform(X)
        logger.info("Preprocessing complete: rows=%d columns=%d", len(df), len(df.columns))
        return df


def run_preprocessing_pipeline(
    h5_path: str | Path,
    config: dict[str, Any] | str | Path,
    *,
    output_joblib_path: str | Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Build the Aramis preprocessing DataFrame declared by YAML."""
    pipeline = AramisPreprocessingPipeline(config=config, verbose=verbose)
    df = pipeline.fit_transform(h5_path)
    _write_joblib_if_requested(
        df,
        output_joblib_path,
        effective_config=pipeline.config_,
        input_h5_path=h5_path,
    )
    return df


def run_preprocessing_artifact_from_config(
    config_path: str | Path,
    *,
    output_joblib_path: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run preprocessing and return the written artifact without reloading joblib."""
    config_path = Path(config_path)
    config = load_preprocessing_config(config_path)
    h5_path = _config_path(config, config_path, "input_h5_path")
    output_joblib_path = output_joblib_path or _config_path(
        config, config_path, "output_joblib_path"
    )
    pipeline = AramisPreprocessingPipeline(config=config, verbose=verbose)
    df = pipeline.fit_transform(h5_path)
    return _write_joblib_if_requested(
        df,
        output_joblib_path,
        effective_config=pipeline.config_,
        input_h5_path=h5_path,
    )


def run_preprocessing_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run Aramis preprocessing using only paths stored in YAML."""
    config_path = Path(config_path)
    config = load_preprocessing_config(config_path)
    h5_path = _config_path(config, config_path, "input_h5_path")
    output_joblib_path = _config_path(config, config_path, "output_joblib_path")
    return run_preprocessing_pipeline(
        h5_path,
        config,
        output_joblib_path=output_joblib_path,
        verbose=verbose,
    )


def _config_path(config: dict[str, Any], config_path: Path, key: str) -> Path:
    value = config.get("io", {}).get(key)
    if value in {None, ""}:
        raise ValueError(f"Missing io.{key} in preprocessing config: {config_path}")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


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
    effective_config: dict[str, Any],
    input_h5_path: str | Path,
) -> dict[str, Any] | None:
    if output_joblib_path is None:
        return None
    output_path = Path(output_joblib_path)
    config_text = yaml.safe_dump(effective_config, sort_keys=False)
    return save_preprocessing_artifact(
        df,
        output_path,
        preprocessing_config_text=config_text,
        metadata={
            "input_h5_sha256": _file_sha256(input_h5_path),
            "aramis_version": _aramis_version(),
            "aramis_git_sha": _aramis_git_sha(),
        },
    )


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
