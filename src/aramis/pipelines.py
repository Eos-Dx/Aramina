"""Aramis preprocessing pipeline entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from xrd_preprocessing import build_pipeline_from_config, load_preprocessing_config


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
        self.pipeline_ = build_pipeline_from_config(config)
        return self.pipeline_.fit_transform(X)


class AramisOneToOnePreprocessingPipeline(AramisPreprocessingPipeline):
    """One-to-one paired-breast preprocessing route."""


class AramisOneToManyPreprocessingPipeline(AramisPreprocessingPipeline):
    """One-to-many specimen-level preprocessing route."""


def run_one_to_one_preprocessing_pipeline(
    h5_path: str | Path,
    config: dict[str, Any] | str | Path,
    *,
    output_joblib_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the one-to-one paired-breast preprocessing DataFrame."""
    df = AramisOneToOnePreprocessingPipeline(config=config).fit_transform(h5_path)
    _write_joblib_if_requested(df, output_joblib_path)
    return df


def run_one_to_many_preprocessing_pipeline(
    h5_path: str | Path,
    config: dict[str, Any] | str | Path,
    *,
    output_joblib_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the one-to-many specimen-level preprocessing DataFrame."""
    df = AramisOneToManyPreprocessingPipeline(config=config).fit_transform(h5_path)
    _write_joblib_if_requested(df, output_joblib_path)
    return df


def run_preprocessing_from_config(config_path: str | Path) -> pd.DataFrame:
    """Run Aramis preprocessing using only paths stored in YAML."""
    config_path = Path(config_path)
    config = load_preprocessing_config(config_path)
    branch = config["aramis_preprocessing"]["branch"]
    h5_path = _config_path(config, config_path, "input_h5_path")
    output_joblib_path = _config_path(config, config_path, "output_joblib_path")
    if branch == "one_to_one":
        return run_one_to_one_preprocessing_pipeline(
            h5_path,
            config,
            output_joblib_path=output_joblib_path,
        )
    if branch == "one_to_many":
        return run_one_to_many_preprocessing_pipeline(
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


def _write_joblib_if_requested(df: pd.DataFrame, output_joblib_path: str | Path | None) -> None:
    if output_joblib_path is None:
        return
    output_path = Path(output_joblib_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(df, output_path)
