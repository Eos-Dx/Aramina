"""Aramina preprocessing pipeline entrypoints."""

from __future__ import annotations

import gc
import logging
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

from .config_paths import resolve_config_path
from .preprocessing_contract import (
    ARAMINA_PREPROCESSING_CONTRACT,
    validate_aramina_preprocessing_config,
    validate_if_aramina_product_config,
)
from .preprocessing_lineage import build_preprocessing_lineage
from .runtime_identity import aramina_git_sha, aramina_version, file_sha256


logger = logging.getLogger(__name__)


class AraminaPreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Aramina preprocessing route declared by YAML `pipeline.steps`."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | str | Path,
        verbose: bool = False,
        allow_legacy_product_config: bool = False,
    ) -> None:
        self.config = config
        self.verbose = verbose
        self.allow_legacy_product_config = allow_legacy_product_config

    def fit(self, X: str | Path, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: str | Path) -> pd.DataFrame:
        config = _load_config(
            self.config,
            allow_legacy_product_config=self.allow_legacy_product_config,
        )
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
    allow_legacy_product_config: bool = False,
) -> pd.DataFrame:
    """Build the Aramina preprocessing DataFrame declared by YAML."""
    pipeline = AraminaPreprocessingPipeline(
        config=config,
        verbose=verbose,
        allow_legacy_product_config=allow_legacy_product_config,
    )
    df = pipeline.fit_transform(h5_path).copy(deep=True)
    _write_joblib_if_requested(
        df,
        output_joblib_path,
        effective_config=pipeline.config_,
        input_h5_path=h5_path,
    )
    # Fitted H5 readers retain calibration frames that are not part of the
    # returned DataFrame. Release them before downstream model training.
    del pipeline
    gc.collect()
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
    validate_aramina_preprocessing_config(config)
    h5_path = _config_path(config, config_path, "input_h5_path")
    output_joblib_path = output_joblib_path or _config_path(
        config, config_path, "output_joblib_path"
    )
    pipeline = AraminaPreprocessingPipeline(config=config, verbose=verbose)
    df = pipeline.fit_transform(h5_path).copy(deep=True)
    artifact = _write_joblib_if_requested(
        df,
        output_joblib_path,
        effective_config=pipeline.config_,
        input_h5_path=h5_path,
    )
    # Keep only the compact artifact; fitted readers can retain raw calibration
    # frames and would otherwise overlap with the following training stage.
    del pipeline
    gc.collect()
    return artifact


def run_preprocessing_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run Aramina preprocessing using only paths stored in YAML."""
    config_path = Path(config_path)
    config = load_preprocessing_config(config_path)
    validate_aramina_preprocessing_config(config)
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
    return resolve_config_path(value, config_path)


def _load_config(
    config: dict[str, Any] | str | Path,
    *,
    allow_legacy_product_config: bool = False,
) -> dict[str, Any]:
    if isinstance(config, str | Path):
        config = load_preprocessing_config(config)
    validate_if_aramina_product_config(
        config,
        allow_legacy=allow_legacy_product_config,
    )
    return config


def _require_output_columns(config: dict[str, Any]) -> None:
    if not config.get("metadata", {}).get("output_columns"):
        raise ValueError("Aramina preprocessing config requires metadata.output_columns.")


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
    metadata = {
        "input_h5_sha256": file_sha256(input_h5_path),
        "aramina_version": aramina_version(),
        "aramina_git_sha": aramina_git_sha(),
    }
    if (
        effective_config.get("aramina_preprocessing", {}).get("contract")
        == ARAMINA_PREPROCESSING_CONTRACT
    ):
        metadata["aramina_preprocessing_lineage"] = build_preprocessing_lineage(
            effective_config
        )
    return save_preprocessing_artifact(
        df,
        output_path,
        preprocessing_config_text=config_text,
        preprocessing_config=effective_config,
        metadata=metadata,
    )
