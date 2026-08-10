"""Input lineage, preprocessing, and cohort validation."""

from __future__ import annotations

import copy
import gc
import inspect
import logging
from hashlib import sha256
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from xrd_preprocessing import (
    build_pipeline_from_config,
    load_preprocessing_config,
    save_preprocessing_artifact,
)

from aramina.patient_features import TARGET_CASE_ID

from .config import resolve_path
from .model import build_dataset_context, validate_profile_grid


LOGGER = logging.getLogger(__name__)
MEASUREMENT_ID_COLUMNS = (
    "patientId",
    "specimenId",
    "side",
    "position",
    "started_at",
)


def create_research_npt256_artifact(
    input_h5_path: str | Path,
    *,
    base_config_path: str | Path,
    output_artifact_path: str | Path,
    lineage: dict[str, Any],
) -> Path:
    """Create npt256 input without changing the product preprocessing contract."""
    validate_pyfai_runtime(lineage)
    h5_path = Path(input_h5_path).expanduser().resolve()
    base_path = Path(base_config_path).expanduser().resolve()
    output_path = Path(output_artifact_path).expanduser().resolve()
    _require_file_sha256(
        h5_path,
        lineage["source_h5_sha256"],
        description="raw H5 input",
    )
    _require_file_sha256(
        base_path,
        lineage["base_preprocessing_config_sha256"],
        description="base preprocessing config",
    )
    config = copy.deepcopy(load_preprocessing_config(base_path))
    config["integration"]["npt"] = 256
    config["integration"]["method"] = list(lineage["integration_method"])
    config.setdefault("io", {})["input_h5_path"] = str(h5_path)
    config["io"]["output_joblib_path"] = str(output_path)
    config.setdefault("provenance", {})["status"] = (
        "research_only_npt256_fpca_profile_encoder_input"
    )
    pipeline = build_pipeline_from_config(config, verbose=True)
    LOGGER.info("Research-only npt256 preprocessing input: %s", h5_path)
    df = pipeline.fit_transform(h5_path).copy(deep=True)
    validate_profile_grid(
        df,
        profile_column="radial_profile_data",
        q_column="q_range",
        expected_npt=256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_preprocessing_artifact(
        df,
        output_path,
        preprocessing_config_text=yaml.safe_dump(config, sort_keys=False),
        metadata={
            "clinical_stage": "research_only",
            "controlled_change": "integration.npt=256",
            "input_h5_sha256": lineage["source_h5_sha256"],
            "base_preprocessing_config_sha256": lineage[
                "base_preprocessing_config_sha256"
            ],
            "aramina_base_main_git_sha": lineage["aramina_base_main_git_sha"],
            "integration_variant": "npt256_bbox",
            "integration_npt": 256,
            "integration_method": list(lineage["integration_method"]),
            "pyfai_version": lineage["pyfai_version"],
            "integration_method_source": lineage["integration_method_source"],
        },
    )
    del pipeline, df
    gc.collect()
    return output_path


def validate_common_cohort(
    npt100: pd.DataFrame,
    npt256: pd.DataFrame,
    contexts: dict[int, pd.DataFrame],
) -> None:
    """Require exact common measurement identities, cases, labels, and folds."""
    missing = [
        column
        for column in MEASUREMENT_ID_COLUMNS
        if column not in npt100 or column not in npt256
    ]
    if missing:
        raise ValueError(f"Common cohort identity columns are missing: {missing}")
    keys100 = _measurement_keys(npt100)
    keys256 = _measurement_keys(npt256)
    if keys100.duplicated().any() or keys256.duplicated().any():
        raise ValueError("Common cohort measurement identity must be unique.")
    if set(keys100) != set(keys256):
        raise ValueError("npt100 and npt256 common measurement identities differ.")
    metadata_columns = [*MEASUREMENT_ID_COLUMNS, "product_status_group", "biopsy"]
    left = npt100[metadata_columns].copy()
    right = npt256[metadata_columns].copy()
    left["_identity"] = keys100
    right["_identity"] = keys256
    compare_columns = ["product_status_group", "biopsy"]
    left = left.set_index("_identity")[compare_columns].sort_index().astype(str)
    right = right.set_index("_identity")[compare_columns].sort_index().astype(str)
    if not left.equals(right):
        raise ValueError("Common cohort labels or biopsy flags differ by measurement.")
    require_matching_case_order(
        contexts[100].sort_values(TARGET_CASE_ID).reset_index(drop=True),
        contexts[256].sort_values(TARGET_CASE_ID).reset_index(drop=True),
    )


def load_cohort_datasets(
    config: dict[str, Any],
    source: Path,
    *,
    cohort_name: str,
    enforce_expected: bool,
    generated_mode: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, Any]]]:
    """Load and validate every artifact required by one cohort mode."""
    section = config["cohorts"][cohort_name]
    artifact_pins = {256: section["npt256_artifact"]}
    if cohort_name == "common":
        artifact_pins[100] = section["npt100_artifact"]
    datasets: dict[int, pd.DataFrame] = {}
    lineage: dict[int, dict[str, Any]] = {}
    for npt, pin in artifact_pins.items():
        path = resolve_path(pin["path"], source)
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or "dataframe" not in artifact:
            raise TypeError(f"Expected preprocessing artifact with dataframe: {path}")
        actual_lineage = validate_artifact_lineage(
            path,
            artifact,
            expected=pin,
            experiment_lineage=config["lineage"],
            generated_mode=generated_mode,
        )
        df = artifact["dataframe"].copy()
        validate_profile_grid(
            df,
            profile_column=config["model"]["profile_column"],
            q_column=config["model"]["q_column"],
            expected_npt=npt,
        )
        datasets[npt] = df
        lineage[npt] = actual_lineage
    if enforce_expected:
        _validate_expected_counts(datasets[256], config, section)
    return datasets, lineage


def validate_artifact_lineage(
    path: Path,
    artifact: dict[str, Any],
    *,
    expected: dict[str, Any],
    experiment_lineage: dict[str, Any],
    generated_mode: bool,
) -> dict[str, Any]:
    """Reject any artifact that does not match its declared immutable lineage."""
    actual_sha256 = file_sha256(path)
    actual_fingerprint = artifact.get("pipeline_fingerprint")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Preprocessing artifact metadata is missing: {path}")
    if not generated_mode:
        if actual_sha256 != expected["sha256"]:
            raise ValueError(
                f"Input artifact SHA-256 mismatch for {path}: "
                f"expected {expected['sha256']}, received {actual_sha256}."
            )
        if actual_fingerprint != expected["pipeline_fingerprint"]:
            raise ValueError(
                f"Pipeline fingerprint mismatch for {path}: expected "
                f"{expected['pipeline_fingerprint']}, received {actual_fingerprint}."
            )
    elif not isinstance(actual_fingerprint, str) or not actual_fingerprint:
        raise ValueError("Generated artifact must record its pipeline fingerprint.")
    expected_h5 = experiment_lineage["source_h5_sha256"]
    checks = {
        "input_h5_sha256": expected_h5,
        "integration_variant": expected["integration_variant"],
        "integration_npt": expected["integration_npt"],
        "integration_method": expected["integration_method"],
    }
    for key, expected_value in checks.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Artifact metadata {key!r} mismatch for {path}: expected "
                f"{expected_value!r}, received {metadata.get(key)!r}."
            )
    if expected["input_h5_sha256"] != expected_h5:
        raise ValueError(
            "Artifact pin input_h5_sha256 does not match experiment lineage."
        )
    if generated_mode:
        generated_checks = {
            "base_preprocessing_config_sha256": experiment_lineage[
                "base_preprocessing_config_sha256"
            ],
            "aramina_base_main_git_sha": experiment_lineage[
                "aramina_base_main_git_sha"
            ],
        }
        for key, expected_value in generated_checks.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"Generated artifact metadata {key!r} mismatch: expected "
                    f"{expected_value!r}, received {metadata.get(key)!r}."
                )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "sha256_validation": (
            "recorded_generated_artifact_not_pinned"
            if generated_mode
            else "matched_pinned_artifact"
        ),
        "pipeline_fingerprint": actual_fingerprint,
        "input_h5_sha256": metadata["input_h5_sha256"],
        "integration_variant": metadata["integration_variant"],
        "integration_npt": metadata["integration_npt"],
        "integration_method": metadata["integration_method"],
        "pyfai_version": experiment_lineage["pyfai_version"],
        "integration_method_source": experiment_lineage[
            "integration_method_source"
        ],
    }


def validate_pyfai_runtime(
    lineage: dict[str, Any],
    *,
    installed_version: str | None = None,
    integrate1d_default: Any | None = None,
) -> dict[str, Any]:
    """Require the pinned PyFAI release and integrate1d method default."""
    actual_version = (
        distribution_version("pyFAI")
        if installed_version is None
        else installed_version
    )
    if integrate1d_default is None:
        integrate1d_default = inspect.signature(
            AzimuthalIntegrator.integrate1d
        ).parameters["method"].default
    expected_method = tuple(lineage["integration_method"])
    actual_method = integrate1d_default
    if actual_version != lineage["pyfai_version"]:
        raise ValueError(
            "PyFAI runtime version mismatch: expected "
            f"{lineage['pyfai_version']}, received {actual_version}."
        )
    if actual_method != expected_method:
        raise ValueError(
            "PyFAI integrate1d method default mismatch: expected "
            f"{expected_method!r}, received {actual_method!r}."
        )
    if lineage["integration_method_source"] != "pyfai_integrate1d_default":
        raise ValueError(
            "Integration method source must be pyfai_integrate1d_default."
        )
    return {
        "pyfai_version": actual_version,
        "integration_method": list(actual_method),
        "integration_method_source": lineage["integration_method_source"],
    }


def require_matching_case_order(
    canonical: pd.DataFrame,
    candidate: pd.DataFrame,
) -> None:
    """Require identical ordered target cases across profile representations."""
    columns = [TARGET_CASE_ID, "patientId", "label"]
    left = canonical[columns].reset_index(drop=True).astype(str)
    right = candidate[columns].reset_index(drop=True).astype(str)
    if not left.equals(right):
        raise ValueError("Datasets do not contain identical ordered target cases.")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_expected_counts(
    df: pd.DataFrame,
    config: dict[str, Any],
    section: dict[str, Any],
) -> None:
    context = build_dataset_context(df, config["model"])
    actual = {
        "expected_rows": int(len(df)),
        "expected_patients": int(df[config["model"]["group_column"]].nunique()),
        "expected_target_cases": int(len(context)),
    }
    expected = {key: int(section[key]) for key in actual}
    if actual != expected:
        raise ValueError(f"Cohort count mismatch: expected={expected}, actual={actual}")


def _measurement_keys(df: pd.DataFrame) -> pd.Series:
    return df.loc[:, MEASUREMENT_ID_COLUMNS].astype(str).agg("||".join, axis=1)


def _require_file_sha256(
    path: str | Path,
    expected: str,
    *,
    description: str,
) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{description} SHA-256 mismatch for {path}: "
            f"expected {expected}, received {actual}."
        )
