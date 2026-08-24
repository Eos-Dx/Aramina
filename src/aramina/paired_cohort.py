"""Matched-cohort construction and traceability for paired evaluation."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .model_utils import profile_matrix
from .paired_contract import (
    MATCHED_METADATA_COLUMNS,
    MEASUREMENT_ID_COLUMNS,
    PROFILE_SCORE_COLUMNS,
)
from .patient_features import (
    TARGET_CASE_ID,
    empty_lr1_scores,
    patient_feature_table,
)
from .training_config import PRODUCT_MODEL_NAME, resolve_model_definition


def load_dataframe_artifact(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a DataFrame artifact and return reproducibility metadata."""
    source = Path(path).expanduser().resolve()
    value = joblib.load(source)
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        artifact_metadata: dict[str, Any] = {}
    elif isinstance(value, dict) and isinstance(value.get("dataframe"), pd.DataFrame):
        frame = value["dataframe"].copy()
        artifact_metadata = value.get("metadata", {})
    else:
        raise ValueError(
            "Input joblib must contain a DataFrame or {'dataframe': DataFrame}."
        )
    return frame, {
        "path": str(source),
        "sha256": file_sha256(source),
        "artifact_metadata": artifact_metadata,
    }


def construct_common_cohort(
    raw100: pd.DataFrame,
    fpca256: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Intersect exact measurements and verify one matched target-case cohort."""
    _require_columns(
        raw100,
        [*MEASUREMENT_ID_COLUMNS, *MATCHED_METADATA_COLUMNS],
        dataset="raw100",
    )
    _require_columns(
        fpca256,
        [*MEASUREMENT_ID_COLUMNS, *MATCHED_METADATA_COLUMNS],
        dataset="fpca256",
    )
    raw_keys = measurement_keys(raw100)
    fpca_keys = measurement_keys(fpca256)
    if raw_keys.duplicated().any() or fpca_keys.duplicated().any():
        raise ValueError("Measurement identity must be unique within each input.")
    common_keys = set(raw_keys).intersection(fpca_keys)
    if not common_keys:
        raise ValueError("raw100 and fpca256 inputs share no measurements.")
    raw_mask = raw_keys.isin(common_keys)
    fpca_mask = fpca_keys.isin(common_keys)
    raw_common = raw100.loc[raw_mask].copy()
    fpca_common = fpca256.loc[fpca_mask].copy()
    _require_matching_measurement_metadata(
        raw_common,
        fpca_common,
        raw_keys=raw_keys.loc[raw_mask],
        fpca_keys=fpca_keys.loc[fpca_mask],
    )
    _validate_profile_grid(raw_common, expected_npt=100, dataset="raw100")
    _validate_profile_grid(fpca_common, expected_npt=256, dataset="fpca256")

    raw_context = dataset_context(raw_common)
    fpca_context = dataset_context(fpca_common)
    _require_matching_cases(raw_context, fpca_context)
    case_manifest = (
        raw_context[
            [TARGET_CASE_ID, "patientId", "target_side", "label", "label_name"]
        ]
        .sort_values(TARGET_CASE_ID, kind="stable")
        .reset_index(drop=True)
    )
    return (
        raw_common.reset_index(drop=True),
        fpca_common.reset_index(drop=True),
        case_manifest,
    )


def model_columns() -> dict[str, Any]:
    """Return controlled product columns and fixed regularization."""
    model = resolve_model_definition(PRODUCT_MODEL_NAME)["model"]
    return {
        key: model[key]
        for key in (
            "profile_column",
            "label_column",
            "group_column",
            "specimen_column",
            "side_column",
            "q_column",
            "age_column",
            "biopsy_column",
            "lr1_row_policy",
            "lr1_logreg_c",
            "lr2_logreg_c",
        )
    }


def dataset_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Build target-case context without retaining neutral LR1 placeholders."""
    model = model_columns()
    scores = empty_lr1_scores(
        frame,
        group_column=model["group_column"],
        side_column=model["side_column"],
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
    )
    context = patient_feature_table(
        frame,
        scores,
        profile_column=model["profile_column"],
        label_column=model["label_column"],
        group_column=model["group_column"],
        specimen_column=model["specimen_column"],
        side_column=model["side_column"],
        q_column=model["q_column"],
        age_column=model["age_column"],
        biopsy_column=model["biopsy_column"],
        require_two_classes=False,
    )
    return context.drop(columns=list(PROFILE_SCORE_COLUMNS))


def ordered_context(
    frame: pd.DataFrame,
    case_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Order one representation by the canonical case manifest."""
    context = dataset_context(frame).set_index(TARGET_CASE_ID)
    return context.loc[case_manifest[TARGET_CASE_ID]].reset_index()


def measurement_manifest(
    raw100: pd.DataFrame,
    fpca256: pd.DataFrame,
) -> pd.DataFrame:
    """Record every included or excluded measurement identity."""
    raw = pd.DataFrame(
        {
            "measurement_identity": measurement_keys(raw100),
            "patientId": raw100["patientId"].astype(str),
            "in_raw100": True,
        }
    )
    fpca = pd.DataFrame(
        {
            "measurement_identity": measurement_keys(fpca256),
            "patientId": fpca256["patientId"].astype(str),
            "in_fpca256": True,
        }
    )
    manifest = raw.merge(
        fpca,
        on="measurement_identity",
        how="outer",
        suffixes=("_raw100", "_fpca256"),
        validate="one_to_one",
    )
    manifest["patientId"] = manifest["patientId_raw100"].fillna(
        manifest["patientId_fpca256"]
    )
    manifest["in_raw100"] = manifest["in_raw100"].eq(True)
    manifest["in_fpca256"] = manifest["in_fpca256"].eq(True)
    manifest["in_common_cohort"] = manifest["in_raw100"] & manifest["in_fpca256"]
    manifest["exclusion_reason"] = np.select(
        [
            manifest["in_common_cohort"],
            manifest["in_raw100"] & ~manifest["in_fpca256"],
            ~manifest["in_raw100"] & manifest["in_fpca256"],
        ],
        ["included", "missing_from_fpca256", "missing_from_raw100"],
        default="unknown",
    )
    return manifest[
        [
            "measurement_identity",
            "patientId",
            "in_raw100",
            "in_fpca256",
            "in_common_cohort",
            "exclusion_reason",
        ]
    ].sort_values("measurement_identity", kind="stable")


def require_common_source_h5(
    raw_metadata: dict[str, Any],
    fpca_metadata: dict[str, Any],
) -> None:
    """Require both preprocessing artifacts to identify one source H5."""
    raw_h5 = raw_metadata.get("artifact_metadata", {}).get("input_h5_sha256")
    fpca_h5 = fpca_metadata.get("artifact_metadata", {}).get("input_h5_sha256")
    if not raw_h5 or not fpca_h5:
        raise ValueError("Both artifacts must record input_h5_sha256.")
    if raw_h5 != fpca_h5:
        raise ValueError("raw100 and fpca256 artifacts use different source H5 files.")


def measurement_keys(frame: pd.DataFrame) -> pd.Series:
    """Return stable composite measurement identities."""
    return frame.loc[:, MEASUREMENT_ID_COLUMNS].astype(str).agg("||".join, axis=1)


def patient_subset(frame: pd.DataFrame, patient_ids: set[str]) -> pd.DataFrame:
    """Return all rows belonging to one patient set."""
    return frame.loc[frame["patientId"].astype(str).isin(patient_ids)].copy()


def file_sha256(path: str | Path) -> str:
    """Return SHA-256 for one file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_matching_measurement_metadata(
    raw100: pd.DataFrame,
    fpca256: pd.DataFrame,
    *,
    raw_keys: pd.Series,
    fpca_keys: pd.Series,
) -> None:
    left = raw100.loc[:, MATCHED_METADATA_COLUMNS].copy()
    right = fpca256.loc[:, MATCHED_METADATA_COLUMNS].copy()
    left["_measurement_identity"] = raw_keys.to_numpy()
    right["_measurement_identity"] = fpca_keys.to_numpy()
    left = left.set_index("_measurement_identity").sort_index()
    right = right.set_index("_measurement_identity").sort_index()
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            "Common measurements differ in label, biopsy, or age metadata."
        ) from exc


def _require_matching_cases(left: pd.DataFrame, right: pd.DataFrame) -> None:
    columns = [TARGET_CASE_ID, "patientId", "target_side", "label", "label_name"]
    left_cases = left[columns].sort_values(TARGET_CASE_ID).reset_index(drop=True)
    right_cases = right[columns].sort_values(TARGET_CASE_ID).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left_cases, right_cases, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Matched inputs do not produce identical target cases.") from exc
    if left_cases[TARGET_CASE_ID].duplicated().any():
        raise ValueError("Target-case identity must be unique.")


def _validate_profile_grid(
    frame: pd.DataFrame,
    *,
    expected_npt: int,
    dataset: str,
) -> None:
    _require_columns(frame, ["radial_profile_data", "q_range"], dataset=dataset)
    matrix = profile_matrix(frame, "radial_profile_data")
    if matrix.shape[1] != int(expected_npt):
        raise ValueError(
            f"{dataset} requires {expected_npt}-bin profiles; "
            f"received {matrix.shape[1]}."
        )
    grids = [np.asarray(value, dtype=float).ravel() for value in frame["q_range"]]
    reference = grids[0]
    if reference.size != expected_npt or not np.isfinite(reference).all():
        raise ValueError(f"{dataset} q grid must be finite and match profile length.")
    if any(
        grid.shape != reference.shape
        or not np.allclose(grid, reference, rtol=1e-9, atol=1e-10)
        for grid in grids[1:]
    ):
        raise ValueError(f"{dataset} requires one shared q grid.")
    spacing = np.diff(reference)
    if not np.all(spacing > 0.0) or not np.allclose(
        spacing, spacing[0], rtol=1e-6, atol=1e-10
    ):
        raise ValueError(f"{dataset} requires a uniformly increasing q grid.")


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    dataset: str,
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")
