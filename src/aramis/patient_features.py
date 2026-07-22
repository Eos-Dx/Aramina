"""Patient-level target-breast feature construction shared by training and prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .model_utils import LABEL_MAP, profile_matrix
from .symmetry_features import (
    SK_FEATURE_CONTRACT_V0_1,
    SK_FEATURE_CONTRACT_V0_2,
    target_contralateral_symmetry_features,
)


TARGET_CASE_ID = "target_case_id"

PREDICTION_METADATA_COLUMNS = (
    "session_uid",
    "session_id",
    "scan_date_time",
    "started_at",
    "operator_id",
    "hardware_version",
    "eoscan_version",
    "experimental_protocol_version",
    "product_protocol_version",
    "mammography_suspicious_field",
    "mammography_conclusion",
)


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Fail before model work when a required DataFrame column is absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Required training columns are missing: {missing}")


def boolean_series(values: pd.Series) -> pd.Series:
    """Return explicit boolean values from product metadata."""
    clean = values.astype("object").where(values.notna(), False)
    if clean.dtype == bool:
        return clean
    return clean.astype(str).str.lower().isin(["true", "1", "yes"])


def normalize_side(value: Any) -> str | None:
    """Map supported left/right encodings to stable internal values."""
    clean = str(value).strip().upper()
    if clean.startswith("LEFT"):
        return "LEFT"
    if clean.startswith("RIGHT"):
        return "RIGHT"
    return None


def display_side(side_norm: str | None) -> str:
    """Return the report-facing spelling for an internal breast side."""
    if side_norm == "LEFT":
        return "Left"
    if side_norm == "RIGHT":
        return "Right"
    return ""


def lr1_training_rows(
    df: pd.DataFrame,
    *,
    label_column: str,
    biopsy_column: str,
    lr1_row_policy: str,
    require_two_classes: bool = True,
) -> pd.DataFrame:
    """Select the labelled measurement rows admitted to LR1 training."""
    require_columns(df, [label_column])
    out = df[df[label_column].isin(LABEL_MAP)].copy()
    if lr1_row_policy == "biopsy_only":
        require_columns(out, [biopsy_column])
        out = out[boolean_series(out[biopsy_column])].copy()
    elif lr1_row_policy != "all_rows":
        raise ValueError(f"Unsupported lr1_row_policy: {lr1_row_policy!r}")
    if require_two_classes and out[label_column].nunique() != 2:
        label_counts = {
            str(label): int(count)
            for label, count in out[label_column].value_counts(dropna=False).items()
        }
        patient_count = (
            int(out["patientId"].astype(str).nunique()) if "patientId" in out else 0
        )
        raise ValueError(
            "LR1 training rows must contain BENIGN and CANCER after "
            f"lr1_row_policy={lr1_row_policy!r}; rows={len(out)}, "
            f"patients={patient_count}, label_counts={label_counts}. "
            "Inspect preprocessing/cohort_summary.json in this workflow run and "
            "verify the bundled H5 SHA256 matches bundle_manifest.json."
        )
    return out.reset_index(drop=True)


def row_labels(df: pd.DataFrame, label_column: str) -> np.ndarray:
    """Map product status groups to binary LR labels."""
    return df[label_column].map(LABEL_MAP).astype(int).to_numpy()


def target_breast_cases(
    df: pd.DataFrame,
    *,
    group_column: str,
    side_column: str,
    label_column: str,
    biopsy_column: str,
) -> pd.DataFrame:
    """Return one historical target case for every biopsied breast."""
    biopsy_rows = df[
        df[label_column].isin(LABEL_MAP) & boolean_series(df[biopsy_column])
    ].copy()
    records: list[dict[str, Any]] = []
    for patient_id, patient_df in df.groupby(group_column, sort=True):
        patient_biopsy = biopsy_rows[
            biopsy_rows[group_column].astype(str) == str(patient_id)
        ]
        for target_side in sorted(
            side
            for side in patient_biopsy[side_column]
            .map(normalize_side)
            .dropna()
            .unique()
        ):
            target_rows = patient_biopsy[
                patient_biopsy[side_column].map(normalize_side) == target_side
            ]
            labels = target_rows[label_column].map(LABEL_MAP).dropna().unique()
            if len(labels) != 1:
                raise ValueError(
                    f"Target breast {patient_id!r}/{target_side!r} has ambiguous labels."
                )
            sides = set(patient_df[side_column].map(normalize_side).dropna())
            contralateral = next((side for side in sides if side != target_side), None)
            records.append(
                {
                    TARGET_CASE_ID: f"{patient_id}::{target_side}",
                    group_column: str(patient_id),
                    "target_side_norm": target_side,
                    "target_side": display_side(target_side),
                    "contralateral_side_norm": contralateral,
                    "contralateral_side": display_side(contralateral),
                    "label": int(labels[0]),
                }
            )
    cases = pd.DataFrame(records)
    if cases.empty:
        raise ValueError("No biopsied target-breast cases are available.")
    return cases


def logit_average_probability(scores: Sequence[float]) -> float:
    """Aggregate measurement probabilities as the mean LR1 logit."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return 0.5
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return float(1.0 / (1.0 + np.exp(-float(np.mean(logits)))))


def score_lr1_rows(
    lr1_model: Pipeline,
    rows: pd.DataFrame,
    *,
    full_df: pd.DataFrame,
    profile_column: str,
    group_column: str,
    side_column: str,
    label_column: str,
    biopsy_column: str,
) -> pd.DataFrame:
    """Score LR1 rows and aggregate target-side evidence for each target case."""
    require_columns(rows, [group_column, side_column])
    out = rows[[group_column, side_column]].copy()
    out[group_column] = out[group_column].astype(str)
    out["_side_norm"] = out[side_column].map(normalize_side)
    out["lr1_measurement_p_cancer"] = lr1_model.predict_proba(
        profile_matrix(rows, profile_column)
    )[:, 1]
    cases = target_breast_cases(
        full_df,
        group_column=group_column,
        side_column=side_column,
        label_column=label_column,
        biopsy_column=biopsy_column,
    )
    grouped_rows = []
    for target in cases.itertuples(index=False):
        group = out[out[group_column].astype(str) == str(getattr(target, group_column))]
        target_scores = group.loc[
            group["_side_norm"] == target.target_side_norm,
            "lr1_measurement_p_cancer",
        ].to_numpy(dtype=float)
        if target_scores.size == 0:
            raise ValueError(
                f"No LR1 target-side scores for {target.target_case_id!r}; "
                "check target-side policy and lr1_row_policy."
            )
        grouped_rows.append(
            {
                TARGET_CASE_ID: target.target_case_id,
                "profile_p_cancer_probability_mean": float(np.mean(target_scores)),
                "profile_p_cancer_logit_average": logit_average_probability(
                    target_scores
                ),
                "profile_p_cancer_n_measurements": int(target_scores.size),
            }
        )
    return pd.DataFrame(grouped_rows)


def empty_lr1_scores(
    df: pd.DataFrame,
    *,
    group_column: str,
    side_column: str,
    label_column: str,
    biopsy_column: str,
) -> pd.DataFrame:
    """Create neutral LR1 scores used solely to define patient-safe folds."""
    cases = target_breast_cases(
        df,
        group_column=group_column,
        side_column=side_column,
        label_column=label_column,
        biopsy_column=biopsy_column,
    )
    return cases[[TARGET_CASE_ID]].assign(
        profile_p_cancer_probability_mean=0.5,
        profile_p_cancer_logit_average=0.5,
        profile_p_cancer_n_measurements=0,
    )


def patient_feature_table(
    df: pd.DataFrame,
    lr1_scores: pd.DataFrame,
    *,
    profile_column: str,
    label_column: str,
    group_column: str,
    specimen_column: str,
    side_column: str,
    q_column: str,
    age_column: str,
    biopsy_column: str,
    require_two_classes: bool = True,
) -> pd.DataFrame:
    """Build one final-model feature row for each historical target breast."""
    require_columns(
        df,
        [
            group_column,
            specimen_column,
            label_column,
            profile_column,
            side_column,
            q_column,
            biopsy_column,
        ],
    )
    rows = []
    cases = target_breast_cases(
        df,
        group_column=group_column,
        side_column=side_column,
        label_column=label_column,
        biopsy_column=biopsy_column,
    )
    for target_case in cases.itertuples(index=False):
        patient_id = str(getattr(target_case, group_column))
        patient_df = df[df[group_column].astype(str) == patient_id]
        symmetry = target_contralateral_symmetry_features(
            patient_df,
            profile_column=profile_column,
            q_column=q_column,
            side_column=side_column,
            target_side_norm=target_case.target_side_norm,
            contralateral_side_norm=target_case.contralateral_side_norm,
            feature_contract=SK_FEATURE_CONTRACT_V0_2,
        )
        rows.append(
            {
                TARGET_CASE_ID: target_case.target_case_id,
                group_column: patient_id,
                "label": int(target_case.label),
                "label_name": "CANCER" if int(target_case.label) == 1 else "BENIGN",
                "target_side": target_case.target_side,
                "contralateral_side": target_case.contralateral_side,
                "specimens": int(patient_df[specimen_column].astype(str).nunique()),
                "measurements": int(len(patient_df)),
                "age": numeric_median(patient_df, age_column, default=0.0),
                "age_available": int(has_numeric(patient_df, age_column)),
                **symmetry,
            }
        )
    feature_table = pd.DataFrame(rows)
    out = feature_table.merge(lr1_scores, on=TARGET_CASE_ID, how="inner")
    out = add_patient_reliability_columns(out)
    if require_two_classes and out["label"].nunique() != 2:
        raise ValueError("Patient feature table must contain BENIGN and CANCER.")
    return out.reset_index(drop=True)


def build_patient_prediction_feature_row(
    df: pd.DataFrame,
    model_info: dict[str, Any],
    *,
    patient_id: str,
    target_side: str,
    profile_column: str = "radial_profile_data",
    group_column: str = "patientId",
    specimen_column: str = "specimenId",
    side_column: str = "side",
    q_column: str = "q_range",
    age_column: str = "age",
) -> pd.DataFrame:
    """Build one prediction feature row from a clinician-supplied target side."""
    require_columns(
        df,
        [group_column, specimen_column, side_column, profile_column, q_column],
    )
    lr1_model = model_info.get("lr1_model")
    if lr1_model is None:
        raise ValueError("Model artifact is missing lr1_model.")
    patient_df = df[df[group_column].astype(str) == str(patient_id)].copy()
    if patient_df.empty:
        raise ValueError(f"Patient not found in prediction DataFrame: {patient_id!r}")
    target_side_norm = normalize_side(target_side)
    if target_side_norm is None:
        raise ValueError(f"Invalid target_side: {target_side!r}")
    side_norms = patient_df[side_column].map(normalize_side)
    available_sides = sorted(side for side in side_norms.dropna().unique())
    if target_side_norm not in available_sides:
        raise ValueError(
            f"Target side {target_side!r} is absent for patient {patient_id!r}; "
            f"available sides: {available_sides}"
        )
    contralateral = [side for side in available_sides if side != target_side_norm]
    contralateral_side_norm = contralateral[0] if contralateral else None
    target_df = patient_df[side_norms == target_side_norm].copy()
    target_scores = lr1_model.predict_proba(profile_matrix(target_df, profile_column))[
        :, 1
    ]
    symmetry = target_contralateral_symmetry_features(
        patient_df,
        profile_column=profile_column,
        q_column=q_column,
        side_column=side_column,
        target_side_norm=target_side_norm,
        contralateral_side_norm=contralateral_side_norm,
        feature_contract=str(
            model_info.get("symmetry_feature_contract", SK_FEATURE_CONTRACT_V0_1)
        ),
    )
    row = {
        TARGET_CASE_ID: f"{patient_id}::{target_side_norm}",
        "patientId": str(patient_id),
        "target_side": display_side(target_side_norm),
        "contralateral_side": display_side(contralateral_side_norm),
        "specimens": int(patient_df[specimen_column].astype(str).nunique()),
        "measurements": int(len(patient_df)),
        "age": numeric_median(patient_df, age_column, default=0.0),
        "age_available": int(has_numeric(patient_df, age_column)),
        "profile_p_cancer_probability_mean": float(np.mean(target_scores)),
        "profile_p_cancer_logit_average": logit_average_probability(target_scores),
        "profile_p_cancer_n_measurements": int(target_scores.size),
        **prediction_metadata_from_target_rows(target_df),
        **symmetry,
    }
    return add_patient_reliability_columns(pd.DataFrame([row]))


def add_patient_reliability_columns(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Attach report-only measurement sufficiency and reliability fields."""
    out = feature_table.copy()
    out["min_measurements_per_breast"] = np.minimum(
        out["target_measurements"].astype(int),
        out["contralateral_measurements"].astype(int),
    )
    out["target_measurements_ok"] = (
        out["target_measurements"].astype(int) >= 2
    ).astype(int)
    out["contralateral_measurements_ok"] = (
        out["contralateral_measurements"].astype(int) >= 2
    ).astype(int)
    out["paired_measurements_ok"] = (
        (out["symmetry_available"].astype(int) == 1)
        & (out["min_measurements_per_breast"].astype(int) >= 2)
    ).astype(int)
    out["profile_measurements_ok"] = (
        out["profile_p_cancer_n_measurements"].astype(int) >= 2
    ).astype(int)
    out["result_reliability"] = np.select(
        [
            out["paired_measurements_ok"].astype(bool),
            out["target_measurements_ok"].astype(bool),
        ],
        ["high", "medium"],
        default="low",
    )
    symmetry_reason = out.get("symmetry_reason", pd.Series("", index=out.index))
    symmetry_reason = symmetry_reason.fillna("").astype(str)
    out["result_reliability_reason"] = np.select(
        [
            out["paired_measurements_ok"].astype(bool),
            symmetry_reason.eq("sk_core4_not_computable"),
            out["contralateral_measurements"].astype(int).eq(0),
            out["target_measurements_ok"].astype(bool),
        ],
        [
            "at least 2 valid measurements per breast; symmetry refinement applied",
            "symmetry features could not be computed; symmetry refinement not applied",
            "contralateral breast unavailable after preprocessing; symmetry refinement not applied",
            "fewer than 2 valid contralateral-breast measurements; symmetry refinement not applied",
        ],
        default="fewer than 2 valid target-breast measurements",
    )
    return out


def prediction_metadata_from_target_rows(target_df: pd.DataFrame) -> dict[str, Any]:
    """Keep one non-conflicting target-side metadata value for report generation."""
    metadata: dict[str, Any] = {}
    for column in PREDICTION_METADATA_COLUMNS:
        if column not in target_df.columns:
            continue
        values = [
            value
            for value in target_df[column].tolist()
            if not _metadata_value_is_empty(value)
        ]
        unique = _unique_metadata_values(values)
        if len(unique) == 1:
            metadata[column] = unique[0]
    return metadata


def _metadata_value_is_empty(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _unique_metadata_values(values: Sequence[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if not any(value == existing for existing in unique):
            unique.append(value.strip() if isinstance(value, str) else value)
    return unique


def numeric_median(df: pd.DataFrame, column: str, *, default: float) -> float:
    """Return a numeric patient metadata median or the explicit fallback."""
    if column not in df.columns:
        return float(default)
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.median()) if values.notna().any() else float(default)


def has_numeric(df: pd.DataFrame, column: str) -> bool:
    """Return whether patient metadata has at least one numeric value."""
    return (
        column in df.columns
        and pd.to_numeric(df[column], errors="coerce").notna().any()
    )
