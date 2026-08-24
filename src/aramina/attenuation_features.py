"""Validated three-point attenuation feature construction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .attenuation_contract import (
    ATTENUATION_SYMMETRY_COLUMNS,
    ATTENUATION_VALUE_COLUMNS,
    AttenuationExperimentUnavailable,
    AttenuationFeatureResult,
    STANDARDIZED_ATTENUATION_POSITIONS,
    VALIDATED_ATTENUATION_STATUS,
    _canonical_kind,
    _has_text,
    _normalize_position,
    _normalize_side,
    _normalized_text,
)


def extract_three_point_attenuation_features(
    measurements: pd.DataFrame,
    *,
    patient_column: str = "patientId",
    side_column: str = "side",
    position_column: str = "position",
    value_column: str = "attenuation_value",
    provenance_status_column: str = "attenuation_provenance_status",
    formula_column: str = "attenuation_formula_id",
    reference_column: str = "attenuation_reference_id",
    units_column: str = "attenuation_units",
    standardized_positions: Sequence[str] = STANDARDIZED_ATTENUATION_POSITIONS,
    validated_status: str = VALIDATED_ATTENUATION_STATUS,
    strict: bool = False,
) -> AttenuationFeatureResult:
    """Build complete three-point breast features and bilateral symmetry fields.

    ``value_column`` must be a numeric, validated attenuation/optical-density
    value. Categorical mammographic density and unvalidated transmission fields
    are rejected rather than interpreted or transformed.
    """
    _reject_non_measurement_column(value_column)
    positions = tuple(
        str(position).strip().upper() for position in standardized_positions
    )
    required_columns = (
        patient_column,
        side_column,
        position_column,
        value_column,
        provenance_status_column,
        formula_column,
        reference_column,
        units_column,
    )
    missing = [column for column in required_columns if column not in measurements]
    if missing:
        result = AttenuationFeatureResult(
            features=_empty_feature_frame(),
            coverage=pd.DataFrame(
                [
                    {
                        "patientId": "",
                        "side": "",
                        "input_rows": int(len(measurements)),
                        "validated_rows": 0,
                        "complete_three_point": 0,
                        "bilateral_symmetry_available": 0,
                        "evaluation_eligible": 0,
                        "availability_reason": "required_columns_missing",
                        "missing_columns": ",".join(missing),
                    }
                ]
            ),
            status="unavailable",
            unavailable_reason=f"Required attenuation columns are missing: {missing}",
        )
        if strict:
            raise AttenuationExperimentUnavailable(result.unavailable_reason)
        return result

    values = measurements.loc[:, required_columns].copy()
    values[patient_column] = values[patient_column].astype(str)
    values["_side"] = values[side_column].map(_normalize_side)
    values["_position"] = values[position_column].map(_normalize_position)
    values["_value"] = pd.to_numeric(values[value_column], errors="coerce")
    values["_validated"] = (
        values["_value"].map(np.isfinite)
        & values[provenance_status_column].map(_normalized_text).eq(
            _normalized_text(validated_status)
        )
        & values[formula_column].map(_has_text)
        & values[reference_column].map(_has_text)
        & values[units_column].map(_has_text)
    )
    invalid_side_rows = values[values["_side"].isna()].copy()
    values = values[values["_side"].notna()].copy()
    coverage_rows = []
    feature_rows = []
    for (patient_id, side), breast in values.groupby([patient_column, "_side"], sort=True):
        position_counts = {
            position: int((breast["_position"] == position).sum())
            for position in positions
        }
        valid_counts = {
            position: int(
                (
                    (breast["_position"] == position) & breast["_validated"]
                ).sum()
            )
            for position in positions
        }
        reasons = []
        for position in positions:
            if position_counts[position] != 1:
                reasons.append(f"{position.lower()}_row_count_{position_counts[position]}")
            elif valid_counts[position] != 1:
                reasons.append(f"{position.lower()}_not_validated")
        complete = not reasons
        coverage_rows.append(
            {
                "patientId": str(patient_id),
                "side": side,
                "input_rows": int(len(breast)),
                "validated_rows": int(breast["_validated"].sum()),
                "complete_three_point": int(complete),
                "bilateral_symmetry_available": 0,
                "evaluation_eligible": 0,
                "availability_reason": "" if complete else ";".join(reasons),
                "missing_columns": "",
                **{
                    f"{position.lower()}_rows": position_counts[position]
                    for position in positions
                },
                **{
                    f"{position.lower()}_validated_rows": valid_counts[position]
                    for position in positions
                },
            }
        )
        if not complete:
            continue
        point_values = {
            position: float(
                breast.loc[
                    (breast["_position"] == position) & breast["_validated"],
                    "_value",
                ].iloc[0]
            )
            for position in positions
        }
        vector = np.asarray(
            [point_values[position] for position in positions], dtype=float
        )
        feature_rows.append(
            {
                "patientId": str(patient_id),
                "side": side,
                **{
                    f"attenuation_{position.lower()}": point_values[position]
                    for position in positions
                },
                "attenuation_mean": float(np.mean(vector)),
                "attenuation_std": float(np.std(vector, ddof=0)),
                "attenuation_range": float(np.ptp(vector)),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    if not invalid_side_rows.empty:
        invalid_coverage = pd.DataFrame(
            [
                {
                    "patientId": str(patient_id),
                    "side": "",
                    "input_rows": int(len(group)),
                    "validated_rows": int(group["_validated"].sum()),
                    "complete_three_point": 0,
                    "bilateral_symmetry_available": 0,
                    "evaluation_eligible": 0,
                    "availability_reason": "side_not_left_or_right",
                    "missing_columns": "",
                }
                for patient_id, group in invalid_side_rows.groupby(
                    patient_column, sort=True
                )
            ]
        )
        coverage = pd.concat([coverage, invalid_coverage], ignore_index=True)
    features = pd.DataFrame(feature_rows)
    if not features.empty:
        features, coverage = _add_bilateral_symmetry(features, coverage, positions)
    else:
        features = _empty_feature_frame()
    available = not features.empty and bool(
        features["attenuation_evaluation_eligible"].any()
    )
    result = AttenuationFeatureResult(
        features=features,
        coverage=coverage,
        status="available" if available else "unavailable",
        unavailable_reason=(
            "No breast has validated P1-P3 attenuation and a complete contralateral breast."
            if not available
            else ""
        ),
    )
    if strict and result.status != "available":
        raise AttenuationExperimentUnavailable(result.unavailable_reason)
    return result


def _add_bilateral_symmetry(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    positions: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = features.copy()
    for column in ATTENUATION_SYMMETRY_COLUMNS:
        out[column] = np.nan
    out["attenuation_symmetry_available"] = 0
    out["attenuation_evaluation_eligible"] = 0
    for index, row in out.iterrows():
        contralateral = "RIGHT" if row["side"] == "LEFT" else "LEFT"
        paired = out[
            (out["patientId"] == row["patientId"])
            & (out["side"] == contralateral)
        ]
        if len(paired) != 1:
            continue
        paired_row = paired.iloc[0]
        delta = np.asarray(
            [
                row[f"attenuation_{position.lower()}"]
                - paired_row[f"attenuation_{position.lower()}"]
                for position in positions
            ],
            dtype=float,
        )
        for point_index, position in enumerate(positions):
            out.at[index, f"attenuation_delta_{position.lower()}"] = delta[point_index]
            out.at[index, f"attenuation_abs_delta_{position.lower()}"] = abs(
                delta[point_index]
            )
        out.at[index, "attenuation_mean_delta"] = float(np.mean(delta))
        out.at[index, "attenuation_mean_abs_delta"] = float(np.mean(np.abs(delta)))
        out.at[index, "attenuation_rms_delta"] = float(np.sqrt(np.mean(delta**2)))
        out.at[index, "attenuation_symmetry_available"] = 1
        out.at[index, "attenuation_evaluation_eligible"] = 1
    symmetry = out.set_index(["patientId", "side"])[
        "attenuation_symmetry_available"
    ].to_dict()
    evaluation = out.set_index(["patientId", "side"])[
        "attenuation_evaluation_eligible"
    ].to_dict()
    coverage = coverage.copy()
    coverage["bilateral_symmetry_available"] = [
        int(symmetry.get((row.patientId, row.side), 0))
        for row in coverage.itertuples(index=False)
    ]
    coverage["evaluation_eligible"] = [
        int(evaluation.get((row.patientId, row.side), 0))
        for row in coverage.itertuples(index=False)
    ]
    coverage.loc[
        coverage["complete_three_point"].eq(1)
        & coverage["bilateral_symmetry_available"].eq(0),
        "availability_reason",
    ] = "contralateral_three_point_data_unavailable"
    return out, coverage


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patientId",
            "side",
            *ATTENUATION_VALUE_COLUMNS,
            *ATTENUATION_SYMMETRY_COLUMNS,
            "attenuation_symmetry_available",
            "attenuation_evaluation_eligible",
        ]
    )


def _reject_non_measurement_column(value_column: str) -> None:
    if _canonical_kind(value_column) in {
        "breastdensity",
        "transmissionpct",
        "correctionfactor",
    }:
        raise AttenuationExperimentUnavailable(
            f"{value_column!r} is not a validated numeric attenuation/optical-density value."
        )
