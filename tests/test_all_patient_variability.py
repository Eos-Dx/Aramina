"""Tests for the K-beta-inclusive all-patient variability experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.profile_variability.all_patient_variability import (
    build_all_patient_variability_table,
    run_all_patient_variability_analysis,
)


def _side_rows(
    patient_id: str,
    side: str,
    *,
    biopsy: bool,
    group: str,
    scale: float,
) -> list[dict[str, object]]:
    q_grid = np.array([2.0, 7.0, 15.0, 23.0])
    profiles = [
        [1.0, scale, 1.0, 1.0],
        [1.0, scale + 0.2, 1.0, 1.0],
        [1.0, scale - 0.2, 1.0, 1.0],
    ]
    return [
        {
            "patientId": patient_id,
            "side": side,
            "position": f"P{index + 1}",
            "biopsy": biopsy,
            "product_status_group": group,
            "q_range": q_grid.copy(),
            "radial_profile_data": np.asarray(profile, dtype=float),
        }
        for index, profile in enumerate(profiles)
    ]


def _all_patient_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        _side_rows(
            "unilateral_cancer",
            "Left",
            biopsy=True,
            group="CANCER",
            scale=1.4,
        )
    )
    rows.extend(
        _side_rows(
            "unilateral_cancer",
            "Right",
            biopsy=False,
            group="BENIGN",
            scale=1.1,
        )
    )
    rows.extend(
        _side_rows(
            "unilateral_benign",
            "Left",
            biopsy=False,
            group="BENIGN",
            scale=1.1,
        )
    )
    rows.extend(
        _side_rows(
            "unilateral_benign",
            "Right",
            biopsy=True,
            group="BENIGN",
            scale=1.4,
        )
    )
    rows.extend(
        _side_rows(
            "no_biopsy",
            "Left",
            biopsy=False,
            group="BENIGN",
            scale=1.2,
        )
    )
    rows.extend(
        _side_rows(
            "no_biopsy",
            "Right",
            biopsy=False,
            group="BENIGN",
            scale=1.3,
        )
    )
    rows.extend(
        _side_rows(
            "bilateral",
            "Left",
            biopsy=True,
            group="CANCER",
            scale=1.2,
        )
    )
    rows.extend(
        _side_rows(
            "bilateral",
            "Right",
            biopsy=True,
            group="BENIGN",
            scale=1.3,
        )
    )
    rows.extend(
        _side_rows(
            "unresolved_biopsy",
            "Left",
            biopsy=True,
            group="EXCLUDE",
            scale=1.4,
        )
    )
    rows.extend(
        _side_rows(
            "unresolved_biopsy",
            "Right",
            biopsy=False,
            group="BENIGN",
            scale=1.1,
        )
    )
    return pd.DataFrame(rows)


def test_all_patients_are_assigned_to_explicit_cohorts():
    cases = build_all_patient_variability_table(_all_patient_frame())
    by_patient = cases.set_index("patient_id")
    assert by_patient.loc["unilateral_cancer", "cohort"] == "BIOPSY_CANCER"
    assert by_patient.loc["unilateral_cancer", "numerator_side"] == "LEFT"
    assert by_patient.loc["unilateral_benign", "cohort"] == "BIOPSY_BENIGN"
    assert by_patient.loc["unilateral_benign", "numerator_side"] == "RIGHT"
    assert by_patient.loc["no_biopsy", "cohort"] == "NO_BIOPSY"
    assert by_patient.loc["no_biopsy", "comparison_orientation"] == "LEFT_OVER_RIGHT"
    assert by_patient.loc["bilateral", "cohort"] == "BILATERAL_BIOPSY"
    assert by_patient.loc["unresolved_biopsy", "cohort"] == "BIOPSY_UNRESOLVED"


def test_all_patient_analysis_records_inclusion_policy():
    analysis = run_all_patient_variability_analysis(
        _all_patient_frame(),
        bootstrap_iterations=200,
    )
    assert analysis.metadata["eligible_patients"] == 5
    assert analysis.metadata["kbeta_historical_exclusions_applied"] is False
    assert analysis.metadata["biopsy_patient_filter_applied"] is False
    assert set(analysis.summary["cohort"]) == {
        "BIOPSY_BENIGN",
        "BIOPSY_CANCER",
        "BIOPSY_UNRESOLVED",
        "NO_BIOPSY",
        "BILATERAL_BIOPSY",
    }
