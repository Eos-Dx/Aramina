"""Tests for the research-only profile variability experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.profile_variability.profile_variability import (
    PRIMARY_METRIC,
    build_variability_table,
    load_profile_dataframe,
    run_variability_analysis,
)


def _breast_rows(
    patient: str,
    side: str,
    *,
    biopsy: bool,
    label: str,
    profiles: list[list[float]],
) -> list[dict]:
    q_grid = np.array([2.0, 7.0, 15.0, 23.0])
    return [
        {
            "patientId": patient,
            "side": side,
            "position": f"P{index + 1}",
            "product_diagnosis": label,
            "biopsy": biopsy,
            "q_range": q_grid.copy(),
            "radial_profile_data": np.asarray(profile, dtype=float),
        }
        for index, profile in enumerate(profiles)
    ]


def _frame() -> pd.DataFrame:
    rows = []
    rows.extend(
        _breast_rows(
            "P1",
            "Left",
            biopsy=True,
            label="CANCER",
            profiles=[[1, 1, 1, 1], [1, 2, 1, 1], [1, 3, 1, 1]],
        )
    )
    rows.extend(
        _breast_rows(
            "P1",
            "Right",
            biopsy=False,
            label="BENIGN",
            profiles=[[1, 1, 1, 1], [1, 1.1, 1, 1], [1, 0.9, 1, 1]],
        )
    )
    rows.extend(
        _breast_rows(
            "P2",
            "Right",
            biopsy=True,
            label="BENIGN",
            profiles=[[1, 1, 1, 1], [1, 1.1, 1, 1], [1, 0.9, 1, 1]],
        )
    )
    rows.extend(
        _breast_rows(
            "P2",
            "Left",
            biopsy=False,
            label="BENIGN",
            profiles=[[1, 1, 1, 1], [1, 2, 1, 1], [1, 3, 1, 1]],
        )
    )
    return pd.DataFrame(rows)


def test_build_variability_table_is_patient_paired_and_directional():
    cases = build_variability_table(_frame(), min_measurements=3)
    assert len(cases) == 2
    cancer = cases.loc[cases["label"].eq("CANCER")].iloc[0]
    benign = cases.loc[cases["label"].eq("BENIGN")].iloc[0]
    assert cancer[f"log_ratio_{PRIMARY_METRIC}"] > 0.0
    assert benign[f"log_ratio_{PRIMARY_METRIC}"] < 0.0
    assert set(cases["target_measurements"]) == {3}
    assert not cases["contralateral_biopsy"].any()


def test_bilateral_biopsy_is_excluded_by_default():
    frame = _frame()
    bilateral = frame.loc[frame["patientId"].eq("P1")].copy()
    bilateral["patientId"] = "P3"
    bilateral["biopsy"] = True
    combined = pd.concat([frame, bilateral], ignore_index=True)
    primary = build_variability_table(combined, min_measurements=3)
    sensitivity = build_variability_table(
        combined,
        min_measurements=3,
        include_bilateral_biopsy=True,
    )
    assert len(primary) == 2
    assert len(sensitivity) == 4


def test_duplicate_positions_are_collapsed_before_counting():
    frame = _frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["radial_profile_data"] = [np.array([1.0, 5.0, 1.0, 1.0])]
    cases = build_variability_table(
        pd.concat([frame, duplicate], ignore_index=True),
        min_measurements=3,
    )
    patient = cases.loc[cases["patientId"].eq("P1")].iloc[0]
    assert patient["target_measurements"] == 3
    assert patient["target_raw_measurements"] == 4


def test_run_analysis_returns_fixed_primary_outputs():
    analysis = run_variability_analysis(
        _frame(),
        min_measurements=3,
        bootstrap_iterations=200,
    )
    assert analysis.metadata["primary_metric"] == PRIMARY_METRIC
    assert analysis.metadata["cases"] == 2
    assert analysis.metadata["q_min_nm_inv"] == 2.0
    assert analysis.metadata["q_max_nm_inv"] == 23.0
    assert set(analysis.paired_summary["group"]) == {"ALL", "BENIGN", "CANCER"}
    assert len(analysis.secondary_metrics) == 11
    assert len(analysis.q_variability) == 16


def test_load_rejects_non_common_q_grid(tmp_path):
    frame = _frame()
    frame.at[0, "q_range"] = np.array([2.0, 7.0, 16.0, 23.0])
    path = tmp_path / "profiles.joblib"
    import joblib

    joblib.dump({"dataframe": frame}, path)
    with pytest.raises(ValueError, match="common q-grid"):
        load_profile_dataframe(path)
