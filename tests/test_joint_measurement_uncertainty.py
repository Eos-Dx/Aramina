from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aramina.experiments.joint_measurement_uncertainty import (
    Scenario,
    _load_config,
    effective_detector_distance_m,
    sample_nuisance_draws,
    summarize_case_uncertainty,
)


ROOT = Path(__file__).parents[1]


def _patient_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patientId": ["P1", "P1", "P1"],
            "specimenId": ["S1", "S2", "S3"],
            "side": ["Left", "Left", "Right"],
            "position": ["P1", "P2", "P1"],
            "started_at": ["T1", "T2", "T3"],
            "sample_thickness_mm": [40.0, 60.0, 60.0],
            "calibration_session_uid": ["A", "A", "B"],
        }
    )


def _nuisance_configs(correlation: str = "visit_shared"):
    return (
        {
            "thin_max_mm": 50.0,
            "thin_half_width_mm": 5.0,
            "thick_half_width_mm": 10.0,
            "correlation": correlation,
        },
        {"radius_px": 5.0},
        {"half_width_mm": 10.0},
    )


def test_effective_distance_uses_half_thickness_correction():
    actual = effective_detector_distance_m(
        0.705,
        60.0,
        20.0,
        sample_thickness_delta_mm=10.0,
        detector_distance_delta_mm=-5.0,
    )

    assert actual == pytest.approx(0.675)


def test_visit_shared_thickness_and_session_shared_geometry():
    thickness, center, distance = _nuisance_configs()
    draws = sample_nuisance_draws(
        _patient_frame(),
        draws=100,
        seed=17,
        thickness_config=thickness,
        beam_center_config=center,
        detector_distance_config=distance,
    )

    np.testing.assert_allclose(
        draws.thickness_delta_mm[:, 0] / 5.0,
        draws.thickness_delta_mm[:, 1] / 10.0,
    )
    np.testing.assert_array_equal(
        draws.beam_center_row_delta_px[:, 0],
        draws.beam_center_row_delta_px[:, 1],
    )
    assert not np.array_equal(
        draws.beam_center_row_delta_px[:, 0],
        draws.beam_center_row_delta_px[:, 2],
    )
    radius = np.hypot(
        draws.beam_center_row_delta_px,
        draws.beam_center_col_delta_px,
    )
    assert float(radius.max()) <= 5.0
    assert float(np.abs(draws.detector_distance_delta_mm).max()) <= 10.0


def test_measurement_independent_thickness_does_not_share_latent_draw():
    thickness, center, distance = _nuisance_configs("measurement_independent")
    draws = sample_nuisance_draws(
        _patient_frame(),
        draws=20,
        seed=19,
        thickness_config=thickness,
        beam_center_config=center,
        detector_distance_config=distance,
    )

    assert not np.array_equal(
        draws.thickness_delta_mm[:, 0] / 5.0,
        draws.thickness_delta_mm[:, 1] / 10.0,
    )


def test_geometry_stream_is_shared_across_patients_in_same_session():
    thickness, center, distance = _nuisance_configs()
    first = _patient_frame().iloc[[0]].copy()
    second = _patient_frame().iloc[[1]].copy()
    second["patientId"] = "P2"
    first_draws = sample_nuisance_draws(
        first,
        draws=20,
        seed=23,
        thickness_config=thickness,
        beam_center_config=center,
        detector_distance_config=distance,
    )
    second_draws = sample_nuisance_draws(
        second,
        draws=20,
        seed=23,
        thickness_config=thickness,
        beam_center_config=center,
        detector_distance_config=distance,
    )

    np.testing.assert_array_equal(
        first_draws.beam_center_row_delta_px,
        second_draws.beam_center_row_delta_px,
    )
    np.testing.assert_array_equal(
        first_draws.detector_distance_delta_mm,
        second_draws.detector_distance_delta_mm,
    )
    assert not np.array_equal(
        first_draws.thickness_delta_mm,
        second_draws.thickness_delta_mm,
    )


def test_summary_reports_threshold_crossing_and_flip_probability():
    scenarios = (Scenario("joint", True, True, True, True),)
    probabilities = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)
    cases = pd.DataFrame(
        {
            "target_case_id": ["P1::LEFT"],
            "patient_id": ["P1"],
            "target_side": ["left"],
            "label": [1],
            "deterministic_p_cancer": [0.3],
            "decision_threshold": [0.25],
        }
    )

    summary = summarize_case_uncertainty(
        probabilities,
        cases,
        scenarios=scenarios,
        quantiles=(0.025, 0.5, 0.975),
    ).iloc[0]

    assert bool(summary["threshold_crossing"])
    assert summary["probability_at_or_above_threshold"] == pytest.approx(0.5)
    assert summary["class_flip_probability"] == pytest.approx(0.5)


def test_pilot_and_full_configs_are_valid():
    for filename in (
        "config_joint_measurement_uncertainty_pilot_v0_1.yaml",
        "config_joint_measurement_uncertainty_full_v0_1.yaml",
    ):
        config = _load_config(ROOT / "config/experiments" / filename)
        assert config["experiment"]["model_version"] == "0.2.15-beta"
