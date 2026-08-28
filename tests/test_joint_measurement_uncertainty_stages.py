from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import aramina.experiments.joint_measurement_uncertainty as joint_uncertainty
from aramina.experiments.joint_measurement_uncertainty import (
    NuisanceDraws,
    Scenario,
    _initialize_run_checkpoint,
    _open_run_checkpoint,
    _run_patient_scenario,
    _stage_plateau_metrics,
    _stage_ranges,
)


def _resume_inputs():
    frame = pd.DataFrame({"patientId": ["P1", "P2"], "value": [1.0, 2.0]})
    cases = pd.DataFrame(
        {
            "target_case_id": ["P1::LEFT", "P2::RIGHT"],
            "patient_id": ["P1", "P2"],
            "target_side": ["left", "right"],
            "label": [1, 0],
        }
    )
    parity = pd.DataFrame({"parity_pass": [True]})
    identity = {
        "config_fingerprint": "config-a",
        "model_fingerprint": "model-a",
        "data_fingerprint": "data-a",
        "scenario_fingerprint": "scenario-a",
    }
    return frame, cases, parity, identity


def _nuisance(draws: int) -> NuisanceDraws:
    values = np.zeros((draws, 2), dtype=float)
    return NuisanceDraws(
        thickness_delta_mm=values,
        beam_center_row_delta_px=values,
        beam_center_col_delta_px=values,
        detector_distance_delta_mm=values,
        photon_measurement_seeds=np.array([101, 102], dtype=np.uint64),
    )


def test_patient_scenario_stage_uses_global_draw_offsets(
    monkeypatch: pytest.MonkeyPatch,
):
    patient_frame = pd.DataFrame(
        {
            "patientId": ["P1", "P1"],
            "specimenId": ["S1", "S2"],
            "side": ["Left", "Right"],
            "position": ["P1", "P1"],
        }
    )
    patient_cases = pd.DataFrame(
        {
            "target_case_id": ["P1::LEFT"],
            "patient_id": ["P1"],
            "target_side": ["left"],
        }
    )
    context = SimpleNamespace(q_grid=np.array([2.0, 3.0, 4.0]))
    calls: list[tuple[int, int, int]] = []

    def fake_profile_chunk(
        *_args,
        start: int,
        stop: int,
        audit_draw_start: int,
        **_kwargs,
    ):
        calls.append((start, stop, audit_draw_start))
        profiles = np.full((stop - start, 2, 3), start / 1000.0)
        audit_draws = 1 if start == audit_draw_start else 0
        audit = np.zeros((audit_draws, 2, 3), dtype=float)
        return profiles, audit, audit.copy(), context.q_grid, []

    def fake_score(cube, **_kwargs):
        return SimpleNamespace(
            target_case_ids=("P1::LEFT",),
            p_cancer=np.mean(cube, axis=(1, 2))[:, np.newaxis],
            threshold=0.5,
        )

    monkeypatch.setattr(joint_uncertainty, "_metal_profile_chunk", fake_profile_chunk)
    monkeypatch.setattr(
        joint_uncertainty, "score_frozen_aramina_0_2_15_cube", fake_score
    )
    values, parity, _ = _run_patient_scenario(
        patient_frame,
        patient_cases,
        model_artifact={},
        model_info={},
        metal_context=context,
        scenario=Scenario("joint", True, True, True, True),
        nuisance=_nuisance(500),
        draws=500,
        draw_start=250,
        draw_stop=255,
        draw_chunk_size=2,
        geometry_audit_draws=1,
        normalization_q_range=(2.0, 4.0),
        metal_profile_max_tolerance=0.002,
        metal_profile_p99_tolerance=0.0001,
        metal_p_cancer_parity_tolerance=1.0,
        random_seed=43,
    )

    assert calls == [(250, 252, 250), (252, 254, 250), (254, 255, 250)]
    assert values["P1::LEFT"].shape == (5,)
    assert parity[0]["draw_start"] == 250


def test_stage_checkpoint_preserves_only_completed_draw_range(tmp_path: Path):
    frame, cases, parity, identity = _resume_inputs()
    checkpoint = _initialize_run_checkpoint(
        tmp_path,
        base_identity=identity,
        selected_frame=frame,
        selected_cases=cases,
        parity=parity,
        probability_shape=(2, 1, 500),
    )
    checkpoint.complete_unit(
        patient_id="P1",
        scenario_name="photon_only",
        scenario_index=0,
        case_values={"P1::LEFT": np.linspace(0.1, 0.2, 250)},
        case_index={"P1::LEFT": 0, "P2::RIGHT": 1},
        parity_rows=[],
        geometry_rows=[],
        draw_start=0,
        draw_stop=250,
    )

    resumed, *_ = _open_run_checkpoint(
        tmp_path,
        expected_base_identity=identity,
        probability_shape=(2, 1, 500),
    )
    assert resumed.completed_unit(
        patient_id="P1",
        scenario_name="photon_only",
        case_ids=["P1::LEFT"],
        case_indices=[0],
        scenario_index=0,
        draw_start=0,
        draw_stop=250,
    )
    assert resumed.completed_unit(
        patient_id="P1",
        scenario_name="photon_only",
        case_ids=["P1::LEFT"],
        case_indices=[0],
        scenario_index=0,
        draw_start=250,
        draw_stop=500,
    ) is None
    assert np.isnan(resumed.probabilities[0, 0, 250:]).all()


def test_global_stage_schedule_includes_short_final_stage():
    assert _stage_ranges(1100, 250) == (
        (0, 250),
        (250, 500),
        (500, 750),
        (750, 1000),
        (1000, 1100),
    )


def test_plateau_requires_consecutive_stable_checkpoints():
    current = pd.DataFrame(
        {
            "target_case_id": ["P1::LEFT", "P2::RIGHT"],
            "scenario": ["joint", "joint"],
            "p_cancer_p025": [0.10, 0.20],
            "p_cancer_p50": [0.20, 0.30],
            "p_cancer_p975": [0.30, 0.40],
            "threshold_crossing": [False, True],
        }
    )
    config = {
        "convergence": {
            "minimum_draws": 1000,
            "required_stable_checkpoints": 3,
            "median_endpoint_change_tolerance": 0.0025,
            "p90_endpoint_change_tolerance": 0.01,
            "max_threshold_crossing_count_change": 1,
            "max_threshold_status_change_count": 1,
        }
    }

    _, status = _stage_plateau_metrics(
        current,
        current.copy(),
        config=config,
        completed_draws=1000,
        stable_checkpoint_count=1,
    )
    assert status["consecutive_stable_checkpoints"] == 2
    assert status["plateau"] is False
    _, status = _stage_plateau_metrics(
        current,
        current.copy(),
        config=config,
        completed_draws=1250,
        stable_checkpoint_count=2,
    )
    assert status["plateau"] is True
