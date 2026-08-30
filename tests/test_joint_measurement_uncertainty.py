from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import aramina.experiments.joint_measurement_uncertainty as joint_uncertainty
from aramina.experiments.joint_measurement_uncertainty import (
    NuisanceDraws,
    PatientMetalContext,
    Scenario,
    _initialize_run_checkpoint,
    _metal_profile_chunk,
    _open_run_checkpoint,
    _profile_parity_metrics,
    _profile_parity_tolerances,
    _prepared_geometry_seed,
    _run_patient_scenario,
    effective_detector_distance_m,
    sample_cohort_nuisance_draws,
    sample_nuisance_draws,
)
from aramina.experiments.measurement_uncertainty import MeasurementUncertaintyError


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
            "ponifile": ["PONI-A", "PONI-A", "PONI-B"],
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
        {"radius_px": 5.0, "correlation": "poni_file_shared"},
        {"half_width_mm": 10.0, "correlation": "poni_file_shared"},
    )


def _array_nuisance(draws: int = 5, measurements: int = 2) -> NuisanceDraws:
    values = np.arange(draws * measurements, dtype=float).reshape(
        draws, measurements
    )
    return NuisanceDraws(
        thickness_delta_mm=values + 1.0,
        beam_center_row_delta_px=values + 2.0,
        beam_center_col_delta_px=values + 3.0,
        detector_distance_delta_mm=values + 4.0,
        photon_measurement_seeds=np.arange(
            101, 101 + measurements, dtype=np.uint64
        ),
    )


class _FakeGeometrySession:
    def __init__(self, bins: int):
        self.bins = bins
        self.calls: list[tuple[str, int, int]] = []
        self.closed = False

    def _profiles(
        self,
        draws: int,
        *,
        effective_distance_m: np.ndarray,
        poni1_m: np.ndarray,
        poni2_m: np.ndarray,
        draw_offset: int,
        draw_chunk_size: int,
    ) -> np.ndarray:
        assert draw_chunk_size == draws
        draw_key = np.arange(draw_offset, draw_offset + draws)[:, np.newaxis]
        base = effective_distance_m + poni1_m + poni2_m + draw_key * 1e-6
        return np.repeat(base[:, :, np.newaxis], self.bins, axis=2)

    def integrate(self, draws: int, **kwargs) -> np.ndarray:
        self.calls.append(("integrate", int(kwargs["draw_offset"]), draws))
        return self._profiles(draws, **kwargs)

    def run(self, scales, draws: int, *, seed: int, **kwargs) -> np.ndarray:
        assert scales == (1.0,)
        self.calls.append(("run", int(kwargs["draw_offset"]), draws))
        profiles = self._profiles(draws, **kwargs) + seed * 1e-12
        return profiles[np.newaxis, ...]

    def run_nested(
        self,
        scales,
        geometry_draws: int,
        photon_replicates: int,
        *,
        seed: int,
        **kwargs,
    ) -> np.ndarray:
        assert scales == (1.0,)
        geometry_offset = int(kwargs["geometry_draw_offset"])
        self.calls.append(("run_nested", geometry_offset, geometry_draws))
        profiles = self._profiles(
            geometry_draws,
            effective_distance_m=kwargs["effective_distance_m"],
            poni1_m=kwargs["poni1_m"],
            poni2_m=kwargs["poni2_m"],
            draw_offset=geometry_offset,
            draw_chunk_size=int(kwargs["geometry_chunk_size"]),
        )
        repeated = np.repeat(profiles[:, np.newaxis], photon_replicates, axis=1)
        return (repeated + seed * 1e-12)[np.newaxis, ...]

    def close(self) -> None:
        self.closed = True

    def run_geometry(
        self,
        plans,
        photon_replicates: int,
        *,
        seed: int,
        include_deterministic: bool,
    ):
        draw_index = int(plans[0].draw_index)
        self.calls.append(("run_geometry", draw_index, photon_replicates))
        base = np.arange(len(plans), dtype=float) + draw_index / 1000.0
        deterministic = np.repeat(base[:, np.newaxis], self.bins, axis=1)
        profiles = np.repeat(
            deterministic[np.newaxis, ...], photon_replicates, axis=0
        )
        return SimpleNamespace(
            profiles=profiles + (seed % 997) * 1e-12,
            deterministic_profiles=deterministic if include_deterministic else None,
            unique_plan_count=len(plans),
        )


def _fake_metal_context(
    session: _FakeGeometrySession,
    *,
    backend_kind: str = "pyfai_prepared_csr_metal_photon_mc",
) -> PatientMetalContext:
    return PatientMetalContext(
        session=session,
        q_grid=np.array([2.0, 3.0, 4.0]),
        images=np.zeros((2, 2, 2)),
        backend_kind=backend_kind,
        nominal_effective_distance_m=np.array([0.7, 0.8]),
        nominal_poni1_m=np.array([0.01, 0.02]),
        nominal_poni2_m=np.array([0.03, 0.04]),
        pixel1_m=np.array([1e-4, 1e-4]),
        pixel2_m=np.array([1e-4, 1e-4]),
    )


def _fake_prepare_geometry_draw(
    patient_frame,
    *,
    draw_index: int,
    q_grid,
    **_kwargs,
):
    base = np.arange(len(patient_frame), dtype=float) + draw_index / 1000.0
    expected = np.repeat(base[:, np.newaxis], len(q_grid), axis=1)
    plans = [SimpleNamespace(draw_index=draw_index) for _ in range(len(patient_frame))]
    return expected, plans, []


def _resume_inputs():
    selected_frame = pd.DataFrame(
        {"patientId": ["P1", "P2"], "value": [1.0, 2.0]}
    )
    selected_cases = pd.DataFrame(
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
    return selected_frame, selected_cases, parity, identity


def test_effective_distance_uses_half_thickness_correction():
    actual = effective_detector_distance_m(
        0.705,
        60.0,
        20.0,
        sample_thickness_delta_mm=10.0,
        detector_distance_delta_mm=-5.0,
    )

    assert actual == pytest.approx(0.675)


def test_visit_shared_thickness_and_poni_shared_geometry():
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
    assert draws.photon_measurement_seeds.shape == (3,)


def test_photon_seed_is_stable_by_measurement_when_rows_are_reordered():
    thickness, center, distance = _nuisance_configs()
    frame = _patient_frame()
    reordered = frame.iloc[[2, 0, 1]].reset_index(drop=True)
    first = sample_nuisance_draws(
        frame,
        draws=4,
        seed=29,
        thickness_config=thickness,
        beam_center_config=center,
        detector_distance_config=distance,
    )
    second = sample_nuisance_draws(
        reordered,
        draws=4,
        seed=29,
        thickness_config=thickness,
        beam_center_config=center,
        detector_distance_config=distance,
    )

    first_by_specimen = dict(
        zip(frame["specimenId"], first.photon_measurement_seeds, strict=True)
    )
    second_by_specimen = dict(
        zip(reordered["specimenId"], second.photon_measurement_seeds, strict=True)
    )
    assert first_by_specimen == second_by_specimen


def test_cohort_geometry_is_shared_by_poni_not_patient_or_session():
    thickness, center, distance = _nuisance_configs()
    first = _patient_frame().iloc[[0]].copy()
    second = _patient_frame().iloc[[1]].copy()
    second["patientId"] = "P2"
    second["specimenId"] = "S4"
    second["calibration_session_uid"] = "DIFFERENT-SESSION"
    cohort = pd.concat((first, second), ignore_index=True)

    field = sample_cohort_nuisance_draws(
        cohort,
        draws=20,
        seed=29,
        thickness_config=thickness,
        beam_center_config=center,
        detector_distance_config=distance,
    )
    first_draws = field.for_frame(first)
    second_draws = field.for_frame(second)

    np.testing.assert_array_equal(
        first_draws.beam_center_row_delta_px,
        second_draws.beam_center_row_delta_px,
    )
    np.testing.assert_array_equal(
        first_draws.beam_center_col_delta_px,
        second_draws.beam_center_col_delta_px,
    )
    np.testing.assert_array_equal(
        first_draws.detector_distance_delta_mm,
        second_draws.detector_distance_delta_mm,
    )
    assert not np.array_equal(
        first_draws.thickness_delta_mm,
        second_draws.thickness_delta_mm,
    )


def test_prepared_geometry_seed_is_reproducible_and_draw_specific():
    assert _prepared_geometry_seed(43, 7) == _prepared_geometry_seed(43, 7)
    assert _prepared_geometry_seed(43, 7) != _prepared_geometry_seed(43, 8)
    assert _prepared_geometry_seed(43, 7) != _prepared_geometry_seed(44, 7)


def test_profile_parity_gate_records_max_and_p99_separately():
    expected = np.zeros((1, 1, 100), dtype=float)
    isolated_edge_error = expected.copy()
    isolated_edge_error[0, 0, -1] = 0.001468

    metrics = _profile_parity_metrics(
        isolated_edge_error,
        expected,
        np.linspace(2.0, 22.895, 100),
        draw_start=7,
        maximum_tolerance=0.002,
        p99_tolerance=0.0001,
    )

    assert metrics["maximum_absolute_error"] == pytest.approx(0.001468)
    assert metrics["p99_absolute_error"] < 0.0001
    assert metrics["maximum_error_q_nm_inv"] == pytest.approx(22.895)
    assert metrics["maximum_error_draw_index"] == 7
    assert metrics["parity_pass"] is True

    broad_error = expected.copy()
    broad_error[0, 0, -5:] = 0.0002
    broad_metrics = _profile_parity_metrics(
        broad_error,
        expected,
        np.linspace(2.0, 22.895, 100),
        draw_start=0,
        maximum_tolerance=0.002,
        p99_tolerance=0.0001,
    )
    assert broad_metrics["parity_pass"] is False


def test_profile_parity_tolerances_are_explicit():
    with pytest.raises(MeasurementUncertaintyError, match="explicit max and p99"):
        _profile_parity_tolerances({"metal_parity_tolerance": 1e-4})

    assert _profile_parity_tolerances(
        {
            "metal_profile_max_abs_tolerance": 0.002,
            "metal_profile_p99_abs_tolerance": 0.0001,
        }
    ) == (0.002, 0.0001)


def test_prepared_geometry_chunk_output_is_resume_invariant(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        joint_uncertainty,
        "_prepare_pyfai_geometry_draw",
        _fake_prepare_geometry_draw,
    )
    patient_frame = _patient_frame().iloc[:2].reset_index(drop=True)
    nuisance = _array_nuisance()
    scenario = Scenario("joint", True, True, True, True)

    full_session = _FakeGeometrySession(3)
    full_context = _fake_metal_context(full_session)
    full = _metal_profile_chunk(
        patient_frame,
        metal_context=full_context,
        scenario=scenario,
        nuisance=nuisance,
        start=0,
        stop=5,
        geometry_audit_draws=0,
        normalization_q_range=(2.0, 4.0),
        random_seed=43,
    )[0]

    chunked_session = _FakeGeometrySession(3)
    chunked_context = _fake_metal_context(chunked_session)
    first = _metal_profile_chunk(
        patient_frame,
        metal_context=chunked_context,
        scenario=scenario,
        nuisance=nuisance,
        start=0,
        stop=2,
        geometry_audit_draws=0,
        normalization_q_range=(2.0, 4.0),
        random_seed=43,
    )[0]
    second = _metal_profile_chunk(
        patient_frame,
        metal_context=chunked_context,
        scenario=scenario,
        nuisance=nuisance,
        start=2,
        stop=5,
        geometry_audit_draws=0,
        normalization_q_range=(2.0, 4.0),
        random_seed=43,
    )[0]

    np.testing.assert_allclose(np.concatenate((first, second)), full)
    assert chunked_session.calls == [
        ("run_geometry", 0, 1),
        ("run_geometry", 1, 1),
        ("run_geometry", 2, 1),
        ("run_geometry", 3, 1),
        ("run_geometry", 4, 1),
    ]
    assert chunked_context.session is chunked_session


def test_nested_profile_chunk_reuses_prepared_plan_for_photon_replicates(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        joint_uncertainty,
        "_prepare_pyfai_geometry_draw",
        _fake_prepare_geometry_draw,
    )
    patient_frame = _patient_frame().iloc[:2].reset_index(drop=True)
    nuisance = _array_nuisance()
    session = _FakeGeometrySession(3)
    profiles = _metal_profile_chunk(
        patient_frame,
        metal_context=_fake_metal_context(session),
        scenario=Scenario("joint", True, True, True, True),
        nuisance=nuisance,
        start=1,
        stop=3,
        geometry_audit_draws=0,
        normalization_q_range=(2.0, 4.0),
        random_seed=43,
        photon_replicates=5,
    )[0]

    assert profiles.shape == (10, 2, 3)
    assert session.calls == [
        ("run_geometry", 1, 5),
        ("run_geometry", 2, 5),
    ]
    np.testing.assert_allclose(profiles[0], profiles[1])


def test_p_cancer_parity_is_fail_closed_on_bounded_audit_draws(
    monkeypatch: pytest.MonkeyPatch,
):
    patient_frame = _patient_frame().iloc[:2].reset_index(drop=True)
    patient_cases = pd.DataFrame(
        {
            "target_case_id": ["P1::LEFT"],
            "patient_id": ["P1"],
            "target_side": ["left"],
        }
    )
    context = _fake_metal_context(_FakeGeometrySession(3))
    nuisance = _array_nuisance(draws=5)
    audit_calls: list[tuple[int, int]] = []

    def fake_profile_chunk(*_args, start: int, stop: int, **_kwargs):
        profiles = np.zeros((stop - start, 2, 3), dtype=float)
        audit_stop = min(stop, 2)
        audit_draws = max(0, audit_stop - start)
        if audit_draws:
            audit_calls.append((start, audit_stop))
        metal = np.zeros((audit_draws, 2, 3), dtype=float)
        expected = np.full_like(metal, 5e-5)
        return profiles, metal, expected, context.q_grid, []

    def fake_score(cube, **_kwargs):
        values = np.mean(cube, axis=(1, 2))[:, np.newaxis] * 3.0
        return SimpleNamespace(
            target_case_ids=("P1::LEFT",),
            p_cancer=values,
            threshold=0.5,
        )

    monkeypatch.setattr(
        joint_uncertainty, "_metal_profile_chunk", fake_profile_chunk
    )
    monkeypatch.setattr(
        joint_uncertainty, "score_frozen_aramina_0_2_15_cube", fake_score
    )

    with pytest.raises(MeasurementUncertaintyError, match="p_cancer parity"):
        _run_patient_scenario(
            patient_frame,
            patient_cases,
            model_artifact={},
            model_info={},
            metal_context=context,
            scenario=Scenario("joint", True, True, True, True),
            nuisance=nuisance,
            draws=5,
            draw_chunk_size=2,
            geometry_audit_draws=2,
            normalization_q_range=(2.0, 4.0),
            metal_profile_max_tolerance=0.002,
            metal_profile_p99_tolerance=0.0001,
            metal_p_cancer_parity_tolerance=0.0001,
            random_seed=43,
        )

    assert audit_calls == [(0, 2)]


def test_decision_class_parity_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    patient_frame = _patient_frame().iloc[:2].reset_index(drop=True)
    patient_cases = pd.DataFrame(
        {
            "target_case_id": ["P1::LEFT"],
            "patient_id": ["P1"],
            "target_side": ["left"],
        }
    )
    context = _fake_metal_context(_FakeGeometrySession(3))

    def fake_profile_chunk(*_args, start: int, stop: int, **_kwargs):
        profiles = np.zeros((stop - start, 2, 3), dtype=float)
        metal = np.zeros((1, 2, 3), dtype=float)
        expected = np.full_like(metal, 5e-5)
        return profiles, metal, expected, context.q_grid, []

    def fake_score(cube, **_kwargs):
        values = np.mean(cube, axis=(1, 2))[:, np.newaxis] * 3.0
        return SimpleNamespace(
            target_case_ids=("P1::LEFT",),
            p_cancer=values,
            threshold=1e-4,
        )

    monkeypatch.setattr(
        joint_uncertainty, "_metal_profile_chunk", fake_profile_chunk
    )
    monkeypatch.setattr(
        joint_uncertainty, "score_frozen_aramina_0_2_15_cube", fake_score
    )

    with pytest.raises(MeasurementUncertaintyError, match="decision class"):
        _run_patient_scenario(
            patient_frame,
            patient_cases,
            model_artifact={},
            model_info={},
            metal_context=context,
            scenario=Scenario("joint", True, True, True, True),
            nuisance=_array_nuisance(draws=1),
            draws=1,
            draw_chunk_size=1,
            geometry_audit_draws=1,
            normalization_q_range=(2.0, 4.0),
            metal_profile_max_tolerance=0.002,
            metal_profile_p99_tolerance=0.0001,
            metal_p_cancer_parity_tolerance=1.0,
            random_seed=43,
        )


def test_interrupted_partial_probability_slice_is_not_completed(tmp_path: Path):
    frame, cases, parity, identity = _resume_inputs()
    checkpoint = _initialize_run_checkpoint(
        tmp_path,
        base_identity=identity,
        selected_frame=frame,
        selected_cases=cases,
        parity=parity,
        probability_shape=(2, 2, 3),
    )
    checkpoint.probabilities[0, 0] = np.array([0.1, 0.2, 0.3])
    checkpoint.probabilities.flush()

    resumed, *_ = _open_run_checkpoint(
        tmp_path,
        expected_base_identity=identity,
        probability_shape=(2, 2, 3),
    )

    assert (
        resumed.completed_unit(
            patient_id="P1",
            scenario_name="photon_only",
            case_ids=["P1::LEFT"],
            case_indices=[0],
            scenario_index=0,
        )
        is None
    )
    assert resumed.progress["completed_units"] == {}


def test_successful_resume_skips_only_atomic_completed_units(tmp_path: Path):
    frame, cases, parity, identity = _resume_inputs()
    checkpoint = _initialize_run_checkpoint(
        tmp_path,
        base_identity=identity,
        selected_frame=frame,
        selected_cases=cases,
        parity=parity,
        probability_shape=(2, 2, 3),
    )
    checkpoint.complete_unit(
        patient_id="P1",
        scenario_name="photon_only",
        scenario_index=0,
        case_values={"P1::LEFT": np.array([0.1, 0.2, 0.3])},
        case_index={"P1::LEFT": 0, "P2::RIGHT": 1},
        parity_rows=[{"parity_pass": True}],
        geometry_rows=[{"draw_index": 0}],
    )

    resumed, resumed_frame, resumed_cases, resumed_parity = _open_run_checkpoint(
        tmp_path,
        expected_base_identity=identity,
        probability_shape=(2, 2, 3),
    )
    payload = resumed.completed_unit(
        patient_id="P1",
        scenario_name="photon_only",
        case_ids=["P1::LEFT"],
        case_indices=[0],
        scenario_index=0,
    )

    assert payload is not None
    assert payload["parity_rows"] == [{"parity_pass": True}]
    pd.testing.assert_frame_equal(resumed_frame, frame)
    pd.testing.assert_frame_equal(resumed_cases, cases)
    pd.testing.assert_frame_equal(resumed_parity, parity)
    resumed.complete_unit(
        patient_id="P1",
        scenario_name="thickness_only",
        scenario_index=1,
        case_values={"P1::LEFT": np.array([0.4, 0.5, 0.6])},
        case_index={"P1::LEFT": 0, "P2::RIGHT": 1},
        parity_rows=[],
        geometry_rows=[],
    )
    assert len(resumed.progress["completed_units"]) == 2


def test_resume_fingerprint_mismatch_fails_closed(tmp_path: Path):
    frame, cases, parity, identity = _resume_inputs()
    _initialize_run_checkpoint(
        tmp_path,
        base_identity=identity,
        selected_frame=frame,
        selected_cases=cases,
        parity=parity,
        probability_shape=(2, 2, 3),
    )
    mismatched = {**identity, "model_fingerprint": "model-b"}

    with pytest.raises(
        MeasurementUncertaintyError,
        match="Resume fingerprint mismatch: model_fingerprint",
    ):
        _open_run_checkpoint(
            tmp_path,
            expected_base_identity=mismatched,
            probability_shape=(2, 2, 3),
        )


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
