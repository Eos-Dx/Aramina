import numpy as np
import pandas as pd
import pytest

import aramina.experiments.joint_measurement_uncertainty as joint_uncertainty
from aramina.experiments.joint_measurement_uncertainty import (
    GeometryPlanCache,
    NuisanceDraws,
    PatientMetalContext,
    PreparedGeometryDraw,
    Scenario,
    _geometry_plan_cache_key,
    _metal_profile_chunk,
)


class _FakeGeometrySession:
    def __init__(self, bins: int) -> None:
        self.bins = bins
        self.calls: list[tuple[str, int, int]] = []

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


def _patient_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patientId": ["P1", "P1"],
            "specimenId": ["S1", "S2"],
            "side": ["Left", "Right"],
            "position": ["P1", "P2"],
        }
    )


def _nuisance(draws: int = 5) -> NuisanceDraws:
    values = np.arange(draws * 2, dtype=float).reshape(draws, 2)
    return NuisanceDraws(
        thickness_delta_mm=values + 1.0,
        beam_center_row_delta_px=values + 2.0,
        beam_center_col_delta_px=values + 3.0,
        detector_distance_delta_mm=values + 4.0,
        photon_measurement_seeds=np.array([101, 102], dtype=np.uint64),
    )


def _geometry_context(session: _FakeGeometrySession) -> PatientMetalContext:
    return PatientMetalContext(
        session=session,
        q_grid=np.array([2.0, 3.0, 4.0]),
        images=np.zeros((2, 2, 2)),
        backend_kind="metal_geometry_aware_nested",
        nominal_effective_distance_m=np.array([0.7, 0.8]),
        nominal_poni1_m=np.array([0.01, 0.02]),
        nominal_poni2_m=np.array([0.03, 0.04]),
        pixel1_m=np.array([1e-4, 1e-4]),
        pixel2_m=np.array([1e-4, 1e-4]),
    )


def test_geometry_aware_chunk_avoids_pyfai_plan_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_pyfai(*_args, **_kwargs):
        raise AssertionError("Dynamic geometry must not build a pyFAI plan.")

    monkeypatch.setattr(
        joint_uncertainty,
        "_prepare_pyfai_geometry_draw",
        unexpected_pyfai,
    )
    session = _FakeGeometrySession(3)
    profiles = _metal_profile_chunk(
        _patient_frame(),
        metal_context=_geometry_context(session),
        scenario=Scenario("joint", True, True, True, True),
        nuisance=_nuisance(),
        start=1,
        stop=3,
        geometry_audit_draws=0,
        normalization_q_range=(2.0, 4.0),
        random_seed=43,
        photon_replicates=5,
    )[0]

    assert profiles.shape == (10, 2, 3)
    assert session.calls == [("run_nested", 1, 2)]


def test_geometry_aware_deterministic_chunk_repeats_each_geometry_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_pyfai(*_args, **_kwargs):
        raise AssertionError("Dynamic geometry must not build a pyFAI plan.")

    monkeypatch.setattr(
        joint_uncertainty,
        "_prepare_pyfai_geometry_draw",
        unexpected_pyfai,
    )
    session = _FakeGeometrySession(3)
    profiles = _metal_profile_chunk(
        _patient_frame(),
        metal_context=_geometry_context(session),
        scenario=Scenario("geometry", False, True, True, True),
        nuisance=_nuisance(),
        start=1,
        stop=3,
        geometry_audit_draws=0,
        normalization_q_range=(2.0, 4.0),
        random_seed=43,
        photon_replicates=5,
    )[0]

    assert profiles.shape == (10, 2, 3)
    np.testing.assert_allclose(profiles[:5], np.repeat(profiles[0:1], 5, axis=0))
    np.testing.assert_allclose(profiles[5:], np.repeat(profiles[5:6], 5, axis=0))
    assert session.calls == [("integrate", 1, 2)]


def test_geometry_plan_cache_key_preserves_frame_local_mask_identity() -> None:
    frame = pd.DataFrame(
        {
            "measurement_data": [np.zeros((2, 2), dtype=float)],
            "faulty_pixel_mask": [np.array([[0, 1]], dtype=int)],
            "ponifile": ["PONI-A"],
            "sample_thickness_mm": [40.0],
            "calibrant_thickness_mm": [20.0],
            "interpolation_q_range": [(2.0, 4.0)],
            "azimuthal_range": [None],
        }
    )
    kwargs = {
        "thickness": np.zeros((1, 1)),
        "row_delta": np.zeros((1, 1)),
        "column_delta": np.zeros((1, 1)),
        "distance_delta": np.zeros((1, 1)),
        "q_grid": np.array([2.0, 3.0, 4.0]),
        "normalization_q_range": (2.0, 4.0),
    }
    initial = _geometry_plan_cache_key(frame, **kwargs)
    changed_mask = frame.copy()
    changed_mask.at[0, "faulty_pixel_mask"] = np.array([[1, 0]], dtype=int)
    changed_geometry = {**kwargs, "distance_delta": np.array([[0.001]])}

    assert _geometry_plan_cache_key(changed_mask, **kwargs) != initial
    assert _geometry_plan_cache_key(frame, **changed_geometry) != initial


def test_geometry_plan_cache_is_bounded_lru() -> None:
    cache = GeometryPlanCache(max_entries=1)
    draw = PreparedGeometryDraw(
        expected=np.zeros((1, 3)),
        plans=(),
        geometry_rows=(),
    )
    cache.put("first", draw)
    assert cache.get("first") is draw
    cache.put("second", draw)

    assert cache.get("first") is None
    assert cache.get("second") is draw
