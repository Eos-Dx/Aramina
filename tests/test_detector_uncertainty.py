from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aramina.experiments import detector_uncertainty


def _detector_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patientId": ["P00", "P00"],
            "specimenId": ["S-left", "S-right"],
            "side": ["Left", "Right"],
            "position": ["P1", "P1"],
            "started_at": ["2026-01-01", "2026-01-01"],
            "measurement_data": [np.ones((4, 4)), np.full((4, 4), 2.0)],
            "faulty_pixel_mask": [np.empty((0, 2), dtype=int)] * 2,
            "ponifile": ["fake"] * 2,
            "sample_thickness_mm": [20.0] * 2,
            "calibrant_thickness_mm": [10.0] * 2,
            "calibration_session_uid": ["calib-a"] * 2,
        }
    )


def test_detector_poisson_profiles_reintegrate_every_measurement(monkeypatch):
    frame = _detector_frame()

    def fake_poisson(expected, *_args, **_kwargs):
        return SimpleNamespace(values=expected[np.newaxis, ...])

    def fake_integration(row, **_kwargs):
        q = np.asarray([6.8, 7.0, 8.0])
        intensity = np.asarray([2.0, 4.0, float(np.mean(row["measurement_data"]))])
        return q, intensity, np.ones(3), 0.1

    monkeypatch.setattr(
        detector_uncertainty,
        "sample_detector_centered_poisson",
        fake_poisson,
    )
    monkeypatch.setattr(
        detector_uncertainty,
        "perform_azimuthal_integration",
        fake_integration,
    )

    draws = list(
        detector_uncertainty.iter_detector_poisson_profiles(
            frame,
            draws=2,
            random_state=42,
            normalization_q_range=(6.7, 7.1),
        )
    )

    assert len(draws) == 2
    assert all(len(draw) == 2 for draw in draws)
    assert np.allclose(draws[0]["radial_profile_data"].iloc[0], [2 / 3, 4 / 3, 1 / 3])


def test_detector_poisson_profiles_zero_masked_invalid_pixels(monkeypatch):
    frame = _detector_frame().iloc[[0]].copy()
    image = np.ones((4, 4))
    image[0, 1] = -2.0
    image[2, 3] = np.nan
    frame.at[0, "measurement_data"] = image
    frame.at[0, "faulty_pixel_mask"] = np.asarray([[0, 1], [2, 3]])
    captured = {}

    def fake_poisson(expected, *_args, **_kwargs):
        captured["expected"] = expected.copy()
        return SimpleNamespace(values=expected[np.newaxis, ...])

    monkeypatch.setattr(
        detector_uncertainty,
        "sample_detector_centered_poisson",
        fake_poisson,
    )
    monkeypatch.setattr(
        detector_uncertainty,
        "perform_azimuthal_integration",
        lambda *_args, **_kwargs: (
            np.asarray([6.8, 7.0, 8.0]),
            np.asarray([2.0, 4.0, 3.0]),
            np.ones(3),
            0.1,
        ),
    )

    list(
        detector_uncertainty.iter_detector_poisson_profiles(
            frame,
            draws=1,
            random_state=42,
            normalization_q_range=(6.7, 7.1),
        )
    )

    assert captured["expected"][0, 0, 1] == 0.0
    assert captured["expected"][0, 2, 3] == 0.0


def test_detector_poisson_profiles_keep_unmasked_negative_pixel(monkeypatch):
    frame = _detector_frame().iloc[[0]].copy()
    image = np.ones((4, 4))
    image[0, 1] = -2.0
    frame.at[0, "measurement_data"] = image
    captured = {}

    def fake_poisson(observed, *_args, **_kwargs):
        captured["observed"] = observed.copy()
        return SimpleNamespace(values=observed[np.newaxis, ...])

    monkeypatch.setattr(
        detector_uncertainty,
        "sample_detector_centered_poisson",
        fake_poisson,
    )
    monkeypatch.setattr(
        detector_uncertainty,
        "perform_azimuthal_integration",
        lambda *_args, **_kwargs: (
            np.asarray([6.8, 7.0, 8.0]),
            np.asarray([2.0, 4.0, 3.0]),
            np.ones(3),
            0.1,
        ),
    )

    next(
        detector_uncertainty.iter_detector_poisson_profiles(
            frame,
            draws=1,
            random_state=42,
            normalization_q_range=(6.7, 7.1),
        )
    )

    assert captured["observed"][0, 0, 1] == -2.0


def test_polar_cake_artifacts_store_axes_uncertainty_and_parity(
    tmp_path: Path,
    monkeypatch,
):
    frame = _detector_frame().iloc[[0]].copy()
    cake = SimpleNamespace(
        intensity=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        q=np.asarray([7.0, 8.0]),
        azimuth=np.asarray([-90.0, 90.0]),
        sigma=np.full((2, 2), 0.1),
        sum_variance=np.full((2, 2), 0.2),
        count=np.ones((2, 2)),
    )
    monkeypatch.setattr(
        detector_uncertainty,
        "perform_polar_cake_integration",
        lambda *_args, **_kwargs: cake,
    )
    monkeypatch.setattr(
        detector_uncertainty,
        "perform_azimuthal_integration",
        lambda *_args, **_kwargs: (
            cake.q,
            np.asarray([2.0, 3.0]),
            np.ones(2),
            0.1,
        ),
    )

    manifest = detector_uncertainty.write_polar_cake_artifacts(
        frame,
        target_cases=[("P00", "left")],
        output_folder=tmp_path / "polar_cakes",
        n_q=2,
        n_chi=2,
        parity_max_relative_rmse=1e-12,
    )

    assert manifest["parity_pass"].tolist() == [True]
    artifact = tmp_path / manifest["artifact"].iloc[0]
    with np.load(artifact) as stored:
        assert stored["intensity"].shape == (2, 2)
        assert stored["sigma"].shape == (2, 2)
