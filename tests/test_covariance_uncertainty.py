from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from aramina.experiments.covariance_uncertainty import (
    covariance_eigen_spectrum_frame,
    fit_full_shrinkage_covariance,
    fit_low_rank_covariance,
    normalized_profile_sigma,
)


def test_low_rank_covariance_preserves_detector_mc_correlation_and_is_seeded():
    rng = np.random.default_rng(17)
    latent = rng.normal(size=(200, 1))
    samples = np.column_stack(
        (
            latent[:, 0] + rng.normal(scale=0.1, size=200),
            latent[:, 0] + rng.normal(scale=0.1, size=200),
            rng.normal(scale=0.3, size=200),
        )
    )
    model = fit_low_rank_covariance(
        {"profile_0": samples},
        pd.DataFrame({"profile_key": ["profile_0"]}),
        explained_variance=0.9,
        max_rank=2,
        minimum_diagonal_variance=1e-12,
    )

    first = model.sample(
        draws=4,
        sigma_scale=np.asarray([0.1, 0.1, 0.1]),
        rng=np.random.default_rng(31),
    )
    second = model.sample(
        draws=4,
        sigma_scale=np.asarray([0.1, 0.1, 0.1]),
        rng=np.random.default_rng(31),
    )

    np.testing.assert_allclose(first, second)
    reconstructed = model.reconstructed_covariance()
    assert reconstructed.shape == (3, 3)
    assert reconstructed[0, 1] > 0.5
    assert model.diagnostics["selected_rank"] <= 2
    spectrum = covariance_eigen_spectrum_frame(model)
    assert spectrum["component"].tolist() == list(range(1, len(spectrum) + 1))
    assert spectrum["cumulative_explained_variance_fraction"].is_monotonic_increasing


def test_low_rank_covariance_supports_explicit_fixed_rank() -> None:
    rng = np.random.default_rng(91)
    samples = rng.normal(size=(200, 5))
    manifest = pd.DataFrame({"profile_key": ["profile"]})

    adaptive = fit_low_rank_covariance(
        {"profile": samples},
        manifest,
        explained_variance=0.5,
        max_rank=5,
        minimum_diagonal_variance=1e-12,
    )
    fixed = fit_low_rank_covariance(
        {"profile": samples},
        manifest,
        explained_variance=0.5,
        max_rank=5,
        minimum_diagonal_variance=1e-12,
        fixed_rank=5,
    )

    assert adaptive.diagnostics["selected_rank"] < 5
    assert adaptive.diagnostics["rank_policy"] == (
        "explained_variance_capped_by_max_rank"
    )
    assert fixed.diagnostics["selected_rank"] == 5
    assert fixed.diagnostics["rank_policy"] == "fixed_rank"


def test_normalized_profile_sigma_uses_frozen_median_normalization():
    sigma = normalized_profile_sigma(
        np.asarray([6.8, 7.0, 7.2]),
        np.asarray([2.0, 4.0, 8.0]),
        np.asarray([0.2, 0.4, 0.8]),
    )

    np.testing.assert_allclose(sigma, [1 / 15, 2 / 15, 4 / 15])


def test_full_shrinkage_covariance_matches_auditable_ledoit_wolf_fit():
    rng = np.random.default_rng(41)
    first = rng.multivariate_normal(
        mean=[2.0, 5.0, -3.0],
        cov=[[4.0, 1.5, 0.0], [1.5, 1.0, 0.2], [0.0, 0.2, 0.5]],
        size=80,
    )
    second = rng.multivariate_normal(
        mean=[-4.0, 1.0, 7.0],
        cov=[[1.0, 0.6, 0.1], [0.6, 3.0, 0.0], [0.1, 0.0, 2.0]],
        size=60,
    )
    draws = {"first": first, "second": second}
    manifest = pd.DataFrame({"profile_key": ["first", "second"]})

    model = fit_full_shrinkage_covariance(draws, manifest)

    standardized = []
    for samples in (first, second):
        residuals = samples - samples.mean(axis=0, keepdims=True)
        standardized.append(residuals / residuals.std(axis=0, ddof=1))
    expected_covariance = LedoitWolf(
        assume_centered=True, store_precision=False
    ).fit(np.vstack(standardized))
    expected_scale = np.sqrt(np.diag(expected_covariance.covariance_))
    expected_correlation = expected_covariance.covariance_ / np.outer(
        expected_scale, expected_scale
    )

    reconstructed = model.reconstructed_covariance()
    np.testing.assert_allclose(reconstructed, expected_correlation, atol=1e-12)
    np.testing.assert_allclose(np.diag(reconstructed), np.ones(3), atol=1e-12)
    assert np.linalg.eigvalsh(reconstructed).min() >= -1e-12
    assert model.basis.shape == (3, 3)
    assert model.eigenvalues.shape == (3,)
    np.testing.assert_array_equal(model.diagonal, np.zeros(3))
    assert model.diagnostics["estimator"] == "ledoit_wolf"
    assert model.diagnostics["shrinkage_coefficient"] == (
        expected_covariance.shrinkage_
    )
    assert model.diagnostics["selected_rank"] == 3
    assert model.diagnostics["reference_measurements"] == 2
    assert model.diagnostics["reference_observations"] == 140


def test_full_shrinkage_covariance_is_seeded_through_existing_interface():
    rng = np.random.default_rng(9)
    samples = rng.normal(size=(50, 4))
    model = fit_full_shrinkage_covariance(
        {"profile": samples},
        pd.DataFrame({"profile_key": ["profile"]}),
    )

    first = model.sample(
        draws=12,
        sigma_scale=np.full(4, 0.1),
        rng=np.random.default_rng(3),
    )
    second = model.sample(
        draws=12,
        sigma_scale=np.full(4, 0.1),
        rng=np.random.default_rng(3),
    )

    np.testing.assert_allclose(first, second)
