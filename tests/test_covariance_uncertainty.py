from __future__ import annotations

import numpy as np
import pandas as pd

from aramina.experiments.covariance_uncertainty import (
    covariance_eigen_spectrum_frame,
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


def test_normalized_profile_sigma_uses_frozen_median_normalization():
    sigma = normalized_profile_sigma(
        np.asarray([6.8, 7.0, 7.2]),
        np.asarray([2.0, 4.0, 8.0]),
        np.asarray([0.2, 0.4, 0.8]),
    )

    np.testing.assert_allclose(sigma, [1 / 15, 2 / 15, 4 / 15])
