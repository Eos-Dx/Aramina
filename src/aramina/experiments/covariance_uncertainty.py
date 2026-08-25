"""Empirical low-rank profile covariance for measurement-uncertainty research.

The covariance is estimated only from detector-level centered-Poisson draws
after the unchanged product integration and q-range normalization path.  It is
therefore a fast approximation to the available photon/statistical component,
not a model of calibration, positioning, or biological variability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class CovarianceUncertaintyError(ValueError):
    """Raised when covariance construction or sampling cannot be audited."""


@dataclass(frozen=True)
class LowRankCovarianceModel:
    """Memory-bounded pooled correlation ``R = B Lambda B^T + D``.

    For one measurement with a normalized pyFAI scale vector ``S``, the
    transferred covariance is ``Sigma = S R S``.  This is a heteroscedastic
    engineering approximation, not empirical measurement repeatability.
    """

    basis: np.ndarray
    eigenvalues: np.ndarray
    diagonal: np.ndarray
    diagnostics: dict[str, float | int | str]

    def sample(
        self,
        *,
        draws: int,
        sigma_scale: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate zero-mean correlated normalized-profile perturbations."""
        if draws < 1:
            raise CovarianceUncertaintyError("draws must be positive.")
        scale = np.asarray(sigma_scale, dtype=float).ravel()
        if (
            scale.shape != self.diagonal.shape
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise CovarianceUncertaintyError(
                "sigma_scale must be a positive vector matching covariance bins."
            )
        low_rank = rng.normal(size=(draws, self.eigenvalues.size))
        low_rank *= np.sqrt(self.eigenvalues)
        perturbation = low_rank @ self.basis.T
        perturbation += rng.normal(size=(draws, self.diagonal.size)) * np.sqrt(
            self.diagonal
        )
        return perturbation * scale[np.newaxis, :]

    def reconstructed_covariance(self) -> np.ndarray:
        """Materialize the compact covariance for diagnostics only."""
        return (self.basis * self.eigenvalues) @ self.basis.T + np.diag(
            self.diagonal
        )


def normalized_profile_sigma(
    q: np.ndarray,
    raw_profile: np.ndarray,
    profile_sigma: np.ndarray,
    *,
    q_range: tuple[float, float] = (6.7, 7.1),
) -> np.ndarray:
    """Return pyFAI sigma after the frozen median-normalization scale.

    This transfers a pooled normalized covariance shape to one measurement
    without claiming that pyFAI provides the missing cross-bin covariance.
    """
    q_values = np.asarray(q, dtype=float).ravel()
    raw_values = np.asarray(raw_profile, dtype=float).ravel()
    sigma_values = np.asarray(profile_sigma, dtype=float).ravel()
    if (
        q_values.size < 2
        or q_values.shape != raw_values.shape
        or q_values.shape != sigma_values.shape
        or not np.isfinite(q_values).all()
        or not np.isfinite(raw_values).all()
        or not np.isfinite(sigma_values).all()
        or np.any(sigma_values < 0.0)
    ):
        raise CovarianceUncertaintyError("Invalid q/raw-profile/sigma arrays.")
    band = (q_values >= q_range[0]) & (q_values <= q_range[1])
    if not np.any(band):
        raise CovarianceUncertaintyError("Normalization q range has no profile bins.")
    normalization = float(np.median(raw_values[band]))
    if not np.isfinite(normalization) or normalization <= 1e-12:
        raise CovarianceUncertaintyError("Normalization scale must be positive.")
    normalized_sigma = sigma_values / normalization
    if not np.isfinite(normalized_sigma).all() or np.any(normalized_sigma <= 0.0):
        raise CovarianceUncertaintyError("Normalized profile sigma is invalid.")
    return normalized_sigma


def fit_low_rank_covariance(
    profile_draws: dict[str, np.ndarray],
    measurement_manifest: pd.DataFrame,
    *,
    explained_variance: float,
    max_rank: int,
    minimum_diagonal_variance: float,
    fixed_rank: int | None = None,
) -> LowRankCovarianceModel:
    """Estimate a pooled covariance from exact normalized detector draws.

    Each measurement covariance is divided by its own normalized pyFAI sigma
    scale before pooling.  Thus the learned object is a covariance *shape*;
    sampling later restores the scale of each target measurement.
    """
    if not 0.5 <= float(explained_variance) < 1.0:
        raise CovarianceUncertaintyError("explained_variance must be in [0.5, 1).")
    if max_rank < 1:
        raise CovarianceUncertaintyError("max_rank must be positive.")
    if fixed_rank is not None and (
        isinstance(fixed_rank, bool) or not isinstance(fixed_rank, int) or fixed_rank < 1
    ):
        raise CovarianceUncertaintyError("fixed_rank must be a positive integer.")
    if minimum_diagonal_variance < 0.0:
        raise CovarianceUncertaintyError("minimum_diagonal_variance must be nonnegative.")
    required = {"profile_key"}
    missing = required.difference(measurement_manifest.columns)
    if missing:
        raise CovarianceUncertaintyError(
            f"Measurement manifest lacks covariance fields: {sorted(missing)}."
        )
    covariances: list[np.ndarray] = []
    dimension: int | None = None
    draw_counts: set[int] = set()
    for row in measurement_manifest.itertuples(index=False):
        key = str(row.profile_key)
        samples = np.asarray(profile_draws.get(key), dtype=float)
        if samples.ndim != 2 or samples.shape[0] < 3 or not np.isfinite(samples).all():
            raise CovarianceUncertaintyError(
                f"Detector profile draws for {key!r} are incomplete or non-finite."
            )
        if dimension is None:
            dimension = int(samples.shape[1])
        if samples.shape[1] != dimension:
            raise CovarianceUncertaintyError("Detector profiles have inconsistent bins.")
        covariance = np.cov(samples, rowvar=False, ddof=1)
        standard_deviation = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        if np.any(standard_deviation <= 0.0):
            raise CovarianceUncertaintyError(
                f"Detector profile draws for {key!r} have zero-variance bins."
            )
        correlation = covariance / np.outer(standard_deviation, standard_deviation)
        correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
        if not np.isfinite(correlation).all():
            raise CovarianceUncertaintyError(
                f"Detector profile draws for {key!r} yield invalid correlation."
            )
        covariances.append(correlation)
        draw_counts.add(int(samples.shape[0]))
    if not covariances or dimension is None:
        raise CovarianceUncertaintyError("No detector profiles were available for fitting.")
    empirical = np.mean(np.stack(covariances), axis=0)
    empirical = (empirical + empirical.T) / 2.0
    np.fill_diagonal(empirical, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(empirical)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    total_variance = float(eigenvalues.sum())
    if total_variance <= 0.0:
        raise CovarianceUncertaintyError("Empirical covariance has no positive variance.")
    cumulative = np.cumsum(eigenvalues) / total_variance
    required_rank = int(np.searchsorted(cumulative, float(explained_variance)) + 1)
    positive_rank = int(np.count_nonzero(eigenvalues > 0.0))
    if fixed_rank is None:
        rank = min(max_rank, required_rank, positive_rank)
        rank_policy = "explained_variance_capped_by_max_rank"
    else:
        rank = min(fixed_rank, positive_rank)
        rank_policy = "fixed_rank"
        if rank != fixed_rank:
            raise CovarianceUncertaintyError(
                f"Requested fixed rank {fixed_rank} exceeds positive covariance rank "
                f"{positive_rank}."
            )
    if rank < 1:
        raise CovarianceUncertaintyError("Covariance has no usable positive eigenvalues.")
    basis = eigenvectors[:, :rank]
    retained = eigenvalues[:rank]
    low_rank = (basis * retained) @ basis.T
    diagonal = np.maximum(
        np.diag(empirical - low_rank), float(minimum_diagonal_variance)
    )
    reconstructed = low_rank + np.diag(diagonal)
    frobenius = float(np.linalg.norm(empirical, ord="fro"))
    reconstruction_error = float(
        np.linalg.norm(empirical - reconstructed, ord="fro") / frobenius
    )
    return LowRankCovarianceModel(
        basis=basis,
        eigenvalues=retained,
        diagonal=diagonal,
        diagnostics={
            "feature_count": dimension,
            "reference_measurements": len(covariances),
            "reference_draws_min": min(draw_counts),
            "reference_draws_max": max(draw_counts),
            "selected_rank": rank,
            "requested_max_rank": int(max_rank),
            "rank_policy": rank_policy,
            "explained_variance_target": float(explained_variance),
            "explained_variance_retained": float(cumulative[rank - 1]),
            "pooled_correlation_trace": total_variance,
            "relative_frobenius_reconstruction_error": reconstruction_error,
            "minimum_diagonal_variance": float(minimum_diagonal_variance),
        },
    )


def fit_full_shrinkage_covariance(
    profile_draws: dict[str, np.ndarray],
    measurement_manifest: pd.DataFrame,
) -> LowRankCovarianceModel:
    """Fit full-rank Ledoit-Wolf correlation from detector-MC residuals.

    Draws are centered and standardized within each measurement before they
    are pooled. Ledoit-Wolf then shrinks their covariance toward a scaled
    identity target. The result is normalized to a unit-diagonal correlation
    matrix and represented by the existing covariance-model interface without
    spectral truncation.
    """
    required = {"profile_key"}
    missing = required.difference(measurement_manifest.columns)
    if missing:
        raise CovarianceUncertaintyError(
            f"Measurement manifest lacks covariance fields: {sorted(missing)}."
        )

    standardized_residuals: list[np.ndarray] = []
    dimension: int | None = None
    draw_counts: set[int] = set()
    for row in measurement_manifest.itertuples(index=False):
        key = str(row.profile_key)
        samples = np.asarray(profile_draws.get(key), dtype=float)
        if samples.ndim != 2 or samples.shape[0] < 3 or not np.isfinite(samples).all():
            raise CovarianceUncertaintyError(
                f"Detector profile draws for {key!r} are incomplete or non-finite."
            )
        if dimension is None:
            dimension = int(samples.shape[1])
        if samples.shape[1] != dimension:
            raise CovarianceUncertaintyError("Detector profiles have inconsistent bins.")

        residuals = samples - samples.mean(axis=0, keepdims=True)
        standard_deviation = residuals.std(axis=0, ddof=1)
        if np.any(standard_deviation <= 0.0):
            raise CovarianceUncertaintyError(
                f"Detector profile draws for {key!r} have zero-variance bins."
            )
        standardized_residuals.append(residuals / standard_deviation)
        draw_counts.add(int(samples.shape[0]))

    if not standardized_residuals or dimension is None:
        raise CovarianceUncertaintyError("No detector profiles were available for fitting.")

    pooled = np.vstack(standardized_residuals)
    estimator = LedoitWolf(assume_centered=True, store_precision=False).fit(pooled)
    covariance = np.asarray(estimator.covariance_, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    scale = np.sqrt(np.diag(covariance))
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise CovarianceUncertaintyError(
            "Ledoit-Wolf covariance has invalid diagonal variance."
        )
    correlation = covariance / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    correlation = (eigenvectors * eigenvalues) @ eigenvectors.T
    correlation_scale = np.sqrt(np.diag(correlation))
    if not np.isfinite(correlation_scale).all() or np.any(correlation_scale <= 0.0):
        raise CovarianceUncertaintyError(
            "Ledoit-Wolf correlation normalization is invalid."
        )
    correlation /= np.outer(correlation_scale, correlation_scale)
    correlation = (correlation + correlation.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    basis = eigenvectors[:, order]
    diagonal = np.zeros(dimension, dtype=float)
    reconstructed = (basis * eigenvalues) @ basis.T
    trace = float(np.trace(correlation))
    reconstruction_error = float(
        np.linalg.norm(correlation - reconstructed, ord="fro")
        / np.linalg.norm(correlation, ord="fro")
    )

    return LowRankCovarianceModel(
        basis=basis,
        eigenvalues=eigenvalues,
        diagonal=diagonal,
        diagnostics={
            "estimator": "ledoit_wolf",
            "shrinkage_coefficient": float(estimator.shrinkage_),
            "feature_count": dimension,
            "reference_measurements": len(standardized_residuals),
            "reference_observations": int(pooled.shape[0]),
            "reference_draws_min": min(draw_counts),
            "reference_draws_max": max(draw_counts),
            "selected_rank": dimension,
            "requested_max_rank": dimension,
            "explained_variance_target": 1.0,
            "explained_variance_retained": 1.0,
            "pooled_correlation_trace": trace,
            "relative_frobenius_reconstruction_error": reconstruction_error,
            "minimum_diagonal_variance": 0.0,
            "minimum_correlation_eigenvalue": float(eigenvalues[-1]),
        },
    )


def write_low_rank_covariance(
    path: str,
    model: LowRankCovarianceModel,
) -> None:
    """Persist compact matrices and auditable diagnostics without pickle."""
    np.savez_compressed(
        path,
        basis=model.basis,
        eigenvalues=model.eigenvalues,
        diagonal=model.diagonal,
    )


def covariance_diagnostics_frame(model: LowRankCovarianceModel) -> pd.DataFrame:
    """Return one-row diagnostic table suitable for CSV and MLflow logging."""
    return pd.DataFrame([model.diagnostics])


def covariance_eigen_spectrum_frame(model: LowRankCovarianceModel) -> pd.DataFrame:
    """Return retained eigen spectrum and its share of pooled correlation trace."""
    trace = float(model.diagnostics["pooled_correlation_trace"])
    values = np.asarray(model.eigenvalues, dtype=float)
    return pd.DataFrame(
        {
            "component": np.arange(1, values.size + 1),
            "eigenvalue": values,
            "explained_variance_fraction": values / trace,
            "cumulative_explained_variance_fraction": np.cumsum(values) / trace,
        }
    )
