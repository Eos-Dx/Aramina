"""SK target-versus-contralateral symmetry feature calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .target_breast_model import SK_CORE4_FEATURE_COLUMNS


SK_SYMMETRY_COLUMNS = (
    "sk_meanrms1",
    "sk_weightedrms1",
    "sk_sigma_target1",
    "sk_sigma_contralateral1",
    "sk_mahalanobis1",
    "sk_meanrms2",
    "sk_weightedrms2",
    "sk_sigma_target2",
    "sk_sigma_contralateral2",
    "sk_mahalanobis2",
    "sk_peak14_intensity_abs_delta",
    "sk_mean_peak_value_abs_delta",
    "sk_wasserstein_distance_mu_tc",
    "sk_cosine_distance_full_q2",
    "sk_wasserstein_distance_full_q2",
)


SYMMETRY_MIN_MEASUREMENTS_PER_BREAST = 2
SK_FEATURE_CONTRACT_V0_1 = "aramina_sk_symmetry_v0_1"
SK_FEATURE_CONTRACT_V0_2 = "aramina_sk_symmetry_v0_2"


def target_contralateral_symmetry_features(
    patient_df: pd.DataFrame,
    *,
    profile_column: str,
    q_column: str,
    side_column: str,
    target_side_norm: str,
    contralateral_side_norm: str | None,
    feature_contract: str = SK_FEATURE_CONTRACT_V0_2,
) -> dict[str, Any]:
    """Calculate paired-breast features when the Core4 refinement is usable."""
    if feature_contract not in {
        SK_FEATURE_CONTRACT_V0_1,
        SK_FEATURE_CONTRACT_V0_2,
    }:
        raise ValueError(f"Unsupported SK feature contract: {feature_contract!r}")
    legacy = feature_contract == SK_FEATURE_CONTRACT_V0_1
    target = _side_profiles(patient_df, profile_column, side_column, target_side_norm)
    contralateral = (
        _side_profiles(patient_df, profile_column, side_column, contralateral_side_norm)
        if contralateral_side_norm is not None
        else []
    )
    target_within = _mean_pairwise_cosine(target)
    contralateral_within = _mean_pairwise_cosine(contralateral)
    if not contralateral:
        out = {
            "symmetry_available": 0,
            "symmetry_reason": "contralateral_breast_unavailable",
            "target_measurements": int(len(target)),
            "contralateral_measurements": int(len(contralateral)),
            "target_within_cosine_distance_mean": _finite_or_zero(target_within),
            "contralateral_within_cosine_distance_mean": _finite_or_zero(
                contralateral_within
            ),
            "between_breasts_cosine_distance_mean": 0.0,
            "symmetry_cosine_score": 0.0,
        }
        out.update(empty_sk_symmetry_features())
        return out

    if not legacy and (
        len(target) < SYMMETRY_MIN_MEASUREMENTS_PER_BREAST
        or len(contralateral) < SYMMETRY_MIN_MEASUREMENTS_PER_BREAST
    ):
        out = {
            "symmetry_available": 0,
            "symmetry_reason": "fewer_than_2_valid_measurements_per_breast",
            "target_measurements": int(len(target)),
            "contralateral_measurements": int(len(contralateral)),
            "target_within_cosine_distance_mean": _finite_or_zero(target_within),
            "contralateral_within_cosine_distance_mean": _finite_or_zero(
                contralateral_within
            ),
            "between_breasts_cosine_distance_mean": 0.0,
            "symmetry_cosine_score": 0.0,
        }
        out.update(empty_sk_symmetry_features())
        return out

    target_mean = np.mean(np.vstack(target), axis=0)
    contralateral_mean = np.mean(np.vstack(contralateral), axis=0)
    between = _cosine_distance(target_mean, contralateral_mean)
    within_values = [
        value for value in (target_within, contralateral_within) if np.isfinite(value)
    ]
    within_mean = float(np.mean(within_values)) if within_values else 0.0
    raw_sk = _sk_target_contralateral_features(
        patient_df,
        profile_column=profile_column,
        q_column=q_column,
        side_column=side_column,
        target_side_norm=target_side_norm,
        contralateral_side_norm=contralateral_side_norm,
        legacy=legacy,
    )
    if not legacy and not all(
        np.isfinite(raw_sk[column]) for column in SK_CORE4_FEATURE_COLUMNS
    ):
        out = {
            "symmetry_available": 0,
            "symmetry_reason": "sk_core4_not_computable",
            "target_measurements": int(len(target)),
            "contralateral_measurements": int(len(contralateral)),
            "target_within_cosine_distance_mean": _finite_or_zero(target_within),
            "contralateral_within_cosine_distance_mean": _finite_or_zero(
                contralateral_within
            ),
            "between_breasts_cosine_distance_mean": _finite_or_zero(between),
            "symmetry_cosine_score": _finite_or_zero(between - within_mean),
        }
        out.update(empty_sk_symmetry_features())
        return out

    out = {
        "symmetry_available": 1,
        "symmetry_reason": "",
        "target_measurements": int(len(target)),
        "contralateral_measurements": int(len(contralateral)),
        "target_within_cosine_distance_mean": _finite_or_zero(target_within),
        "contralateral_within_cosine_distance_mean": _finite_or_zero(
            contralateral_within
        ),
        "between_breasts_cosine_distance_mean": _finite_or_zero(between),
        "symmetry_cosine_score": _finite_or_zero(between - within_mean),
    }
    out.update({column: _finite_or_zero(value) for column, value in raw_sk.items()})
    return out


def empty_sk_symmetry_features() -> dict[str, float]:
    """Return the neutral SK feature vector used when pairing is unavailable."""
    return {column: 0.0 for column in SK_SYMMETRY_COLUMNS}


def _sk_target_contralateral_features(
    patient_df: pd.DataFrame,
    *,
    profile_column: str,
    q_column: str,
    side_column: str,
    target_side_norm: str,
    contralateral_side_norm: str | None,
    legacy: bool = False,
) -> dict[str, float]:
    if contralateral_side_norm is None:
        return empty_sk_symmetry_features()
    metrics = _side_mean_metrics(
        patient_df,
        profile_column=profile_column,
        q_column=q_column,
        side_column=side_column,
        target_side_norm=target_side_norm,
        contralateral_side_norm=contralateral_side_norm,
        q_roi=(7.5, 23.0) if legacy else (6.7, 23.0),
        legacy=legacy,
    )
    metrics_full = _side_mean_metrics(
        patient_df,
        profile_column=profile_column,
        q_column=q_column,
        side_column=side_column,
        target_side_norm=target_side_norm,
        contralateral_side_norm=contralateral_side_norm,
        q_roi=(2.0, 23.0),
        legacy=legacy,
    )
    if not metrics or not metrics_full:
        return (
            empty_sk_symmetry_features()
            if legacy
            else {column: float("nan") for column in SK_SYMMETRY_COLUMNS}
        )
    q = metrics["q"]
    mu_target = metrics["mu_target"]
    mu_contralateral = metrics["mu_contralateral"]
    std_target = metrics["std_target"]
    std_contralateral = metrics["std_contralateral"]
    mask1 = (q >= (7.0 if legacy else 6.7)) & (q <= 15.0)
    mask2 = (q >= 15.0) & (q <= 23.0)
    raw = {
        "sk_meanrms1": _rms_difference(mu_target, mu_contralateral, mask1),
        "sk_weightedrms1": (
            _weighted_rms_difference(
                mu_target, mu_contralateral, std_target, std_contralateral, mask1
            )
        ),
        "sk_sigma_target1": _sigma_rms(std_target, mask1),
        "sk_sigma_contralateral1": _sigma_rms(std_contralateral, mask1),
        "sk_mahalanobis1": (
            _mahalanobis_difference(
                mu_target, mu_contralateral, std_target, std_contralateral, mask1
            )
        ),
        "sk_meanrms2": _rms_difference(mu_target, mu_contralateral, mask2),
        "sk_weightedrms2": (
            _weighted_rms_difference(
                mu_target, mu_contralateral, std_target, std_contralateral, mask2
            )
        ),
        "sk_sigma_target2": _sigma_rms(std_target, mask2),
        "sk_sigma_contralateral2": _sigma_rms(std_contralateral, mask2),
        "sk_mahalanobis2": (
            _mahalanobis_difference(
                mu_target, mu_contralateral, std_target, std_contralateral, mask2
            )
        ),
        "sk_peak14_intensity_abs_delta": _peak14_intensity_abs_delta(
            q, mu_target, mu_contralateral
        ),
        "sk_mean_peak_value_abs_delta": (
            _mean_peak_value_abs_delta(
                patient_df,
                q_column=q_column,
                profile_column=profile_column,
                side_column=side_column,
                target_side_norm=target_side_norm,
                contralateral_side_norm=contralateral_side_norm,
            )
        ),
        "sk_wasserstein_distance_mu_tc": _profile_wasserstein(
            q, mu_target, mu_contralateral
        ),
        "sk_cosine_distance_full_q2": (
            _cosine_distance(
                metrics_full["mu_target"], metrics_full["mu_contralateral"]
            )
        ),
        "sk_wasserstein_distance_full_q2": _profile_wasserstein(
            metrics_full["q"],
            metrics_full["mu_target"],
            metrics_full["mu_contralateral"],
        ),
    }
    return {column: _finite_or_zero(value) for column, value in raw.items()} if legacy else raw


def _side_profiles(
    df: pd.DataFrame,
    profile_column: str,
    side_column: str,
    side_norm: str | None,
) -> list[np.ndarray]:
    if side_norm is None:
        return []
    rows = df[df[side_column].map(_normalize_side) == side_norm]
    return [np.asarray(value, dtype=float).ravel() for value in rows[profile_column]]


def _side_mean_metrics(
    df: pd.DataFrame,
    *,
    profile_column: str,
    q_column: str,
    side_column: str,
    target_side_norm: str,
    contralateral_side_norm: str,
    q_roi: tuple[float, float],
    legacy: bool = False,
) -> dict[str, np.ndarray] | None:
    target_profiles: list[np.ndarray] = []
    contralateral_profiles: list[np.ndarray] = []
    q_common: np.ndarray | None = None
    for row in df.itertuples(index=False):
        side = _normalize_side(getattr(row, side_column))
        if side not in {target_side_norm, contralateral_side_norm}:
            continue
        q = np.asarray(getattr(row, q_column), dtype=float).ravel()
        y = np.asarray(getattr(row, profile_column), dtype=float).ravel()
        q, y = _profile_roi(q, y, q_roi, fallback_full_range=legacy)
        if q.size < 5:
            continue
        y = (
            _normalize_profile_near_minimum(q, _smooth_profile(y))
            if legacy
            else _smooth_profile(y)
        )
        if q_common is None:
            q_common = q
        y_common = np.interp(q_common, q, y)
        if side == target_side_norm:
            target_profiles.append(y_common)
        else:
            contralateral_profiles.append(y_common)
    if q_common is None or not target_profiles or not contralateral_profiles:
        return None
    target = np.vstack(target_profiles)
    contralateral = np.vstack(contralateral_profiles)
    return {
        "q": q_common,
        "mu_target": np.mean(target, axis=0),
        "mu_contralateral": np.mean(contralateral, axis=0),
        "std_target": _profile_std(target),
        "std_contralateral": _profile_std(contralateral),
    }


def _profile_roi(
    q: np.ndarray,
    y: np.ndarray,
    q_roi: tuple[float, float],
    *,
    fallback_full_range: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (q >= float(q_roi[0])) & (q <= float(q_roi[1]))
    if fallback_full_range and int(mask.sum()) < 5:
        return q, y
    return q[mask], y[mask]


def _smooth_profile(y: np.ndarray) -> np.ndarray:
    if y.size < 7:
        return y
    window = min(11, y.size if y.size % 2 else y.size - 1)
    if window < 5:
        return y
    return savgol_filter(y, window_length=window, polyorder=min(3, window - 2))


def _normalize_profile_near_minimum(
    q: np.ndarray,
    y: np.ndarray,
    *,
    q0: float = 6.7,
    halfwidth: float = 0.25,
) -> np.ndarray:
    """Legacy v0.1 SK-only normalization retained for released artifacts."""
    mask = (q >= q0 - halfwidth) & (q <= q0 + halfwidth) & np.isfinite(y)
    baseline = (
        float(np.nanpercentile(y[mask], 5))
        if int(mask.sum()) >= 2
        else float(np.nanpercentile(y, 5))
    )
    if not np.isfinite(baseline) or abs(baseline) < 1e-12:
        baseline = 1.0
    return y / baseline


def _profile_std(values: np.ndarray) -> np.ndarray:
    return (
        np.zeros(values.shape[1])
        if values.shape[0] < 2
        else np.std(values, axis=0, ddof=1)
    )


def _rms_difference(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = np.asarray(a - b, dtype=float)
    good = mask & np.isfinite(diff)
    return float(np.sqrt(np.mean(diff[good] ** 2))) if int(good.sum()) >= 5 else np.nan


def _weighted_rms_difference(
    a: np.ndarray, b: np.ndarray, std_a: np.ndarray, std_b: np.ndarray, mask: np.ndarray
) -> float:
    diff = np.asarray(a - b, dtype=float)
    variance = np.asarray(std_a**2 + std_b**2, dtype=float)
    good = mask & np.isfinite(diff) & np.isfinite(variance)
    if int(good.sum()) < 5:
        return np.nan
    floor = float(np.nanpercentile(variance[good], 5))
    weight = 1.0 / np.maximum(variance[good], floor + 1e-12)
    return float(np.sqrt(np.sum(weight * diff[good] ** 2) / np.sum(weight)))


def _mahalanobis_difference(
    a: np.ndarray, b: np.ndarray, std_a: np.ndarray, std_b: np.ndarray, mask: np.ndarray
) -> float:
    diff = np.asarray(a - b, dtype=float)
    variance = np.asarray(std_a**2 + std_b**2, dtype=float)
    good = mask & np.isfinite(diff) & np.isfinite(variance)
    if int(good.sum()) < 5:
        return np.nan
    return float(np.sqrt(np.sum(diff[good] ** 2 / (variance[good] + 1e-12))))


def _sigma_rms(std: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(std)
    return float(np.sqrt(np.mean(std[good] ** 2))) if int(good.sum()) >= 5 else np.nan


def _peak14_intensity_abs_delta(
    q: np.ndarray, target: np.ndarray, contralateral: np.ndarray
) -> float:
    target_peak = _peak_value(q, target, q_min=13.5, q_max=14.5)
    contralateral_peak = _peak_value(q, contralateral, q_min=13.5, q_max=14.5)
    if not np.isfinite(target_peak) or not np.isfinite(contralateral_peak):
        return np.nan
    return float(abs(target_peak - contralateral_peak))


def _peak_value(q: np.ndarray, y: np.ndarray, *, q_min: float, q_max: float) -> float:
    mask = (q >= q_min) & (q <= q_max) & np.isfinite(y)
    return float(np.nanmax(y[mask])) if int(mask.sum()) >= 3 else np.nan


def _mean_peak_value_abs_delta(
    df: pd.DataFrame,
    *,
    q_column: str,
    profile_column: str,
    side_column: str,
    target_side_norm: str,
    contralateral_side_norm: str,
) -> float:
    target_peak = _mean_peak_value_for_side(
        df,
        q_column=q_column,
        profile_column=profile_column,
        side_column=side_column,
        side_norm=target_side_norm,
    )
    contralateral_peak = _mean_peak_value_for_side(
        df,
        q_column=q_column,
        profile_column=profile_column,
        side_column=side_column,
        side_norm=contralateral_side_norm,
    )
    if not np.isfinite(target_peak) or not np.isfinite(contralateral_peak):
        return np.nan
    return float(abs(target_peak - contralateral_peak))


def _mean_peak_value_for_side(
    df: pd.DataFrame,
    *,
    q_column: str,
    profile_column: str,
    side_column: str,
    side_norm: str,
) -> float:
    values = []
    for side, q_raw, y_raw in df[[side_column, q_column, profile_column]].itertuples(
        index=False, name=None
    ):
        if _normalize_side(side) != side_norm:
            continue
        peak = _peak_value(
            np.asarray(q_raw, dtype=float).ravel(),
            np.asarray(y_raw, dtype=float).ravel(),
            q_min=13.0,
            q_max=14.8,
        )
        if np.isfinite(peak):
            values.append(peak)
    return float(np.mean(values)) if values else np.nan


def _profile_wasserstein(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(q) & np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 5:
        return np.nan
    qv = q[good]
    av = np.clip(a[good], 0.0, None)
    bv = np.clip(b[good], 0.0, None)
    if float(av.sum()) <= 1e-12 or float(bv.sum()) <= 1e-12:
        return np.nan
    order = np.argsort(qv)
    qv, av, bv = qv[order], av[order] / float(av.sum()), bv[order] / float(bv.sum())
    return float(np.sum(np.abs(np.cumsum(av)[:-1] - np.cumsum(bv)[:-1]) * np.diff(qv)))


def _normalize_side(value: Any) -> str | None:
    clean = str(value).strip().upper()
    if clean.startswith("LEFT"):
        return "LEFT"
    if clean.startswith("RIGHT"):
        return "RIGHT"
    return None


def _mean_pairwise_cosine(profiles: list[np.ndarray]) -> float:
    if len(profiles) < 2:
        return np.nan
    values = [
        _cosine_distance(profiles[i], profiles[j])
        for i in range(len(profiles))
        for j in range(i + 1, len(profiles))
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 3:
        return np.nan
    av, bv = np.asarray(a[good], dtype=float), np.asarray(b[good], dtype=float)
    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denominator <= 1e-12:
        return np.nan
    return float(1.0 - np.clip(np.dot(av, bv) / denominator, -1.0, 1.0))


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
