"""Paired statistics for completed polar-harmonic research runs.

The utility is deliberately fail-closed.  A paired result is meaningful only
when every variant scored the same held-out target cases in the same folds.
It is research-only and does not alter product thresholds or model artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..model_metrics import ratio


MODE_A0 = "A0"
MODE_A0_A2 = "A0+A2"
MODE_A0_A2_A4 = "A0+A2+A4"
PRIMARY_CONTRASTS = (
    ("A0+A2 minus A0", MODE_A0, MODE_A0_A2),
    ("A0+A2+A4 minus A0+A2", MODE_A0_A2, MODE_A0_A2_A4),
)
PRIMARY_METRICS = ("sensitivity", "specificity", "roc_auc")
_PREDICTION_KEY = ("n_chi", "mode_set", "split_id", "target_case_id")
_METRIC_KEY = ("n_chi", "mode_set", "split_id")


class PolarHarmonicStatisticsError(ValueError):
    """Raised when inputs cannot support a valid paired comparison."""


@dataclass(frozen=True)
class PolarHarmonicStatisticsConfig:
    """Deterministic settings for the research-only paired analysis."""

    seed: int = 20260826
    bootstrap_iterations: int = 2_000
    reference_n_chi: int = 36


def analyze_polar_harmonic_runs(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    config: PolarHarmonicStatisticsConfig = PolarHarmonicStatisticsConfig(),
) -> dict[str, pd.DataFrame]:
    """Calculate paired contrasts and chi-resolution stability statistics.

    ``predictions`` must contain one held-out target-case prediction per
    ``(n_chi, mode_set, split_id, target_case_id)``.  ``fold_metrics`` must
    contain the same variant/split grid and the product metric columns.  The
    function derives cohort and fold fingerprints from prediction rows, so a
    supplied result cannot be treated as paired merely because its metadata
    says it is.
    """
    _validate_config(config)
    prepared_predictions = _prepare_predictions(predictions)
    prepared_metrics = _prepare_metrics(fold_metrics)
    fingerprints = _validate_common_folds(prepared_predictions)
    _validate_metric_grid(prepared_predictions, prepared_metrics)

    paired_split_deltas = _paired_split_deltas(prepared_metrics)
    bootstrap_confidence_intervals = _bootstrap_confidence_intervals(
        prepared_predictions,
        config=config,
    )
    holm_correction = _holm_correction(paired_split_deltas)
    chi_resolution_per_split, chi_resolution_summary = _chi_resolution_stability(
        prepared_predictions,
        reference_n_chi=config.reference_n_chi,
    )
    direction_consistency = _direction_consistency(paired_split_deltas)
    paired_contrasts = _paired_contrast_summary(
        paired_split_deltas,
        bootstrap_confidence_intervals,
        holm_correction,
    )
    return {
        "fingerprints": fingerprints,
        "paired_split_deltas": paired_split_deltas,
        "bootstrap_confidence_intervals": bootstrap_confidence_intervals,
        "holm_correction": holm_correction,
        "paired_contrasts": paired_contrasts,
        "chi_resolution_per_split": chi_resolution_per_split,
        "chi_resolution_summary": chi_resolution_summary,
        "direction_consistency": direction_consistency,
    }


def _validate_config(config: PolarHarmonicStatisticsConfig) -> None:
    if config.bootstrap_iterations < 100:
        raise PolarHarmonicStatisticsError("bootstrap_iterations must be at least 100.")
    if config.reference_n_chi <= 0:
        raise PolarHarmonicStatisticsError("reference_n_chi must be positive.")


def _prepare_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = _rename_aliases(
        frame,
        {
            "target_case_id": ("target_case", "case_id", "held_out_case_id"),
            "patientId": ("patient_id",),
            "mode_set": ("mode", "modes"),
            "n_chi": ("chi_bins", "n_azimuthal_bins"),
        },
    )
    required = (*_PREDICTION_KEY, "patientId", "label", "p_cancer", "threshold")
    _require_columns(renamed, required, "predictions")
    out = renamed.loc[:, _present_prediction_columns(renamed)].copy()
    out["n_chi"] = pd.to_numeric(out["n_chi"], errors="raise").astype(int)
    out["split_id"] = pd.to_numeric(out["split_id"], errors="raise").astype(int)
    out["mode_set"] = out["mode_set"].map(_canonical_mode_set)
    out["target_case_id"] = out["target_case_id"].astype(str)
    out["patientId"] = out["patientId"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    out["p_cancer"] = pd.to_numeric(out["p_cancer"], errors="raise").astype(float)
    out["threshold"] = pd.to_numeric(out["threshold"], errors="raise").astype(float)
    if not out["label"].isin((0, 1)).all():
        raise PolarHarmonicStatisticsError("predictions.label must contain only 0 and 1.")
    if not np.isfinite(out[["p_cancer", "threshold"]].to_numpy(dtype=float)).all():
        raise PolarHarmonicStatisticsError("predictions contain non-finite scores or thresholds.")
    if out.duplicated(_PREDICTION_KEY).any():
        raise PolarHarmonicStatisticsError(
            "predictions contain duplicate variant/split/target-case rows."
        )
    return out.sort_values(list(_PREDICTION_KEY)).reset_index(drop=True)


def _prepare_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = _rename_aliases(
        frame,
        {"mode_set": ("mode", "modes"), "n_chi": ("chi_bins", "n_azimuthal_bins")},
    )
    _require_columns(renamed, (*_METRIC_KEY, *PRIMARY_METRICS), "fold_metrics")
    out = renamed.loc[:, list(_METRIC_KEY) + list(PRIMARY_METRICS)].copy()
    out["n_chi"] = pd.to_numeric(out["n_chi"], errors="raise").astype(int)
    out["split_id"] = pd.to_numeric(out["split_id"], errors="raise").astype(int)
    out["mode_set"] = out["mode_set"].map(_canonical_mode_set)
    for metric in PRIMARY_METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="raise").astype(float)
    if out.duplicated(_METRIC_KEY).any():
        raise PolarHarmonicStatisticsError("fold_metrics contain duplicate variant/split rows.")
    return out.sort_values(list(_METRIC_KEY)).reset_index(drop=True)


def _present_prediction_columns(frame: pd.DataFrame) -> list[str]:
    fingerprint_columns = (
        "cohort_fingerprint",
        "fold_fingerprint",
        "held_out_case_fingerprint",
    )
    return list(_PREDICTION_KEY) + [
        "patientId",
        "label",
        "p_cancer",
        "threshold",
        *[column for column in fingerprint_columns if column in frame.columns],
    ]


def _rename_aliases(
    frame: pd.DataFrame,
    aliases: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    out = frame.copy()
    rename: dict[str, str] = {}
    for canonical, alternatives in aliases.items():
        if canonical in out.columns:
            continue
        found = [name for name in alternatives if name in out.columns]
        if len(found) == 1:
            rename[found[0]] = canonical
        elif len(found) > 1:
            raise PolarHarmonicStatisticsError(
                f"Ambiguous aliases for {canonical}: {found}."
            )
    return out.rename(columns=rename)


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise PolarHarmonicStatisticsError(f"{name} missing required columns: {missing}.")


def _canonical_mode_set(value: Any) -> str:
    compact = str(value).replace(" ", "").replace("_", "+").upper()
    aliases = {
        "A0": MODE_A0,
        "A0+A2": MODE_A0_A2,
        "A0+A2+A4": MODE_A0_A2_A4,
    }
    if compact not in aliases:
        raise PolarHarmonicStatisticsError(f"Unsupported mode_set: {value!r}.")
    return aliases[compact]


def _validate_common_folds(predictions: pd.DataFrame) -> pd.DataFrame:
    variants = predictions[["n_chi", "mode_set"]].drop_duplicates().sort_values(
        ["n_chi", "mode_set"]
    )
    if variants.empty:
        raise PolarHarmonicStatisticsError("predictions are empty.")
    reference_variant = tuple(variants.iloc[0])
    reference = predictions[
        (predictions["n_chi"] == reference_variant[0])
        & (predictions["mode_set"] == reference_variant[1])
    ]
    reference_cohort = set(reference["target_case_id"])
    reference_folds = _fold_case_sets(reference)
    reference_assignment = _fold_assignment_fingerprint(reference)
    rows: list[dict[str, Any]] = []
    for variant in variants.itertuples(index=False):
        subset = predictions[
            (predictions["n_chi"] == variant.n_chi)
            & (predictions["mode_set"] == variant.mode_set)
        ]
        cohort = set(subset["target_case_id"])
        fold_sets = _fold_case_sets(subset)
        if cohort != reference_cohort:
            raise PolarHarmonicStatisticsError(
                "Cohort fingerprint mismatch between polar variants."
            )
        if fold_sets != reference_folds:
            raise PolarHarmonicStatisticsError(
                "Held-out case sets differ between polar variants."
            )
        if _fold_assignment_fingerprint(subset) != reference_assignment:
            raise PolarHarmonicStatisticsError(
                "Fold fingerprint mismatch between polar variants."
            )
        _validate_labels_and_patients(reference, subset)
        rows.append(
            {
                "n_chi": int(variant.n_chi),
                "mode_set": variant.mode_set,
                "cohort_fingerprint": _text_fingerprint(reference_cohort),
                "fold_fingerprint": reference_assignment,
                "held_out_case_fingerprint": _held_out_sets_fingerprint(reference_folds),
                "target_cases": len(reference_cohort),
                "splits": len(reference_folds),
            }
        )
    _validate_declared_fingerprints(predictions, rows)
    return pd.DataFrame(rows)


def _fold_case_sets(frame: pd.DataFrame) -> dict[int, frozenset[str]]:
    return {
        int(split_id): frozenset(group["target_case_id"].astype(str))
        for split_id, group in frame.groupby("split_id", sort=True)
    }


def _fold_assignment_fingerprint(frame: pd.DataFrame) -> str:
    pairs = frame.loc[:, ["split_id", "target_case_id"]].drop_duplicates()
    values = [f"{row.split_id}\t{row.target_case_id}" for row in pairs.itertuples()]
    return _text_fingerprint(values)


def _held_out_sets_fingerprint(sets: dict[int, frozenset[str]]) -> str:
    values = [
        f"{split_id}\t{case_id}"
        for split_id, case_ids in sorted(sets.items())
        for case_id in sorted(case_ids)
    ]
    return _text_fingerprint(values)


def _validate_labels_and_patients(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> None:
    keys = ["split_id", "target_case_id"]
    merged = reference.merge(
        candidate,
        on=keys,
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    for column in ("patientId", "label"):
        if not (
            merged[f"{column}_reference"].to_numpy()
            == merged[f"{column}_candidate"].to_numpy()
        ).all():
            raise PolarHarmonicStatisticsError(
                f"{column} mismatch between polar variants."
            )


def _validate_declared_fingerprints(
    predictions: pd.DataFrame,
    derived_rows: list[dict[str, Any]],
) -> None:
    if "cohort_fingerprint" in predictions.columns:
        values = predictions["cohort_fingerprint"].dropna().astype(str).unique()
        if len(values) > 1:
            raise PolarHarmonicStatisticsError("Declared cohort fingerprints differ.")
    if "fold_fingerprint" in predictions.columns:
        values = predictions["fold_fingerprint"].dropna().astype(str).unique()
        if len(values) > 1:
            raise PolarHarmonicStatisticsError("Declared fold fingerprints differ.")
    if "held_out_case_fingerprint" in predictions.columns:
        for split_id, group in predictions.groupby("split_id", sort=False):
            values = group["held_out_case_fingerprint"].dropna().astype(str).unique()
            if len(values) > 1:
                raise PolarHarmonicStatisticsError(
                    f"Declared held-out case fingerprints differ for split {split_id}."
                )
    del derived_rows


def _validate_metric_grid(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    prediction_grid = set(
        map(tuple, predictions.loc[:, list(_METRIC_KEY)].drop_duplicates().to_numpy())
    )
    metric_grid = set(map(tuple, metrics.loc[:, list(_METRIC_KEY)].to_numpy()))
    if prediction_grid != metric_grid:
        raise PolarHarmonicStatisticsError(
            "fold_metrics variant/split grid does not match predictions."
        )


def _paired_split_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n_chi, group in metrics.groupby("n_chi", sort=True):
        for contrast, baseline, candidate in PRIMARY_CONTRASTS:
            baseline_rows = group[group["mode_set"] == baseline].set_index("split_id")
            candidate_rows = group[group["mode_set"] == candidate].set_index("split_id")
            if baseline_rows.empty or candidate_rows.empty:
                continue
            if set(baseline_rows.index) != set(candidate_rows.index):
                raise PolarHarmonicStatisticsError(
                    f"Split grid differs for {contrast} at n_chi={n_chi}."
                )
            for split_id in sorted(baseline_rows.index):
                row: dict[str, Any] = {
                    "n_chi": int(n_chi),
                    "contrast": contrast,
                    "baseline_mode_set": baseline,
                    "candidate_mode_set": candidate,
                    "split_id": int(split_id),
                }
                for metric in PRIMARY_METRICS:
                    row[f"baseline_{metric}"] = float(
                        baseline_rows.loc[split_id, metric]
                    )
                    row[f"candidate_{metric}"] = float(
                        candidate_rows.loc[split_id, metric]
                    )
                    row[f"delta_{metric}"] = (
                        row[f"candidate_{metric}"] - row[f"baseline_{metric}"]
                    )
                rows.append(row)
    if not rows:
        raise PolarHarmonicStatisticsError("No complete primary contrasts were found.")
    return pd.DataFrame(rows).sort_values(
        ["n_chi", "contrast", "split_id"]
    ).reset_index(drop=True)


def _bootstrap_confidence_intervals(
    predictions: pd.DataFrame,
    *,
    config: PolarHarmonicStatisticsConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, Any]] = []
    for n_chi, group in predictions.groupby("n_chi", sort=True):
        for contrast, baseline, candidate in PRIMARY_CONTRASTS:
            paired = _paired_prediction_rows(group, baseline, candidate)
            if paired is None:
                continue
            samples = _cluster_bootstrap_deltas(
                paired,
                rng=rng,
                iterations=config.bootstrap_iterations,
            )
            for metric in PRIMARY_METRICS:
                values = samples[metric]
                finite = values[np.isfinite(values)]
                if not len(finite):
                    raise PolarHarmonicStatisticsError(
                        f"Bootstrap yielded no valid {metric} samples for {contrast}."
                    )
                rows.append(
                    {
                        "n_chi": int(n_chi),
                        "contrast": contrast,
                        "metric": metric,
                        "delta_mean": float(np.mean(finite)),
                        "ci_low": float(np.quantile(finite, 0.025)),
                        "ci_high": float(np.quantile(finite, 0.975)),
                        "bootstrap_valid_samples": int(len(finite)),
                        "bootstrap_iterations": config.bootstrap_iterations,
                    }
                )
    return pd.DataFrame(rows)


def _paired_prediction_rows(
    predictions: pd.DataFrame,
    baseline: str,
    candidate: str,
) -> pd.DataFrame | None:
    keys = ["split_id", "target_case_id"]
    left = predictions[predictions["mode_set"] == baseline]
    right = predictions[predictions["mode_set"] == candidate]
    if left.empty or right.empty:
        return None
    paired = left.merge(
        right,
        on=keys,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    for column in ("patientId", "label"):
        if not (
            paired[f"{column}_baseline"].to_numpy()
            == paired[f"{column}_candidate"].to_numpy()
        ).all():
            raise PolarHarmonicStatisticsError(
                f"{column} mismatch while pairing contrast {candidate} minus {baseline}."
            )
    return paired


def _cluster_bootstrap_deltas(
    paired: pd.DataFrame,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> dict[str, np.ndarray]:
    patient_indices = {
        patient: group.index.to_numpy()
        for patient, group in paired.groupby("patientId_baseline", sort=False)
    }
    patients = np.asarray(sorted(patient_indices), dtype=object)
    if len(patients) < 2:
        raise PolarHarmonicStatisticsError("Patient-cluster bootstrap requires two patients.")
    output = {metric: np.full(iterations, np.nan, dtype=float) for metric in PRIMARY_METRICS}
    for index in range(iterations):
        selected = rng.choice(patients, size=len(patients), replace=True)
        row_indices = np.concatenate([patient_indices[patient] for patient in selected])
        sampled = paired.loc[row_indices]
        baseline = _aggregate_split_metrics(sampled, suffix="baseline")
        candidate = _aggregate_split_metrics(sampled, suffix="candidate")
        for metric in PRIMARY_METRICS:
            output[metric][index] = candidate[metric] - baseline[metric]
    return output


def _aggregate_split_metrics(sampled: pd.DataFrame, *, suffix: str) -> dict[str, float]:
    rows = []
    for _, group in sampled.groupby("split_id", sort=False):
        rows.append(
            _safe_binary_metric_values(
                group["label_baseline"].to_numpy(dtype=int),
                group[f"p_cancer_{suffix}"].to_numpy(dtype=float),
                group[f"threshold_{suffix}"].to_numpy(dtype=float),
            )
        )
    summary: dict[str, float] = {}
    for metric in PRIMARY_METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        summary[metric] = float(np.mean(finite)) if len(finite) else float("nan")
    return summary


def _safe_binary_metric_values(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    predicted = (scores >= thresholds).astype(int)
    true_positive = int(np.sum((labels == 1) & (predicted == 1)))
    false_negative = int(np.sum((labels == 1) & (predicted == 0)))
    true_negative = int(np.sum((labels == 0) & (predicted == 0)))
    false_positive = int(np.sum((labels == 0) & (predicted == 1)))
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if len(positive_scores) and len(negative_scores):
        pairwise = positive_scores[:, None] - negative_scores[None, :]
        roc_auc = float(np.mean((pairwise > 0.0) + 0.5 * (pairwise == 0.0)))
    else:
        roc_auc = float("nan")
    return {
        "sensitivity": ratio(true_positive, true_positive + false_negative),
        "specificity": ratio(true_negative, true_negative + false_positive),
        "roc_auc": roc_auc,
    }


def _holm_correction(paired_deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n_chi, group in paired_deltas.groupby("n_chi", sort=True):
        for metric in PRIMARY_METRICS:
            family = []
            for contrast, _, _ in PRIMARY_CONTRASTS:
                values = group.loc[group["contrast"] == contrast, f"delta_{metric}"].to_numpy(
                    dtype=float
                )
                family.append((contrast, _paired_wilcoxon_p_value(values)))
            adjusted = _holm_adjust([value for _, value in family])
            for (contrast, raw_p_value), adjusted_p_value in zip(family, adjusted):
                rows.append(
                    {
                        "n_chi": int(n_chi),
                        "metric": metric,
                        "contrast": contrast,
                        "raw_p_value": raw_p_value,
                        "holm_adjusted_p_value": adjusted_p_value,
                        "reject_at_0_05": bool(adjusted_p_value <= 0.05),
                    }
                )
    return pd.DataFrame(rows)


def _paired_wilcoxon_p_value(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite) or np.allclose(finite, 0.0):
        return 1.0
    return float(wilcoxon(finite, alternative="two-sided", method="auto").pvalue)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _chi_resolution_stability(
    predictions: pd.DataFrame,
    *,
    reference_n_chi: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reference_n_chi not in set(predictions["n_chi"]):
        raise PolarHarmonicStatisticsError(
            f"Reference n_chi={reference_n_chi} is absent from predictions."
        )
    reference = predictions[predictions["n_chi"] == reference_n_chi]
    rows: list[dict[str, Any]] = []
    for (n_chi, mode_set), candidate in predictions.groupby(["n_chi", "mode_set"]):
        if n_chi == reference_n_chi:
            continue
        baseline = reference[reference["mode_set"] == mode_set]
        if baseline.empty:
            raise PolarHarmonicStatisticsError(
                f"Missing n_chi={reference_n_chi} reference for {mode_set}."
            )
        paired = baseline.merge(
            candidate,
            on=["split_id", "target_case_id"],
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        for split_id, group in paired.groupby("split_id", sort=True):
            reference_scores = group["p_cancer_reference"].to_numpy(dtype=float)
            candidate_scores = group["p_cancer_candidate"].to_numpy(dtype=float)
            correlation = _pearson_correlation(reference_scores, candidate_scores)
            reference_class = reference_scores >= group["threshold_reference"].to_numpy(
                dtype=float
            )
            candidate_class = candidate_scores >= group["threshold_candidate"].to_numpy(
                dtype=float
            )
            rows.append(
                {
                    "n_chi": int(n_chi),
                    "reference_n_chi": int(reference_n_chi),
                    "mode_set": mode_set,
                    "split_id": int(split_id),
                    "p_cancer_correlation": correlation,
                    "median_absolute_p_cancer_delta": float(
                        np.median(np.abs(candidate_scores - reference_scores))
                    ),
                    "threshold_class_disagreement": float(
                        np.mean(reference_class != candidate_class)
                    ),
                    "held_out_target_cases": int(len(group)),
                }
            )
    per_split = pd.DataFrame(rows)
    if per_split.empty:
        return per_split, pd.DataFrame(
            columns=[
                "n_chi",
                "reference_n_chi",
                "mode_set",
                "splits",
                "p_cancer_correlation_mean",
                "median_absolute_p_cancer_delta_median",
                "threshold_class_disagreement_mean",
            ]
        )
    summary = (
        per_split.groupby(["n_chi", "reference_n_chi", "mode_set"], as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            p_cancer_correlation_mean=("p_cancer_correlation", "mean"),
            median_absolute_p_cancer_delta_median=(
                "median_absolute_p_cancer_delta",
                "median",
            ),
            threshold_class_disagreement_mean=("threshold_class_disagreement", "mean"),
        )
        .sort_values(["mode_set", "n_chi"])
        .reset_index(drop=True)
    )
    return per_split, summary


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _direction_consistency(paired_deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (contrast, metric), group in _long_deltas(paired_deltas).groupby(
        ["contrast", "metric"], sort=True
    ):
        per_resolution = group.groupby("n_chi", sort=True)["delta"].mean()
        directions = np.sign(per_resolution.to_numpy(dtype=float))
        nonzero = directions[directions != 0.0]
        consensus = float(np.sign(np.sum(nonzero))) if len(nonzero) else 0.0
        rows.append(
            {
                "contrast": contrast,
                "metric": metric,
                "n_chi_resolutions": int(len(per_resolution)),
                "n_chi_values": ",".join(str(value) for value in per_resolution.index),
                "positive_resolutions": int(np.sum(directions > 0.0)),
                "negative_resolutions": int(np.sum(directions < 0.0)),
                "zero_resolutions": int(np.sum(directions == 0.0)),
                "direction_consistent": bool(
                    len(nonzero) > 0 and np.all(nonzero == nonzero[0])
                ),
                "consensus_direction": int(consensus),
            }
        )
    return pd.DataFrame(rows)


def _paired_contrast_summary(
    paired_deltas: pd.DataFrame,
    confidence_intervals: pd.DataFrame,
    holm: pd.DataFrame,
) -> pd.DataFrame:
    long = _long_deltas(paired_deltas)
    summary = (
        long.groupby(["n_chi", "contrast", "metric"], as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            paired_delta_mean=("delta", "mean"),
            paired_delta_median=("delta", "median"),
        )
        .sort_values(["n_chi", "contrast", "metric"])
    )
    return (
        summary.merge(
            confidence_intervals,
            on=["n_chi", "contrast", "metric"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            holm,
            on=["n_chi", "contrast", "metric"],
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )


def _long_deltas(paired_deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in paired_deltas.itertuples(index=False):
        for metric in PRIMARY_METRICS:
            rows.append(
                {
                    "n_chi": row.n_chi,
                    "contrast": row.contrast,
                    "split_id": row.split_id,
                    "metric": metric,
                    "delta": getattr(row, f"delta_{metric}"),
                }
            )
    return pd.DataFrame(rows)


def _text_fingerprint(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
