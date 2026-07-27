"""Shared binary-classification metrics for evaluation and final model artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def binary_metric_values(
    y: np.ndarray,
    score: np.ndarray,
    threshold: np.ndarray,
) -> dict[str, float]:
    """Calculate thresholded and threshold-independent binary metrics."""
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    calibration_intercept, calibration_slope = calibration_parameters(y, score)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": mean_finite([sensitivity, specificity]),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "brier_score": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def final_fit_training_metrics(
    y: np.ndarray,
    score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Describe fit-to-training-cohort performance without implying validation."""
    decision_thresholds = np.full(len(y), threshold, dtype=float)
    pred = (score >= decision_thresholds).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "evaluation_status": "in_sample_not_independent",
        "target_cases": int(len(y)),
        "cancer_target_cases": int((y == 1).sum()),
        "benign_target_cases": int((y == 0).sum()),
        "decision_threshold": threshold,
        **binary_metric_values(y, score, decision_thresholds),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def ratio(numerator: int, denominator: int) -> float:
    """Return a finite ratio when its denominator is non-zero."""
    return float(numerator / denominator) if denominator else float("nan")


def mean_finite(values: Sequence[float]) -> float:
    """Mean finite scalar values, or NaN when none are available."""
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def calibration_parameters(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Fit logistic calibration intercept and slope for a fixed score vector."""
    clipped = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if np.unique(y).size != 2:
        return float("nan"), float("nan")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    calibrator.fit(logits, y)
    return float(calibrator.intercept_[0]), float(calibrator.coef_[0, 0])
