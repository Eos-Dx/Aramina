"""Small shared utilities for the product model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


LABEL_MAP = {"BENIGN": 0, "CANCER": 1}


def compute_binary_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_sensitivity: float = 0.95,
) -> dict[str, Any]:
    """Compute Youden and target-sensitivity thresholds on training scores."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_idx = int(np.argmax(tpr - fpr))
    target_mask = tpr >= float(target_sensitivity)
    if bool(np.any(target_mask)):
        candidates = np.where(target_mask)[0]
        target_idx = int(candidates[np.argmin(fpr[candidates])])
        target_reached = True
    else:
        target_idx = youden_idx
        target_reached = False
    return {
        "threshold_youden": float(thresholds[youden_idx]),
        "threshold_target": float(thresholds[target_idx]),
        "target_sensitivity": float(target_sensitivity),
        "target_reached": bool(target_reached),
    }


def profile_matrix(df: pd.DataFrame, profile_column: str) -> np.ndarray:
    """Stack one profile-array column into a finite 2D model matrix."""
    if profile_column not in df.columns:
        raise KeyError(f"Missing required columns: {[profile_column]}")
    profiles = [np.asarray(value, dtype=float) for value in df[profile_column]]
    lengths = {profile.size for profile in profiles}
    if len(lengths) != 1:
        raise ValueError(f"Profiles in {profile_column!r} must have equal length.")
    matrix = np.vstack(profiles)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Profiles in {profile_column!r} contain non-finite values.")
    return matrix
