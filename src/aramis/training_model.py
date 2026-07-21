"""Fit LR1 and final M2Q estimators from patient feature tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from .m2q_model import (
    GatedSymmetryLogistic,
    build_profile_logistic as _profile_logistic,
)
from .model_utils import compute_binary_thresholds, profile_matrix
from .model_schema import m2q_model_input_columns
from .patient_features import (
    lr1_training_rows as _lr1_training_rows,
    patient_feature_table as _patient_feature_table,
    row_labels as _row_labels,
    score_lr1_rows as _score_lr1_rows,
)


def _fit_patient_model_input(
    df: pd.DataFrame,
    *,
    profile_column: str,
    label_column: str,
    group_column: str,
    specimen_column: str,
    side_column: str,
    q_column: str,
    age_column: str,
    biopsy_column: str,
    lr1_row_policy: str,
    lr1_logreg_c: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lr1_rows = _lr1_training_rows(
        df,
        label_column=label_column,
        biopsy_column=biopsy_column,
        lr1_row_policy=lr1_row_policy,
    )
    lr1_model = _profile_logistic(logreg_c=lr1_logreg_c, random_state=random_state)
    lr1_model.fit(
        profile_matrix(lr1_rows, profile_column), _row_labels(lr1_rows, label_column)
    )
    scored_lr1 = _score_lr1_rows(
        lr1_model,
        lr1_rows,
        full_df=df,
        profile_column=profile_column,
        group_column=group_column,
        side_column=side_column,
        label_column=label_column,
        biopsy_column=biopsy_column,
    )
    feature_table = _patient_feature_table(
        df,
        scored_lr1,
        profile_column=profile_column,
        label_column=label_column,
        group_column=group_column,
        specimen_column=specimen_column,
        side_column=side_column,
        q_column=q_column,
        age_column=age_column,
        biopsy_column=biopsy_column,
    )
    return feature_table, lr1_rows


def _fit_m2q_model(
    feature_table: pd.DataFrame,
    lr1_rows: pd.DataFrame,
    *,
    profile_column: str,
    label_column: str,
    lr1_logreg_c: float,
    lr2_logreg_c: float,
    random_state: int,
    target_sensitivity: float,
) -> dict[str, Any]:
    """Fit the fixed two-layer M2Q model on all accepted target cases."""
    lr1_model = _profile_logistic(logreg_c=lr1_logreg_c, random_state=random_state)
    lr1_model.fit(
        profile_matrix(lr1_rows, profile_column), _row_labels(lr1_rows, label_column)
    )
    y = feature_table["label"].to_numpy(dtype=int)
    final_model = GatedSymmetryLogistic(
        logreg_c=lr2_logreg_c,
        random_state=random_state,
    ).fit(feature_table, y)
    score = final_model.predict_proba(feature_table)[:, 1]
    thresholds = compute_binary_thresholds(
        y,
        score,
        target_sensitivity=target_sensitivity,
    )
    final_score_reference = _score_reference_distribution(
        score,
        y,
        score="final_prediction.p_cancer",
    )
    return {
        "name": "Aramis T100 profile, optional SK symmetry refinement, and age",
        "lr1_model": lr1_model,
        "final_model": final_model,
        "feature_columns": m2q_model_input_columns(),
        "symmetry_policy": "single_model_gated_optional_refinement",
        "symmetry_gate": "symmetry_available",
        "thresholds": thresholds,
        "final_fit_training_metrics": _final_fit_training_metrics(
            y,
            score,
            threshold=float(thresholds["threshold_target"]),
        ),
        "prediction_reference_scores": {
            "contract": "aramis_prediction_score_percentiles_v0_2",
            "final_prediction": final_score_reference,
        },
        "tissue_risk_assessment": {
            "contract": "aramis_tra_v0_1",
            "reference_score": "final_prediction.p_cancer",
            "reference_population": "train_on_all target-breast cases",
            "levels": [
                {
                    "level": "TRA 1",
                    "minimum_percentile": 0.0,
                    "maximum_percentile": 20.0,
                },
                {
                    "level": "TRA 2",
                    "minimum_percentile": 20.0,
                    "maximum_percentile": 50.0,
                },
                {
                    "level": "TRA 3",
                    "minimum_percentile": 50.0,
                    "maximum_percentile": 80.0,
                },
                {
                    "level": "TRA 4",
                    "minimum_percentile": 80.0,
                    "maximum_percentile": 90.0,
                },
                {
                    "level": "TRA 5",
                    "minimum_percentile": 90.0,
                    "maximum_percentile": 100.0,
                },
            ],
        },
    }


def _sorted_scores(values: np.ndarray) -> list[float]:
    """Return finite fitted probabilities for descriptive report quantiles."""
    return sorted(float(value) for value in values if np.isfinite(value))


def _score_reference_distribution(
    score_values: np.ndarray,
    labels: np.ndarray,
    *,
    score: str,
) -> dict[str, Any]:
    """Freeze one score-specific target-case reference distribution."""
    return {
        "score": score,
        "population": "train_on_all target-breast cases",
        "all_target_cases": _sorted_scores(score_values),
        "benign_target_cases": _sorted_scores(score_values[labels == 0]),
        "cancer_target_cases": _sorted_scores(score_values[labels == 1]),
    }


def _final_fit_training_metrics(
    y: np.ndarray,
    score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Describe train-on-all fit without representing it as validation."""
    decision_thresholds = np.full(len(y), threshold, dtype=float)
    pred = (score >= decision_thresholds).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "evaluation_status": "in_sample_not_independent",
        "target_cases": int(len(y)),
        "cancer_target_cases": int((y == 1).sum()),
        "benign_target_cases": int((y == 0).sum()),
        "decision_threshold": threshold,
        **_binary_metrics(y, score, decision_thresholds),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def _binary_metrics(
    y: np.ndarray,
    score: np.ndarray,
    threshold: np.ndarray,
) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": _mean_finite([sensitivity, specificity]),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "brier_score": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _mean_finite(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")
