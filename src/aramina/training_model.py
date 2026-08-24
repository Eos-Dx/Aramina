"""Fit profile and final target-breast estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .target_breast_model import (
    GatedSymmetryLogistic,
    PROFILE_FPCA_COMPONENTS,
    PROFILE_INTEGRATION_NPT,
    build_profile_logistic as _profile_logistic,
)
from .model_utils import compute_binary_thresholds, profile_matrix
from .model_metrics import final_fit_training_metrics as _final_fit_training_metrics
from .model_schema import target_breast_model_input_columns
from .symmetry_features import SK_FEATURE_CONTRACT_V0_2
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


def _fit_target_breast_model(
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
    """Fit the fixed two-stage model on all accepted target cases."""
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
        "name": "Aramina 256-bin FPCA30 profile, optional SK symmetry refinement, and age",
        "profile_encoder": {
            "type": "discrete_fpca",
            "input_q_bins": PROFILE_INTEGRATION_NPT,
            "components": PROFILE_FPCA_COMPONENTS,
            "output_dimensions": PROFILE_FPCA_COMPONENTS,
            "fit_scope": "fold_local_during_evaluation_train_all_for_final_fit",
        },
        "lr1_model": lr1_model,
        "final_model": final_model,
        "feature_columns": target_breast_model_input_columns(),
        "symmetry_policy": (
            "single_model_gated_optional_refinement_requires_2_valid_measurements_"
            "per_breast_and_finite_core4"
        ),
        "symmetry_gate": "symmetry_available",
        "symmetry_feature_contract": SK_FEATURE_CONTRACT_V0_2,
        "thresholds": thresholds,
        "final_fit_training_metrics": _final_fit_training_metrics(
            y,
            score,
            threshold=float(thresholds["threshold_target"]),
        ),
        "prediction_reference_scores": {
            "contract": "aramina_prediction_score_percentiles_v0_2",
            "final_prediction": final_score_reference,
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
        "population": "train_on_all_target-breast_cases",
        "all_target_cases": _sorted_scores(score_values),
        "benign_target_cases": _sorted_scores(score_values[labels == 0]),
        "cancer_target_cases": _sorted_scores(score_values[labels == 1]),
    }
