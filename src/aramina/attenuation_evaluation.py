"""Patient-safe paired evaluation for attenuation research features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .attenuation_contract import (
    DEFAULT_ATTENUATION_EVALUATION_COLUMNS,
    AttenuationExperimentUnavailable,
    PairedAttenuationEvaluation,
)
from .model_metrics import binary_metric_values
from .model_utils import compute_binary_thresholds


def evaluate_paired_attenuation_contribution(
    cases: pd.DataFrame,
    *,
    baseline_feature_columns: Sequence[str],
    attenuation_feature_columns: Sequence[str] = DEFAULT_ATTENUATION_EVALUATION_COLUMNS,
    patient_column: str = "patientId",
    label_column: str = "label",
    case_id_column: str = "target_case_id",
    eligibility_column: str = "attenuation_evaluation_eligible",
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    target_sensitivity: float = 0.95,
) -> PairedAttenuationEvaluation:
    """Compare research models on identical patient-safe folds and cases.

    The caller supplies pre-existing baseline predictors. Both models are fresh
    research LogisticRegression fits per fold: baseline predictors alone versus
    the same predictors plus attenuation features. This never modifies or
    scores the product model.
    """
    if not baseline_feature_columns:
        raise ValueError("At least one baseline feature column is required.")
    required = [
        patient_column,
        label_column,
        case_id_column,
        eligibility_column,
        *baseline_feature_columns,
        *attenuation_feature_columns,
    ]
    missing = [column for column in required if column not in cases]
    if missing:
        raise AttenuationExperimentUnavailable(
            f"Paired attenuation evaluation is unavailable; missing columns: {missing}"
        )
    values = cases.loc[:, required].copy()
    values[patient_column] = values[patient_column].astype(str)
    values[label_column] = pd.to_numeric(values[label_column], errors="coerce")
    numeric_columns = [*baseline_feature_columns, *attenuation_feature_columns]
    values.loc[:, numeric_columns] = values.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    eligibility = pd.to_numeric(values[eligibility_column], errors="coerce").fillna(0)
    eligible_mask = (
        eligibility.eq(1)
        & values[label_column].isin([0, 1])
        & np.isfinite(values.loc[:, numeric_columns]).all(axis=1)
    )
    eligible = values.loc[eligible_mask].copy().reset_index(drop=True)
    coverage = _evaluation_coverage(
        values,
        eligible,
        patient_column=patient_column,
        label_column=label_column,
        eligibility_column=eligibility_column,
    )
    _validate_evaluation_cases(
        eligible,
        patient_column=patient_column,
        label_column=label_column,
        case_id_column=case_id_column,
        n_splits=n_splits,
    )
    patient_labels = (
        eligible.groupby(patient_column, as_index=False)[label_column]
        .max()
        .sort_values(patient_column)
        .reset_index(drop=True)
    )
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    split_metrics = []
    predictions = []
    paired_deltas = []
    for split_id, (train_patient_index, test_patient_index) in enumerate(
        splitter.split(patient_labels, patient_labels[label_column].to_numpy(dtype=int))
    ):
        train_patients = set(
            patient_labels.iloc[train_patient_index][patient_column].astype(str)
        )
        test_patients = set(
            patient_labels.iloc[test_patient_index][patient_column].astype(str)
        )
        if train_patients.intersection(test_patients):
            raise RuntimeError("Patient leakage detected in attenuation evaluation split.")
        train = eligible[eligible[patient_column].isin(train_patients)].copy()
        test = eligible[eligible[patient_column].isin(test_patients)].copy()
        model_outputs: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        for model_name, feature_columns in (
            ("baseline", tuple(baseline_feature_columns)),
            (
                "baseline_plus_attenuation",
                (*baseline_feature_columns, *attenuation_feature_columns),
            ),
        ):
            model = _research_logistic(random_state + split_id)
            model.fit(
                train.loc[:, feature_columns].to_numpy(dtype=float),
                train[label_column].to_numpy(dtype=int),
            )
            train_score = model.predict_proba(
                train.loc[:, feature_columns].to_numpy(dtype=float)
            )[:, 1]
            test_score = model.predict_proba(
                test.loc[:, feature_columns].to_numpy(dtype=float)
            )[:, 1]
            thresholds = compute_binary_thresholds(
                train[label_column].to_numpy(dtype=int),
                train_score,
                target_sensitivity=target_sensitivity,
            )
            threshold = np.full(len(test), thresholds["threshold_target"], dtype=float)
            metric_row = _paired_metric_row(
                model_name,
                split_id,
                train,
                test,
                test_score,
                threshold,
                thresholds,
                patient_column=patient_column,
                label_column=label_column,
            )
            split_metrics.append(metric_row)
            predictions.append(
                _paired_prediction_frame(
                    test,
                    test_score,
                    threshold,
                    model_name=model_name,
                    split_id=split_id,
                    patient_column=patient_column,
                    label_column=label_column,
                    case_id_column=case_id_column,
                )
            )
            model_outputs[model_name] = (test_score, metric_row)
        paired_deltas.append(
            _paired_delta_row(
                split_id,
                model_outputs["baseline"][1],
                model_outputs["baseline_plus_attenuation"][1],
            )
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    _assert_identical_paired_cases(prediction_frame, case_id_column=case_id_column)
    return PairedAttenuationEvaluation(
        eligible_cases=eligible,
        coverage=coverage,
        split_metrics=pd.DataFrame(split_metrics),
        predictions=prediction_frame,
        paired_deltas=pd.DataFrame(paired_deltas),
    )


def _evaluation_coverage(
    values: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    patient_column: str,
    label_column: str,
    eligibility_column: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_cases": int(len(values)),
                "input_patients": int(values[patient_column].nunique()),
                "flagged_attenuation_eligible_cases": int(
                    pd.to_numeric(values[eligibility_column], errors="coerce")
                    .fillna(0)
                    .eq(1)
                    .sum()
                ),
                "paired_evaluation_cases": int(len(eligible)),
                "paired_evaluation_patients": int(eligible[patient_column].nunique()),
                "excluded_cases": int(len(values) - len(eligible)),
                "paired_benign_cases": int((eligible[label_column] == 0).sum()),
                "paired_cancer_cases": int((eligible[label_column] == 1).sum()),
            }
        ]
    )


def _validate_evaluation_cases(
    eligible: pd.DataFrame,
    *,
    patient_column: str,
    label_column: str,
    case_id_column: str,
    n_splits: int,
) -> None:
    if eligible.empty:
        raise AttenuationExperimentUnavailable(
            "Paired attenuation evaluation is unavailable: no complete validated cases."
        )
    if eligible[case_id_column].duplicated().any():
        duplicate_cases = eligible.loc[
            eligible[case_id_column].duplicated(), case_id_column
        ].astype(str).tolist()
        raise AttenuationExperimentUnavailable(
            f"Paired attenuation evaluation has duplicate target cases: {duplicate_cases}"
        )
    patient_labels = eligible.groupby(patient_column)[label_column].max()
    if patient_labels.nunique() != 2:
        raise AttenuationExperimentUnavailable(
            "Paired attenuation evaluation requires BENIGN and CANCER patients."
        )
    minimum_class_patients = int(patient_labels.value_counts().min())
    if minimum_class_patients < n_splits:
        raise AttenuationExperimentUnavailable(
            "Paired attenuation evaluation requires at least n_splits patients "
            "in each class after complete-case eligibility filtering."
        )


def _research_logistic(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=int(random_state),
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _paired_metric_row(
    model_name: str,
    split_id: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    score: np.ndarray,
    threshold: np.ndarray,
    thresholds: dict[str, Any],
    *,
    patient_column: str,
    label_column: str,
) -> dict[str, Any]:
    labels = test[label_column].to_numpy(dtype=int)
    values = binary_metric_values(labels, score, threshold)
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "model_name": model_name,
        "split_id": int(split_id),
        "roc_auc": values["roc_auc"],
        "sensitivity": values["sensitivity"],
        "specificity": values["specificity"],
        "balanced_accuracy": values["balanced_accuracy"],
        "ppv": values["ppv"],
        "npv": values["npv"],
        "threshold": float(thresholds["threshold_target"]),
        "target_sensitivity": float(thresholds["target_sensitivity"]),
        "target_reached": bool(thresholds["target_reached"]),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "train_patients": int(train[patient_column].nunique()),
        "test_patients": int(test[patient_column].nunique()),
        "train_cases": int(len(train)),
        "test_cases": int(len(test)),
    }


def _paired_prediction_frame(
    test: pd.DataFrame,
    score: np.ndarray,
    threshold: np.ndarray,
    *,
    model_name: str,
    split_id: int,
    patient_column: str,
    label_column: str,
    case_id_column: str,
) -> pd.DataFrame:
    out = test[[case_id_column, patient_column, label_column]].copy()
    out["model_name"] = model_name
    out["split_id"] = int(split_id)
    out["p_cancer"] = np.asarray(score, dtype=float)
    out["threshold"] = np.asarray(threshold, dtype=float)
    out["y_pred"] = (out["p_cancer"] >= out["threshold"]).astype(int)
    return out


def _paired_delta_row(
    split_id: int,
    baseline: dict[str, Any],
    augmented: dict[str, Any],
) -> dict[str, Any]:
    metric_names = (
        "roc_auc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "ppv",
        "npv",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    )
    return {
        "split_id": int(split_id),
        **{
            f"delta_{metric}": float(augmented[metric] - baseline[metric])
            for metric in metric_names
        },
    }


def _assert_identical_paired_cases(
    predictions: pd.DataFrame,
    *,
    case_id_column: str,
) -> None:
    for split_id, split in predictions.groupby("split_id", sort=True):
        grouped = split.groupby("model_name")[case_id_column].agg(lambda values: set(values))
        case_sets = list(grouped)
        if len(case_sets) != 2 or case_sets[0] != case_sets[1]:
            raise RuntimeError(
                f"Paired comparison mismatch: split {split_id} does not use identical cases."
            )
