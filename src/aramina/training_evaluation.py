"""Patient-safe repeated-stratified evaluation and metric summaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from .m2q_model import (
    GatedSymmetryLogistic,
    build_profile_logistic as _profile_logistic,
)
from .model_utils import compute_binary_thresholds, profile_matrix
from .model_metrics import binary_metric_values as _binary_metric_values
from .patient_features import (
    TARGET_CASE_ID,
    empty_lr1_scores as _empty_lr1_scores,
    lr1_training_rows as _lr1_training_rows,
    patient_feature_table as _patient_feature_table,
    row_labels as _row_labels,
    score_lr1_rows as _score_lr1_rows,
)
from .training_config import PRODUCT_MODEL_NAME
from .training_model import _fit_patient_model_input


def _fit_split_feature_tables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
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
    train_features, train_lr1_rows = _fit_patient_model_input(
        train_df,
        profile_column=profile_column,
        label_column=label_column,
        group_column=group_column,
        specimen_column=specimen_column,
        side_column=side_column,
        q_column=q_column,
        age_column=age_column,
        biopsy_column=biopsy_column,
        lr1_row_policy=lr1_row_policy,
        lr1_logreg_c=lr1_logreg_c,
        random_state=random_state,
    )
    lr1_model = _profile_logistic(
        logreg_c=lr1_logreg_c,
        random_state=random_state,
    )
    lr1_model.fit(
        profile_matrix(train_lr1_rows, profile_column),
        _row_labels(train_lr1_rows, label_column),
    )
    test_lr1_rows = _lr1_training_rows(
        test_df,
        label_column=label_column,
        biopsy_column=biopsy_column,
        lr1_row_policy=lr1_row_policy,
        require_two_classes=False,
    )
    test_features = _patient_feature_table(
        test_df,
        _score_lr1_rows(
            lr1_model,
            test_lr1_rows,
            full_df=test_df,
            profile_column=profile_column,
            group_column=group_column,
            side_column=side_column,
            label_column=label_column,
            biopsy_column=biopsy_column,
        ),
        profile_column=profile_column,
        label_column=label_column,
        group_column=group_column,
        specimen_column=specimen_column,
        side_column=side_column,
        q_column=q_column,
        age_column=age_column,
        biopsy_column=biopsy_column,
        require_two_classes=False,
    )
    return train_features, test_features


def _evaluate_m2q_model(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
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
    lr2_logreg_c: float,
    random_state: int,
    target_sensitivity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_config = config.get("evaluation", {})
    mode = _evaluation_mode(evaluation_config)
    n_splits = int(evaluation_config["n_splits"])
    n_repeats = int(evaluation_config["n_repeats"])
    base_features = _patient_feature_table(
        df,
        _empty_lr1_scores(
            df,
            group_column=group_column,
            side_column=side_column,
            label_column=label_column,
            biopsy_column=biopsy_column,
        ),
        profile_column=profile_column,
        label_column=label_column,
        group_column=group_column,
        specimen_column=specimen_column,
        side_column=side_column,
        q_column=q_column,
        age_column=age_column,
        biopsy_column=biopsy_column,
    )
    metrics = []
    predictions = []
    y_patients = base_features["label"].to_numpy(dtype=int)
    split_pairs = _patient_split_pairs(
        mode=mode,
        base_features=base_features,
        y_patients=y_patients,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    for split_id, (train_idx, test_idx) in enumerate(split_pairs):
        train_patients = set(base_features.iloc[train_idx]["patientId"].astype(str))
        test_patients = set(base_features.iloc[test_idx]["patientId"].astype(str))
        train_df = df[df[group_column].astype(str).isin(train_patients)].copy()
        test_df = df[df[group_column].astype(str).isin(test_patients)].copy()
        if set(train_df[group_column].astype(str)).intersection(
            set(test_df[group_column].astype(str))
        ):
            raise RuntimeError("Patient leakage detected in training split.")
        train_features, test_features = _fit_split_feature_tables(
            train_df,
            test_df,
            profile_column=profile_column,
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            side_column=side_column,
            q_column=q_column,
            age_column=age_column,
            biopsy_column=biopsy_column,
            lr1_row_policy=lr1_row_policy,
            lr1_logreg_c=lr1_logreg_c,
            random_state=random_state + split_id,
        )
        final_model = GatedSymmetryLogistic(
            logreg_c=lr2_logreg_c,
            random_state=random_state + split_id,
        ).fit(train_features, train_features["label"].to_numpy(dtype=int))
        train_score = final_model.predict_proba(train_features)[:, 1]
        test_score = final_model.predict_proba(test_features)[:, 1]
        thresholds = compute_binary_thresholds(
            train_features["label"].to_numpy(dtype=int),
            train_score,
            target_sensitivity=target_sensitivity,
        )
        thresholds["selected_lr1_c"] = lr1_logreg_c
        thresholds["selected_lr2_c"] = lr2_logreg_c
        test_thresholds = np.full(
            len(test_features),
            thresholds["threshold_target"],
            dtype=float,
        )
        metrics.append(
            _patient_metric_row(
                PRODUCT_MODEL_NAME,
                split_id,
                train_features,
                test_features,
                test_score,
                thresholds,
                test_thresholds,
                evaluation_mode=mode,
            )
        )
        predictions.append(
            _patient_prediction_frame(
                PRODUCT_MODEL_NAME,
                split_id,
                test_features,
                test_score,
                thresholds,
                _default_routes(test_features),
                test_thresholds,
                evaluation_mode=mode,
            )
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    return pd.DataFrame(metrics), prediction_frame


def _evaluation_mode(evaluation_config: dict[str, Any]) -> str:
    if evaluation_config.get("mode") != "stratified_kfold":
        raise ValueError("The product evaluator supports only stratified_kfold.")
    return "stratified_kfold"


def _patient_split_pairs(
    *,
    mode: str,
    base_features: pd.DataFrame,
    y_patients: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    patient_table = (
        base_features.groupby("patientId", as_index=False)["label"]
        .max()
        .assign(patientId=lambda frame: frame["patientId"].astype(str))
    )
    patient_labels = patient_table["label"].to_numpy(dtype=int)

    def case_indices(patient_index: np.ndarray) -> np.ndarray:
        patient_ids = set(patient_table.iloc[patient_index]["patientId"])
        return base_features.index[
            base_features["patientId"].astype(str).isin(patient_ids)
        ].to_numpy()

    if mode != "stratified_kfold":
        raise ValueError(f"Unsupported product split mode: {mode!r}")
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return [
        (case_indices(train_index), case_indices(test_index))
        for train_index, test_index in splitter.split(patient_table, patient_labels)
    ]


def _default_routes(values: Any) -> np.ndarray:
    return np.full(len(values), "default", dtype=object)


def _patient_metric_row(
    model_name: str,
    split_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score: np.ndarray,
    thresholds: dict[str, Any],
    decision_thresholds: np.ndarray,
    *,
    evaluation_mode: str,
) -> dict[str, Any]:
    y = test_df["label"].to_numpy(dtype=int)
    values = _binary_metric_values(y, score, decision_thresholds)
    pred = (score >= decision_thresholds).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "model_name": model_name,
        "split_id": int(split_id),
        "evaluation_mode": evaluation_mode,
        "roc_auc": values["roc_auc"],
        "pr_auc": values["pr_auc"],
        "brier_score": values["brier_score"],
        "log_loss": values["log_loss"],
        "calibration_intercept": values["calibration_intercept"],
        "calibration_slope": values["calibration_slope"],
        "sensitivity_target": values["sensitivity"],
        "specificity_target": values["specificity"],
        "balanced_accuracy_target": values["balanced_accuracy"],
        "ppv_target": values["ppv"],
        "npv_target": values["npv"],
        "tp_target": tp,
        "tn_target": tn,
        "fp_target": fp,
        "fn_target": fn,
        "train_patients": int(train_df["patientId"].nunique()),
        "test_patients": int(test_df["patientId"].nunique()),
        "train_target_cases": int(len(train_df)),
        "test_target_cases": int(len(test_df)),
        "train_cancer_target_cases": int((train_df["label"] == 1).sum()),
        "test_cancer_target_cases": int((test_df["label"] == 1).sum()),
        **thresholds,
    }


def _patient_prediction_frame(
    model_name: str,
    split_id: int,
    test_df: pd.DataFrame,
    score: np.ndarray,
    thresholds: dict[str, Any],
    routes: np.ndarray,
    decision_thresholds: np.ndarray,
    *,
    evaluation_mode: str,
) -> pd.DataFrame:
    out = test_df[[TARGET_CASE_ID, "patientId", "label", "label_name"]].copy()
    out["model_name"] = model_name
    out["split_id"] = int(split_id)
    out["evaluation_mode"] = evaluation_mode
    out["p_cancer"] = np.asarray(score, dtype=float)
    out["model_route"] = np.asarray(routes, dtype=str)
    out["threshold_youden"] = float(thresholds["threshold_youden"])
    out["threshold_target"] = np.asarray(decision_thresholds, dtype=float)
    out["y_pred_target"] = (out["p_cancer"] >= out["threshold_target"]).astype(int)
    return out


def _summarize_patient_model_metrics(
    split_metrics: pd.DataFrame,
    split_predictions: pd.DataFrame,
    *,
    random_state: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    for model_name, group in split_metrics.groupby("model_name", sort=False):
        evaluation_modes = sorted(
            group["evaluation_mode"].dropna().astype(str).unique()
        )
        rows.append(
            {
                "model_name": model_name,
                "evaluation_mode": evaluation_modes[0]
                if len(evaluation_modes) == 1
                else ",".join(evaluation_modes),
                "evidence_status": "patient_safe_validation",
                "splits": int(len(group)),
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_std": float(group["roc_auc"].std(ddof=0)),
                "pr_auc_mean": float(group["pr_auc"].mean()),
                "pr_auc_std": float(group["pr_auc"].std(ddof=0)),
                "brier_score_mean": float(group["brier_score"].mean()),
                "brier_score_std": float(group["brier_score"].std(ddof=0)),
                "log_loss_mean": float(group["log_loss"].mean()),
                "log_loss_std": float(group["log_loss"].std(ddof=0)),
                "calibration_intercept_mean": float(
                    group["calibration_intercept"].mean()
                ),
                "calibration_slope_mean": float(group["calibration_slope"].mean()),
                "sensitivity_target_mean": float(group["sensitivity_target"].mean()),
                "sensitivity_target_std": float(
                    group["sensitivity_target"].std(ddof=0)
                ),
                "specificity_target_mean": float(group["specificity_target"].mean()),
                "specificity_target_std": float(
                    group["specificity_target"].std(ddof=0)
                ),
                "balanced_accuracy_target_mean": float(
                    group["balanced_accuracy_target"].mean()
                ),
                "ppv_target_mean": float(group["ppv_target"].mean()),
                "npv_target_mean": float(group["npv_target"].mean()),
                "true_positives_mean": float(group["tp_target"].mean()),
                "true_negatives_mean": float(group["tn_target"].mean()),
                "false_negatives_mean": float(group["fn_target"].mean()),
                "false_positives_mean": float(group["fp_target"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    intervals = _patient_bootstrap_intervals(
        split_predictions,
        random_state=random_state,
        bootstrap_samples=bootstrap_samples,
    )
    return summary.merge(
        intervals,
        on=["model_name"],
        how="left",
    )


def _patient_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    random_state: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_state)
    for model_name, group in predictions.groupby("model_name", sort=False):
        cases = (
            group.groupby(TARGET_CASE_ID, as_index=False)
            .agg(
                patientId=("patientId", "first"),
                label=("label", "first"),
                p_cancer=("p_cancer", "mean"),
                threshold_target=("threshold_target", "mean"),
            )
            .reset_index(drop=True)
        )
        y = cases["label"].to_numpy(dtype=int)
        score = cases["p_cancer"].to_numpy(dtype=float)
        threshold = cases["threshold_target"].to_numpy(dtype=float)
        point = _binary_metric_values(y, score, threshold)
        sampled = {name: [] for name in point}
        for _ in range(max(0, bootstrap_samples)):
            patient_ids = cases["patientId"].drop_duplicates().to_numpy()
            sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
            sample = pd.concat(
                [
                    cases.loc[cases["patientId"] == patient_id]
                    for patient_id in sampled_ids
                ],
                ignore_index=True,
            )
            sample_y = sample["label"].to_numpy(dtype=int)
            if np.unique(sample_y).size != 2:
                continue
            values = _binary_metric_values(
                sample_y,
                sample["p_cancer"].to_numpy(dtype=float),
                sample["threshold_target"].to_numpy(dtype=float),
            )
            for name, value in values.items():
                if np.isfinite(value):
                    sampled[name].append(value)
        row: dict[str, Any] = {
            "model_name": model_name,
            "pooled_patients": int(cases["patientId"].nunique()),
            "pooled_target_cases": int(len(cases)),
        }
        for name, value in point.items():
            values = sampled[name]
            row[f"{name}_pooled"] = value
            row[f"{name}_ci_low"] = (
                float(np.quantile(values, 0.025)) if values else float("nan")
            )
            row[f"{name}_ci_high"] = (
                float(np.quantile(values, 0.975)) if values else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _patient_dataset_summary(
    df: pd.DataFrame,
    feature_table: pd.DataFrame,
    lr1_rows: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "measurements": int(len(df)),
                "patients": int(df["patientId"].astype(str).nunique()),
                "specimens": int(df["specimenId"].astype(str).nunique()),
                "lr1_measurements": int(len(lr1_rows)),
                "lr1_patients": int(lr1_rows["patientId"].astype(str).nunique()),
                "final_patients": int(feature_table["patientId"].astype(str).nunique()),
                "final_target_cases": int(len(feature_table)),
                "final_cancer_target_cases": int((feature_table["label"] == 1).sum()),
                "final_benign_target_cases": int((feature_table["label"] == 0).sum()),
            }
        ]
    )


def _require_training_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing training columns: {missing}")
