"""Screen Random Forest as LR1 while retaining the product LR2 unchanged.

Research-only experiment. The frozen Aramis architecture is preserved:
measurement-level profile estimator -> target-breast aggregation -> fixed
patient-level logistic refinement using age and optional SK Core4 symmetry.
Only the measurement-level profile estimator is varied between the released
logistic regression and constrained Random Forest candidates.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from xrd_preprocessing import load_preprocessing_dataframe

from aramis.m2q_model import GatedSymmetryLogistic, build_profile_logistic
from aramis.model_metrics import binary_metric_values, final_fit_training_metrics
from aramis.model_utils import compute_binary_thresholds, profile_matrix
from aramis.patient_features import (
    empty_lr1_scores,
    lr1_training_rows,
    patient_feature_table,
    row_labels,
    score_lr1_rows,
)
from aramis.training_evaluation import _patient_split_pairs


PROFILE_COLUMN = "radial_profile_data"
LABEL_COLUMN = "product_status_group"
GROUP_COLUMN = "patientId"
SPECIMEN_COLUMN = "specimenId"
SIDE_COLUMN = "side"
Q_COLUMN = "q_range"
AGE_COLUMN = "age"
BIOPSY_COLUMN = "biopsy"
LR1_C = 0.1
LR2_C = 0.3
TARGET_SENSITIVITY = 0.95
FOLDS = 5
REPEATS = 20
RANDOM_SEED = 42
N_ESTIMATORS = 200
LEAF_SIZES = (1, 2, 4, 8)
LR1_ROW_POLICY = "biopsy_only"


class ProbabilityEstimator(Protocol):
    """Minimal common scoring interface for LR1 candidates."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return binary probabilities for profile rows."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataframe-joblib", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    args = parser.parse_args()

    dataframe_path = args.dataframe_joblib.resolve()
    df = load_preprocessing_dataframe(dataframe_path)
    output = args.output_folder
    output.mkdir(parents=True, exist_ok=True)

    base_features = _base_feature_table(df)
    splits = _patient_split_pairs(
        mode="stratified_kfold",
        base_features=base_features,
        y_patients=base_features["label"].to_numpy(dtype=int),
        n_splits=FOLDS,
        n_repeats=REPEATS,
        random_state=RANDOM_SEED,
    )
    fold_metrics, importances = _evaluate(df, base_features, splits)
    metrics = pd.DataFrame(fold_metrics)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(importances).to_csv(output / "lr1_feature_importances.csv", index=False)

    final_fit = _fit_all(df)
    final_fit.to_csv(output / "train_all_metrics.csv", index=False)
    summary = _summary(df, base_features, metrics, final_fit, dataframe_path)
    (output / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )


def _base_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create label-only target cases for patient-safe outer folds."""
    return patient_feature_table(
        df,
        empty_lr1_scores(
            df,
            group_column=GROUP_COLUMN,
            side_column=SIDE_COLUMN,
            label_column=LABEL_COLUMN,
            biopsy_column=BIOPSY_COLUMN,
        ),
        profile_column=PROFILE_COLUMN,
        label_column=LABEL_COLUMN,
        group_column=GROUP_COLUMN,
        specimen_column=SPECIMEN_COLUMN,
        side_column=SIDE_COLUMN,
        q_column=Q_COLUMN,
        age_column=AGE_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
    )


def _evaluate(
    df: pd.DataFrame,
    base_features: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    for split_id, (train_idx, test_idx) in enumerate(splits):
        if split_id % 10 == 0:
            print(f"Outer split {split_id + 1}/{len(splits)}", flush=True)
        train_patients = set(base_features.iloc[train_idx][GROUP_COLUMN].astype(str))
        test_patients = set(base_features.iloc[test_idx][GROUP_COLUMN].astype(str))
        if train_patients.intersection(test_patients):
            raise RuntimeError("Patient leakage detected in outer split.")
        train_df = df[df[GROUP_COLUMN].astype(str).isin(train_patients)].copy()
        test_df = df[df[GROUP_COLUMN].astype(str).isin(test_patients)].copy()

        train_rows = _lr1_rows(train_df)
        test_rows = _lr1_rows(test_df, require_two_classes=False)
        candidates: list[tuple[str, ProbabilityEstimator]] = [
            (
                "profile_logistic_lr1",
                build_profile_logistic(
                    logreg_c=LR1_C, random_state=RANDOM_SEED + split_id
                ).fit(
                    profile_matrix(train_rows, PROFILE_COLUMN),
                    row_labels(train_rows, LABEL_COLUMN),
                ),
            )
        ]
        for leaf_size in LEAF_SIZES:
            candidates.append(
                (
                    f"profile_random_forest_leaf_{leaf_size}",
                    _random_forest(leaf_size, RANDOM_SEED + split_id).fit(
                        profile_matrix(train_rows, PROFILE_COLUMN),
                        row_labels(train_rows, LABEL_COLUMN),
                    ),
                )
            )

        for name, lr1_model in candidates:
            train_features = _feature_table(train_df, train_rows, lr1_model)
            test_features = _feature_table(test_df, test_rows, lr1_model)
            y_train = train_features["label"].to_numpy(dtype=int)
            y_test = test_features["label"].to_numpy(dtype=int)
            lr2 = GatedSymmetryLogistic(
                logreg_c=LR2_C, random_state=RANDOM_SEED + split_id
            ).fit(train_features, y_train)
            train_score = lr2.predict_proba(train_features)[:, 1]
            test_score = lr2.predict_proba(test_features)[:, 1]
            threshold = float(
                compute_binary_thresholds(
                    y_train, train_score, target_sensitivity=TARGET_SENSITIVITY
                )["threshold_target"]
            )
            rows.append(
                _metric_row(
                    name=name,
                    split_id=split_id,
                    y_train=y_train,
                    y_test=y_test,
                    train_score=train_score,
                    test_score=test_score,
                    threshold=threshold,
                    train_patients=len(train_patients),
                    test_patients=len(test_patients),
                )
            )
            if isinstance(lr1_model, RandomForestClassifier):
                for q_value, importance in zip(
                    _q_grid(train_rows), lr1_model.feature_importances_, strict=True
                ):
                    importances.append(
                        {
                            "model_name": name,
                            "split_id": split_id,
                            "q_nm_inv": float(q_value),
                            "importance": float(importance),
                        }
                    )
    return rows, importances


def _fit_all(df: pd.DataFrame) -> pd.DataFrame:
    """Report in-sample capacity only; it is not independent validation."""
    rows = _lr1_rows(df)
    candidates: list[tuple[str, ProbabilityEstimator]] = [
        (
            "profile_logistic_lr1",
            build_profile_logistic(logreg_c=LR1_C, random_state=RANDOM_SEED).fit(
                profile_matrix(rows, PROFILE_COLUMN), row_labels(rows, LABEL_COLUMN)
            ),
        )
    ]
    for leaf_size in LEAF_SIZES:
        candidates.append(
            (
                f"profile_random_forest_leaf_{leaf_size}",
                _random_forest(leaf_size, RANDOM_SEED).fit(
                    profile_matrix(rows, PROFILE_COLUMN), row_labels(rows, LABEL_COLUMN)
                ),
            )
        )

    result = []
    for name, lr1_model in candidates:
        features = _feature_table(df, rows, lr1_model)
        y = features["label"].to_numpy(dtype=int)
        lr2 = GatedSymmetryLogistic(
            logreg_c=LR2_C, random_state=RANDOM_SEED
        ).fit(features, y)
        score = lr2.predict_proba(features)[:, 1]
        threshold = float(
            compute_binary_thresholds(
                y, score, target_sensitivity=TARGET_SENSITIVITY
            )["threshold_target"]
        )
        result.append(
            {
                "model_name": name,
                **final_fit_training_metrics(y, score, threshold=threshold),
            }
        )
    return pd.DataFrame(result)


def _lr1_rows(
    df: pd.DataFrame, *, require_two_classes: bool = True
) -> pd.DataFrame:
    return lr1_training_rows(
        df,
        label_column=LABEL_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
        lr1_row_policy=LR1_ROW_POLICY,
        require_two_classes=require_two_classes,
    )


def _feature_table(
    full_df: pd.DataFrame,
    rows: pd.DataFrame,
    lr1_model: ProbabilityEstimator,
) -> pd.DataFrame:
    scores = score_lr1_rows(
        lr1_model,
        rows,
        full_df=full_df,
        profile_column=PROFILE_COLUMN,
        group_column=GROUP_COLUMN,
        side_column=SIDE_COLUMN,
        label_column=LABEL_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
    )
    return patient_feature_table(
        full_df,
        scores,
        profile_column=PROFILE_COLUMN,
        label_column=LABEL_COLUMN,
        group_column=GROUP_COLUMN,
        specimen_column=SPECIMEN_COLUMN,
        side_column=SIDE_COLUMN,
        q_column=Q_COLUMN,
        age_column=AGE_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
        require_two_classes=False,
    )


def _random_forest(leaf_size: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=leaf_size,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def _q_grid(rows: pd.DataFrame) -> np.ndarray:
    values = np.asarray(rows.iloc[0][Q_COLUMN], dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{Q_COLUMN!r} must contain a one-dimensional q grid.")
    return values


def _metric_row(
    *,
    name: str,
    split_id: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_score: np.ndarray,
    test_score: np.ndarray,
    threshold: float,
    train_patients: int,
    test_patients: int,
) -> dict[str, Any]:
    train_values = binary_metric_values(
        y_train, train_score, np.full(len(y_train), threshold)
    )
    test_values = binary_metric_values(
        y_test, test_score, np.full(len(y_test), threshold)
    )
    return {
        "model_name": name,
        "split_id": split_id,
        "threshold_target": threshold,
        "train_patients": train_patients,
        "test_patients": test_patients,
        **{f"train_{key}": value for key, value in train_values.items()},
        **{f"test_{key}": value for key, value in test_values.items()},
    }


def _summary(
    df: pd.DataFrame,
    base_features: pd.DataFrame,
    metrics: pd.DataFrame,
    final_fit: pd.DataFrame,
    input_path: Path,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name, group in metrics.groupby("model_name", sort=False):
        final = final_fit.loc[final_fit["model_name"] == name].iloc[0]
        models[name] = {
            "held_out_patient_safe": {
                metric: {
                    "mean": float(group[f"test_{metric}"].mean()),
                    "std": float(group[f"test_{metric}"].std(ddof=0)),
                }
                for metric in (
                    "roc_auc",
                    "sensitivity",
                    "specificity",
                    "balanced_accuracy",
                    "brier_score",
                    "log_loss",
                )
            },
            "train_test_roc_auc_gap": {
                "mean": float(
                    (group["train_roc_auc"] - group["test_roc_auc"]).mean()
                ),
                "std": float(
                    (group["train_roc_auc"] - group["test_roc_auc"]).std(ddof=0)
                ),
            },
            "train_on_all_not_independent": {
                key: _plain_scalar(value)
                for key, value in final.drop(labels=["model_name"]).to_dict().items()
            },
        }
    return {
        "experiment": "random_forest_lr1_screening_v0_1",
        "status": "research_only_not_a_product_candidate",
        "question": (
            "Does Random Forest improve the measurement-level profile estimator "
            "while the released patient-level LR2 remains unchanged?"
        ),
        "input_dataframe_joblib": str(input_path),
        "input_dataframe_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "dataset": {
            "measurements": int(len(df)),
            "patients": int(base_features[GROUP_COLUMN].nunique()),
            "target_cases": int(len(base_features)),
            "cancer_target_cases": int((base_features["label"] == 1).sum()),
            "benign_target_cases": int((base_features["label"] == 0).sum()),
        },
        "architecture": {
            "lr1": "profile estimator candidate: logistic regression or Random Forest",
            "aggregation": "mean LR1 logit over target-breast measurements",
            "lr2": "unchanged GatedSymmetryLogistic with C=0.3, age, and optional SK Core4 symmetry",
        },
        "lr1_training": {
            "row_policy": LR1_ROW_POLICY,
            "logistic_regularization_c": LR1_C,
            "random_forest_candidates": {
                "n_estimators": N_ESTIMATORS,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
                "min_samples_leaf": list(LEAF_SIZES),
            },
        },
        "outer_evaluation": {
            "method": "repeated_patient_safe_stratified_kfold",
            "folds": FOLDS,
            "repeats": REPEATS,
            "splits": FOLDS * REPEATS,
            "random_seed": RANDOM_SEED,
            "threshold_policy": (
                "threshold selected on each outer training fold at target sensitivity "
                "0.95, then applied unchanged to the held-out fold"
            ),
        },
        "important_limitations": [
            "Multiple measurement rows from one breast are correlated; outer folds are patient-safe, but LR1 is still fitted on measurement-level rows within each training fold.",
            "No candidate is selected from these outer-fold results. Any new product candidate requires an independent selection/validation procedure.",
        ],
        "baseline_reproduction": {
            "status": "reproduced",
            "detail": (
                "The logistic LR1 plus unchanged LR2 baseline reproduces the frozen "
                "0.2.12-beta held-out footprint from the same dataframe and split protocol."
            ),
        },
        "models": models,
    }


def _plain_scalar(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


if __name__ == "__main__":
    main()
