"""Rank T100 M2Q SK symmetry features without changing product code.

This research script keeps the M2Q architecture and both regularization values
fixed. It tests only the SK symmetry block on repeated, patient-safe,
stratified 5-fold splits. Every held-out patient is excluded from both LR1 and
LR2 fitting for its split.

Three complementary signals are written:

* LOFO: refit LR2 without one SK feature and measure the held-out change;
* permutation: shuffle one feature only inside held-out patients;
* coefficient stability: inspect the standardized LR2 coefficient over folds.

The output ranks candidates for a smaller M2Q schema. It does not select a
final product schema automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xrd_preprocessing import load_preprocessing_dataframe

from aramis.modeling import compute_binary_thresholds, profile_matrix
from aramis.training import (
    _empty_lr1_scores,
    _lr1_training_rows,
    _patient_feature_table,
    _patient_model_feature_columns,
    _profile_logistic,
    _row_labels,
    _scalar_logistic,
    _score_lr1_rows,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_JOBLIB = (
    ROOT
    / "examples"
    / "outputs"
    / "model_selection_m1q_v0_1"
    / "preprocessing"
    / "aramis_t100_biopsy_patients_model_input.joblib"
)
RESULT_DIR = ROOT / "docs" / "modeling" / "results"
PER_SPLIT_CSV = RESULT_DIR / "t100_m2q_sk_feature_selection_per_split_v0_1.csv"
SUMMARY_CSV = RESULT_DIR / "t100_m2q_sk_feature_selection_summary_v0_1.csv"
COEFFICIENT_CSV = RESULT_DIR / "t100_m2q_sk_feature_coefficients_v0_1.csv"
SUBSET_CSV = RESULT_DIR / "t100_m2q_sk_feature_subset_comparison_v0_1.csv"

LR1_C = 0.3
LR2_C = 0.1
N_SPLITS = 5
N_REPEATS = 20
PERMUTATION_REPEATS = 30
RANDOM_STATE = 42
TARGET_SENSITIVITY = 0.95

PROFILE_COLUMN = "radial_profile_data"
LABEL_COLUMN = "product_status_group"
GROUP_COLUMN = "patientId"
SPECIMEN_COLUMN = "specimenId"
SIDE_COLUMN = "side"
Q_COLUMN = "q_range"
AGE_COLUMN = "age"
BIOPSY_COLUMN = "biopsy"
LR1_ROW_POLICY = "biopsy_only"

# These fields are the only candidates in this experiment. M2Q core fields
# (LR1 risk, reliability, age, and symmetry availability) remain mandatory.
SK_FEATURES = [
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
]

# These sets are derived only after the individual importance screen. The
# comparison decides whether a smaller set preserves the held-out M2Q signal.
SK_SUBSETS = {
    "all_sk": SK_FEATURES,
    "distance_tail": [
        "sk_wasserstein_distance_mu_tc",
        "sk_cosine_distance_full_q2",
        "sk_wasserstein_distance_full_q2",
    ],
    "screened_core_3": [
        "sk_wasserstein_distance_full_q2",
        "sk_weightedrms2",
        "sk_mean_peak_value_abs_delta",
    ],
    "screened_core_4": [
        "sk_wasserstein_distance_full_q2",
        "sk_weightedrms1",
        "sk_weightedrms2",
        "sk_mean_peak_value_abs_delta",
    ],
    "no_sk": [],
}


def _feature_table_without_lr1_scores(df: pd.DataFrame) -> pd.DataFrame:
    return _patient_feature_table(
        df,
        _empty_lr1_scores(df, GROUP_COLUMN),
        profile_column=PROFILE_COLUMN,
        label_column=LABEL_COLUMN,
        group_column=GROUP_COLUMN,
        specimen_column=SPECIMEN_COLUMN,
        side_column=SIDE_COLUMN,
        q_column=Q_COLUMN,
        age_column=AGE_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
    )


def _fit_lr1_and_build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LR1 on train rows only and construct matching train/test features."""
    train_lr1_rows = _lr1_training_rows(
        train_df,
        label_column=LABEL_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
        lr1_row_policy=LR1_ROW_POLICY,
    )
    lr1_model = _profile_logistic(logreg_c=LR1_C, random_state=random_state)
    lr1_model.fit(
        profile_matrix(train_lr1_rows, PROFILE_COLUMN),
        _row_labels(train_lr1_rows, LABEL_COLUMN),
    )

    train_scores = _score_lr1_rows(
        lr1_model,
        train_lr1_rows,
        full_df=train_df,
        profile_column=PROFILE_COLUMN,
        group_column=GROUP_COLUMN,
        side_column=SIDE_COLUMN,
        label_column=LABEL_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
    )
    train_features = _patient_feature_table(
        train_df,
        train_scores,
        profile_column=PROFILE_COLUMN,
        label_column=LABEL_COLUMN,
        group_column=GROUP_COLUMN,
        specimen_column=SPECIMEN_COLUMN,
        side_column=SIDE_COLUMN,
        q_column=Q_COLUMN,
        age_column=AGE_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
    )

    test_lr1_rows = _lr1_training_rows(
        test_df,
        label_column=LABEL_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
        lr1_row_policy=LR1_ROW_POLICY,
        require_two_classes=False,
    )
    test_scores = _score_lr1_rows(
        lr1_model,
        test_lr1_rows,
        full_df=test_df,
        profile_column=PROFILE_COLUMN,
        group_column=GROUP_COLUMN,
        side_column=SIDE_COLUMN,
        label_column=LABEL_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
    )
    test_features = _patient_feature_table(
        test_df,
        test_scores,
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
    return train_features, test_features


def _fit_lr2(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    columns: list[str],
    *,
    random_state: int,
) -> tuple[object, np.ndarray, np.ndarray]:
    model = _scalar_logistic(logreg_c=LR2_C, random_state=random_state)
    y_train = train_features["label"].to_numpy(dtype=int)
    model.fit(train_features[columns], y_train)
    return (
        model,
        model.predict_proba(train_features[columns])[:, 1],
        model.predict_proba(test_features[columns])[:, 1],
    )


def _threshold_metrics(
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    return sensitivity, specificity


def _summary(per_split: pd.DataFrame, coefficients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in per_split.groupby("feature", sort=False):
        coef = coefficients[coefficients["feature"] == feature]
        positive = float((coef["standardized_coefficient"] > 0).mean())
        negative = float((coef["standardized_coefficient"] < 0).mean())
        rows.append(
            {
                "feature": feature,
                "folds": int(len(group)),
                "full_roc_auc_mean": group["full_roc_auc"].mean(),
                "lofo_roc_auc_mean": group["lofo_roc_auc"].mean(),
                "lofo_delta_roc_auc_mean": group["lofo_delta_roc_auc"].mean(),
                "lofo_delta_roc_auc_std": group["lofo_delta_roc_auc"].std(ddof=1),
                "lofo_delta_sensitivity_mean": group[
                    "lofo_delta_sensitivity"
                ].mean(),
                "lofo_delta_specificity_mean": group[
                    "lofo_delta_specificity"
                ].mean(),
                "permutation_auc_drop_mean": group["permutation_auc_drop"].mean(),
                "permutation_auc_drop_std": group["permutation_auc_drop"].std(
                    ddof=1
                ),
                "coefficient_mean": coef["standardized_coefficient"].mean(),
                "coefficient_std": coef["standardized_coefficient"].std(ddof=1),
                "coefficient_positive_fraction": positive,
                "coefficient_negative_fraction": negative,
                "coefficient_dominant_sign_fraction": max(positive, negative),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["lofo_delta_roc_auc_mean", "permutation_auc_drop_mean"],
        ascending=False,
    )


def _subset_columns(full_columns: list[str], active_sk: list[str]) -> list[str]:
    return [
        column
        for column in full_columns
        if column not in SK_FEATURES or column in active_sk
    ]


def _subset_row(
    *,
    candidate: str,
    mode: str,
    split_id: int,
    columns: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_score: np.ndarray,
    test_score: np.ndarray,
) -> dict[str, float | int | str]:
    threshold = compute_binary_thresholds(
        y_train,
        train_score,
        target_sensitivity=TARGET_SENSITIVITY,
    )["threshold_target"]
    sensitivity, specificity = _threshold_metrics(y_test, test_score, float(threshold))
    return {
        "candidate": candidate,
        "mode": mode,
        "split_id": split_id,
        "sk_features": int(sum(column in SK_FEATURES for column in columns)),
        "roc_auc": float(roc_auc_score(y_test, test_score)),
        "sensitivity_target": sensitivity,
        "specificity_target": specificity,
    }


def _subset_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["candidate", "mode", "sk_features"], as_index=False)
        .agg(
            folds=("split_id", "count"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            sensitivity_target_mean=("sensitivity_target", "mean"),
            sensitivity_target_std=("sensitivity_target", "std"),
            specificity_target_mean=("specificity_target", "mean"),
            specificity_target_std=("specificity_target", "std"),
        )
        .sort_values(["mode", "roc_auc_mean"], ascending=[True, False])
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_preprocessing_dataframe(INPUT_JOBLIB)
    patient_table = _feature_table_without_lr1_scores(df)
    y = patient_table["label"].to_numpy(dtype=int)
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    full_columns = _patient_model_feature_columns([])["M2Q"]
    unknown = set(SK_FEATURES).difference(full_columns)
    if unknown:
        raise ValueError(f"SK feature list is not in M2Q schema: {sorted(unknown)}")

    per_split_rows: list[dict[str, float | int | str]] = []
    coefficient_rows: list[dict[str, float | int | str]] = []
    subset_rows: list[dict[str, float | int | str]] = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(patient_table, y)):
        train_patients = set(patient_table.iloc[train_idx]["patientId"].astype(str))
        test_patients = set(patient_table.iloc[test_idx]["patientId"].astype(str))
        train_df = df[df[GROUP_COLUMN].astype(str).isin(train_patients)].copy()
        test_df = df[df[GROUP_COLUMN].astype(str).isin(test_patients)].copy()
        if train_patients.intersection(test_patients):
            raise RuntimeError("Patient leakage detected in feature-selection split.")

        train_features, test_features = _fit_lr1_and_build_features(
            train_df,
            test_df,
            random_state=RANDOM_STATE + split_id,
        )
        full_model, train_score, test_score = _fit_lr2(
            train_features,
            test_features,
            full_columns,
            random_state=RANDOM_STATE + split_id,
        )
        y_train = train_features["label"].to_numpy(dtype=int)
        y_test = test_features["label"].to_numpy(dtype=int)
        threshold = compute_binary_thresholds(
            y_train,
            train_score,
            target_sensitivity=TARGET_SENSITIVITY,
        )["threshold_target"]
        full_auc = float(roc_auc_score(y_test, test_score))
        full_sensitivity, full_specificity = _threshold_metrics(
            y_test,
            test_score,
            float(threshold),
        )
        subset_rows.append(
            _subset_row(
                candidate="all_sk",
                mode="repeated_stratified_5fold",
                split_id=split_id,
                columns=full_columns,
                y_train=y_train,
                y_test=y_test,
                train_score=train_score,
                test_score=test_score,
            )
        )
        coefficients = full_model.named_steps["logreg"].coef_[0]
        coefficient_rows.extend(
            {
                "split_id": split_id,
                "feature": feature,
                "standardized_coefficient": float(coefficient),
            }
            for feature, coefficient in zip(full_columns, coefficients, strict=True)
            if feature in SK_FEATURES
        )

        rng = np.random.default_rng(RANDOM_STATE + split_id)
        for feature in SK_FEATURES:
            lofo_columns = [column for column in full_columns if column != feature]
            _, lofo_train_score, lofo_test_score = _fit_lr2(
                train_features,
                test_features,
                lofo_columns,
                random_state=RANDOM_STATE + split_id,
            )
            lofo_threshold = compute_binary_thresholds(
                y_train,
                lofo_train_score,
                target_sensitivity=TARGET_SENSITIVITY,
            )["threshold_target"]
            lofo_auc = float(roc_auc_score(y_test, lofo_test_score))
            lofo_sensitivity, lofo_specificity = _threshold_metrics(
                y_test,
                lofo_test_score,
                float(lofo_threshold),
            )

            shuffled_auc = []
            for _ in range(PERMUTATION_REPEATS):
                shuffled = test_features[full_columns].copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                shuffled_score = full_model.predict_proba(shuffled)[:, 1]
                shuffled_auc.append(float(roc_auc_score(y_test, shuffled_score)))

            per_split_rows.append(
                {
                    "split_id": split_id,
                    "feature": feature,
                    "full_roc_auc": full_auc,
                    "lofo_roc_auc": lofo_auc,
                    "lofo_delta_roc_auc": full_auc - lofo_auc,
                    "full_sensitivity": full_sensitivity,
                    "lofo_sensitivity": lofo_sensitivity,
                    "lofo_delta_sensitivity": full_sensitivity - lofo_sensitivity,
                    "full_specificity": full_specificity,
                    "lofo_specificity": lofo_specificity,
                    "lofo_delta_specificity": full_specificity - lofo_specificity,
                    "permutation_auc_drop": full_auc - float(np.mean(shuffled_auc)),
                }
            )

        for candidate, active_sk in SK_SUBSETS.items():
            if candidate == "all_sk":
                continue
            columns = _subset_columns(full_columns, active_sk)
            _, subset_train_score, subset_test_score = _fit_lr2(
                train_features,
                test_features,
                columns,
                random_state=RANDOM_STATE + split_id,
            )
            subset_rows.append(
                _subset_row(
                    candidate=candidate,
                    mode="repeated_stratified_5fold",
                    split_id=split_id,
                    columns=columns,
                    y_train=y_train,
                    y_test=y_test,
                    train_score=subset_train_score,
                    test_score=subset_test_score,
                )
            )

    all_train_features, _ = _fit_lr1_and_build_features(
        df,
        df,
        random_state=RANDOM_STATE,
    )
    all_y = all_train_features["label"].to_numpy(dtype=int)
    for candidate, active_sk in SK_SUBSETS.items():
        columns = _subset_columns(full_columns, active_sk)
        _, all_train_score, all_test_score = _fit_lr2(
            all_train_features,
            all_train_features,
            columns,
            random_state=RANDOM_STATE,
        )
        subset_rows.append(
            _subset_row(
                candidate=candidate,
                mode="train_all",
                split_id=0,
                columns=columns,
                y_train=all_y,
                y_test=all_y,
                train_score=all_train_score,
                test_score=all_test_score,
            )
        )

    per_split = pd.DataFrame(per_split_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    summary = _summary(per_split, coefficients)
    subset_summary = _subset_summary(pd.DataFrame(subset_rows))
    per_split.to_csv(PER_SPLIT_CSV, index=False)
    coefficients.to_csv(COEFFICIENT_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    subset_summary.to_csv(SUBSET_CSV, index=False)
    print(f"patients={len(patient_table)}")
    print(f"splits={N_SPLITS * N_REPEATS}")
    print(summary.to_string(index=False))
    print(subset_summary.to_string(index=False))
    print(SUMMARY_CSV)
    print(SUBSET_CSV)


if __name__ == "__main__":
    main()
