"""Compare M0Q, M1Q, and M2Q on the same T100 patient-safe splits.

This control resolves a specific question: how much of the held-out signal is
provided by the SK symmetry block itself, separately from age and reliability.
It does not choose a new product model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from xrd_preprocessing import load_preprocessing_dataframe

from aramis.training import _patient_model_feature_columns
from aramis_t100_m2q_symmetry_feature_selection_v0_1 import (
    INPUT_JOBLIB,
    N_REPEATS,
    N_SPLITS,
    RANDOM_STATE,
    SK_FEATURES,
    _feature_table_without_lr1_scores,
    _fit_lr1_and_build_features,
    _fit_lr2,
    _subset_row,
    _subset_summary,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_CSV = ROOT / "docs" / "modeling" / "results" / "t100_m0q_m1q_m2q_controlled_comparison_v0_1.csv"
OOF_PREDICTIONS_CSV = (
    ROOT
    / "docs"
    / "modeling"
    / "results"
    / "t100_m0q_m1q_m2q_controlled_oof_predictions_v0_1.csv"
)
OOF_OPERATING_CSV = (
    ROOT
    / "docs"
    / "modeling"
    / "results"
    / "t100_m0q_m1q_m2q_fixed_sensitivity_oof_v0_1.csv"
)

SCREENED_CORE_4 = [
    "sk_wasserstein_distance_full_q2",
    "sk_weightedrms1",
    "sk_weightedrms2",
    "sk_mean_peak_value_abs_delta",
]


def _keep_sk(columns: list[str], retained_sk: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column not in SK_FEATURES or column in retained_sk
    ]


def _candidates() -> dict[str, list[str]]:
    schema = _patient_model_feature_columns([])
    return {
        "M0Q_profile_plus_reliability": schema["M0Q"],
        "M1Q_all_15_SK": schema["M1Q"],
        "M1Q_screened_core_4": _keep_sk(schema["M1Q"], SCREENED_CORE_4),
        "M2Q_all_15_SK_plus_age": schema["M2Q"],
        "M2Q_screened_core_4_plus_age": _keep_sk(
            schema["M2Q"], SCREENED_CORE_4
        ),
        "M2Q_no_SK_plus_age": _keep_sk(schema["M2Q"], []),
    }


def _pooled_oof_operating_point(predictions: pd.DataFrame) -> pd.DataFrame:
    """Describe pooled OOF ROC operating points; not a deployable threshold."""
    rows = []
    for candidate, group in predictions.groupby("candidate", sort=False):
        patient = (
            group.groupby(["patientId", "label"], as_index=False)["p_cancer"]
            .mean()
            .reset_index(drop=True)
        )
        y = patient["label"].to_numpy(dtype=int)
        score = patient["p_cancer"].to_numpy(dtype=float)
        fpr, tpr, thresholds = roc_curve(y, score)
        eligible = np.flatnonzero(tpr >= 0.95)
        best = eligible[np.argmin(fpr[eligible])]
        rows.append(
            {
                "candidate": candidate,
                "patients": int(len(patient)),
                "roc_auc_oof_mean": float(roc_auc_score(y, score)),
                "sensitivity_target": float(tpr[best]),
                "specificity_at_target_sensitivity": float(1.0 - fpr[best]),
                "threshold_from_pooled_oof": float(thresholds[best]),
                "note": "descriptive pooled OOF ROC point; threshold uses OOF labels",
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc_oof_mean", ascending=False)


def main() -> None:
    df = load_preprocessing_dataframe(INPUT_JOBLIB)
    patient_table = _feature_table_without_lr1_scores(df)
    candidates = _candidates()
    y = patient_table["label"].to_numpy(dtype=int)
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    rows = []
    prediction_rows = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(patient_table, y)):
        train_patients = set(patient_table.iloc[train_idx]["patientId"].astype(str))
        test_patients = set(patient_table.iloc[test_idx]["patientId"].astype(str))
        if train_patients.intersection(test_patients):
            raise RuntimeError("Patient leakage detected in controlled comparison.")
        train_df = df[df["patientId"].astype(str).isin(train_patients)].copy()
        test_df = df[df["patientId"].astype(str).isin(test_patients)].copy()
        train_features, test_features = _fit_lr1_and_build_features(
            train_df,
            test_df,
            random_state=RANDOM_STATE + split_id,
        )
        y_train = train_features["label"].to_numpy(dtype=int)
        y_test = test_features["label"].to_numpy(dtype=int)
        for candidate, columns in candidates.items():
            _, train_score, test_score = _fit_lr2(
                train_features,
                test_features,
                columns,
                random_state=RANDOM_STATE + split_id,
            )
            rows.append(
                _subset_row(
                    candidate=candidate,
                    mode="repeated_stratified_5fold",
                    split_id=split_id,
                    columns=columns,
                    y_train=y_train,
                    y_test=y_test,
                    train_score=train_score,
                    test_score=test_score,
                )
            )
            prediction_rows.extend(
                {
                    "candidate": candidate,
                    "split_id": split_id,
                    "patientId": str(patient_id),
                    "label": int(label),
                    "p_cancer": float(score),
                }
                for patient_id, label, score in zip(
                    test_features["patientId"],
                    y_test,
                    test_score,
                    strict=True,
                )
            )

    all_features, _ = _fit_lr1_and_build_features(df, df, random_state=RANDOM_STATE)
    all_y = all_features["label"].to_numpy(dtype=int)
    for candidate, columns in candidates.items():
        _, train_score, test_score = _fit_lr2(
            all_features,
            all_features,
            columns,
            random_state=RANDOM_STATE,
        )
        rows.append(
            _subset_row(
                candidate=candidate,
                mode="train_all",
                split_id=0,
                columns=columns,
                y_train=all_y,
                y_test=all_y,
                train_score=train_score,
                test_score=test_score,
            )
        )

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary = _subset_summary(pd.DataFrame(rows))
    oof_predictions = pd.DataFrame(prediction_rows)
    oof_operating = _pooled_oof_operating_point(oof_predictions)
    summary.to_csv(RESULT_CSV, index=False)
    oof_predictions.to_csv(OOF_PREDICTIONS_CSV, index=False)
    oof_operating.to_csv(OOF_OPERATING_CSV, index=False)
    print(summary.to_string(index=False))
    print(oof_operating.to_string(index=False))
    print(RESULT_CSV)
    print(OOF_OPERATING_CSV)


if __name__ == "__main__":
    main()
