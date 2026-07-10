"""Check whether the T100 M0Q route learns above a patient-label null.

Target-side assignment and profiles remain unchanged. BENIGN/CANCER labels are
permuted between patients before LR1 and LR2 fitting. Held-out ROC should be
near 0.5 if the patient-safe route does not retain label leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xrd_preprocessing import load_preprocessing_dataframe

from aramis.modeling import profile_matrix
from aramis.training import (
    _lr1_training_rows,
    _patient_model_feature_columns,
    _patient_feature_table,
    _profile_logistic,
    _row_labels,
    _score_lr1_rows,
)
from aramis_t100_m2q_symmetry_feature_selection_v0_1 import (
    AGE_COLUMN,
    BIOPSY_COLUMN,
    GROUP_COLUMN,
    INPUT_JOBLIB,
    LABEL_COLUMN,
    LR1_C,
    LR1_ROW_POLICY,
    PROFILE_COLUMN,
    Q_COLUMN,
    SIDE_COLUMN,
    SPECIMEN_COLUMN,
    _feature_table_without_lr1_scores,
    _fit_lr1_and_build_features,
    _fit_lr2,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_CSV = (
    ROOT
    / "docs"
    / "modeling"
    / "results"
    / "t100_m0q_patient_label_permutation_v0_1.csv"
)
N_PERMUTATIONS = 20
N_SPLITS = 5
RANDOM_STATE = 20260710


def _label_text(value: int) -> str:
    return "CANCER" if value else "BENIGN"


def _permuted_patient_labels(
    patient_table: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> dict[str, int]:
    patient_ids = patient_table["patientId"].astype(str).to_numpy()
    labels = patient_table["label"].to_numpy(dtype=int).copy()
    rng.shuffle(labels)
    return dict(zip(patient_ids, labels, strict=True))


def _replace_feature_labels(
    features: pd.DataFrame,
    labels: dict[str, int],
) -> pd.DataFrame:
    out = features.copy()
    out["label"] = out["patientId"].astype(str).map(labels).astype(int)
    return out


def _permuted_lr1_rows(
    df: pd.DataFrame,
    labels: dict[str, int],
    *,
    require_two_classes: bool,
) -> pd.DataFrame:
    rows = _lr1_training_rows(
        df,
        label_column=LABEL_COLUMN,
        biopsy_column=BIOPSY_COLUMN,
        lr1_row_policy=LR1_ROW_POLICY,
        require_two_classes=require_two_classes,
    ).copy()
    rows[LABEL_COLUMN] = rows[GROUP_COLUMN].astype(str).map(labels).map(_label_text)
    return rows


def _build_permuted_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: dict[str, int],
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_lr1_rows = _permuted_lr1_rows(train_df, labels, require_two_classes=True)
    lr1_model = _profile_logistic(logreg_c=LR1_C, random_state=random_state)
    lr1_model.fit(
        profile_matrix(train_lr1_rows, PROFILE_COLUMN),
        _row_labels(train_lr1_rows, LABEL_COLUMN),
    )

    test_lr1_rows = _permuted_lr1_rows(test_df, labels, require_two_classes=False)
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
    train_features = _replace_feature_labels(
        _patient_feature_table(
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
        ),
        labels,
    )
    test_features = _replace_feature_labels(
        _patient_feature_table(
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
        ),
        labels,
    )
    return train_features, test_features


def _observed_auc(
    df: pd.DataFrame,
    patient_table: pd.DataFrame,
    splitter: StratifiedKFold,
    columns: list[str],
) -> tuple[float, float]:
    labels = patient_table["label"].to_numpy(dtype=int)
    fold_scores = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(patient_table, labels)):
        train_patients = set(patient_table.iloc[train_idx]["patientId"].astype(str))
        test_patients = set(patient_table.iloc[test_idx]["patientId"].astype(str))
        train_features, test_features = _fit_lr1_and_build_features(
            df[df[GROUP_COLUMN].astype(str).isin(train_patients)].copy(),
            df[df[GROUP_COLUMN].astype(str).isin(test_patients)].copy(),
            random_state=RANDOM_STATE + fold_id,
        )
        _, _, test_score = _fit_lr2(
            train_features,
            test_features,
            columns,
            random_state=RANDOM_STATE + fold_id,
        )
        fold_scores.append(
            roc_auc_score(test_features["label"].to_numpy(dtype=int), test_score)
        )
    return float(np.mean(fold_scores)), float(np.std(fold_scores, ddof=1))


def main() -> None:
    df = load_preprocessing_dataframe(INPUT_JOBLIB)
    patient_table = _feature_table_without_lr1_scores(df)
    original_labels = patient_table["label"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    candidate_columns = _patient_model_feature_columns([])["M0Q"]
    observed_mean, observed_std = _observed_auc(
        df,
        patient_table,
        splitter,
        candidate_columns,
    )
    rows = [
        {
            "kind": "observed",
            "run_id": 0,
            "folds": N_SPLITS,
            "roc_auc_mean": observed_mean,
            "roc_auc_std": observed_std,
        }
    ]
    rng = np.random.default_rng(RANDOM_STATE)

    for permutation_id in range(N_PERMUTATIONS):
        labels = _permuted_patient_labels(patient_table, rng=rng)
        fold_scores = []
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(patient_table, original_labels)):
            train_patients = set(patient_table.iloc[train_idx]["patientId"].astype(str))
            test_patients = set(patient_table.iloc[test_idx]["patientId"].astype(str))
            if train_patients.intersection(test_patients):
                raise RuntimeError("Patient leakage detected in permutation test.")
            train_features, test_features = _build_permuted_features(
                df[df[GROUP_COLUMN].astype(str).isin(train_patients)].copy(),
                df[df[GROUP_COLUMN].astype(str).isin(test_patients)].copy(),
                labels,
                random_state=RANDOM_STATE + permutation_id * N_SPLITS + fold_id,
            )
            _, _, test_score = _fit_lr2(
                train_features,
                test_features,
                candidate_columns,
                random_state=RANDOM_STATE + permutation_id * N_SPLITS + fold_id,
            )
            fold_scores.append(
                roc_auc_score(test_features["label"].to_numpy(dtype=int), test_score)
            )
        rows.append(
            {
                "kind": "patient_label_permutation",
                "run_id": permutation_id,
                "folds": N_SPLITS,
                "roc_auc_mean": float(np.mean(fold_scores)),
                "roc_auc_std": float(np.std(fold_scores, ddof=1)),
            }
        )

    result = pd.DataFrame(rows)
    null = result[result["kind"] == "patient_label_permutation"]
    empirical_p = float(
        (1 + (null["roc_auc_mean"] >= observed_mean).sum()) / (1 + len(null))
    )
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULT_CSV, index=False)
    print(result.to_string(index=False))
    print(
        "null ROC AUC mean +/- std: "
        f"{null['roc_auc_mean'].mean():.3f} +/- {null['roc_auc_mean'].std(ddof=1):.3f}"
    )
    print(f"observed ROC AUC: {observed_mean:.3f} +/- {observed_std:.3f}")
    print(f"empirical p(null ROC >= observed): {empirical_p:.3f}")
    print(RESULT_CSV)


if __name__ == "__main__":
    main()
