"""Test the original pairwise-cosine symmetry block in the T100 M2Q model.

The block follows the earlier Aramis symmetry formulation: all target x
contralateral measurement pairs define between-breast distance; replicate pairs
within each breast define expected within-breast variability; their difference
is the cosine asymmetry score. This is distinct from SK cosine distance between
two aggregated profiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from xrd_preprocessing import load_preprocessing_dataframe

from aramis.modeling import _distance_summary, _profile_array_list
from aramis.training import _patient_model_feature_columns
from aramis_t100_m2q_symmetry_feature_selection_v0_1 import (
    GROUP_COLUMN,
    INPUT_JOBLIB,
    N_REPEATS,
    N_SPLITS,
    PROFILE_COLUMN,
    RANDOM_STATE,
    SIDE_COLUMN,
    SK_FEATURES,
    _feature_table_without_lr1_scores,
    _fit_lr1_and_build_features,
    _fit_lr2,
    _subset_columns,
    _subset_row,
    _subset_summary,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_CSV = ROOT / "docs" / "modeling" / "results" / "t100_m2q_pairwise_cosine_comparison_v0_1.csv"
CORRELATION_CSV = ROOT / "docs" / "modeling" / "results" / "t100_m2q_pairwise_cosine_correlations_v0_1.csv"

BASE_SK = [
    "sk_wasserstein_distance_full_q2",
    "sk_weightedrms1",
    "sk_weightedrms2",
    "sk_mean_peak_value_abs_delta",
]
PAIRWISE_VALUES = [
    "cosine_between_mean_x_available",
    "cosine_within_target_mean_x_available",
    "cosine_within_contralateral_mean_x_available",
    "cosine_asymmetry_score_x_available",
    "cosine_target_contralateral_centroid_x_available",
    "cosine_replicate_variability_delta_x_available",
    "cosine_replicate_variability_ratio_x_available",
]
PAIRWISE_FLAGS = [
    "target_within_available",
    "contralateral_within_available",
]
PAIRWISE_ASYMMETRY = [
    "cosine_asymmetry_score_x_available",
    *PAIRWISE_FLAGS,
]
PAIRWISE_BETWEEN_WITHIN = [
    "cosine_between_mean_x_available",
    "cosine_within_target_mean_x_available",
    "cosine_within_contralateral_mean_x_available",
    "cosine_asymmetry_score_x_available",
    *PAIRWISE_FLAGS,
]
PAIRWISE_FULL = [*PAIRWISE_VALUES, *PAIRWISE_FLAGS]


def _side_profiles(patient_df: pd.DataFrame, side: str) -> list[np.ndarray]:
    mask = patient_df[SIDE_COLUMN].astype(str).str.lower() == side.lower()
    return _profile_array_list(patient_df[mask], PROFILE_COLUMN)


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _pairwise_cosine_block(
    df: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Add original all-pairs cosine summaries without replacing M2Q fields."""
    rows = []
    for row in feature_table.itertuples(index=False):
        patient_id = str(row.patientId)
        patient_df = df[df[GROUP_COLUMN].astype(str) == patient_id]
        target_profiles = _side_profiles(patient_df, str(row.inferred_target_side))
        contralateral_profiles = _side_profiles(
            patient_df,
            str(row.inferred_contralateral_side),
        )
        summary = _distance_summary(
            target_profiles,
            contralateral_profiles,
            lambda left, right: _cosine_distance(left, right),
        )
        symmetry_available = int(np.isfinite(summary["asymmetry_score"]))
        target_within_available = int(np.isfinite(summary["within_target_mean"]))
        contralateral_within_available = int(
            np.isfinite(summary["within_contralateral_mean"])
        )
        rows.append(
            {
                "patientId": patient_id,
                "cosine_between_mean": _finite(summary["between_mean"]),
                "cosine_within_target_mean": _finite(summary["within_target_mean"]),
                "cosine_within_contralateral_mean": _finite(
                    summary["within_contralateral_mean"]
                ),
                "cosine_asymmetry_score": _finite(summary["asymmetry_score"]),
                "cosine_target_contralateral_centroid": _finite(
                    summary["target_contralateral_centroid"]
                ),
                "cosine_replicate_variability_delta": _finite(
                    summary["replicate_variability_delta"]
                ),
                "cosine_replicate_variability_ratio": _finite(
                    summary["replicate_variability_ratio"]
                ),
                "pairwise_cosine_available": symmetry_available,
                "target_within_available": target_within_available,
                "contralateral_within_available": contralateral_within_available,
            }
        )
    out = feature_table.merge(pd.DataFrame(rows), on="patientId", how="left")
    for column in [
        "cosine_between_mean",
        "cosine_within_target_mean",
        "cosine_within_contralateral_mean",
        "cosine_asymmetry_score",
        "cosine_target_contralateral_centroid",
        "cosine_replicate_variability_delta",
        "cosine_replicate_variability_ratio",
    ]:
        out[f"{column}_x_available"] = (
            out[column].fillna(0.0) * out["pairwise_cosine_available"]
        )
    return out


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    dim = min(left.size, right.size)
    if dim < 2:
        return float("nan")
    x = np.nan_to_num(left[:dim], nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(right[:dim], nan=0.0, posinf=0.0, neginf=0.0)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-12:
        return float("nan")
    return float(1.0 - np.dot(x, y) / denominator)


def _candidate_columns(base_columns: list[str]) -> dict[str, list[str]]:
    no_sk = _subset_columns(base_columns, [])
    return {
        "m2q_screened_core_4": base_columns,
        "m2q_core_4_plus_pairwise_asymmetry": [
            *base_columns,
            *PAIRWISE_ASYMMETRY,
        ],
        "m2q_core_4_plus_pairwise_between_within": [
            *base_columns,
            *PAIRWISE_BETWEEN_WITHIN,
        ],
        "m2q_core_4_plus_pairwise_full": [*base_columns, *PAIRWISE_FULL],
        "m2q_no_sk_plus_pairwise_full": [*no_sk, *PAIRWISE_FULL],
    }


def _pairwise_correlation_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    pairwise_raw = [
        "cosine_between_mean",
        "cosine_within_target_mean",
        "cosine_within_contralateral_mean",
        "cosine_asymmetry_score",
        "cosine_target_contralateral_centroid",
        "cosine_replicate_variability_delta",
        "cosine_replicate_variability_ratio",
    ]
    audit = [
        "target_within_cosine_distance_mean",
        "contralateral_within_cosine_distance_mean",
        "between_breasts_cosine_distance_mean",
        "symmetry_cosine_score",
    ]
    return (
        feature_table[[*pairwise_raw, *audit]]
        .corr(numeric_only=True)
        .loc[pairwise_raw, audit]
        .reset_index(names="pairwise_feature")
        .melt(id_vars="pairwise_feature", var_name="audit_feature", value_name="pearson_r")
    )


def main() -> None:
    df = load_preprocessing_dataframe(INPUT_JOBLIB)
    patient_table = _pairwise_cosine_block(df, _feature_table_without_lr1_scores(df))
    CORRELATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    _pairwise_correlation_table(patient_table).to_csv(CORRELATION_CSV, index=False)

    full_m2q_columns = _patient_model_feature_columns([])["M2Q"]
    base_columns = [
        column
        for column in full_m2q_columns
        if column not in SK_FEATURES or column in BASE_SK
    ]
    candidates = _candidate_columns(base_columns)
    y = patient_table["label"].to_numpy(dtype=int)
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    rows: list[dict[str, float | int | str]] = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(patient_table, y)):
        train_patients = set(patient_table.iloc[train_idx]["patientId"].astype(str))
        test_patients = set(patient_table.iloc[test_idx]["patientId"].astype(str))
        if train_patients.intersection(test_patients):
            raise RuntimeError("Patient leakage detected in pairwise-cosine split.")
        train_df = df[df[GROUP_COLUMN].astype(str).isin(train_patients)].copy()
        test_df = df[df[GROUP_COLUMN].astype(str).isin(test_patients)].copy()
        train_features, test_features = _fit_lr1_and_build_features(
            train_df,
            test_df,
            random_state=RANDOM_STATE + split_id,
        )
        train_features = _pairwise_cosine_block(train_df, train_features)
        test_features = _pairwise_cosine_block(test_df, test_features)
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

    all_features, _ = _fit_lr1_and_build_features(df, df, random_state=RANDOM_STATE)
    all_features = _pairwise_cosine_block(df, all_features)
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

    summary = _subset_summary(pd.DataFrame(rows))
    summary.to_csv(RESULT_CSV, index=False)
    print(summary.to_string(index=False))
    print(RESULT_CSV)
    print(CORRELATION_CSV)


if __name__ == "__main__":
    main()
