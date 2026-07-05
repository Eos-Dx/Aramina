"""M1 protocol comparison for Aramis patient-level research draft.

Compares the same M1 structure on two datasets:
- all_patients: all labelled patients in the wide t130 pool.
- biopsy_patients: patients with at least one biopsy row; LR1 uses biopsy rows.

NORMAL is mapped to BENIGN at label construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from aramis_biopsy_patient_model_structures_experiment import (
    DATASETS,
    INPUT_JOBLIB,
    RANDOM_STATE,
    TARGET_SENSITIVITY,
    _build_dataset,
    _fit_split_tables,
    _leave_one_patient_out,
    _load_dataframe,
    _metric_row,
    _model_scores,
    _repeated_patient_splits,
    _score_models,
    _summarize_repeated,
    _threshold_for_sensitivity,
    _write_markdown_table,
)
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "outputs" / "m1_protocol_comparison_v0_1"
M1_NAME = "M1_profile_plus_symmetry_raw_LR"
K_FOLDS = 5
K_REPEATS = 20


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide_df = _load_dataframe(INPUT_JOBLIB)
    summaries = []
    predictions = []

    for spec in DATASETS:
        cohort_df, lr1_df, feature_df = _build_dataset(wide_df, spec)
        feature_df.to_csv(OUTPUT_DIR / f"{spec.name}_feature_table.csv", index=False)

        train_all = _m1_only(
            _score_models(
                spec,
                cohort_df,
                lr1_df,
                _fit_train_all_final_df(lr1_df, feature_df),
                protocol="train_all_discovery",
                threshold_source="same_data",
            )
        )
        split_70_30, split_70_30_predictions = _repeated_patient_splits(
            spec,
            cohort_df,
            lr1_df,
            feature_df,
        )
        loocv, loocv_predictions = _leave_one_patient_out(
            spec,
            cohort_df,
            lr1_df,
            feature_df,
        )
        kfold = _repeated_stratified_kfold(spec, cohort_df, lr1_df, feature_df)

        summaries.extend(
            [
                train_all,
                _filter_m1(split_70_30),
                _filter_m1(loocv),
                kfold,
            ]
        )
        predictions.extend(
            [
                _filter_m1(split_70_30_predictions),
                _filter_m1(loocv_predictions),
            ]
        )

    summary = pd.concat(summaries, ignore_index=True)
    prediction_df = pd.concat(predictions, ignore_index=True)
    compact = _compact_table(summary)

    summary.to_csv(OUTPUT_DIR / "m1_protocol_summary.csv", index=False)
    prediction_df.to_csv(OUTPUT_DIR / "m1_protocol_predictions.csv", index=False)
    compact.to_csv(OUTPUT_DIR / "m1_protocol_comparison.csv", index=False)
    _write_markdown_table(compact, OUTPUT_DIR / "m1_protocol_comparison.md")

    print(compact.to_string(index=False))
    print(f"\nOUT {OUTPUT_DIR}")


def _fit_train_all_final_df(lr1_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    from aramis_biopsy_patient_model_structures_experiment import (
        _final_table,
        _fit_lr1,
        _patient_lr1_scores,
    )

    lr1_model = _fit_lr1(lr1_df)
    profile_scores = _patient_lr1_scores(lr1_model, lr1_df)
    return _final_table(feature_df, profile_scores)


def _m1_only(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([row for row in rows if row["model"] == M1_NAME])


def _filter_m1(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"].eq(M1_NAME)].copy()


def _repeated_stratified_kfold(
    spec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    splitter = RepeatedStratifiedKFold(
        n_splits=K_FOLDS,
        n_repeats=K_REPEATS,
        random_state=RANDOM_STATE,
    )
    y = feature_df["label"].to_numpy(dtype=int)
    rows = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(feature_df, y)):
        train_features = feature_df.iloc[train_idx].copy()
        test_features = feature_df.iloc[test_idx].copy()
        train_final, test_final = _fit_split_tables(train_features, test_features, lr1_df)
        if train_final["label"].nunique() < 2 or test_final["label"].nunique() < 2:
            continue
        for model_name, train_score, test_score in _model_scores(train_final, test_final):
            if model_name != M1_NAME:
                continue
            threshold = _threshold_for_sensitivity(
                train_final["label"].to_numpy(dtype=int),
                train_score,
                TARGET_SENSITIVITY,
            )
            row = _metric_row(
                spec,
                cohort_df,
                lr1_df,
                feature_df,
                protocol=f"stratified_{K_FOLDS}fold_x{K_REPEATS}_train_threshold",
                model_name=model_name,
                y=test_final["label"].to_numpy(dtype=int),
                score=test_score,
                threshold=threshold,
                split_count=1,
                threshold_source="train",
            )
            row["split_id"] = split_id
            rows.append(row)
    return _summarize_repeated(pd.DataFrame(rows), spec, cohort_df, lr1_df, feature_df)


def _compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    data["metric"] = data.apply(_format_metric, axis=1)
    return (
        data.pivot_table(
            index=[
                "protocol",
                "threshold_source",
                "dataset",
                "patients_for_final_model",
                "cancer_patients",
                "non_cancer_patients",
                "split_count",
            ],
            columns="model",
            values="metric",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["dataset", "protocol"])
    )


def _format_metric(row: pd.Series) -> str:
    if int(row["split_count"]) > 1 and row["protocol"] != "patient_loocv_pooled":
        return (
            f"R {row.R_ROC_AUC:.3f}+/-{row.R_ROC_AUC_std:.3f}; "
            f"S {row.S_sensitivity:.3f}+/-{row.S_sensitivity_std:.3f}; "
            f"Sp {row.Sp_specificity:.3f}+/-{row.Sp_specificity_std:.3f}"
        )
    return (
        f"R {row.R_ROC_AUC:.3f}; "
        f"S {row.S_sensitivity:.3f}; "
        f"Sp {row.Sp_specificity:.3f}"
    )


if __name__ == "__main__":
    main()
