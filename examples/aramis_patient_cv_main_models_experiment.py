"""Patient-level repeated CV comparison for Aramis M0-M3 research models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from aramis_loocv_main_models_experiment import (
    INPUT_JOBLIB,
    _coerce_features,
    _global_summary,
    _load_pool,
    _run_fold,
    _save_roc_plot,
    _target_training_rows,
)
from aramis.modeling import build_fusion_feature_table
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "examples" / "outputs" / "modeling_patient_cv"
RANDOM_STATE = 42


def main() -> None:
    pool_df = _load_pool(INPUT_JOBLIB)
    target_df = _target_training_rows(pool_df)
    feature_table = _coerce_features(build_fusion_feature_table(target_df, pool_df))
    split_base = _specimen_split_base(target_df)

    runs = {
        "repeated_5fold_20x": _repeated_stratified_group_kfold(split_base),
        "group_shuffle_70_30_50x": _group_shuffle_splits(split_base),
    }
    for run_name, splits in runs.items():
        _run_protocol(
            run_name=run_name,
            splits=splits,
            pool_df=pool_df,
            target_df=target_df,
            feature_table=feature_table,
            split_base=split_base,
        )


def _specimen_split_base(target_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for specimen_id, group_df in target_df.groupby("specimenId", sort=True):
        labels = group_df["product_status_group"].astype(str).unique().tolist()
        patients = group_df["patientId"].astype(str).unique().tolist()
        if len(labels) != 1 or len(patients) != 1:
            raise ValueError((specimen_id, labels, patients))
        rows.append(
            {
                "specimenId": str(specimen_id),
                "patientId": patients[0],
                "y_true": int(labels[0] == "CANCER"),
            }
        )
    return pd.DataFrame(rows)


def _repeated_stratified_group_kfold(
    split_base: pd.DataFrame,
    *,
    repeats: int = 20,
    n_splits: int = 5,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    x = np.zeros(len(split_base))
    y = split_base["y_true"].to_numpy(dtype=int)
    groups = split_base["patientId"].astype(str).to_numpy()
    splits = []
    split_id = 0
    for repeat_id in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE + repeat_id,
        )
        for train_idx, test_idx in splitter.split(x, y, groups):
            _check_split(y, groups, train_idx, test_idx)
            splits.append((split_id, train_idx, test_idx))
            split_id += 1
    return splits


def _group_shuffle_splits(
    split_base: pd.DataFrame,
    *,
    n_splits: int = 50,
    test_size: float = 0.30,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    x = np.zeros(len(split_base))
    y = split_base["y_true"].to_numpy(dtype=int)
    groups = split_base["patientId"].astype(str).to_numpy()
    splits = []
    attempts = 0
    while len(splits) < n_splits and attempts < n_splits * 200:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=RANDOM_STATE + attempts,
        )
        train_idx, test_idx = next(splitter.split(x, y, groups))
        attempts += 1
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        _check_split(y, groups, train_idx, test_idx)
        splits.append((len(splits), train_idx, test_idx))
    if len(splits) != n_splits:
        raise ValueError(f"Only created {len(splits)} GroupShuffleSplit folds.")
    return splits


def _check_split(
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    if len(np.unique(y[train_idx])) < 2:
        raise ValueError("Train split has one class.")
    if set(groups[train_idx]).intersection(set(groups[test_idx])):
        raise RuntimeError("Patient leakage detected.")


def _run_protocol(
    *,
    run_name: str,
    splits: list[tuple[int, np.ndarray, np.ndarray]],
    pool_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    split_base: pd.DataFrame,
) -> None:
    output_dir = OUTPUT_ROOT / f"monochromatic_biopsy_m0_m3_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    fold_metrics = []
    for split_id, train_idx, test_idx in splits:
        train_specimens = set(split_base.iloc[train_idx]["specimenId"].astype(str))
        test_specimens = set(split_base.iloc[test_idx]["specimenId"].astype(str))
        train_df = target_df[target_df["specimenId"].astype(str).isin(train_specimens)]
        test_df = target_df[target_df["specimenId"].astype(str).isin(test_specimens)]
        fold_predictions, _ = _run_fold(
            fold_id=split_id,
            patient_id=f"{run_name}_{split_id}",
            train_df=train_df.copy(),
            test_df=test_df.copy(),
            feature_table=feature_table,
        )
        fold_prediction_df = pd.concat(fold_predictions, ignore_index=True)
        fold_prediction_df["cv_protocol"] = run_name
        predictions.append(fold_prediction_df)
        fold_metrics.extend(_fold_metrics(fold_prediction_df))

    predictions_df = pd.concat(predictions, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics)
    global_summary_df = _global_summary(predictions_df)
    repeated_summary_df = _repeated_summary(fold_metrics_df)

    feature_table.to_csv(output_dir / "feature_table.csv", index=False)
    predictions_df.to_csv(output_dir / "cv_predictions.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    global_summary_df.to_csv(output_dir / "cv_global_summary.csv", index=False)
    repeated_summary_df.to_csv(output_dir / "cv_repeated_summary.csv", index=False)
    _save_roc_plot(predictions_df, output_dir / "cv_roc_m0_m3.png")
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "input_joblib": str(INPUT_JOBLIB),
                "cv_protocol": run_name,
                "splits": len(splits),
                "pool_rows": int(len(pool_df)),
                "target_rows": int(len(target_df)),
                "target_patients": int(target_df["patientId"].astype(str).nunique()),
                "target_specimens": int(
                    target_df["specimenId"].astype(str).nunique()
                ),
                "global_summary": global_summary_df.to_dict(orient="records"),
                "repeated_summary": repeated_summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{run_name}")
    print(global_summary_df.to_string(index=False))
    print("\nMean across folds")
    print(repeated_summary_df.to_string(index=False))
    print(f"OUT {output_dir}")


def _fold_metrics(predictions_df: pd.DataFrame) -> list[dict]:
    rows = []
    for model_name, group_df in predictions_df.groupby("model_name", sort=False):
        y_true = group_df["y_true"].to_numpy(dtype=int)
        score = group_df["p_cancer"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        rows.append(
            {
                "model_name": str(model_name),
                "fold_id": int(group_df["fold_id"].iloc[0]),
                "test_patients": int(group_df["patientId"].astype(str).nunique()),
                "test_specimens": int(len(group_df)),
                "benign_specimens": int((y_true == 0).sum()),
                "cancer_specimens": int((y_true == 1).sum()),
                "roc_auc": float(roc_auc_score(y_true, score)),
                "pr_auc": float(average_precision_score(y_true, score)),
                "target_sensitivity": float(
                    _ratio(
                        int(((y_true == 1) & (group_df["y_pred_target"] == 1)).sum()),
                        int((y_true == 1).sum()),
                    )
                ),
                "target_specificity": float(
                    _ratio(
                        int(((y_true == 0) & (group_df["y_pred_target"] == 0)).sum()),
                        int((y_true == 0).sum()),
                    )
                ),
                "youden_sensitivity": float(
                    _ratio(
                        int(((y_true == 1) & (group_df["y_pred_youden"] == 1)).sum()),
                        int((y_true == 1).sum()),
                    )
                ),
                "youden_specificity": float(
                    _ratio(
                        int(((y_true == 0) & (group_df["y_pred_youden"] == 0)).sum()),
                        int((y_true == 0).sum()),
                    )
                ),
            }
        )
    return rows


def _repeated_summary(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group_df in fold_metrics_df.groupby("model_name", sort=False):
        rows.append(
            {
                "model_name": str(model_name),
                "folds": int(len(group_df)),
                "roc_auc_mean": float(group_df["roc_auc"].mean()),
                "roc_auc_std": float(group_df["roc_auc"].std(ddof=0)),
                "pr_auc_mean": float(group_df["pr_auc"].mean()),
                "pr_auc_std": float(group_df["pr_auc"].std(ddof=0)),
                "target_sensitivity_mean": float(
                    group_df["target_sensitivity"].mean()
                ),
                "target_specificity_mean": float(
                    group_df["target_specificity"].mean()
                ),
                "youden_sensitivity_mean": float(
                    group_df["youden_sensitivity"].mean()
                ),
                "youden_specificity_mean": float(
                    group_df["youden_specificity"].mean()
                ),
                "test_patients_mean": float(group_df["test_patients"].mean()),
                "test_specimens_mean": float(group_df["test_specimens"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


if __name__ == "__main__":
    main()
