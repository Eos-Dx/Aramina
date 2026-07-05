"""Train-on-all LR1 profile score plus Kubitskii symmetry SVM-poly2 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from aramis_ka_symmetry_only_experiment import (
    FEATURE_COLUMNS,
    INPUT_JOBLIB,
    OUTPUT_DIR as KA_OUTPUT_DIR,
    RAW_PEAK_THRESHOLD,
    TARGET_SENSITIVITY,
    _build_patient_feature_table,
    _load_pool,
    _ratio,
    _save_roc_plot,
    _score_model,
    _summary_row,
    _threshold_for_sensitivity,
)
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT
    / "examples"
    / "outputs"
    / "modeling_ka_symmetry"
    / "profile_lr1_plus_ka_symmetry_svm_poly2_train_all"
)

LR1_SCORE_COLUMN = "lr1_patient_mean_p_cancer"
FINAL_FEATURE_COLUMNS = [LR1_SCORE_COLUMN, *FEATURE_COLUMNS]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_df = _load_pool(INPUT_JOBLIB)
    feature_df_all = _build_patient_feature_table(pool_df)
    feature_df = feature_df_all[
        feature_df_all["mean_peak_value_raw"].ge(RAW_PEAK_THRESHOLD)
    ].copy()
    feature_df.reset_index(drop=True, inplace=True)

    lr1_patient_scores = _fit_lr1_profile_train_all(pool_df, feature_df)
    model_df = feature_df.merge(lr1_patient_scores, on="patient_id", how="inner")
    if model_df["label"].nunique() < 2:
        raise ValueError("Final model table has one class.")

    predictions_df, summary_df = _train_all_models(model_df)
    model_df.to_csv(OUTPUT_DIR / "final_model_feature_table.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "train_all_predictions.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "train_all_summary.csv", index=False)
    _save_roc_plot(
        predictions_df,
        OUTPUT_DIR / "train_all_roc_lr1_plus_ka_symmetry.png",
        title="Train-on-all: LR1 profile mean + KA symmetry",
        score_column="score",
    )
    _plot_feature_importance(model_df, OUTPUT_DIR / "final_svm_poly2_feature_values.png")
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "input_joblib": str(INPUT_JOBLIB),
                "ka_symmetry_output_dir": str(KA_OUTPUT_DIR),
                "output_dir": str(OUTPUT_DIR),
                "protocol": "train-on-all feature-discovery prototype",
                "lr1": "LogisticRegression L2 on measurement radial_profile_data; averaged per patient",
                "final_model": "SVM polynomial degree 2 on LR1 mean score plus Kubitskii symmetry features",
                "raw_peak_threshold": RAW_PEAK_THRESHOLD,
                "target_sensitivity": TARGET_SENSITIVITY,
                "patients": int(model_df["patient_id"].nunique()),
                "label_counts": model_df["label_name"].value_counts().to_dict(),
                "final_feature_columns": FINAL_FEATURE_COLUMNS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary_df.to_string(index=False))
    print(f"\nOUT {OUTPUT_DIR}")


def _fit_lr1_profile_train_all(
    pool_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    label_map = feature_df.set_index("patient_id")["label"].astype(int).to_dict()
    measurement_df = pool_df[pool_df["patientId"].astype(str).isin(label_map)].copy()
    measurement_df["patient_id"] = measurement_df["patientId"].astype(str)
    measurement_df["label"] = measurement_df["patient_id"].map(label_map).astype(int)

    x = _profile_matrix(measurement_df)
    y = measurement_df["label"].to_numpy(dtype=int)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x, y)
    measurement_df["lr1_measurement_p_cancer"] = model.predict_proba(x)[:, 1]
    return (
        measurement_df.groupby("patient_id", as_index=False)
        .agg(
            lr1_patient_mean_p_cancer=("lr1_measurement_p_cancer", "mean"),
            lr1_patient_median_p_cancer=("lr1_measurement_p_cancer", "median"),
            lr1_patient_measurements=("lr1_measurement_p_cancer", "size"),
        )
        .reset_index(drop=True)
    )


def _profile_matrix(df: pd.DataFrame) -> np.ndarray:
    matrix = np.vstack(
        [np.asarray(value, dtype=float).ravel() for value in df["radial_profile_data"]]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Profile matrix contains non-finite values.")
    return matrix


def _train_all_models(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []

    y = model_df["label"].to_numpy(dtype=int)
    lr1_score = model_df[LR1_SCORE_COLUMN].to_numpy(dtype=float)
    rows.append(_score_summary(model_df, "LR1_mean_only", lr1_score))
    predictions.append(_prediction_frame(model_df, "LR1_mean_only", lr1_score))

    ka_model = _svm_poly2()
    ka_model.fit(model_df[FEATURE_COLUMNS], y)
    ka_score = _score_model(ka_model, model_df[FEATURE_COLUMNS])
    rows.append(_summary_row(model_df, "KA_SVM_poly2_only", "train_all", ka_score))
    predictions.append(_prediction_frame(model_df, "KA_SVM_poly2_only", ka_score))

    final_model = _svm_poly2()
    final_model.fit(model_df[FINAL_FEATURE_COLUMNS], y)
    final_score = _score_model(final_model, model_df[FINAL_FEATURE_COLUMNS])
    rows.append(
        _summary_row(
            model_df,
            "LR1_mean_plus_KA_SVM_poly2",
            "train_all",
            final_score,
        )
    )
    predictions.append(
        _prediction_frame(model_df, "LR1_mean_plus_KA_SVM_poly2", final_score)
    )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(rows)


def _svm_poly2() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    C=1.0,
                    class_weight="balanced",
                    degree=2,
                    gamma="scale",
                    kernel="poly",
                ),
            ),
        ]
    )


def _score_summary(
    df: pd.DataFrame,
    model_name: str,
    score: np.ndarray,
) -> dict:
    y = df["label"].to_numpy(dtype=int)
    threshold = _threshold_for_sensitivity(y, score, TARGET_SENSITIVITY)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "protocol": "train_all",
        "model_name": model_name,
        "patients": int(df["patient_id"].nunique()),
        "rows": int(len(df)),
        "cancer": int((y == 1).sum()),
        "non_cancer": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "balanced_accuracy_at_95sens": float(balanced_accuracy_score(y, pred)),
        "threshold_95sens": float(threshold),
        "sensitivity_95": _ratio(tp, tp + fn),
        "specificity_95": _ratio(tn, tn + fp),
        "ppv_95": _ratio(tp, tp + fp),
        "npv_95": _ratio(tn, tn + fn),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def _prediction_frame(
    df: pd.DataFrame,
    model_name: str,
    score: np.ndarray,
) -> pd.DataFrame:
    out = df[["patient_id", "label", "label_name", LR1_SCORE_COLUMN]].copy()
    out["model_name"] = model_name
    out["split_id"] = "train_all"
    out["score"] = np.asarray(score, dtype=float)
    return out


def _plot_feature_importance(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y0 = df.loc[df["label"].eq(0), LR1_SCORE_COLUMN]
    y1 = df.loc[df["label"].eq(1), LR1_SCORE_COLUMN]
    ax.hist(y0, bins=18, alpha=0.65, label="NON-CANCER", color="#7b61ff")
    ax.hist(y1, bins=18, alpha=0.65, label="CANCER", color="#d62728")
    ax.set_xlabel("LR1 patient mean p_cancer")
    ax.set_ylabel("patients")
    ax.set_title("LR1 profile score distribution before final SVM")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
