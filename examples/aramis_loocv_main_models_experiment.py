"""Patient-level leave-one-out comparison for Aramis M0-M3 research models."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from aramis.modeling import (
    build_fusion_feature_table,
    compute_binary_thresholds,
    default_fusion_feature_sets,
)
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
INPUT_JOBLIB = (
    ROOT
    / "examples"
    / "outputs"
    / "real_h5_yaml_validation"
    / "aramis_monochromatic_metadata_pool_max_v0_1.joblib"
)
OUTPUT_DIR = (
    ROOT
    / "examples"
    / "outputs"
    / "modeling_loocv"
    / "monochromatic_biopsy_m0_m3_patient_loocv"
)

LABEL = {"BENIGN": 0, "CANCER": 1}
LOGREG_C = 1.0
TARGET_SENSITIVITY = 0.95
RANDOM_STATE = 42


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pool_df = _load_pool(INPUT_JOBLIB)
    target_df = _target_training_rows(pool_df)
    feature_table = _coerce_features(build_fusion_feature_table(target_df, pool_df))

    predictions = []
    metrics = []
    patients = sorted(target_df["patientId"].astype(str).unique())
    for fold_id, patient_id in enumerate(patients):
        train_df = target_df[target_df["patientId"].astype(str) != patient_id].copy()
        test_df = target_df[target_df["patientId"].astype(str) == patient_id].copy()
        if train_df["product_status_group"].nunique() < 2:
            continue

        fold_predictions, fold_metrics = _run_fold(
            fold_id=fold_id,
            patient_id=patient_id,
            train_df=train_df,
            test_df=test_df,
            feature_table=feature_table,
        )
        predictions.extend(fold_predictions)
        metrics.extend(fold_metrics)

    predictions_df = pd.concat(predictions, ignore_index=True)
    fold_metrics_df = pd.DataFrame(metrics)
    summary_df = _global_summary(predictions_df)
    run_summary = _run_summary(pool_df, target_df, feature_table, summary_df)

    feature_table.to_csv(OUTPUT_DIR / "feature_table.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "loocv_predictions.csv", index=False)
    fold_metrics_df.to_csv(OUTPUT_DIR / "loocv_fold_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "loocv_summary.csv", index=False)
    _save_roc_plot(predictions_df, OUTPUT_DIR / "loocv_roc_m0_m3.png")
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )

    print(summary_df.to_string(index=False))
    print(f"\nOUT {OUTPUT_DIR}")


def _load_pool(path: Path) -> pd.DataFrame:
    artifact = joblib.load(path)
    df = artifact["dataframe"].copy() if isinstance(artifact, dict) else artifact.copy()
    status = df["specimen_status"].fillna("NA").astype(str).str.upper().str.strip()
    df["product_status_group"] = np.select(
        [
            status.isin(["CANCER", "ATYPICAL", "PRE_CANCEROUS"]),
            status.eq("BENIGN"),
            status.eq("NORMAL"),
        ],
        ["CANCER", "BENIGN", "NORMAL"],
        default="EXCLUDE",
    )
    return df


def _target_training_rows(pool_df: pd.DataFrame) -> pd.DataFrame:
    target_df = pool_df[
        (pool_df["biopsy"] == True)  # noqa: E712
        & pool_df["product_status_group"].isin(["BENIGN", "CANCER"])
    ].copy()
    target_df.reset_index(drop=True, inplace=True)
    if (target_df.groupby("specimenId")["product_status_group"].nunique() > 1).any():
        raise ValueError("Mixed product labels inside one specimenId.")
    if (target_df.groupby("specimenId")["patientId"].nunique() > 1).any():
        raise ValueError("Mixed patients inside one specimenId.")
    return target_df


def _run_fold(
    *,
    fold_id: int,
    patient_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[dict]]:
    y_train = train_df["product_status_group"].map(LABEL).astype(int).to_numpy()
    lr1 = _logistic_model(seed=RANDOM_STATE + fold_id)
    lr1.fit(_profile_matrix(train_df), y_train)

    train_specimens = _aggregate_specimen_scores(
        train_df,
        lr1.predict_proba(_profile_matrix(train_df))[:, 1],
    )
    test_specimens = _aggregate_specimen_scores(
        test_df,
        lr1.predict_proba(_profile_matrix(test_df))[:, 1],
    )

    train_specimens = _merge_features(train_specimens, feature_table)
    test_specimens = _merge_features(test_specimens, feature_table)
    train_specimens["logit_p_cancer_one_to_many"] = _safe_logit(
        train_specimens["p_cancer_one_to_many"]
    )
    test_specimens["logit_p_cancer_one_to_many"] = _safe_logit(
        test_specimens["p_cancer_one_to_many"]
    )
    train_specimens = _coerce_features(train_specimens)
    test_specimens = _coerce_features(test_specimens)

    predictions = []
    metrics = []
    y_train_specimen = train_specimens["y_true"].to_numpy(dtype=int)
    feature_sets = default_fusion_feature_sets()
    for model_name, columns in feature_sets.items():
        model = _logistic_model(seed=10_000 + fold_id)
        model.fit(train_specimens[columns].to_numpy(dtype=float), y_train_specimen)
        train_score = model.predict_proba(
            train_specimens[columns].to_numpy(dtype=float)
        )[:, 1]
        test_score = model.predict_proba(test_specimens[columns].to_numpy(dtype=float))[
            :, 1
        ]
        thresholds = compute_binary_thresholds(
            y_train_specimen,
            train_score,
            target_sensitivity=TARGET_SENSITIVITY,
        )
        predictions.append(
            _prediction_frame(
                model_name=model_name,
                fold_id=fold_id,
                patient_id=patient_id,
                test_df=test_specimens,
                score=test_score,
                thresholds=thresholds,
            )
        )
        metrics.append(
            _fold_metric_row(
                model_name=model_name,
                fold_id=fold_id,
                patient_id=patient_id,
                train_df=train_specimens,
                test_df=test_specimens,
                score=test_score,
                thresholds=thresholds,
            )
        )
    return predictions, metrics


def _logistic_model(*, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=LOGREG_C,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=int(seed),
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _profile_matrix(df: pd.DataFrame) -> np.ndarray:
    matrix = np.vstack(
        [np.asarray(value, dtype=float).ravel() for value in df["radial_profile_data"]]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Profile matrix contains non-finite values.")
    return matrix


def _aggregate_specimen_scores(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    temp = df[["patientId", "specimenId", "product_status_group"]].copy()
    temp["p_cancer_one_to_many"] = np.asarray(score, dtype=float)
    rows = []
    for specimen_id, group_df in temp.groupby("specimenId", sort=True):
        labels = group_df["product_status_group"].astype(str).unique().tolist()
        patients = group_df["patientId"].astype(str).unique().tolist()
        if len(labels) != 1 or len(patients) != 1:
            raise ValueError((specimen_id, labels, patients))
        rows.append(
            {
                "patientId": patients[0],
                "specimenId": str(specimen_id),
                "product_status_group": labels[0],
                "y_true": LABEL[labels[0]],
                "p_cancer_one_to_many": float(group_df["p_cancer_one_to_many"].mean()),
                "n_measurements": int(len(group_df)),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _merge_features(specimens: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    drop = ["product_status_group", "y_true"]
    return specimens.merge(
        feature_table.drop(columns=drop, errors="ignore"),
        on=["patientId", "specimenId"],
        how="left",
    )


def _coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for columns in default_fusion_feature_sets().values():
        for column in columns:
            if column not in out:
                continue
            out[column] = (
                pd.to_numeric(out[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
    return out


def _safe_logit(values: pd.Series) -> np.ndarray:
    probabilities = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(probabilities / (1.0 - probabilities))


def _prediction_frame(
    *,
    model_name: str,
    fold_id: int,
    patient_id: str,
    test_df: pd.DataFrame,
    score: np.ndarray,
    thresholds: dict,
) -> pd.DataFrame:
    out = test_df[
        [
            "patientId",
            "specimenId",
            "product_status_group",
            "y_true",
            "p_cancer_one_to_many",
            "symmetry_available",
            "age_available",
            "bmi_available",
        ]
    ].copy()
    out["model_name"] = model_name
    out["fold_id"] = int(fold_id)
    out["held_out_patientId"] = str(patient_id)
    out["p_cancer"] = np.asarray(score, dtype=float)
    out["threshold_target"] = float(thresholds["threshold_target"])
    out["threshold_youden"] = float(thresholds["threshold_youden"])
    out["y_pred_target"] = (out["p_cancer"] >= out["threshold_target"]).astype(int)
    out["y_pred_youden"] = (out["p_cancer"] >= out["threshold_youden"]).astype(int)
    return out


def _fold_metric_row(
    *,
    model_name: str,
    fold_id: int,
    patient_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score: np.ndarray,
    thresholds: dict,
) -> dict:
    return {
        "model_name": str(model_name),
        "fold_id": int(fold_id),
        "held_out_patientId": str(patient_id),
        "train_patients": int(train_df["patientId"].astype(str).nunique()),
        "test_patients": int(test_df["patientId"].astype(str).nunique()),
        "train_specimens": int(len(train_df)),
        "test_specimens": int(len(test_df)),
        "test_benign_specimens": int((test_df["y_true"].to_numpy(dtype=int) == 0).sum()),
        "test_cancer_specimens": int((test_df["y_true"].to_numpy(dtype=int) == 1).sum()),
        "symmetry_available_test": int(test_df["symmetry_available"].sum()),
        "threshold_target": float(thresholds["threshold_target"]),
        "threshold_youden": float(thresholds["threshold_youden"]),
        "p_cancer_mean": float(np.mean(score)),
    }


def _global_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group_df in predictions_df.groupby("model_name", sort=False):
        y_true = group_df["y_true"].to_numpy(dtype=int)
        score = group_df["p_cancer"].to_numpy(dtype=float)
        target_metrics = _classification_metrics(
            y_true,
            group_df["y_pred_target"].to_numpy(dtype=int),
        )
        youden_metrics = _classification_metrics(
            y_true,
            group_df["y_pred_youden"].to_numpy(dtype=int),
        )
        pooled95_threshold = _threshold_for_sensitivity(
            y_true,
            score,
            target_sensitivity=TARGET_SENSITIVITY,
        )
        pooled95_metrics = _classification_metrics(
            y_true,
            (score >= pooled95_threshold).astype(int),
        )
        rows.append(
            {
                "model_name": model_name,
                "held_out_patients": int(group_df["held_out_patientId"].nunique()),
                "test_specimens": int(len(group_df)),
                "benign_specimens": int((y_true == 0).sum()),
                "cancer_specimens": int((y_true == 1).sum()),
                "roc_auc": float(roc_auc_score(y_true, score)),
                "pr_auc": float(average_precision_score(y_true, score)),
                "target_sensitivity": target_metrics["sensitivity"],
                "target_specificity": target_metrics["specificity"],
                "target_ppv": target_metrics["ppv"],
                "target_npv": target_metrics["npv"],
                "target_false_negatives": target_metrics["fn"],
                "target_false_positives": target_metrics["fp"],
                "pooled95_threshold": float(pooled95_threshold),
                "pooled95_sensitivity": pooled95_metrics["sensitivity"],
                "pooled95_specificity": pooled95_metrics["specificity"],
                "pooled95_ppv": pooled95_metrics["ppv"],
                "pooled95_npv": pooled95_metrics["npv"],
                "pooled95_false_negatives": pooled95_metrics["fn"],
                "pooled95_false_positives": pooled95_metrics["fp"],
                "youden_sensitivity": youden_metrics["sensitivity"],
                "youden_specificity": youden_metrics["specificity"],
                "youden_ppv": youden_metrics["ppv"],
                "youden_npv": youden_metrics["npv"],
                "youden_false_negatives": youden_metrics["fn"],
                "youden_false_positives": youden_metrics["fp"],
            }
        )
    return pd.DataFrame(rows)


def _save_roc_plot(predictions_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for model_name, group_df in predictions_df.groupby("model_name", sort=False):
        y_true = group_df["y_true"].to_numpy(dtype=int)
        score = group_df["p_cancer"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{model_name} ({roc_auc_score(y_true, score):.3f})")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / sensitivity")
    ax.set_title("Patient-level LOOCV ROC, M0-M3")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _threshold_for_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_sensitivity: float,
) -> float:
    positives = np.sort(np.asarray(y_score, dtype=float)[np.asarray(y_true) == 1])
    if positives.size == 0:
        raise ValueError("Cannot set target sensitivity without positive samples.")
    min_true_positives = int(np.ceil(float(target_sensitivity) * positives.size))
    threshold_index = max(0, positives.size - min_true_positives)
    return float(positives[threshold_index])


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _run_summary(
    pool_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict:
    return {
        "input_joblib": str(INPUT_JOBLIB),
        "output_dir": str(OUTPUT_DIR),
        "target_rule": (
            "biopsy=True and product_status_group in BENIGN/CANCER; "
            "NORMAL remains in symmetry pool only"
        ),
        "validation": "patient-level leave-one-out; held-out patient only in test",
        "stage1": "LogisticRegression L2 C=1.0 on radial_profile_data",
        "stage2": "LogisticRegression L2 C=1.0 on M0-M3 feature sets",
        "threshold": "per-fold train-only threshold for sensitivity >= 0.95",
        "pool_rows": int(len(pool_df)),
        "target_rows": int(len(target_df)),
        "target_patients": int(target_df["patientId"].astype(str).nunique()),
        "target_specimens": int(target_df["specimenId"].astype(str).nunique()),
        "target_measurement_labels": target_df["product_status_group"]
        .value_counts()
        .to_dict(),
        "target_specimen_labels": feature_table["product_status_group"]
        .value_counts()
        .to_dict(),
        "feature_sets": default_fusion_feature_sets(),
        "summary": summary_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    main()
