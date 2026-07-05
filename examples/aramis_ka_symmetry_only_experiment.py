"""Kubitskii-style symmetry-only Aramis experiment.

This is a research-draft feature-discovery run. It mirrors the KA notebook:
patient-level LEFT/RIGHT symmetry features, train-on-all ROC, and repeated
patient-level 70/30 splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    / "modeling_ka_symmetry"
    / "monochromatic_patient_level_symmetry_only"
)

TARGET_SENSITIVITY = 0.95
RAW_PEAK_THRESHOLD = 0.60
RANDOM_STATE = 42
N_SPLITS = 50
TEST_SIZE = 0.30

CANCER_STATUSES = {"CANCER", "ATYPICAL", "PRE_CANCEROUS", "PRE-CANCEROUS"}
NONCANCER_STATUSES = {"BENIGN", "NORMAL"}
FEATURE_COLUMNS = [
    "weightedrms1",
    "sigma_l1",
    "sigma_r1",
    "mahalanobis1",
    "weightedrms2",
    "sigma_l2",
    "sigma_r2",
    "mahalanobis2",
    "peak14_intensity",
    "mean_peak_value_raw",
    "wasserstein_distance_muLR",
    "cosine_distance_full_q2",
    "wasserstein_distance_full_q2",
    "meanrms1",
    "meanrms2",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_df = _load_pool(INPUT_JOBLIB)
    feature_df_all = _build_patient_feature_table(pool_df)
    feature_df = feature_df_all[
        feature_df_all["mean_peak_value_raw"].ge(RAW_PEAK_THRESHOLD)
    ].copy()
    feature_df.reset_index(drop=True, inplace=True)
    if feature_df["label"].nunique() < 2:
        raise ValueError("Raw peak filter left only one class.")

    train_all_predictions, train_all_summary = _train_all(feature_df)
    split_predictions, split_summary = _repeated_splits(feature_df)

    feature_df_all.to_csv(OUTPUT_DIR / "ka_symmetry_features_all_patients.csv", index=False)
    feature_df.to_csv(OUTPUT_DIR / "ka_symmetry_features_filtered.csv", index=False)
    train_all_predictions.to_csv(OUTPUT_DIR / "train_all_predictions.csv", index=False)
    train_all_summary.to_csv(OUTPUT_DIR / "train_all_summary.csv", index=False)
    split_predictions.to_csv(OUTPUT_DIR / "repeated_70_30_predictions.csv", index=False)
    split_summary.to_csv(OUTPUT_DIR / "repeated_70_30_summary.csv", index=False)
    _save_roc_plot(
        train_all_predictions,
        OUTPUT_DIR / "train_all_roc_symmetry_only.png",
        title="KA-style symmetry-only train-on-all ROC",
        score_column="score",
    )
    _save_roc_plot(
        split_predictions,
        OUTPUT_DIR / "repeated_70_30_roc_symmetry_only.png",
        title="KA-style symmetry-only repeated 70/30 ROC",
        score_column="score",
    )
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "input_joblib": str(INPUT_JOBLIB),
                "output_dir": str(OUTPUT_DIR),
                "target": "patient-level CANCER vs NON-CANCER",
                "feature_source": "Kubitskii KA notebook symmetry feature set",
                "raw_peak_threshold": RAW_PEAK_THRESHOLD,
                "target_sensitivity": TARGET_SENSITIVITY,
                "patients_before_peak_filter": int(len(feature_df_all)),
                "patients_after_peak_filter": int(len(feature_df)),
                "label_counts_after_peak_filter": feature_df["label_name"]
                .value_counts()
                .to_dict(),
                "feature_columns": FEATURE_COLUMNS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Train-on-all")
    print(train_all_summary.to_string(index=False))
    print("\nRepeated patient 70/30")
    print(split_summary.to_string(index=False))
    print(f"\nOUT {OUTPUT_DIR}")


def _load_pool(path: Path) -> pd.DataFrame:
    artifact = joblib.load(path)
    df = artifact["dataframe"].copy() if isinstance(artifact, dict) else artifact.copy()
    status = df["specimen_status"].fillna("").astype(str).str.upper().str.strip()
    df["patient_symmetry_status"] = np.select(
        [status.isin(CANCER_STATUSES), status.isin(NONCANCER_STATUSES)],
        ["CANCER", "NON-CANCER"],
        default="NA",
    )
    return df


def _build_patient_feature_table(pool_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, patient_df in pool_df.groupby("patientId", sort=True):
        if _patient_label(patient_df) is None:
            continue
        sides = patient_df["side"].fillna("").astype(str).str.upper()
        if not sides.str.startswith("LEFT").any() or not sides.str.startswith("RIGHT").any():
            continue
        metrics = _patient_lr_mean_metrics(patient_df, q_roi=(7.5, 23.0))
        metrics_cos = _patient_lr_mean_metrics(patient_df, q_roi=(2.0, 23.0))
        q = metrics["q"]
        mu_l = metrics["mu_left"]
        mu_r = metrics["mu_right"]
        std_l = metrics["std_left"]
        std_r = metrics["std_right"]
        mask1 = (q >= 7.0) & (q <= 15.0)
        mask2 = (q >= 15.0) & (q <= 23.0)
        label = _patient_label(patient_df)
        rows.append(
            {
                "patient_id": str(patient_id),
                "label": int(label),
                "label_name": "CANCER" if int(label) == 1 else "NON-CANCER",
                "meanrms1": _mean_rms(mu_l, mu_r, mask1),
                "meanrms2": _mean_rms(mu_l, mu_r, mask2),
                "weightedrms1": _weighted_rms(mu_l, mu_r, std_l, std_r, mask1),
                "sigma_l1": _sigma_rms(std_l, mask1),
                "sigma_r1": _sigma_rms(std_r, mask1),
                "mahalanobis1": _mahalanobis(mu_l, mu_r, std_l, std_r, mask1),
                "weightedrms2": _weighted_rms(mu_l, mu_r, std_l, std_r, mask2),
                "sigma_l2": _sigma_rms(std_l, mask2),
                "sigma_r2": _sigma_rms(std_r, mask2),
                "mahalanobis2": _mahalanobis(mu_l, mu_r, std_l, std_r, mask2),
                "peak14_intensity": _peak14(q, mu_l, mu_r),
                "mean_peak_value_raw": _patient_mean_peak_raw(patient_df),
                "wasserstein_distance_muLR": _wasserstein(q, mu_l, mu_r),
                "cosine_distance_full_q2": _cosine(
                    metrics_cos["mu_left"],
                    metrics_cos["mu_right"],
                ),
                "wasserstein_distance_full_q2": _wasserstein(
                    metrics_cos["q"],
                    metrics_cos["mu_left"],
                    metrics_cos["mu_right"],
                ),
                "n_left": int(metrics["n_left"]),
                "n_right": int(metrics["n_right"]),
                "n_q": int(len(q)),
            }
        )
    return pd.DataFrame(rows)


def _patient_label(patient_df: pd.DataFrame) -> int | None:
    statuses = (
        patient_df["specimen_status"].dropna().astype(str).str.upper().str.strip()
    )
    if statuses.isin(CANCER_STATUSES).any():
        return 1
    if statuses.isin(NONCANCER_STATUSES).any():
        return 0
    return None


def _patient_lr_mean_metrics(
    patient_df: pd.DataFrame,
    *,
    q_roi: tuple[float, float],
) -> dict[str, np.ndarray | int]:
    left_profiles = []
    right_profiles = []
    q_common = None
    for row in patient_df.itertuples(index=False):
        side = str(getattr(row, "side", "")).upper()
        if not side.startswith(("LEFT", "RIGHT")):
            continue
        q = np.asarray(getattr(row, "q_range"), dtype=float).ravel()
        y = np.asarray(getattr(row, "radial_profile_data"), dtype=float).ravel()
        q, y = _apply_roi(q, y, q_roi)
        y = _smooth(y)
        y = _normalize_by_minimum(q, y)
        if q_common is None:
            q_common = q
        y_common = np.interp(q_common, q, y)
        if side.startswith("LEFT"):
            left_profiles.append(y_common)
        else:
            right_profiles.append(y_common)
    if q_common is None or not left_profiles or not right_profiles:
        raise ValueError("Patient needs LEFT and RIGHT curves.")
    x_left = np.vstack(left_profiles)
    x_right = np.vstack(right_profiles)
    return {
        "q": q_common,
        "mu_left": np.mean(x_left, axis=0),
        "mu_right": np.mean(x_right, axis=0),
        "std_left": np.std(x_left, axis=0, ddof=1)
        if len(left_profiles) >= 2
        else np.zeros(x_left.shape[1]),
        "std_right": np.std(x_right, axis=0, ddof=1)
        if len(right_profiles) >= 2
        else np.zeros(x_right.shape[1]),
        "n_left": len(left_profiles),
        "n_right": len(right_profiles),
    }


def _apply_roi(
    q: np.ndarray,
    y: np.ndarray,
    q_roi: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    mask = (q >= float(q_roi[0])) & (q <= float(q_roi[1]))
    if int(mask.sum()) < 5:
        return q, y
    return q[mask], y[mask]


def _smooth(y: np.ndarray) -> np.ndarray:
    if y.size < 7:
        return y
    window = min(11, y.size if y.size % 2 else y.size - 1)
    if window < 5:
        return y
    return savgol_filter(y, window_length=window, polyorder=min(3, window - 2))


def _normalize_by_minimum(
    q: np.ndarray,
    y: np.ndarray,
    *,
    q0: float = 6.7,
    halfwidth: float = 0.25,
) -> np.ndarray:
    mask = (q >= q0 - halfwidth) & (q <= q0 + halfwidth) & np.isfinite(y)
    if int(mask.sum()) < 2:
        baseline = float(np.nanpercentile(y, 5))
    else:
        baseline = float(np.nanpercentile(y[mask], 5))
    if not np.isfinite(baseline) or abs(baseline) < 1e-12:
        baseline = 1.0
    return y / baseline


def _mean_rms(mu_l: np.ndarray, mu_r: np.ndarray, mask: np.ndarray) -> float:
    diff = np.asarray(mu_l - mu_r, dtype=float)
    good = mask & np.isfinite(diff)
    return float(np.sqrt(np.mean(diff[good] ** 2))) if int(good.sum()) >= 5 else np.nan


def _weighted_rms(
    mu_l: np.ndarray,
    mu_r: np.ndarray,
    std_l: np.ndarray,
    std_r: np.ndarray,
    mask: np.ndarray,
) -> float:
    diff = np.asarray(mu_l - mu_r, dtype=float)
    var = np.asarray(std_l**2 + std_r**2, dtype=float)
    good = mask & np.isfinite(diff) & np.isfinite(var)
    if int(good.sum()) < 5:
        return np.nan
    floor = float(np.nanpercentile(var[good], 5))
    weight = 1.0 / np.maximum(var[good], floor + 1e-12)
    return float(np.sqrt(np.sum(weight * diff[good] ** 2) / np.sum(weight)))


def _mahalanobis(
    mu_l: np.ndarray,
    mu_r: np.ndarray,
    std_l: np.ndarray,
    std_r: np.ndarray,
    mask: np.ndarray,
) -> float:
    diff = np.asarray(mu_l - mu_r, dtype=float)
    var = np.asarray(std_l**2 + std_r**2, dtype=float)
    good = mask & np.isfinite(diff) & np.isfinite(var)
    if int(good.sum()) < 5:
        return np.nan
    return float(np.sqrt(np.sum(diff[good] ** 2 / (var[good] + 1e-12))))


def _sigma_rms(std: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(std)
    return float(np.sqrt(np.mean(std[good] ** 2))) if int(good.sum()) >= 5 else np.nan


def _peak14(q: np.ndarray, mu_l: np.ndarray, mu_r: np.ndarray) -> float:
    y = 0.5 * (mu_l + mu_r)
    mask = (q >= 13.5) & (q <= 14.5) & np.isfinite(y)
    return float(np.nanmax(y[mask])) if int(mask.sum()) >= 3 else np.nan


def _patient_mean_peak_raw(patient_df: pd.DataFrame) -> float:
    values = []
    for row in patient_df.itertuples(index=False):
        q = np.asarray(getattr(row, "q_range"), dtype=float).ravel()
        if hasattr(row, "radial_profile_data_raw"):
            y = np.asarray(getattr(row, "radial_profile_data_raw"), dtype=float).ravel()
        else:
            y = np.asarray(getattr(row, "radial_profile_data"), dtype=float).ravel()
        mask = (q >= 13.0) & (q <= 14.8) & np.isfinite(y)
        if int(mask.sum()) >= 3:
            values.append(float(np.nanmax(y[mask])))
    return float(np.mean(values)) if values else np.nan


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 5:
        return np.nan
    av = np.asarray(a[good], dtype=float)
    bv = np.asarray(b[good], dtype=float)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - np.clip(np.dot(av, bv) / denom, -1.0, 1.0))


def _wasserstein(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(q) & np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 5:
        return np.nan
    qv = q[good]
    av = np.clip(a[good], 0.0, None)
    bv = np.clip(b[good], 0.0, None)
    if float(av.sum()) <= 1e-12 or float(bv.sum()) <= 1e-12:
        return np.nan
    order = np.argsort(qv)
    qv = qv[order]
    av = av[order] / float(av.sum())
    bv = bv[order] / float(bv.sum())
    return float(np.sum(np.abs(np.cumsum(av)[:-1] - np.cumsum(bv)[:-1]) * np.diff(qv)))


def _train_all(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    rows = []
    x = feature_df[FEATURE_COLUMNS]
    y = feature_df["label"].to_numpy(dtype=int)
    for model_name, model in _models().items():
        model.fit(x, y)
        score = _score_model(model, x)
        predictions.append(_prediction_frame(feature_df, model_name, "train_all", score))
        rows.append(_summary_row(feature_df, model_name, "train_all", score))
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(rows)


def _repeated_splits(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = feature_df[FEATURE_COLUMNS]
    y = feature_df["label"].to_numpy(dtype=int)
    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    predictions = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        train_x = x.iloc[train_idx]
        train_y = y[train_idx]
        test_x = x.iloc[test_idx]
        test_df = feature_df.iloc[test_idx].copy()
        for model_name, model in _models().items():
            model.fit(train_x, train_y)
            score = _score_model(model, test_x)
            frame = _prediction_frame(test_df, model_name, f"split_{split_id:03d}", score)
            predictions.append(frame)
    predictions_df = pd.concat(predictions, ignore_index=True)
    rows = []
    for model_name, model_df in predictions_df.groupby("model_name", sort=False):
        rows.append(_summary_row(model_df, model_name, "repeated_70_30", model_df["score"]))
    return predictions_df, pd.DataFrame(rows)


def _models() -> dict[str, Pipeline]:
    models = {
        "LR_L2": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    }
    for degree in (2, 3, 4):
        models[f"SVM_poly{degree}"] = SVC(
            C=1.0,
            class_weight="balanced",
            degree=degree,
            gamma="scale",
            kernel="poly",
        )
    return {
        name: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }


def _score_model(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def _prediction_frame(
    feature_df: pd.DataFrame,
    model_name: str,
    split_id: str,
    score: np.ndarray,
) -> pd.DataFrame:
    out = feature_df[["patient_id", "label", "label_name"]].copy()
    out["model_name"] = model_name
    out["split_id"] = split_id
    out["score"] = np.asarray(score, dtype=float)
    return out


def _summary_row(
    df: pd.DataFrame,
    model_name: str,
    protocol: str,
    score: pd.Series | np.ndarray,
) -> dict:
    y = df["label"].to_numpy(dtype=int)
    s = np.asarray(score, dtype=float)
    threshold = _threshold_for_sensitivity(y, s, TARGET_SENSITIVITY)
    y_pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "protocol": protocol,
        "model_name": model_name,
        "patients": int(df["patient_id"].nunique()),
        "rows": int(len(df)),
        "cancer": int((y == 1).sum()),
        "non_cancer": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "balanced_accuracy_at_95sens": float(balanced_accuracy_score(y, y_pred)),
        "threshold_95sens": float(threshold),
        "sensitivity_95": _ratio(tp, tp + fn),
        "specificity_95": _ratio(tn, tn + fp),
        "ppv_95": _ratio(tp, tp + fp),
        "npv_95": _ratio(tn, tn + fn),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def _threshold_for_sensitivity(
    y_true: np.ndarray,
    score: np.ndarray,
    target_sensitivity: float,
) -> float:
    positive_scores = np.sort(np.asarray(score, dtype=float)[np.asarray(y_true) == 1])
    n_tp = int(np.ceil(float(target_sensitivity) * positive_scores.size))
    return float(positive_scores[max(0, positive_scores.size - n_tp)])


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _save_roc_plot(
    predictions_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    score_column: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for model_name, group_df in predictions_df.groupby("model_name", sort=False):
        y = group_df["label"].to_numpy(dtype=int)
        score = group_df[score_column].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} ({roc_auc_score(y, score):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / sensitivity")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
