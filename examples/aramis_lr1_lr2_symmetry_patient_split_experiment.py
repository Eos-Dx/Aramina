"""Patient-safe LR1/LR2 symmetry experiment for Aramis research draft.

Target cohort:
    biopsy=True, BENIGN vs CANCER/ATYPICAL/PRE_CANCEROUS.

NORMAL is excluded from target/training, but remains in the source pool used to
calculate contralateral symmetry features.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from aramis.modeling import build_fusion_feature_table, compute_binary_thresholds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
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
    / "modeling_50_splits"
    / "monochromatic_biopsy_benign_vs_cancer_no_normal_lr1_lr2_symmetry_no_oof"
)

N_SPLITS = 50
TEST_SIZE = 0.30
RANDOM_STATE = 42
LABEL = {"BENIGN": 0, "CANCER": 1}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_df = _load_pool(INPUT_JOBLIB)
    target_df = _target_training_rows(pool_df)

    feature_table = build_fusion_feature_table(target_df, pool_df)
    feature_table = _coerce_feature_table(feature_table)
    feature_table.to_csv(OUTPUT_DIR / "feature_table.csv", index=False)

    metrics = []
    predictions = []
    for split_id, train_idx, test_idx in _patient_safe_splits(target_df):
        split_metrics, split_predictions = _run_split(
            split_id=split_id,
            train_df=target_df.iloc[train_idx].copy(),
            test_df=target_df.iloc[test_idx].copy(),
            feature_table=feature_table,
        )
        metrics.extend(split_metrics)
        predictions.extend(split_predictions)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)
    summary_df = _summary_table(metrics_df)
    run_summary = _run_summary(pool_df, target_df, feature_table)

    metrics_df.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
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
        (pool_df["biopsy"] == True)  # noqa: E712 - H5 metadata stores bool-like values.
        & pool_df["product_status_group"].isin(["BENIGN", "CANCER"])
    ].copy()
    target_df.reset_index(drop=True, inplace=True)
    if (target_df.groupby("specimenId")["product_status_group"].nunique() > 1).any():
        raise ValueError("Mixed labels inside specimenId.")
    if (target_df.groupby("specimenId")["patientId"].nunique() > 1).any():
        raise ValueError("Mixed patients inside specimenId.")
    return target_df


def _patient_safe_splits(df: pd.DataFrame):
    y = df["product_status_group"].map(LABEL).astype(int).to_numpy()
    groups = df["patientId"].astype(str).to_numpy()
    seen = 0
    attempts = 0
    while seen < N_SPLITS and attempts < N_SPLITS * 300:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE + attempts,
        )
        train_idx, test_idx = next(splitter.split(np.zeros(len(df)), y, groups))
        attempts += 1
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        if set(groups[train_idx]).intersection(set(groups[test_idx])):
            raise RuntimeError("Patient leakage detected.")
        yield seen, train_idx, test_idx
        seen += 1
    if seen < N_SPLITS:
        raise ValueError(f"Only made {seen} patient-safe splits.")


def _run_split(
    *,
    split_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> tuple[list[dict], list[pd.DataFrame]]:
    y_train = train_df["product_status_group"].map(LABEL).astype(int).to_numpy()
    lr1 = _lr_model(seed=20_000 + split_id)
    lr1.fit(_profile_matrix(train_df), y_train)

    train_specimen_df = _aggregate_scores(
        train_df,
        lr1.predict_proba(_profile_matrix(train_df))[:, 1],
    )
    test_specimen_df = _aggregate_scores(
        test_df,
        lr1.predict_proba(_profile_matrix(test_df))[:, 1],
    )
    train_specimen_df["set_name"] = "train_insample"
    test_specimen_df["set_name"] = "test"

    scored = pd.concat([train_specimen_df, test_specimen_df], ignore_index=True)
    scored = scored.merge(
        feature_table.drop(columns=["product_status_group", "y_true"], errors="ignore"),
        on=["patientId", "specimenId"],
        how="left",
    )
    scored["logit_p_cancer_lr1"] = _safe_logit(scored["p_cancer_lr1"])
    scored = _fill_feature_columns(scored)

    train_s = scored[scored["set_name"] == "train_insample"].copy()
    test_s = scored[scored["set_name"] == "test"].copy()
    y_train_s = train_s["y_true"].to_numpy(dtype=int)

    metrics = []
    predictions = []
    for model_name, columns in _feature_sets().items():
        if model_name == "M0_direct_LR1_mean":
            train_score = train_s["p_cancer_lr1"].to_numpy(dtype=float)
            test_score = test_s["p_cancer_lr1"].to_numpy(dtype=float)
        else:
            lr2 = _lr_model(seed=30_000 + split_id)
            lr2.fit(train_s[columns].to_numpy(dtype=float), y_train_s)
            train_score = lr2.predict_proba(train_s[columns].to_numpy(dtype=float))[:, 1]
            test_score = lr2.predict_proba(test_s[columns].to_numpy(dtype=float))[:, 1]

        thresholds = compute_binary_thresholds(
            y_train_s,
            train_score,
            target_sensitivity=0.95,
        )
        metrics.append(_metric_row(model_name, split_id, train_s, test_s, test_score, thresholds))
        predictions.append(_prediction_frame(model_name, split_id, test_s, test_score))
    return metrics, predictions


def _lr_model(*, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=1.0,
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
        raise ValueError("Non-finite profile matrix.")
    return matrix


def _aggregate_scores(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    temp = df[["patientId", "specimenId", "product_status_group"]].copy()
    temp["p_cancer_measurement"] = scores
    rows = []
    for specimen_id, group_df in temp.groupby("specimenId", sort=True):
        labels = group_df["product_status_group"].unique().tolist()
        patients = group_df["patientId"].astype(str).unique().tolist()
        if len(labels) != 1 or len(patients) != 1:
            raise ValueError((specimen_id, labels, patients))
        rows.append(
            {
                "patientId": patients[0],
                "specimenId": str(specimen_id),
                "product_status_group": labels[0],
                "y_true": LABEL[labels[0]],
                "p_cancer_lr1": float(group_df["p_cancer_measurement"].mean()),
                "n_measurements": int(len(group_df)),
            }
        )
    return pd.DataFrame(rows)


def _feature_sets() -> dict[str, list[str]]:
    symmetry_basic = ["symmetry_available", "symmetry_distance_x_available"]
    symmetry_between_within = [
        "symmetry_available",
        "target_within_available",
        "contralateral_within_available",
        "cosine_between_mean_x_available",
        "cosine_within_target_mean_x_available",
        "cosine_within_contralateral_mean_x_available",
        "cosine_asymmetry_score_x_available",
    ]
    symmetry_variability = [
        "symmetry_available",
        "target_within_available",
        "contralateral_within_available",
        "cosine_target_contralateral_centroid_x_available",
        "cosine_replicate_variability_delta_x_available",
        "cosine_replicate_variability_ratio_x_available",
    ]
    symmetry_full = sorted(set(symmetry_between_within + symmetry_variability))
    return {
        "M0_direct_LR1_mean": [],
        "M0_LR2_p_only_no_oof": ["logit_p_cancer_lr1"],
        "M1_flag_no_oof": ["logit_p_cancer_lr1", "symmetry_available"],
        "M1_basic_asymmetry_no_oof": ["logit_p_cancer_lr1", *symmetry_basic],
        "M1_1_between_within_no_oof": ["logit_p_cancer_lr1", *symmetry_between_within],
        "M1_2_variability_no_oof": ["logit_p_cancer_lr1", *symmetry_variability],
        "M1_N_full_cosine_symmetry_no_oof": ["logit_p_cancer_lr1", *symmetry_full],
        "M2_full_symmetry_age_no_oof": [
            "logit_p_cancer_lr1",
            *symmetry_full,
            "age",
            "age_available",
        ],
        "S0_symmetry_only": symmetry_full,
        "A0_age_only": ["age", "age_available"],
    }


def _coerce_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    out = feature_table.copy()
    for columns in _feature_sets().values():
        for column in columns:
            if column == "logit_p_cancer_lr1":
                continue
            if column not in out:
                raise KeyError(column)
            out[column] = _finite_numeric(out[column])
    return out


def _fill_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for columns in _feature_sets().values():
        for column in columns:
            out[column] = _finite_numeric(out[column])
    return out


def _finite_numeric(values: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _safe_logit(values: pd.Series) -> np.ndarray:
    probabilities = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def _metric_row(
    model_name: str,
    split_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score: np.ndarray,
    thresholds: dict,
) -> dict:
    y_true = test_df["y_true"].to_numpy(dtype=int)
    pred_target = (score >= thresholds["threshold_target"]).astype(int)
    pred_youden = (score >= thresholds["threshold_youden"]).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(
        y_true,
        pred_target,
        labels=[0, 1],
    ).ravel()
    tn_y, fp_y, fn_y, tp_y = confusion_matrix(
        y_true,
        pred_youden,
        labels=[0, 1],
    ).ravel()
    return {
        "model_name": model_name,
        "split_id": int(split_id),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "train_specimens": int(len(train_df)),
        "test_specimens": int(len(test_df)),
        "train_patients": int(train_df["patientId"].nunique()),
        "test_patients": int(test_df["patientId"].nunique()),
        "test_benign_specimens": int((y_true == 0).sum()),
        "test_cancer_specimens": int((y_true == 1).sum()),
        "symmetry_available_test": int(test_df["symmetry_available"].sum()),
        "sensitivity_target": _ratio(tp_t, tp_t + fn_t),
        "specificity_target": _ratio(tn_t, tn_t + fp_t),
        "sensitivity_youden": _ratio(tp_y, tp_y + fn_y),
        "specificity_youden": _ratio(tn_y, tn_y + fp_y),
        "threshold_target": float(thresholds["threshold_target"]),
        "threshold_youden": float(thresholds["threshold_youden"]),
    }


def _prediction_frame(
    model_name: str,
    split_id: int,
    test_df: pd.DataFrame,
    score: np.ndarray,
) -> pd.DataFrame:
    out = test_df[
        ["patientId", "specimenId", "product_status_group", "y_true", "p_cancer_lr1"]
    ].copy()
    out["model_name"] = model_name
    out["split_id"] = int(split_id)
    out["p_cancer_model"] = score
    return out


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group_df in metrics_df.groupby("model_name", sort=False):
        best = group_df.loc[group_df["roc_auc"].idxmax()]
        rows.append(
            {
                "model_name": model_name,
                "splits": int(len(group_df)),
                "roc_auc_mean": float(group_df["roc_auc"].mean()),
                "roc_auc_std": float(group_df["roc_auc"].std(ddof=0)),
                "best_roc_auc": float(best["roc_auc"]),
                "best_split_id": int(best["split_id"]),
                "pr_auc_mean": float(group_df["pr_auc"].mean()),
                "sensitivity_target_mean": float(
                    group_df["sensitivity_target"].mean()
                ),
                "specificity_target_mean": float(
                    group_df["specificity_target"].mean()
                ),
                "sensitivity_youden_mean": float(
                    group_df["sensitivity_youden"].mean()
                ),
                "specificity_youden_mean": float(
                    group_df["specificity_youden"].mean()
                ),
                "train_specimens_mean": float(group_df["train_specimens"].mean()),
                "test_specimens_mean": float(group_df["test_specimens"].mean()),
                "train_patients_mean": float(group_df["train_patients"].mean()),
                "test_patients_mean": float(group_df["test_patients"].mean()),
                "symmetry_available_test_mean": float(
                    group_df["symmetry_available_test"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_summary(
    pool_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> dict:
    return {
        "input_joblib": str(INPUT_JOBLIB),
        "target_rule": (
            "biopsy=True and specimen_status BENIGN vs "
            "CANCER/ATYPICAL/PRE_CANCEROUS; NORMAL excluded from target/training"
        ),
        "symmetry_pool_rule": (
            "NORMAL retained in pool for contralateral breast distance calculation"
        ),
        "split_rule": "GroupShuffleSplit by patientId; no patient in train and test",
        "lr2_train_scores": "in_sample_no_oof",
        "rows_pool": int(len(pool_df)),
        "rows_biopsy_true": int((pool_df["biopsy"] == True).sum()),  # noqa: E712
        "rows_used": int(len(target_df)),
        "patients_used": int(target_df["patientId"].nunique()),
        "specimens_used": int(target_df["specimenId"].nunique()),
        "measurement_label_counts": target_df["product_status_group"]
        .value_counts()
        .to_dict(),
        "specimen_label_counts": feature_table["product_status_group"]
        .value_counts()
        .to_dict(),
        "symmetry_available_specimens": int(feature_table["symmetry_available"].sum()),
        "age_available_specimens": int(feature_table["age_available"].sum()),
        "pool_status_counts": pool_df["product_status_group"]
        .value_counts(dropna=False)
        .to_dict(),
        "feature_sets": _feature_sets(),
    }


if __name__ == "__main__":
    main()
