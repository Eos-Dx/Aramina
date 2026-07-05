"""Aramis threshold-grid experiment.

Research draft. This script builds three wide preprocessing pools by AgBH
monochromaticity threshold, then forms twelve modeling cohorts in memory.

Grid:
3 thresholds x 2 patient biopsy policies x 2 NORMAL label policies.

M0 = LR1 profile model averaged per patient.
M1 = LR1 score + symmetry features -> LogisticRegression.
M2 = LR1 score + symmetry features -> SVM polynomial degree 2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from aramis_ka_symmetry_only_experiment import (
    FEATURE_COLUMNS,
    RAW_PEAK_THRESHOLD,
    TARGET_SENSITIVITY,
    _build_patient_feature_table,
    _ratio,
    _score_model,
    _threshold_for_sensitivity,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config" / "preprocessing"
EXCLUSION_DIR = CONFIG_DIR / "exclusions"
OUTPUT_DIR = ROOT / "examples" / "outputs" / "threshold_grid_patient_cohorts"
META_CONFIG = ROOT / "docs" / "meta" / "aramis_preprocessing_v0_1_config.json"
INPUT_H5 = (
    Path("/Users/sad/dev/eos_play/jupyter_notebooks/Clinical_trials")
    / "data"
    / "product-aramis-data"
    / "combined_archive.h5"
)

CANCER_STATUSES = {"CANCER", "ATYPICAL", "PRE_CANCEROUS", "PRE-CANCEROUS"}
BENIGN_STATUSES = {"BENIGN"}
NORMAL_STATUSES = {"NORMAL"}
LR1_SCORE_COLUMN = "lr1_patient_mean_p_cancer"
N_SPLITS = 30
TEST_SIZE = 0.30
RANDOM_STATE = 42


@dataclass(frozen=True)
class ThresholdSpec:
    """Wide preprocessing pool definition."""

    key: str
    factor: float
    score: float

    @property
    def dataset_id(self) -> str:
        return f"aramis_wide_{self.key}"

    @property
    def config_path(self) -> Path:
        return CONFIG_DIR / f"{self.dataset_id}_max_v0_1.yaml"

    @property
    def joblib_path(self) -> Path:
        return OUTPUT_DIR / "wide_pools" / f"{self.dataset_id}.joblib"


@dataclass(frozen=True)
class CohortSpec:
    """One modeling cohort over a wide preprocessing pool."""

    threshold: ThresholdSpec
    biopsy_policy: str
    normal_policy: str

    @property
    def biopsy_patients_only(self) -> bool:
        return self.biopsy_policy == "biopsy_patients"

    @property
    def normal_as_benign(self) -> bool:
        return self.normal_policy == "normal_as_benign"

    @property
    def dataset_id(self) -> str:
        return (
            f"aramis_{self.threshold.key}_{self.biopsy_policy}_{self.normal_policy}"
        )


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = _threshold_specs()
    cohorts = _cohort_specs(thresholds)
    _write_grid_configs(thresholds)
    if args.run_preprocessing:
        _run_wide_preprocessing(thresholds, force=args.force_preprocessing)
    train_all, repeated = _run_training_grid(cohorts)
    _save_outputs(train_all, repeated)
    print(_compact_table(train_all, repeated).to_string(index=False))
    print(f"\nOUT {OUTPUT_DIR}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-preprocessing", action="store_true")
    parser.add_argument("--force-preprocessing", action="store_true")
    return parser.parse_args()


def _threshold_specs() -> list[ThresholdSpec]:
    base_score = float(_agbh_meta()["parameters"]["max_score"])
    return [
        ThresholdSpec("t70", 0.70, base_score * 0.70),
        ThresholdSpec("t100", 1.00, base_score),
        ThresholdSpec("t130", 1.30, base_score * 1.30),
    ]


def _cohort_specs(thresholds: list[ThresholdSpec]) -> list[CohortSpec]:
    return [
        CohortSpec(threshold, biopsy_policy, normal_policy)
        for threshold in thresholds
        for biopsy_policy in ("all_patients", "biopsy_patients")
        for normal_policy in ("normal_excluded", "normal_as_benign")
    ]


def _write_grid_configs(thresholds: list[ThresholdSpec]) -> None:
    meta = _agbh_meta()
    all_calibrations = meta["accepted_calibrations"] + meta["rejected_calibrations"]
    for threshold in thresholds:
        rejected = [
            row
            for row in all_calibrations
            if float(row["agbh_monochromaticity_score"]) > threshold.score
        ]
        _write_yaml(
            EXCLUSION_DIR / f"agbh_quality_exclusions_{threshold.key}_v0_1.yaml",
            {
                "filters": {
                    "quality_exclusions": {
                        "enabled": True,
                        "reason_doc": "Aramis/docs/agbh_quality_exclusions.md",
                        "monochromaticity_max_score": threshold.score,
                        "primary_key": {
                            "column": "linked_agbh_session_uid",
                            "excluded_values": sorted(
                                str(row["session_uid"]) for row in rejected
                            ),
                        },
                        "fallback_date": {
                            "column": "started_at",
                            "excluded_dates": sorted(
                                {str(row["calibration_day"]) for row in rejected}
                            ),
                            "use_when_primary_key_missing": True,
                        },
                    }
                }
            },
        )
        _write_yaml(threshold.config_path, _wide_root_config(threshold))


def _wide_root_config(threshold: ThresholdSpec) -> dict[str, Any]:
    return {
        "extends": [
            "shared/aramis_policy_v0_1.yaml",
            f"exclusions/agbh_quality_exclusions_{threshold.key}_v0_1.yaml",
            "shared/aramis_pipeline_v0_1.yaml",
            "outputs/max_output_v0_1.yaml",
            "branches/one_to_many_monochromatic_metadata_pool_v0_1.yaml",
        ],
        "aramis_preprocessing": {
            "name": threshold.dataset_id,
            "version": 0.1,
            "branch": "one_to_many",
            "clinical_stage": "research draft",
        },
        "provenance": {
            "status": "research draft wide pool",
            "monochromaticity_source": "Aramis/docs/meta/aramis_preprocessing_v0_1_config.json",
            "quality_exclusion_reason_doc": "Aramis/docs/agbh_quality_exclusions.md",
        },
        "experiment": {
            "monochromaticity_threshold_factor": threshold.factor,
            "monochromaticity_max_score": threshold.score,
            "cohort_policy": "wide pool; model cohorts are built after preprocessing",
        },
        "io": {
            "input_h5_path": str(INPUT_H5),
            "output_joblib_path": str(threshold.joblib_path),
        },
        "branch_settings": {
            "require_biopsy_rows": False,
            "filter_by_specimen_status": False,
            "filter_by_product_status_group": False,
            "output_columns": [],
        },
    }


def _run_wide_preprocessing(thresholds: list[ThresholdSpec], *, force: bool) -> None:
    for threshold in thresholds:
        if threshold.joblib_path.exists() and not force:
            print(f"SKIP {threshold.dataset_id} {threshold.joblib_path}", flush=True)
            continue
        if not force and _copy_previous_wide_pool(threshold):
            continue
        print(f"RUN {threshold.dataset_id}", flush=True)
        subprocess.run(
            [
                "python",
                "-m",
                "aramis",
                "preprocess",
                "--config",
                str(threshold.config_path),
            ],
            cwd=ROOT,
            check=True,
        )


def _copy_previous_wide_pool(threshold: ThresholdSpec) -> bool:
    previous = (
        ROOT
        / "examples"
        / "outputs"
        / "threshold_grid"
        / "datasets"
        / f"aramis_grid_{threshold.key}_all_samples_normal_excluded.joblib"
    )
    if not previous.exists():
        return False
    threshold.joblib_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(previous, threshold.joblib_path)
    print(f"COPY {previous.name} -> {threshold.joblib_path.name}", flush=True)
    return True


def _run_training_grid(
    cohorts: list[CohortSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_all_rows = []
    repeated_rows = []
    for cohort in cohorts:
        wide_df = _load_dataframe(cohort.threshold.joblib_path)
        cohort_df = _cohort_rows(wide_df, cohort)
        lr1_df = _lr1_training_rows(cohort_df, cohort)
        labels = _patient_labels(cohort_df, cohort)
        feature_df = _feature_table(cohort_df, labels)
        if feature_df.empty or feature_df["label"].nunique() < 2:
            train_all_rows.extend(_empty_rows(cohort, cohort_df, lr1_df, feature_df, "train_all"))
            repeated_rows.extend(_empty_rows(cohort, cohort_df, lr1_df, feature_df, "patient_70_30"))
            continue
        train_all_rows.extend(_train_all_models(cohort, cohort_df, lr1_df, feature_df))
        repeated_rows.extend(_repeated_patient_splits(cohort, cohort_df, lr1_df, feature_df))
    return pd.DataFrame(train_all_rows), pd.DataFrame(repeated_rows)


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Wide preprocessing joblib not found: {path}")
    artifact = joblib.load(path)
    return artifact["dataframe"].copy() if isinstance(artifact, dict) else artifact.copy()


def _cohort_rows(df: pd.DataFrame, cohort: CohortSpec) -> pd.DataFrame:
    out = df.copy()
    if not cohort.biopsy_patients_only:
        return out
    biopsy = _boolean_mask(out["biopsy"])
    biopsy_patients = set(out.loc[biopsy, "patientId"].astype(str))
    return out[out["patientId"].astype(str).isin(biopsy_patients)].copy()


def _lr1_training_rows(df: pd.DataFrame, cohort: CohortSpec) -> pd.DataFrame:
    out = df.copy()
    if cohort.biopsy_patients_only:
        out = out[_boolean_mask(out["biopsy"])].copy()
    group = _status_group(out["specimen_status"])
    keep = group.isin(["CANCER", "BENIGN"])
    if cohort.normal_as_benign:
        keep |= group.eq("NORMAL")
    out = out[keep].copy()
    group = _status_group(out["specimen_status"])
    out["model_label"] = np.where(group.eq("CANCER"), 1, 0).astype(int)
    return out


def _patient_labels(df: pd.DataFrame, cohort: CohortSpec) -> pd.Series:
    source = df.copy()
    if cohort.biopsy_patients_only:
        source = source[_boolean_mask(source["biopsy"])].copy()
    group = _status_group(source["specimen_status"])
    keep = group.isin(["CANCER", "BENIGN"])
    if cohort.normal_as_benign:
        keep |= group.eq("NORMAL")
    source = source[keep].copy()
    group = _status_group(source["specimen_status"])
    source["label"] = np.where(group.eq("CANCER"), 1, 0).astype(int)
    labels = {}
    for patient_id, patient_df in source.groupby("patientId", sort=True):
        y = patient_df["label"].to_numpy(dtype=int)
        if (y == 1).any():
            labels[str(patient_id)] = 1
        elif (y == 0).any():
            labels[str(patient_id)] = 0
    return pd.Series(labels, dtype=int)


def _feature_table(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    pool = df[df["patientId"].astype(str).isin(labels.index)].copy()
    pool["specimen_status"] = pool["patientId"].astype(str).map(
        labels.map({1: "CANCER", 0: "BENIGN"})
    )
    feature_df = _build_patient_feature_table(pool)
    feature_df = feature_df[feature_df["mean_peak_value_raw"].ge(RAW_PEAK_THRESHOLD)].copy()
    return feature_df.reset_index(drop=True)


def _train_all_models(
    cohort: CohortSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    lr1_model = _fit_lr1(lr1_df)
    scores = _patient_lr1_scores(lr1_model, lr1_df)
    final_df = _final_table(feature_df, scores)
    if final_df.empty or final_df["label"].nunique() < 2:
        return _empty_rows(cohort, cohort_df, lr1_df, final_df, "train_all")
    return _score_final_models(cohort, cohort_df, lr1_df, final_df, "train_all")


def _repeated_patient_splits(
    cohort: CohortSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    y = feature_df["label"].to_numpy(dtype=int)
    if len(feature_df) < 8 or min(np.bincount(y)) < 2:
        return _empty_rows(cohort, cohort_df, lr1_df, feature_df, "patient_70_30")
    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    predictions = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(feature_df, y)):
        train_features = feature_df.iloc[train_idx].copy()
        test_features = feature_df.iloc[test_idx].copy()
        train_patients = set(train_features["patient_id"])
        train_lr1_df = lr1_df[lr1_df["patientId"].astype(str).isin(train_patients)].copy()
        if train_lr1_df["model_label"].nunique() < 2:
            continue
        lr1_model = _fit_lr1(train_lr1_df)
        train_scores = _patient_lr1_scores(lr1_model, train_lr1_df)
        test_scores = _patient_lr1_scores(
            lr1_model,
            lr1_df[lr1_df["patientId"].astype(str).isin(test_features["patient_id"])],
        )
        train_final = _final_table(train_features, train_scores)
        test_final = _final_table(test_features, test_scores)
        if train_final["label"].nunique() < 2 or test_final["label"].nunique() < 2:
            continue
        predictions.extend(
            _split_predictions(cohort, train_final, test_final, split_id)
        )
    if not predictions:
        return _empty_rows(cohort, cohort_df, lr1_df, feature_df, "patient_70_30")
    prediction_df = pd.DataFrame(predictions)
    rows = []
    for model_name, model_df in prediction_df.groupby("model", sort=False):
        rows.append(
            _prediction_summary(
                cohort,
                cohort_df,
                lr1_df,
                feature_df,
                model_df,
                model_name=model_name,
                protocol="patient_70_30",
            )
        )
    return rows


def _fit_lr1(lr1_df: pd.DataFrame) -> Pipeline:
    if lr1_df["model_label"].nunique() < 2:
        raise ValueError("LR1 training rows contain one class.")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(_profile_matrix(lr1_df), lr1_df["model_label"].to_numpy(dtype=int))
    return model


def _patient_lr1_scores(model: Pipeline, rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["patient_id", LR1_SCORE_COLUMN, "lr1_measurements"])
    out = rows.copy()
    out["patient_id"] = out["patientId"].astype(str)
    out["lr1_measurement_p_cancer"] = model.predict_proba(_profile_matrix(out))[:, 1]
    return (
        out.groupby("patient_id", as_index=False)
        .agg(
            lr1_patient_mean_p_cancer=("lr1_measurement_p_cancer", "mean"),
            lr1_measurements=("lr1_measurement_p_cancer", "size"),
        )
        .reset_index(drop=True)
    )


def _final_table(feature_df: pd.DataFrame, lr1_scores: pd.DataFrame) -> pd.DataFrame:
    return feature_df.merge(lr1_scores, on="patient_id", how="inner")


def _score_final_models(
    cohort: CohortSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    final_df: pd.DataFrame,
    protocol: str,
) -> list[dict[str, Any]]:
    y = final_df["label"].to_numpy(dtype=int)
    x = final_df[[LR1_SCORE_COLUMN, *FEATURE_COLUMNS]]
    return [
        _summary_row(
            cohort,
            cohort_df,
            lr1_df,
            final_df,
            model_name="M0_LR1_mean_only",
            protocol=protocol,
            score=final_df[LR1_SCORE_COLUMN].to_numpy(dtype=float),
        ),
        _summary_row(
            cohort,
            cohort_df,
            lr1_df,
            final_df,
            model_name="M1_LR1_plus_symmetry_LR2",
            protocol=protocol,
            score=_fit_score(_lr2_model(), x, y),
        ),
        _summary_row(
            cohort,
            cohort_df,
            lr1_df,
            final_df,
            model_name="M2_LR1_plus_symmetry_SVM_poly2",
            protocol=protocol,
            score=_fit_score(_svm_poly2(), x, y),
        ),
    ]


def _split_predictions(
    cohort: CohortSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_id: int,
) -> list[dict[str, Any]]:
    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)
    rows = []
    for model_name, train_score, test_score in _split_model_scores(train_df, test_df):
        threshold = _threshold_for_sensitivity(y_train, train_score, TARGET_SENSITIVITY)
        pred = (test_score >= threshold).astype(int)
        for patient_id, label, score, y_hat in zip(
            test_df["patient_id"], y_test, test_score, pred, strict=False
        ):
            rows.append(
                {
                    "dataset_id": cohort.dataset_id,
                    "threshold_key": cohort.threshold.key,
                    "threshold_score": cohort.threshold.score,
                    "biopsy_policy": cohort.biopsy_policy,
                    "normal_policy": cohort.normal_policy,
                    "model": model_name,
                    "protocol": "patient_70_30",
                    "split_id": split_id,
                    "patient_id": patient_id,
                    "label": int(label),
                    "score": float(score),
                    "prediction": int(y_hat),
                }
            )
    return rows


def _split_model_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    y_train = train_df["label"].to_numpy(dtype=int)
    x_train = train_df[[LR1_SCORE_COLUMN, *FEATURE_COLUMNS]]
    x_test = test_df[[LR1_SCORE_COLUMN, *FEATURE_COLUMNS]]
    scores = [
        (
            "M0_LR1_mean_only",
            train_df[LR1_SCORE_COLUMN].to_numpy(dtype=float),
            test_df[LR1_SCORE_COLUMN].to_numpy(dtype=float),
        )
    ]
    for model_name, model in [
        ("M1_LR1_plus_symmetry_LR2", _lr2_model()),
        ("M2_LR1_plus_symmetry_SVM_poly2", _svm_poly2()),
    ]:
        model.fit(x_train, y_train)
        scores.append((model_name, _score_model(model, x_train), _score_model(model, x_test)))
    return scores


def _summary_row(
    cohort: CohortSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    final_df: pd.DataFrame,
    *,
    model_name: str,
    protocol: str,
    score: np.ndarray,
) -> dict[str, Any]:
    y = final_df["label"].to_numpy(dtype=int)
    threshold = _threshold_for_sensitivity(y, score, TARGET_SENSITIVITY)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return _base_summary(cohort, cohort_df, lr1_df, final_df) | {
        "protocol": protocol,
        "model": model_name,
        "split_count": 1,
        "R_ROC_AUC": float(roc_auc_score(y, score)),
        "R_ROC_AUC_std": np.nan,
        "PR_AUC": float(average_precision_score(y, score)),
        "PR_AUC_std": np.nan,
        "S_sensitivity": _ratio(tp, tp + fn),
        "S_sensitivity_std": np.nan,
        "Sp_specificity": _ratio(tn, tn + fp),
        "Sp_specificity_std": np.nan,
        "threshold_95_sensitivity": float(threshold),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def _prediction_summary(
    cohort: CohortSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    *,
    model_name: str,
    protocol: str,
) -> dict[str, Any]:
    split_metrics = []
    for split_id, split_df in prediction_df.groupby("split_id", sort=True):
        y = split_df["label"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        score = split_df["score"].to_numpy(dtype=float)
        pred = split_df["prediction"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        split_metrics.append(
            {
                "split_id": int(split_id),
                "R_ROC_AUC": float(roc_auc_score(y, score)),
                "PR_AUC": float(average_precision_score(y, score)),
                "S_sensitivity": _ratio(tp, tp + fn),
                "Sp_specificity": _ratio(tn, tn + fp),
                "false_negatives": int(fn),
                "false_positives": int(fp),
            }
        )
    split_metric_df = pd.DataFrame(split_metrics)
    if split_metric_df.empty:
        return _empty_rows(cohort, cohort_df, lr1_df, feature_df, protocol)[0] | {
            "model": model_name
        }
    return _base_summary(cohort, cohort_df, lr1_df, feature_df) | {
        "protocol": protocol,
        "model": model_name,
        "split_count": int(len(split_metric_df)),
        "R_ROC_AUC": float(split_metric_df["R_ROC_AUC"].mean()),
        "R_ROC_AUC_std": float(split_metric_df["R_ROC_AUC"].std(ddof=0)),
        "PR_AUC": float(split_metric_df["PR_AUC"].mean()),
        "PR_AUC_std": float(split_metric_df["PR_AUC"].std(ddof=0)),
        "S_sensitivity": float(split_metric_df["S_sensitivity"].mean()),
        "S_sensitivity_std": float(split_metric_df["S_sensitivity"].std(ddof=0)),
        "Sp_specificity": float(split_metric_df["Sp_specificity"].mean()),
        "Sp_specificity_std": float(split_metric_df["Sp_specificity"].std(ddof=0)),
        "threshold_95_sensitivity": np.nan,
        "false_negatives": float(split_metric_df["false_negatives"].mean()),
        "false_positives": float(split_metric_df["false_positives"].mean()),
    }


def _base_summary(
    cohort: CohortSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> dict[str, Any]:
    y = final_df["label"].to_numpy(dtype=int) if "label" in final_df else np.array([])
    return {
        "dataset_id": cohort.dataset_id,
        "threshold_key": cohort.threshold.key,
        "threshold_score": cohort.threshold.score,
        "biopsy_policy": cohort.biopsy_policy,
        "normal_policy": cohort.normal_policy,
        "wide_measurements": int(len(cohort_df)),
        "lr1_measurements": int(len(lr1_df)),
        "patients_for_final_model": int(final_df["patient_id"].nunique())
        if "patient_id" in final_df
        else 0,
        "cancer_patients": int((y == 1).sum()),
        "non_cancer_patients": int((y == 0).sum()),
    }


def _empty_rows(
    cohort: CohortSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    final_df: pd.DataFrame,
    protocol: str,
) -> list[dict[str, Any]]:
    return [
        _base_summary(cohort, cohort_df, lr1_df, final_df)
        | {
            "protocol": protocol,
            "model": model,
            "split_count": 0,
            "R_ROC_AUC": np.nan,
            "R_ROC_AUC_std": np.nan,
            "PR_AUC": np.nan,
            "PR_AUC_std": np.nan,
            "S_sensitivity": np.nan,
            "S_sensitivity_std": np.nan,
            "Sp_specificity": np.nan,
            "Sp_specificity_std": np.nan,
            "threshold_95_sensitivity": np.nan,
            "false_negatives": np.nan,
            "false_positives": np.nan,
        }
        for model in (
            "M0_LR1_mean_only",
            "M1_LR1_plus_symmetry_LR2",
            "M2_LR1_plus_symmetry_SVM_poly2",
        )
    ]


def _fit_score(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    model.fit(x, y)
    return _score_model(model, x)


def _lr2_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


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


def _profile_matrix(df: pd.DataFrame) -> np.ndarray:
    matrix = np.vstack(
        [np.asarray(value, dtype=float).ravel() for value in df["radial_profile_data"]]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Profile matrix contains non-finite values.")
    return matrix


def _status_group(status: pd.Series) -> pd.Series:
    clean = status.fillna("").astype(str).str.upper().str.strip()
    return pd.Series(
        np.select(
            [clean.isin(CANCER_STATUSES), clean.isin(BENIGN_STATUSES), clean.isin(NORMAL_STATUSES)],
            ["CANCER", "BENIGN", "NORMAL"],
            default="EXCLUDE",
        ),
        index=status.index,
    )


def _boolean_mask(values: pd.Series) -> pd.Series:
    clean = values.astype("object").where(values.notna(), False)
    if clean.dtype == bool:
        return clean
    return clean.astype(str).str.lower().isin(["true", "1", "yes"])


def _agbh_meta() -> dict[str, Any]:
    return json.loads(META_CONFIG.read_text(encoding="utf-8"))


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _save_outputs(train_all: pd.DataFrame, repeated: pd.DataFrame) -> None:
    train_all.to_csv(OUTPUT_DIR / "train_all_summary.csv", index=False)
    repeated.to_csv(OUTPUT_DIR / "patient_70_30_summary.csv", index=False)
    compact = _compact_table(train_all, repeated)
    compact.to_csv(OUTPUT_DIR / "comparison_pivot.csv", index=False)
    _write_markdown_table(compact, OUTPUT_DIR / "comparison_pivot.md")


def _compact_table(train_all: pd.DataFrame, repeated: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([train_all, repeated], ignore_index=True)
    data["metric"] = data.apply(_format_metric, axis=1)
    return (
        data.pivot_table(
            index=[
                "protocol",
                "threshold_key",
                "threshold_score",
                "biopsy_policy",
                "normal_policy",
                "patients_for_final_model",
                "cancer_patients",
                "non_cancer_patients",
            ],
            columns="model",
            values="metric",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["protocol", "threshold_key", "biopsy_policy", "normal_policy"])
    )


def _format_metric(row: pd.Series) -> str:
    if row["protocol"] == "patient_70_30":
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


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    path.write_text("\n".join([header, separator, *body]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
