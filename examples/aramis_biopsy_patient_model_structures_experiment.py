"""Aramis patient-level model-structure experiment.

Research draft. Not clinical validation.

Fixed preprocessing pool:
- t130 AgBH monochromaticity threshold.
- NORMAL is treated as BENIGN for model-label construction.

Datasets:
- all_patients: all labelled patients from the wide t130 pool.
- biopsy_patients: patients with at least one biopsy row; LR1 trains only on
  biopsy specimen measurements, while symmetry uses full patient context.

Models:
- AGE_ONLY: age only.
- M0: profile LR1 patient mean probability.
- M1: profile LR1 probability + raw symmetry features -> final LR.
- M2: profile LR1 probability + raw symmetry features + age -> final LR.

Protocols:
- train_all_discovery.
- oracle_70_30_target95: threshold selected on test labels to inspect the
  specificity available at 95% sensitivity.
- honest_70_30_train_threshold: threshold selected on train and applied to test.
- patient_loocv_pooled: one patient out, pooled predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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

ROOT = Path(__file__).resolve().parents[1]
INPUT_JOBLIB = (
    ROOT
    / "examples"
    / "outputs"
    / "threshold_grid_patient_cohorts"
    / "wide_pools"
    / "aramis_wide_t130.joblib"
)
OUTPUT_DIR = ROOT / "examples" / "outputs" / "patient_model_structures_v0_2"

CANCER_STATUSES = {"CANCER", "ATYPICAL", "PRE_CANCEROUS", "PRE-CANCEROUS"}
BENIGN_STATUSES = {"BENIGN", "NORMAL"}
PROFILE_SCORE = "lr1_patient_mean_p_cancer"
AGE_COLUMN = "age"
N_SPLITS = 50
TEST_SIZE = 0.30
RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetSpec:
    """Modeling dataset policy."""

    name: str
    biopsy_patients_only: bool


DATASETS = [
    DatasetSpec("all_patients", biopsy_patients_only=False),
    DatasetSpec("biopsy_patients", biopsy_patients_only=True),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide_df = _load_dataframe(INPUT_JOBLIB)
    all_summaries = []
    all_predictions = []
    for spec in DATASETS:
        cohort_df, lr1_df, feature_df = _build_dataset(wide_df, spec)
        train_all = _train_all(spec, cohort_df, lr1_df, feature_df)
        repeated, repeated_predictions = _repeated_patient_splits(
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
        feature_df.to_csv(OUTPUT_DIR / f"{spec.name}_feature_table.csv", index=False)
        all_summaries.extend([train_all, repeated, loocv])
        all_predictions.extend([repeated_predictions, loocv_predictions])

    summary = pd.concat(all_summaries, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "model_structure_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "model_structure_predictions.csv", index=False)
    compact = _compact_table(summary)
    compact.to_csv(OUTPUT_DIR / "comparison_pivot.csv", index=False)
    _write_markdown_table(compact, OUTPUT_DIR / "comparison_pivot.md")
    print(compact.to_string(index=False))
    print(f"\nOUT {OUTPUT_DIR}")


def _load_dataframe(path: Path) -> pd.DataFrame:
    artifact = joblib.load(path)
    return artifact["dataframe"].copy() if isinstance(artifact, dict) else artifact.copy()


def _build_dataset(
    wide_df: pd.DataFrame,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort_df = wide_df.copy()
    if spec.biopsy_patients_only:
        biopsy_patients = set(
            cohort_df.loc[_boolean_mask(cohort_df["biopsy"]), "patientId"].astype(str)
        )
        cohort_df = cohort_df[cohort_df["patientId"].astype(str).isin(biopsy_patients)].copy()
    lr1_df = _lr1_rows(cohort_df, spec)
    patient_labels = _patient_labels(cohort_df, spec)
    feature_df = _feature_table(cohort_df, patient_labels)
    feature_df = _add_age(feature_df, cohort_df)
    return cohort_df, lr1_df, feature_df


def _lr1_rows(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    out = df.copy()
    if spec.biopsy_patients_only:
        out = out[_boolean_mask(out["biopsy"])].copy()
    group = _status_group(out["specimen_status"])
    out = out[group.isin(["CANCER", "BENIGN"])].copy()
    group = _status_group(out["specimen_status"])
    out["model_label"] = np.where(group.eq("CANCER"), 1, 0).astype(int)
    return out


def _patient_labels(df: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    source = df.copy()
    if spec.biopsy_patients_only:
        source = source[_boolean_mask(source["biopsy"])].copy()
    group = _status_group(source["specimen_status"])
    source = source[group.isin(["CANCER", "BENIGN"])].copy()
    group = _status_group(source["specimen_status"])
    source["label"] = np.where(group.eq("CANCER"), 1, 0).astype(int)
    labels = {}
    for patient_id, patient_df in source.groupby("patientId", sort=True):
        y = patient_df["label"].to_numpy(dtype=int)
        labels[str(patient_id)] = 1 if (y == 1).any() else 0
    return pd.Series(labels, dtype=int)


def _feature_table(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    pool = df[df["patientId"].astype(str).isin(labels.index)].copy()
    pool["specimen_status"] = pool["patientId"].astype(str).map(
        labels.map({1: "CANCER", 0: "BENIGN"})
    )
    features = _build_patient_feature_table(pool)
    features = features[features["mean_peak_value_raw"].ge(RAW_PEAK_THRESHOLD)].copy()
    return features.reset_index(drop=True)


def _add_age(feature_df: pd.DataFrame, cohort_df: pd.DataFrame) -> pd.DataFrame:
    age = (
        cohort_df.assign(patient_id=cohort_df["patientId"].astype(str))
        .groupby("patient_id")["age"]
        .median()
    )
    out = feature_df.copy()
    out[AGE_COLUMN] = out["patient_id"].map(age)
    return out


def _train_all(
    spec: DatasetSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    lr1_model = _fit_lr1(lr1_df)
    profile_scores = _patient_lr1_scores(lr1_model, lr1_df)
    final_df = _final_table(feature_df, profile_scores)
    return pd.DataFrame(
        _score_models(
            spec,
            cohort_df,
            lr1_df,
            final_df,
            protocol="train_all_discovery",
            threshold_source="same_data",
        )
    )


def _repeated_patient_splits(
    spec: DatasetSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = feature_df["label"].to_numpy(dtype=int)
    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    split_metrics = []
    predictions = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(feature_df, y)):
        train_features = feature_df.iloc[train_idx].copy()
        test_features = feature_df.iloc[test_idx].copy()
        train_final, test_final = _fit_split_tables(train_features, test_features, lr1_df)
        if train_final["label"].nunique() < 2 or test_final["label"].nunique() < 2:
            continue
        for model_name, train_score, test_score in _model_scores(train_final, test_final):
            for protocol, threshold_y, threshold_score in [
                (
                    "honest_70_30_train_threshold",
                    train_final["label"].to_numpy(dtype=int),
                    train_score,
                ),
                (
                    "oracle_70_30_target95",
                    test_final["label"].to_numpy(dtype=int),
                    test_score,
                ),
            ]:
                threshold = _threshold_for_sensitivity(
                    threshold_y,
                    threshold_score,
                    TARGET_SENSITIVITY,
                )
                metric_row = _metric_row(
                    spec,
                    cohort_df,
                    lr1_df,
                    feature_df,
                    protocol=protocol,
                    model_name=model_name,
                    y=test_final["label"].to_numpy(dtype=int),
                    score=test_score,
                    threshold=threshold,
                    split_count=1,
                    threshold_source="train" if protocol.startswith("honest") else "test",
                )
                metric_row["split_id"] = split_id
                split_metrics.append(metric_row)
                predictions.extend(
                    _prediction_rows(
                        spec,
                        test_final,
                        model_name=model_name,
                        protocol=protocol,
                        split_id=split_id,
                        score=test_score,
                        threshold=threshold,
                    )
                )
    split_metric_df = pd.DataFrame(split_metrics)
    summary = _summarize_repeated(split_metric_df, spec, cohort_df, lr1_df, feature_df)
    return summary, pd.DataFrame(predictions)


def _leave_one_patient_out(
    spec: DatasetSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    for split_id, test_idx in enumerate(feature_df.index):
        train_features = feature_df.drop(index=test_idx).copy()
        test_features = feature_df.loc[[test_idx]].copy()
        train_final, test_final = _fit_split_tables(train_features, test_features, lr1_df)
        if train_final["label"].nunique() < 2 or test_final.empty:
            continue
        for model_name, _, test_score in _model_scores(train_final, test_final):
            predictions.extend(
                _score_rows(
                    spec,
                    test_final,
                    model_name=model_name,
                    protocol="patient_loocv_pooled",
                    split_id=split_id,
                    score=test_score,
                )
            )
    prediction_df = pd.DataFrame(predictions)
    rows = []
    for model_name, model_df in prediction_df.groupby("model", sort=False):
        y = model_df["label"].to_numpy(dtype=int)
        score = model_df["score"].to_numpy(dtype=float)
        threshold = _threshold_for_sensitivity(y, score, TARGET_SENSITIVITY)
        rows.append(
            _metric_row(
                spec,
                cohort_df,
                lr1_df,
                feature_df,
                protocol="patient_loocv_pooled",
                model_name=model_name,
                y=y,
                score=score,
                threshold=threshold,
                split_count=int(model_df["split_id"].nunique()),
                threshold_source="pooled_loocv",
            )
        )
    return pd.DataFrame(rows), prediction_df


def _fit_split_tables(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    lr1_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_patients = set(train_features["patient_id"])
    test_patients = set(test_features["patient_id"])
    train_lr1 = lr1_df[lr1_df["patientId"].astype(str).isin(train_patients)].copy()
    test_lr1 = lr1_df[lr1_df["patientId"].astype(str).isin(test_patients)].copy()
    lr1_model = _fit_lr1(train_lr1)
    train_final = _final_table(train_features, _patient_lr1_scores(lr1_model, train_lr1))
    test_final = _final_table(test_features, _patient_lr1_scores(lr1_model, test_lr1))
    return train_final, test_final


def _fit_lr1(lr1_df: pd.DataFrame) -> Pipeline:
    model = _lr()
    model.fit(_profile_matrix(lr1_df), lr1_df["model_label"].to_numpy(dtype=int))
    return model


def _patient_lr1_scores(model: Pipeline, rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["patient_id"] = out["patientId"].astype(str)
    out["lr1_measurement_p_cancer"] = model.predict_proba(_profile_matrix(out))[:, 1]
    return (
        out.groupby("patient_id", as_index=False)
        .agg(
            lr1_patient_mean_p_cancer=("lr1_measurement_p_cancer", "mean"),
            lr1_measurements=("lr1_measurement_p_cancer", "size"),
        )
    )


def _final_table(feature_df: pd.DataFrame, profile_scores: pd.DataFrame) -> pd.DataFrame:
    return feature_df.merge(profile_scores, on="patient_id", how="inner")


def _score_models(
    spec: DatasetSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    final_df: pd.DataFrame,
    *,
    protocol: str,
    threshold_source: str,
) -> list[dict[str, Any]]:
    rows = []
    y = final_df["label"].to_numpy(dtype=int)
    for model_name, score in _model_train_all_scores(final_df):
        threshold = _threshold_for_sensitivity(y, score, TARGET_SENSITIVITY)
        rows.append(
            _metric_row(
                spec,
                cohort_df,
                lr1_df,
                final_df,
                protocol=protocol,
                model_name=model_name,
                y=y,
                score=score,
                threshold=threshold,
                split_count=1,
                threshold_source=threshold_source,
            )
        )
    return rows


def _model_train_all_scores(final_df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    y = final_df["label"].to_numpy(dtype=int)
    scores = [
        ("AGE_ONLY", _fit_score(_lr(), final_df[[AGE_COLUMN]], y)),
        ("M0_profile_LR1", final_df[PROFILE_SCORE].to_numpy(dtype=float)),
    ]
    for model_name, columns in _model_columns().items():
        scores.append((model_name, _fit_score(_lr(), final_df[columns], y)))
    return scores


def _model_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    y_train = train_df["label"].to_numpy(dtype=int)
    scores = [
        (
            "AGE_ONLY",
            _fit_score(_lr(), train_df[[AGE_COLUMN]], y_train),
            _score_model(_fit_fitted(_lr(), train_df[[AGE_COLUMN]], y_train), test_df[[AGE_COLUMN]]),
        ),
        (
            "M0_profile_LR1",
            train_df[PROFILE_SCORE].to_numpy(dtype=float),
            test_df[PROFILE_SCORE].to_numpy(dtype=float),
        ),
    ]
    for model_name, columns in _model_columns().items():
        model = _fit_fitted(_lr(), train_df[columns], y_train)
        scores.append(
            (
                model_name,
                _score_model(model, train_df[columns]),
                _score_model(model, test_df[columns]),
            )
        )
    return scores


def _model_columns() -> dict[str, list[str]]:
    return {
        "M1_profile_plus_symmetry_raw_LR": [PROFILE_SCORE, *FEATURE_COLUMNS],
        "M2_profile_plus_symmetry_raw_plus_age_LR": [
            PROFILE_SCORE,
            *FEATURE_COLUMNS,
            AGE_COLUMN,
        ],
    }


def _fit_score(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    model.fit(x, y)
    return _score_model(model, x)


def _fit_fitted(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> Pipeline:
    model.fit(x, y)
    return model


def _metric_row(
    spec: DatasetSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    final_df: pd.DataFrame,
    *,
    protocol: str,
    model_name: str,
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    split_count: int,
    threshold_source: str,
) -> dict[str, Any]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return _base_summary(spec, cohort_df, lr1_df, final_df) | {
        "protocol": protocol,
        "model": model_name,
        "split_count": split_count,
        "threshold_source": threshold_source,
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


def _summarize_repeated(
    split_metric_df: pd.DataFrame,
    spec: DatasetSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (protocol, model_name), group in split_metric_df.groupby(
        ["protocol", "model"],
        sort=False,
    ):
        rows.append(
            _base_summary(spec, cohort_df, lr1_df, feature_df)
            | {
                "protocol": protocol,
                "model": model_name,
                "split_count": int(group["split_id"].nunique()),
                "threshold_source": str(group["threshold_source"].iloc[0]),
                "R_ROC_AUC": float(group["R_ROC_AUC"].mean()),
                "R_ROC_AUC_std": float(group["R_ROC_AUC"].std(ddof=0)),
                "PR_AUC": float(group["PR_AUC"].mean()),
                "PR_AUC_std": float(group["PR_AUC"].std(ddof=0)),
                "S_sensitivity": float(group["S_sensitivity"].mean()),
                "S_sensitivity_std": float(group["S_sensitivity"].std(ddof=0)),
                "Sp_specificity": float(group["Sp_specificity"].mean()),
                "Sp_specificity_std": float(group["Sp_specificity"].std(ddof=0)),
                "threshold_95_sensitivity": np.nan,
                "false_negatives": float(group["false_negatives"].mean()),
                "false_positives": float(group["false_positives"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _prediction_rows(
    spec: DatasetSpec,
    test_df: pd.DataFrame,
    *,
    model_name: str,
    protocol: str,
    split_id: int,
    score: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = _score_rows(
        spec,
        test_df,
        model_name=model_name,
        protocol=protocol,
        split_id=split_id,
        score=score,
    )
    pred = (score >= threshold).astype(int)
    for row, y_hat in zip(rows, pred, strict=False):
        row["prediction"] = int(y_hat)
        row["threshold"] = float(threshold)
    return rows


def _score_rows(
    spec: DatasetSpec,
    test_df: pd.DataFrame,
    *,
    model_name: str,
    protocol: str,
    split_id: int,
    score: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": spec.name,
            "protocol": protocol,
            "split_id": split_id,
            "model": model_name,
            "patient_id": patient_id,
            "label": int(label),
            "score": float(value),
        }
        for patient_id, label, value in zip(
            test_df["patient_id"],
            test_df["label"].to_numpy(dtype=int),
            score,
            strict=False,
        )
    ]


def _base_summary(
    spec: DatasetSpec,
    cohort_df: pd.DataFrame,
    lr1_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> dict[str, Any]:
    y = final_df["label"].to_numpy(dtype=int)
    return {
        "dataset": spec.name,
        "wide_measurements": int(len(cohort_df)),
        "lr1_measurements": int(len(lr1_df)),
        "patients_for_final_model": int(final_df["patient_id"].nunique()),
        "cancer_patients": int((y == 1).sum()),
        "non_cancer_patients": int((y == 0).sum()),
    }


def _lr() -> Pipeline:
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
            [clean.isin(CANCER_STATUSES), clean.isin(BENIGN_STATUSES)],
            ["CANCER", "BENIGN"],
            default="EXCLUDE",
        ),
        index=status.index,
    )


def _boolean_mask(values: pd.Series) -> pd.Series:
    clean = values.astype("object").where(values.notna(), False)
    if clean.dtype == bool:
        return clean
    return clean.astype(str).str.lower().isin(["true", "1", "yes"])


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
    if row["protocol"] in {"honest_70_30_train_threshold", "oracle_70_30_target95"}:
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
