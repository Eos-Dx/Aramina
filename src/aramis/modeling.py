"""Modeling helpers for Aramis research-draft classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xrd_preprocessing import load_preprocessing_dataframe


LABEL_MAP = {"BENIGN": 0, "CANCER": 1}


@dataclass(frozen=True)
class RepeatedLogisticResult:
    """Repeated patient-safe split result for a binary Aramis model."""

    predictions: pd.DataFrame
    split_metrics: pd.DataFrame
    roc_curves: list[dict[str, Any]]
    mean_fpr: np.ndarray
    mean_tpr: np.ndarray
    mean_auc: float
    std_auc: float


@dataclass(frozen=True)
class OneToManyProductLogisticResult:
    """Repeated specimen-level one-to-many product-model result."""

    measurement_predictions: pd.DataFrame
    specimen_predictions: pd.DataFrame
    split_metrics: pd.DataFrame
    roc_curves: list[dict[str, Any]]
    mean_fpr: np.ndarray
    mean_tpr: np.ndarray
    mean_auc: float
    std_auc: float
    threshold_summary: pd.DataFrame


@dataclass(frozen=True)
class OneToManyProductComparisonResult:
    """Side-by-side result for one-to-many product datasets."""

    results: dict[str, OneToManyProductLogisticResult]
    dataset_summary: pd.DataFrame
    metric_summary: pd.DataFrame


@dataclass(frozen=True)
class FusionModelComparisonResult:
    """Repeated patient-safe comparison of Aramis fusion feature sets."""

    one_to_many_result: OneToManyProductLogisticResult
    feature_table: pd.DataFrame
    predictions: pd.DataFrame
    split_metrics: pd.DataFrame
    metric_summary: pd.DataFrame
    roc_curves: dict[str, list[dict[str, Any]]]
    mean_roc_curves: dict[str, dict[str, Any]]
    feature_sets: dict[str, list[str]]


def load_one_to_many_dataframe(path: str | Path) -> pd.DataFrame:
    """Load one-to-many preprocessing DataFrame or artifact joblib."""
    return load_preprocessing_dataframe(Path(path))


def summarize_one_to_many_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return basic row/patient/specimen/label counts."""
    _require_columns(df, ["patientId", "specimenId", "product_status_group"])
    return pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "patients": int(df["patientId"].astype(str).nunique()),
                "specimens": int(df["specimenId"].astype(str).nunique()),
                "BENIGN_rows": int((df["product_status_group"] == "BENIGN").sum()),
                "CANCER_rows": int((df["product_status_group"] == "CANCER").sum()),
            }
        ]
    )


def fit_repeated_one_to_many_logistic(
    df: pd.DataFrame,
    *,
    n_splits: int = 20,
    test_size: float = 0.30,
    random_state: int = 42,
    profile_column: str = "radial_profile_data",
    label_column: str = "product_status_group",
    group_column: str = "patientId",
    logreg_c: float = 1.0,
    extra_feature_columns: Sequence[str] | None = None,
) -> RepeatedLogisticResult:
    """Train LogisticRegression on full normalized profiles over repeated splits.

    Splits are grouped by ``patientId`` so the same patient never appears in
    train and test in one split.
    """
    _validate_binary_frame(df, profile_column, label_column, group_column)
    x = model_matrix(df, profile_column, extra_feature_columns)
    y = df[label_column].map(LABEL_MAP).astype(int).to_numpy()
    groups = df[group_column].astype(str).to_numpy()

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    mean_fpr = np.linspace(0.0, 1.0, 101)
    interpolated_tpr: list[np.ndarray] = []

    for split_id, train_idx, test_idx in _patient_safe_splits(
        y,
        groups,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    ):
        model = _logistic_pipeline(logreg_c=logreg_c, random_state=random_state + split_id)
        model.fit(x[train_idx], y[train_idx])
        y_score = model.predict_proba(x[test_idx])[:, 1]
        fpr, tpr, thresholds = roc_curve(y[test_idx], y_score)
        split_auc = float(roc_auc_score(y[test_idx], y_score))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tpr.append(interp_tpr)

        curves.append(
            {
                "split_id": split_id,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": split_auc,
            }
        )
        metrics.append(
            _split_metric_row(
                df,
                y,
                groups,
                train_idx=train_idx,
                test_idx=test_idx,
                split_id=split_id,
                auc_value=split_auc,
            )
        )
        predictions.append(
            _prediction_frame(
                df,
                y[test_idx],
                y_score,
                test_idx=test_idx,
                split_id=split_id,
            )
        )

    mean_tpr = np.mean(np.vstack(interpolated_tpr), axis=0)
    mean_tpr[-1] = 1.0
    split_metrics = pd.DataFrame(metrics)
    return RepeatedLogisticResult(
        predictions=pd.concat(predictions, ignore_index=True),
        split_metrics=split_metrics,
        roc_curves=curves,
        mean_fpr=mean_fpr,
        mean_tpr=mean_tpr,
        mean_auc=float(auc(mean_fpr, mean_tpr)),
        std_auc=float(split_metrics["roc_auc"].std(ddof=0)),
    )


def fit_repeated_one_to_many_product_logistic(
    df: pd.DataFrame,
    *,
    n_splits: int = 20,
    test_size: float = 0.30,
    random_state: int = 42,
    profile_column: str = "radial_profile_data",
    label_column: str = "product_status_group",
    group_column: str = "patientId",
    specimen_column: str = "specimenId",
    logreg_c: float = 1.0,
    inner_splits: int = 5,
    target_sensitivity: float = 0.95,
    aggregation: str = "mean",
    extra_feature_columns: Sequence[str] | None = None,
    use_profile: bool = True,
) -> OneToManyProductLogisticResult:
    """Evaluate one-to-many LogisticRegression as specimen-level product model.

    The base model learns from measurement profiles. Train-set out-of-fold
    measurement probabilities are aggregated by specimen to choose thresholds.
    Test measurement probabilities are aggregated by specimen for final
    split-level evaluation.
    """
    matrix_profile_column = profile_column if use_profile else None
    _validate_binary_frame(df, matrix_profile_column, label_column, group_column)
    _require_columns(df, [specimen_column])
    y = df[label_column].map(LABEL_MAP).astype(int).to_numpy()
    groups = df[group_column].astype(str).to_numpy()

    measurement_predictions: list[pd.DataFrame] = []
    specimen_predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    mean_fpr = np.linspace(0.0, 1.0, 101)
    interpolated_tpr: list[np.ndarray] = []
    threshold_rows: list[dict[str, Any]] = []

    for split_id, train_idx, test_idx in _patient_safe_splits(
        y,
        groups,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    ):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        oof_score = _measurement_oof_scores(
            train_df,
            profile_column=profile_column,
            label_column=label_column,
            group_column=group_column,
            logreg_c=logreg_c,
            random_state=random_state + split_id,
            inner_splits=inner_splits,
            extra_feature_columns=extra_feature_columns,
            use_profile=use_profile,
        )
        model = _logistic_pipeline(
            logreg_c=logreg_c,
            random_state=random_state + split_id,
        )
        model.fit(
            model_matrix(train_df, matrix_profile_column, extra_feature_columns),
            train_df[label_column].map(LABEL_MAP),
        )
        test_score = model.predict_proba(
            model_matrix(test_df, matrix_profile_column, extra_feature_columns)
        )[:, 1]

        train_scored = train_df.copy()
        train_scored["p_cancer_measurement"] = oof_score
        test_scored = test_df.copy()
        test_scored["p_cancer_measurement"] = test_score
        train_specimen = aggregate_measurement_scores_by_specimen(
            train_scored,
            score_column="p_cancer_measurement",
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            aggregation=aggregation,
        )
        test_specimen = aggregate_measurement_scores_by_specimen(
            test_scored,
            score_column="p_cancer_measurement",
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            aggregation=aggregation,
        )
        thresholds = compute_binary_thresholds(
            train_specimen["y_true"].to_numpy(),
            train_specimen["p_cancer"].to_numpy(),
            target_sensitivity=target_sensitivity,
        )
        y_test = test_specimen["y_true"].to_numpy()
        p_test = test_specimen["p_cancer"].to_numpy()
        fpr, tpr, roc_thresholds = roc_curve(y_test, p_test)
        split_auc = _safe_roc_auc(y_test, p_test)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tpr.append(interp_tpr)
        curves.append(
            {
                "split_id": split_id,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": roc_thresholds,
                "auc": split_auc,
            }
        )
        metrics.append(
            _product_metric_row(
                split_id=split_id,
                train_df=train_df,
                test_df=test_df,
                train_specimen=train_specimen,
                test_specimen=test_specimen,
                y_true=y_test,
                y_score=p_test,
                thresholds=thresholds,
            )
        )
        threshold_rows.append({"split_id": split_id, **thresholds})
        measurement_predictions.append(
            _measurement_prediction_frame(
                train_scored,
                split_id=split_id,
                set_name="train_oof",
                score_column="p_cancer_measurement",
                label_column=label_column,
            )
        )
        measurement_predictions.append(
            _measurement_prediction_frame(
                test_scored,
                split_id=split_id,
                set_name="test",
                score_column="p_cancer_measurement",
                label_column=label_column,
            )
        )
        specimen_predictions.append(
            _specimen_prediction_frame(
                train_specimen,
                split_id=split_id,
                set_name="train_oof",
                thresholds=thresholds,
            )
        )
        specimen_predictions.append(
            _specimen_prediction_frame(
                test_specimen,
                split_id=split_id,
                set_name="test",
                thresholds=thresholds,
            )
        )

    mean_tpr = np.mean(np.vstack(interpolated_tpr), axis=0)
    mean_tpr[-1] = 1.0
    split_metrics = pd.DataFrame(metrics)
    return OneToManyProductLogisticResult(
        measurement_predictions=pd.concat(measurement_predictions, ignore_index=True),
        specimen_predictions=pd.concat(specimen_predictions, ignore_index=True),
        split_metrics=split_metrics,
        roc_curves=curves,
        mean_fpr=mean_fpr,
        mean_tpr=mean_tpr,
        mean_auc=float(auc(mean_fpr, mean_tpr)),
        std_auc=float(split_metrics["roc_auc"].std(ddof=0)),
        threshold_summary=pd.DataFrame(threshold_rows),
    )


def fit_repeated_one_to_many_product_logistic_comparison(
    datasets: dict[str, pd.DataFrame],
    *,
    n_splits: int = 20,
    test_size: float = 0.30,
    random_state: int = 42,
    profile_column: str = "radial_profile_data",
    label_column: str = "product_status_group",
    group_column: str = "patientId",
    specimen_column: str = "specimenId",
    logreg_c: float = 1.0,
    inner_splits: int = 5,
    target_sensitivity: float = 0.95,
    aggregation: str = "mean",
    extra_feature_columns: Sequence[str] | None = None,
    use_profile: bool = True,
) -> OneToManyProductComparisonResult:
    """Run the same one-to-many product model protocol for multiple datasets."""
    results = {
        name: fit_repeated_one_to_many_product_logistic(
            df,
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
            profile_column=profile_column,
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            logreg_c=logreg_c,
            inner_splits=inner_splits,
            target_sensitivity=target_sensitivity,
            aggregation=aggregation,
            extra_feature_columns=extra_feature_columns,
            use_profile=use_profile,
        )
        for name, df in datasets.items()
    }
    dataset_summary = summarize_one_to_many_datasets(datasets)
    metric_summary = summarize_one_to_many_product_results(results)
    return OneToManyProductComparisonResult(
        results=results,
        dataset_summary=dataset_summary,
        metric_summary=metric_summary,
    )


def fit_repeated_one_to_many_product_feature_ablation_comparison(
    datasets: dict[str, pd.DataFrame],
    feature_routes: dict[str, dict[str, Any]],
    *,
    n_splits: int = 20,
    test_size: float = 0.30,
    random_state: int = 42,
    profile_column: str = "radial_profile_data",
    label_column: str = "product_status_group",
    group_column: str = "patientId",
    specimen_column: str = "specimenId",
    logreg_c: float = 1.0,
    inner_splits: int = 5,
    target_sensitivity: float = 0.95,
    aggregation: str = "mean",
) -> OneToManyProductComparisonResult:
    """Compare profile-only, scalar-only, and combined one-to-many routes."""
    results = {}
    summary_rows = []
    for dataset_name, df in datasets.items():
        base_summary = summarize_one_to_many_dataframe(df).iloc[0].to_dict()
        if "biopsy" in df.columns:
            base_summary["biopsy_true_rows"] = int(
                df["biopsy"].fillna(False).astype(bool).sum()
            )
        else:
            base_summary["biopsy_true_rows"] = np.nan
        for route_name, route in feature_routes.items():
            result_name = f"{dataset_name}__{route_name}"
            use_profile = bool(route.get("use_profile", True))
            extra_feature_columns = list(route.get("extra_feature_columns", []))
            results[result_name] = fit_repeated_one_to_many_product_logistic(
                df,
                n_splits=n_splits,
                test_size=test_size,
                random_state=random_state,
                profile_column=profile_column,
                label_column=label_column,
                group_column=group_column,
                specimen_column=specimen_column,
                logreg_c=logreg_c,
                inner_splits=inner_splits,
                target_sensitivity=target_sensitivity,
                aggregation=aggregation,
                extra_feature_columns=extra_feature_columns,
                use_profile=use_profile,
            )
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "feature_route": route_name,
                    "result": result_name,
                    "use_profile": use_profile,
                    "extra_feature_columns": extra_feature_columns,
                    **base_summary,
                }
            )
    metric_summary = summarize_one_to_many_product_results(results)
    metric_summary[["dataset", "feature_route"]] = metric_summary["dataset"].str.split(
        "__",
        n=1,
        expand=True,
    )
    return OneToManyProductComparisonResult(
        results=results,
        dataset_summary=pd.DataFrame(summary_rows),
        metric_summary=metric_summary,
    )


def summarize_one_to_many_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return basic counts for named one-to-many DataFrames."""
    rows = []
    for name, df in datasets.items():
        row = summarize_one_to_many_dataframe(df).iloc[0].to_dict()
        row["dataset"] = str(name)
        row["biopsy_true_rows"] = (
            int(df["biopsy"].fillna(False).astype(bool).sum())
            if "biopsy" in df.columns
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset").reset_index()


def summarize_one_to_many_product_results(
    results: dict[str, OneToManyProductLogisticResult],
) -> pd.DataFrame:
    """Return one row of mean split metrics for each named product result."""
    rows = []
    for name, result in results.items():
        metrics = result.split_metrics
        rows.append(
            {
                "dataset": str(name),
                "splits": int(len(metrics)),
                "roc_auc_mean": float(metrics["roc_auc"].mean()),
                "roc_auc_std": float(metrics["roc_auc"].std(ddof=0)),
                "pr_auc_mean": float(metrics["pr_auc"].mean()),
                "sensitivity_target_mean": float(
                    metrics["sensitivity_target"].mean()
                ),
                "specificity_target_mean": float(
                    metrics["specificity_target"].mean()
                ),
                "sensitivity_youden_mean": float(
                    metrics["sensitivity_youden"].mean()
                ),
                "specificity_youden_mean": float(
                    metrics["specificity_youden"].mean()
                ),
                "mean_auc_curve": float(result.mean_auc),
            }
        )
    return pd.DataFrame(rows)


def default_fusion_feature_sets() -> dict[str, list[str]]:
    """Return the first Aramis fusion feature-set ladder."""
    return {
        "M0_one_to_many_only": ["logit_p_cancer_one_to_many"],
        "M1_one_to_many_plus_symmetry": [
            "logit_p_cancer_one_to_many",
            "symmetry_available",
            "symmetry_distance_x_available",
        ],
        "M2_plus_quality": [
            "logit_p_cancer_one_to_many",
            "symmetry_available",
            "symmetry_distance_x_available",
            "snr_db_mean",
            "n_valid_target_measurements",
            "n_valid_contralateral_measurements",
            "target_profile_replicate_distance",
            "target_profile_replicate_available",
        ],
        "M3_plus_age_bmi": [
            "logit_p_cancer_one_to_many",
            "symmetry_available",
            "symmetry_distance_x_available",
            "snr_db_mean",
            "n_valid_target_measurements",
            "n_valid_contralateral_measurements",
            "target_profile_replicate_distance",
            "target_profile_replicate_available",
            "age",
            "bmi",
            "age_available",
            "bmi_available",
        ],
    }


def fusion_ablation_feature_sets() -> dict[str, list[str]]:
    """Return core M0-M3 plus age/BMI leakage-control ablations."""
    feature_sets = default_fusion_feature_sets()
    feature_sets.update(
        {
            "A0_age_only": ["age", "age_available"],
            "A1_bmi_only": ["bmi", "bmi_available"],
            "A2_availability_only": [
                "symmetry_available",
                "target_profile_replicate_available",
                "age_available",
                "bmi_available",
            ],
            "F0_symmetry_available_only": ["symmetry_available"],
            "F1_bmi_available_only": ["bmi_available"],
            "F2_replicate_available_only": ["target_profile_replicate_available"],
            "F3_symmetry_plus_bmi_availability": [
                "symmetry_available",
                "bmi_available",
            ],
            "T0_thickness_only": [
                "sample_thickness_mm_mean",
                "sample_thickness_mm_min",
                "sample_thickness_available",
            ],
            "S1_cosine_symmetry_block": [
                "logit_p_cancer_one_to_many",
                "cosine_between_mean_x_available",
                "cosine_within_target_mean_x_available",
                "cosine_within_contralateral_mean_x_available",
                "cosine_asymmetry_score_x_available",
                "cosine_target_contralateral_centroid_x_available",
                "cosine_replicate_variability_delta_x_available",
                "cosine_replicate_variability_ratio_x_available",
                "symmetry_available",
                "target_within_available",
                "contralateral_within_available",
            ],
            "M3a_plus_age_no_bmi": [
                "logit_p_cancer_one_to_many",
                "symmetry_available",
                "symmetry_distance_x_available",
                "snr_db_mean",
                "n_valid_target_measurements",
                "n_valid_contralateral_measurements",
                "target_profile_replicate_distance",
                "target_profile_replicate_available",
                "age",
                "age_available",
            ],
            "M3b_plus_bmi_no_age": [
                "logit_p_cancer_one_to_many",
                "symmetry_available",
                "symmetry_distance_x_available",
                "snr_db_mean",
                "n_valid_target_measurements",
                "n_valid_contralateral_measurements",
                "target_profile_replicate_distance",
                "target_profile_replicate_available",
                "bmi",
                "bmi_available",
            ],
        }
    )
    return feature_sets


def build_fusion_feature_table(
    one_to_many_df: pd.DataFrame,
    one_to_one_df: pd.DataFrame,
    *,
    profile_column: str = "radial_profile_data",
    label_column: str = "product_status_group",
) -> pd.DataFrame:
    """Build specimen-level static features for Aramis fusion experiments."""
    _require_columns(
        one_to_many_df,
        ["patientId", "specimenId", label_column, profile_column],
    )
    _require_columns(one_to_one_df, ["patientId", "specimenId", profile_column])
    rows = []
    for specimen_id, target_df in one_to_many_df.groupby("specimenId", sort=True):
        labels = target_df[label_column].astype(str).unique().tolist()
        patients = target_df["patientId"].astype(str).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"Specimen {specimen_id!r} has labels: {labels}")
        if len(patients) != 1:
            raise ValueError(f"Specimen {specimen_id!r} has patients: {patients}")
        patient_id = patients[0]
        side = _first_string(target_df, "side")
        target_profiles = _profile_array_list(target_df, profile_column)
        symmetry = _symmetry_features_for_target(
            patient_id=patient_id,
            specimen_id=str(specimen_id),
            side=side,
            one_to_one_df=one_to_one_df,
            profile_column=profile_column,
        )
        replicate_distance = _mean_pairwise_distance(target_profiles)
        replicate_available = int(np.isfinite(replicate_distance))
        rows.append(
            {
                "patientId": patient_id,
                "specimenId": str(specimen_id),
                "product_status_group": labels[0],
                "y_true": int(LABEL_MAP[labels[0]]),
                "target_side": side,
                "snr_db_mean": _numeric_mean(target_df, "snr_db", default=0.0),
                "snr_db_min": _numeric_min(target_df, "snr_db", default=0.0),
                "n_valid_target_measurements": int(len(target_df)),
                "sample_thickness_mm_mean": _numeric_mean(
                    target_df,
                    "sample_thickness_mm",
                    default=0.0,
                ),
                "sample_thickness_mm_min": _numeric_min(
                    target_df,
                    "sample_thickness_mm",
                    default=0.0,
                ),
                "sample_thickness_available": int(
                    _has_finite(target_df, "sample_thickness_mm")
                ),
                "target_profile_replicate_distance": (
                    float(replicate_distance) if np.isfinite(replicate_distance) else 0.0
                ),
                "target_profile_replicate_available": replicate_available,
                "age": _numeric_mean(target_df, "age", default=0.0),
                "age_available": int(_has_finite(target_df, "age")),
                "bmi": _bmi_from_group(target_df),
                "bmi_available": int(
                    _has_finite(target_df, "height_in")
                    and _has_finite(target_df, "weight_lb")
                ),
                **symmetry,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def fit_repeated_fusion_logistic_models(
    one_to_many_df: pd.DataFrame,
    one_to_one_df: pd.DataFrame,
    *,
    n_splits: int = 20,
    test_size: float = 0.30,
    random_state: int = 42,
    profile_column: str = "radial_profile_data",
    label_column: str = "product_status_group",
    group_column: str = "patientId",
    specimen_column: str = "specimenId",
    logreg_c: float = 1.0,
    inner_splits: int = 5,
    target_sensitivity: float = 0.95,
    aggregation: str = "mean",
    feature_sets: dict[str, list[str]] | None = None,
    extra_feature_columns: Sequence[str] | None = None,
) -> FusionModelComparisonResult:
    """Compare M0-M3 fusion LogisticRegression models on shared splits."""
    feature_sets = default_fusion_feature_sets() if feature_sets is None else feature_sets
    one_to_many_result = fit_repeated_one_to_many_product_logistic(
        one_to_many_df,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        profile_column=profile_column,
        label_column=label_column,
        group_column=group_column,
        specimen_column=specimen_column,
        logreg_c=logreg_c,
        inner_splits=inner_splits,
        target_sensitivity=target_sensitivity,
        aggregation=aggregation,
        extra_feature_columns=extra_feature_columns,
    )
    feature_table = build_fusion_feature_table(
        one_to_many_df,
        one_to_one_df,
        profile_column=profile_column,
        label_column=label_column,
    )
    scored = _fusion_scored_specimens(one_to_many_result.specimen_predictions, feature_table)
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, Any]]] = {name: [] for name in feature_sets}
    mean_fpr = np.linspace(0.0, 1.0, 101)
    interpolated: dict[str, list[np.ndarray]] = {name: [] for name in feature_sets}

    for split_id in sorted(scored["split_id"].unique()):
        split_df = scored[scored["split_id"] == split_id].copy()
        train_df = split_df[split_df["set_name"] == "train_oof"].copy()
        test_df = split_df[split_df["set_name"] == "test"].copy()
        y_train = train_df["y_true"].to_numpy(dtype=int)
        y_test = test_df["y_true"].to_numpy(dtype=int)
        for model_name, columns in feature_sets.items():
            _require_columns(train_df, columns)
            model = _logistic_pipeline(
                logreg_c=logreg_c,
                random_state=random_state + int(split_id),
            )
            model.fit(train_df[columns].to_numpy(dtype=float), y_train)
            train_score = model.predict_proba(train_df[columns].to_numpy(dtype=float))[:, 1]
            test_score = model.predict_proba(test_df[columns].to_numpy(dtype=float))[:, 1]
            thresholds = compute_binary_thresholds(
                y_train,
                train_score,
                target_sensitivity=target_sensitivity,
            )
            fpr, tpr, roc_thresholds = roc_curve(y_test, test_score)
            split_auc = _safe_roc_auc(y_test, test_score)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interpolated[model_name].append(interp_tpr)
            curves[model_name].append(
                {
                    "split_id": int(split_id),
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": roc_thresholds,
                    "auc": split_auc,
                }
            )
            metrics.append(
                _fusion_metric_row(
                    model_name=model_name,
                    split_id=int(split_id),
                    train_df=train_df,
                    test_df=test_df,
                    y_true=y_test,
                    y_score=test_score,
                    thresholds=thresholds,
                )
            )
            predictions.append(
                _fusion_prediction_frame(
                    test_df,
                    model_name=model_name,
                    split_id=int(split_id),
                    y_score=test_score,
                    thresholds=thresholds,
                )
            )

    mean_roc_curves = {}
    for model_name, tprs in interpolated.items():
        mean_tpr = np.mean(np.vstack(tprs), axis=0)
        mean_tpr[-1] = 1.0
        mean_roc_curves[model_name] = {
            "mean_fpr": mean_fpr,
            "mean_tpr": mean_tpr,
            "mean_auc": float(auc(mean_fpr, mean_tpr)),
        }
    split_metrics = pd.DataFrame(metrics)
    return FusionModelComparisonResult(
        one_to_many_result=one_to_many_result,
        feature_table=feature_table,
        predictions=pd.concat(predictions, ignore_index=True),
        split_metrics=split_metrics,
        metric_summary=summarize_fusion_results(split_metrics, mean_roc_curves),
        roc_curves=curves,
        mean_roc_curves=mean_roc_curves,
        feature_sets={name: list(columns) for name, columns in feature_sets.items()},
    )


def summarize_fusion_results(
    split_metrics: pd.DataFrame,
    mean_roc_curves: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Summarize repeated fusion split metrics by model name."""
    rows = []
    for model_name, group_df in split_metrics.groupby("model_name", sort=False):
        rows.append(
            {
                "model_name": str(model_name),
                "splits": int(len(group_df)),
                "roc_auc_mean": float(group_df["roc_auc"].mean()),
                "roc_auc_std": float(group_df["roc_auc"].std(ddof=0)),
                "pr_auc_mean": float(group_df["pr_auc"].mean()),
                "sensitivity_target_mean": float(group_df["sensitivity_target"].mean()),
                "specificity_target_mean": float(group_df["specificity_target"].mean()),
                "sensitivity_youden_mean": float(group_df["sensitivity_youden"].mean()),
                "specificity_youden_mean": float(group_df["specificity_youden"].mean()),
                "mean_auc_curve": float(mean_roc_curves[str(model_name)]["mean_auc"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_measurement_scores_by_specimen(
    df: pd.DataFrame,
    *,
    score_column: str,
    label_column: str = "product_status_group",
    group_column: str = "patientId",
    specimen_column: str = "specimenId",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Aggregate measurement-level probabilities to specimen/breast rows."""
    _require_columns(df, [score_column, label_column, group_column, specimen_column])
    if aggregation not in {"mean", "median"}:
        raise ValueError(f"Unsupported aggregation: {aggregation!r}.")
    rows = []
    for specimen_id, group_df in df.groupby(specimen_column, sort=True):
        labels = group_df[label_column].astype(str).unique().tolist()
        patients = group_df[group_column].astype(str).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"Specimen {specimen_id!r} has labels: {labels}")
        if len(patients) != 1:
            raise ValueError(f"Specimen {specimen_id!r} has patients: {patients}")
        scores = np.asarray(group_df[score_column], dtype=float)
        score = float(np.mean(scores)) if aggregation == "mean" else float(np.median(scores))
        label = labels[0]
        rows.append(
            {
                "patientId": patients[0],
                "specimenId": str(specimen_id),
                "product_status_group": label,
                "y_true": int(LABEL_MAP[label]),
                "p_cancer": score,
                "n_measurements": int(len(group_df)),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def compute_binary_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_sensitivity: float = 0.95,
) -> dict[str, Any]:
    """Compute Youden and target-sensitivity thresholds on training scores."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_idx = int(np.argmax(tpr - fpr))
    target_mask = tpr >= float(target_sensitivity)
    if bool(np.any(target_mask)):
        candidates = np.where(target_mask)[0]
        target_idx = int(candidates[np.argmin(fpr[candidates])])
        target_reached = True
    else:
        target_idx = youden_idx
        target_reached = False
    return {
        "threshold_youden": float(thresholds[youden_idx]),
        "threshold_target": float(thresholds[target_idx]),
        "target_sensitivity": float(target_sensitivity),
        "target_reached": bool(target_reached),
    }


def profile_matrix(df: pd.DataFrame, profile_column: str) -> np.ndarray:
    """Stack one profile-array column into a 2D model matrix."""
    _require_columns(df, [profile_column])
    profiles = [np.asarray(value, dtype=float) for value in df[profile_column]]
    lengths = {profile.size for profile in profiles}
    if len(lengths) != 1:
        raise ValueError(f"Profiles in {profile_column!r} must have equal length.")
    x = np.vstack(profiles)
    if not np.isfinite(x).all():
        raise ValueError(f"Profiles in {profile_column!r} contain non-finite values.")
    return x


def model_matrix(
    df: pd.DataFrame,
    profile_column: str | None,
    extra_feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Stack profile arrays and optional numeric scalar model features."""
    columns = list(extra_feature_columns or [])
    matrices = []
    if profile_column is not None:
        matrices.append(profile_matrix(df, profile_column))
    if columns:
        _require_columns(df, columns)
        extra = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(extra).all():
            raise ValueError(f"Extra model features contain non-finite values: {columns}")
        matrices.append(extra)
    if not matrices:
        raise ValueError("At least one profile or scalar model feature is required.")
    return np.hstack(matrices)


def _measurement_oof_scores(
    df: pd.DataFrame,
    *,
    profile_column: str,
    label_column: str,
    group_column: str,
    logreg_c: float,
    random_state: int,
    inner_splits: int,
    extra_feature_columns: Sequence[str] | None = None,
    use_profile: bool = True,
) -> np.ndarray:
    groups = df[group_column].astype(str).to_numpy()
    n_splits = max(2, min(int(inner_splits), int(pd.Series(groups).nunique())))
    splitter = GroupKFold(n_splits=n_splits)
    scores = np.full(len(df), np.nan, dtype=float)
    y = df[label_column].map(LABEL_MAP).astype(int).to_numpy()
    for fold_id, (train_idx, val_idx) in enumerate(
        splitter.split(df, y, groups=groups)
    ):
        if len(np.unique(y[train_idx])) < 2:
            raise ValueError("Inner train fold has a single class.")
        model = _logistic_pipeline(
            logreg_c=logreg_c,
            random_state=int(random_state + fold_id),
        )
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        matrix_profile_column = profile_column if use_profile else None
        model.fit(
            model_matrix(train_df, matrix_profile_column, extra_feature_columns),
            y[train_idx],
        )
        scores[val_idx] = model.predict_proba(
            model_matrix(val_df, matrix_profile_column, extra_feature_columns)
        )[:, 1]
    if np.isnan(scores).any():
        raise ValueError("Out-of-fold measurement probabilities were not filled.")
    return scores


def _patient_safe_splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    test_size: float,
    random_state: int,
):
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    seen = 0
    seed = int(random_state)
    attempts = 0
    while seen < int(n_splits) and attempts < int(n_splits) * 200:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=float(test_size),
            random_state=seed + attempts,
        )
        train_idx, test_idx = next(splitter.split(np.zeros_like(y), y, groups))
        attempts += 1
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        if set(groups[train_idx]).intersection(set(groups[test_idx])):
            raise RuntimeError("Patient leakage detected in grouped split.")
        yield seen, train_idx, test_idx
        seen += 1

    if seen != int(n_splits):
        raise ValueError("Could not create enough patient-safe two-class splits.")


def _logistic_pipeline(*, logreg_c: float, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(logreg_c),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=int(random_state),
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _split_metric_row(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_id: int,
    auc_value: float,
) -> dict[str, Any]:
    return {
        "split_id": int(split_id),
        "roc_auc": float(auc_value),
        "train_rows": int(train_idx.size),
        "test_rows": int(test_idx.size),
        "train_patients": int(pd.Series(groups[train_idx]).nunique()),
        "test_patients": int(pd.Series(groups[test_idx]).nunique()),
        "train_benign_rows": int((y[train_idx] == 0).sum()),
        "train_cancer_rows": int((y[train_idx] == 1).sum()),
        "test_benign_rows": int((y[test_idx] == 0).sum()),
        "test_cancer_rows": int((y[test_idx] == 1).sum()),
        "test_specimens": int(df.iloc[test_idx]["specimenId"].astype(str).nunique()),
    }


def _prediction_frame(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    test_idx: np.ndarray,
    split_id: int,
) -> pd.DataFrame:
    keep = [
        column
        for column in ("patientId", "specimenId", "measurementId", "product_status_group")
        if column in df.columns
    ]
    out = df.iloc[test_idx][keep].reset_index(drop=True)
    out["split_id"] = int(split_id)
    out["y_true"] = np.asarray(y_true, dtype=int)
    out["p_cancer"] = np.asarray(y_score, dtype=float)
    return out


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(np.asarray(y_true, dtype=int))) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(np.asarray(y_true, dtype=int))) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def _metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= float(threshold)).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _product_metric_row(
    *,
    split_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_specimen: pd.DataFrame,
    test_specimen: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "split_id": int(split_id),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "train_measurements": int(len(train_df)),
        "test_measurements": int(len(test_df)),
        "train_patients": int(train_df["patientId"].astype(str).nunique()),
        "test_patients": int(test_df["patientId"].astype(str).nunique()),
        "train_specimens": int(len(train_specimen)),
        "test_specimens": int(len(test_specimen)),
        "test_benign_specimens": int((test_specimen["y_true"] == 0).sum()),
        "test_cancer_specimens": int((test_specimen["y_true"] == 1).sum()),
        **thresholds,
    }
    for name, threshold_column in [
        ("youden", "threshold_youden"),
        ("target", "threshold_target"),
    ]:
        metrics = _metrics_at_threshold(
            y_true,
            y_score,
            float(thresholds[threshold_column]),
        )
        for key, value in metrics.items():
            row[f"{key}_{name}"] = value
    return row


def _measurement_prediction_frame(
    df: pd.DataFrame,
    *,
    split_id: int,
    set_name: str,
    score_column: str,
    label_column: str,
) -> pd.DataFrame:
    keep = [
        column
        for column in ("patientId", "specimenId", "measurementId", label_column)
        if column in df.columns
    ]
    out = df[keep].copy().reset_index(drop=True)
    out["split_id"] = int(split_id)
    out["set_name"] = str(set_name)
    out["y_true"] = out[label_column].map(LABEL_MAP).astype(int)
    out["p_cancer"] = np.asarray(df[score_column], dtype=float)
    return out


def _specimen_prediction_frame(
    df: pd.DataFrame,
    *,
    split_id: int,
    set_name: str,
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["split_id"] = int(split_id)
    out["set_name"] = str(set_name)
    out["threshold_youden"] = float(thresholds["threshold_youden"])
    out["threshold_target"] = float(thresholds["threshold_target"])
    out["y_pred_youden"] = (out["p_cancer"] >= out["threshold_youden"]).astype(int)
    out["y_pred_target"] = (out["p_cancer"] >= out["threshold_target"]).astype(int)
    return out


def _fusion_scored_specimens(
    specimen_predictions: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    keep = [
        "patientId",
        "specimenId",
        "product_status_group",
        "y_true",
        "p_cancer",
        "split_id",
        "set_name",
    ]
    _require_columns(specimen_predictions, keep)
    out = specimen_predictions[keep].merge(
        feature_table.drop(columns=["product_status_group", "y_true"]),
        on=["patientId", "specimenId"],
        how="left",
        validate="many_to_one",
    )
    if out["symmetry_available"].isna().any():
        raise ValueError("Missing fusion features for at least one specimen.")
    out["p_cancer_one_to_many"] = out["p_cancer"].astype(float)
    out["logit_p_cancer_one_to_many"] = _logit(out["p_cancer_one_to_many"])
    return out


def _fusion_metric_row(
    *,
    model_name: str,
    split_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "model_name": str(model_name),
        "split_id": int(split_id),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "train_specimens": int(len(train_df)),
        "test_specimens": int(len(test_df)),
        "train_patients": int(train_df["patientId"].astype(str).nunique()),
        "test_patients": int(test_df["patientId"].astype(str).nunique()),
        "test_benign_specimens": int((test_df["y_true"] == 0).sum()),
        "test_cancer_specimens": int((test_df["y_true"] == 1).sum()),
        "test_symmetry_available": int(
            test_df["symmetry_available"].astype(bool).sum()
        ),
        **thresholds,
    }
    for name, threshold_column in [
        ("youden", "threshold_youden"),
        ("target", "threshold_target"),
    ]:
        metrics = _metrics_at_threshold(
            y_true,
            y_score,
            float(thresholds[threshold_column]),
        )
        for key, value in metrics.items():
            row[f"{key}_{name}"] = value
    return row


def _fusion_prediction_frame(
    test_df: pd.DataFrame,
    *,
    model_name: str,
    split_id: int,
    y_score: np.ndarray,
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    keep = [
        "patientId",
        "specimenId",
        "product_status_group",
        "y_true",
        "p_cancer_one_to_many",
        "logit_p_cancer_one_to_many",
        "symmetry_available",
        "symmetry_distance_value",
        "symmetry_distance_x_available",
    ]
    out = test_df[keep].copy().reset_index(drop=True)
    out["model_name"] = str(model_name)
    out["split_id"] = int(split_id)
    out["p_cancer_fusion"] = np.asarray(y_score, dtype=float)
    out["threshold_youden"] = float(thresholds["threshold_youden"])
    out["threshold_target"] = float(thresholds["threshold_target"])
    out["y_pred_youden"] = (out["p_cancer_fusion"] >= out["threshold_youden"]).astype(int)
    out["y_pred_target"] = (out["p_cancer_fusion"] >= out["threshold_target"]).astype(int)
    return out


def _symmetry_features_for_target(
    *,
    patient_id: str,
    specimen_id: str,
    side: str,
    one_to_one_df: pd.DataFrame,
    profile_column: str,
) -> dict[str, Any]:
    patient_df = one_to_one_df[one_to_one_df["patientId"].astype(str) == str(patient_id)]
    target_df = patient_df[patient_df["specimenId"].astype(str) == str(specimen_id)]
    contralateral_df = _contralateral_rows(patient_df, specimen_id=specimen_id, side=side)
    target_profiles = _profile_array_list(target_df, profile_column)
    contralateral_profiles = _profile_array_list(contralateral_df, profile_column)
    cosine = _distance_summary(target_profiles, contralateral_profiles, _cosine_distance)
    available = int(np.isfinite(cosine["asymmetry_score"]))
    target_within_available = int(np.isfinite(cosine["within_target_mean"]))
    contralateral_within_available = int(
        np.isfinite(cosine["within_contralateral_mean"])
    )
    return {
        "symmetry_available": available,
        "target_within_available": target_within_available,
        "contralateral_within_available": contralateral_within_available,
        "both_within_available": int(
            bool(target_within_available and contralateral_within_available)
        ),
        "symmetry_distance_value": _finite_or_zero(cosine["asymmetry_score"]),
        "symmetry_distance_x_available": (
            _finite_or_zero(cosine["asymmetry_score"]) * available
        ),
        "symmetry_between_mean": cosine["between_mean"],
        "symmetry_within_target_mean": cosine["within_target_mean"],
        "symmetry_within_contralateral_mean": cosine["within_contralateral_mean"],
        "symmetry_within_mean": cosine["within_mean"],
        **_prefixed_distance_features("cosine", cosine, available),
        "n_valid_contralateral_measurements": int(len(contralateral_profiles)),
        "contralateral_specimen_count": int(
            contralateral_df["specimenId"].astype(str).nunique()
        )
        if len(contralateral_df)
        else 0,
    }


def _contralateral_rows(
    patient_df: pd.DataFrame,
    *,
    specimen_id: str,
    side: str,
) -> pd.DataFrame:
    if "side" not in patient_df.columns or side == "":
        return patient_df[patient_df["specimenId"].astype(str) != str(specimen_id)]
    side_values = patient_df["side"].astype(str).str.lower()
    return patient_df[
        (patient_df["specimenId"].astype(str) != str(specimen_id))
        & (side_values != str(side).lower())
    ]


def _profile_array_list(df: pd.DataFrame, profile_column: str) -> list[np.ndarray]:
    if len(df) == 0:
        return []
    return [np.asarray(value, dtype=float).ravel() for value in df[profile_column]]


def _mean_cross_distance(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    values = [
        _cosine_distance(a, b)
        for a in left
        for b in right
        if _compatible_profiles(a, b)
    ]
    return _finite_mean(values)


def _mean_pairwise_distance(profiles: list[np.ndarray]) -> float:
    if len(profiles) < 2:
        return np.nan
    values = [
        _cosine_distance(profiles[i], profiles[j])
        for i in range(len(profiles) - 1)
        for j in range(i + 1, len(profiles))
        if _compatible_profiles(profiles[i], profiles[j])
    ]
    return _finite_mean(values)


def _distance_summary(
    target_profiles: list[np.ndarray],
    contralateral_profiles: list[np.ndarray],
    distance_fn,
) -> dict[str, float]:
    between = _mean_cross_distance_with(
        target_profiles,
        contralateral_profiles,
        distance_fn,
    )
    within_target = _mean_pairwise_distance_with(target_profiles, distance_fn)
    within_contralateral = _mean_pairwise_distance_with(
        contralateral_profiles,
        distance_fn,
    )
    within = _finite_mean([within_target, within_contralateral])
    asymmetry = (
        float(between - within)
        if np.isfinite(between) and np.isfinite(within)
        else np.nan
    )
    centroid = _centroid_distance(target_profiles, contralateral_profiles, distance_fn)
    variability_delta = (
        float(within_target - within_contralateral)
        if np.isfinite(within_target) and np.isfinite(within_contralateral)
        else np.nan
    )
    variability_ratio = (
        float(within_target / (within_contralateral + 1e-12))
        if np.isfinite(within_target) and np.isfinite(within_contralateral)
        else np.nan
    )
    return {
        "between_mean": between,
        "within_target_mean": within_target,
        "within_contralateral_mean": within_contralateral,
        "within_mean": within,
        "asymmetry_score": asymmetry,
        "target_contralateral_centroid": centroid,
        "replicate_variability_delta": variability_delta,
        "replicate_variability_ratio": variability_ratio,
    }


def _prefixed_distance_features(
    prefix: str,
    summary: dict[str, float],
    available: int,
) -> dict[str, float]:
    out = {}
    for key, value in summary.items():
        clean = _finite_or_zero(value)
        out[f"{prefix}_{key}"] = clean
        out[f"{prefix}_{key}_x_available"] = clean * available
    return out


def _mean_cross_distance_with(
    left: list[np.ndarray],
    right: list[np.ndarray],
    distance_fn,
) -> float:
    values = [
        distance_fn(a, b)
        for a in left
        for b in right
        if _compatible_profiles(a, b)
    ]
    return _finite_mean(values)


def _mean_pairwise_distance_with(profiles: list[np.ndarray], distance_fn) -> float:
    if len(profiles) < 2:
        return np.nan
    values = [
        distance_fn(profiles[i], profiles[j])
        for i in range(len(profiles) - 1)
        for j in range(i + 1, len(profiles))
        if _compatible_profiles(profiles[i], profiles[j])
    ]
    return _finite_mean(values)


def _centroid_distance(
    left: list[np.ndarray],
    right: list[np.ndarray],
    distance_fn,
) -> float:
    left_centroid = _profile_centroid(left)
    right_centroid = _profile_centroid(right)
    if left_centroid.size == 0 or right_centroid.size == 0:
        return np.nan
    return distance_fn(left_centroid, right_centroid)


def _compatible_profiles(a: np.ndarray, b: np.ndarray) -> bool:
    return a.size > 0 and b.size > 0 and min(a.size, b.size) > 1


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dim = min(a.size, b.size)
    x = np.nan_to_num(a[:dim], nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(b[:dim], nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - float(np.dot(x, y)) / denom)


def _profile_centroid(profiles: list[np.ndarray]) -> np.ndarray:
    valid = [profile for profile in profiles if profile.size > 0]
    if not valid:
        return np.asarray([], dtype=float)
    dim = min(profile.size for profile in valid)
    if dim < 2:
        return np.asarray([], dtype=float)
    matrix = np.vstack(
        [
            np.nan_to_num(profile[:dim], nan=0.0, posinf=0.0, neginf=0.0)
            for profile in valid
        ]
    )
    return np.mean(matrix, axis=0)


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.mean(array))


def _first_string(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return ""
    values = df[column].dropna().astype(str)
    return "" if len(values) == 0 else str(values.iloc[0])


def _numeric_mean(df: pd.DataFrame, column: str, *, default: float) -> float:
    if not _has_finite(df, column):
        return float(default)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return float(np.nanmean(values[np.isfinite(values)]))


def _numeric_min(df: pd.DataFrame, column: str, *, default: float) -> float:
    if not _has_finite(df, column):
        return float(default)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return float(np.nanmin(values[np.isfinite(values)]))


def _has_finite(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return bool(np.any(np.isfinite(values)))


def _bmi_from_group(df: pd.DataFrame) -> float:
    if not (_has_finite(df, "height_in") and _has_finite(df, "weight_lb")):
        return 0.0
    height = _numeric_mean(df, "height_in", default=0.0)
    weight = _numeric_mean(df, "weight_lb", default=0.0)
    if height <= 0.0 or weight <= 0.0:
        return 0.0
    return float(weight / (height * height) * 703.0)


def _logit(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _validate_binary_frame(
    df: pd.DataFrame,
    profile_column: str | None,
    label_column: str,
    group_column: str,
) -> None:
    columns = [label_column, group_column, "specimenId"]
    if profile_column is not None:
        columns.append(profile_column)
    _require_columns(df, columns)
    labels = set(df[label_column].dropna().astype(str).unique())
    if labels != set(LABEL_MAP):
        raise ValueError(f"Expected only BENIGN/CANCER labels, got: {sorted(labels)}")
    if df[group_column].isna().any():
        raise ValueError(f"Missing {group_column} values are not allowed.")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
