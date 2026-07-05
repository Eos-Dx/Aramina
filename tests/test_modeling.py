from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aramis.modeling import (
    aggregate_measurement_scores_by_specimen,
    build_fusion_feature_table,
    compute_binary_thresholds,
    default_fusion_feature_sets,
    fit_repeated_fusion_logistic_models,
    fit_repeated_one_to_many_product_feature_ablation_comparison,
    fit_repeated_one_to_many_product_logistic_comparison,
    fit_repeated_one_to_many_product_logistic,
    fit_repeated_one_to_many_logistic,
    fusion_ablation_feature_sets,
    load_one_to_many_dataframe,
    model_matrix,
    profile_matrix,
    summarize_fusion_results,
    summarize_one_to_many_datasets,
    summarize_one_to_many_dataframe,
    summarize_one_to_many_product_results,
)


def _one_to_many_model_frame() -> pd.DataFrame:
    rows = []
    q = np.linspace(2.0, 23.0, 20)
    for patient_idx in range(20):
        label = "CANCER" if patient_idx % 2 else "BENIGN"
        for measurement_idx in range(2):
            baseline = 1.0 if label == "CANCER" else -1.0
            profile = baseline + np.sin(q / 3.0) + measurement_idx * 0.01
            rows.append(
                {
                    "patientId": f"P{patient_idx:02d}",
                    "specimenId": f"P{patient_idx:02d}_RIGHT",
                    "measurementId": f"P{patient_idx:02d}_M{measurement_idx}",
                    "product_status_group": label,
                    "radial_profile_data": profile,
                    "q_range": q,
                    "sample_thickness_mm": 8.0 + measurement_idx * 0.1,
                }
            )
    return pd.DataFrame(rows)


def _one_to_one_model_frame() -> pd.DataFrame:
    rows = []
    q = np.linspace(2.0, 23.0, 20)
    for patient_idx in range(20):
        label = "CANCER" if patient_idx % 2 else "BENIGN"
        for side in ("LEFT", "RIGHT"):
            specimen_id = f"P{patient_idx:02d}_{side}"
            specimen_label = label if side == "RIGHT" else "NORMAL"
            for measurement_idx in range(2):
                baseline = 1.0 if specimen_label == "CANCER" else -1.0
                if specimen_label == "NORMAL":
                    baseline = -0.8
                profile = baseline + np.cos(q / 4.0) + measurement_idx * 0.01
                rows.append(
                    {
                        "patientId": f"P{patient_idx:02d}",
                        "specimenId": specimen_id,
                        "measurementId": f"{specimen_id}_M{measurement_idx}",
                        "product_status_group": specimen_label,
                        "radial_profile_data": profile,
                        "q_range": q,
                        "sample_thickness_mm": 8.0 + measurement_idx * 0.1,
                        "side": "Right" if side == "RIGHT" else "Left",
                        "snr_db": 25.0 + measurement_idx,
                        "age": 45 + patient_idx,
                        "height_in": 65 + patient_idx % 3,
                        "weight_lb": 130 + patient_idx,
                    }
                )
    return pd.DataFrame(rows)


def test_repeated_one_to_many_logistic_has_patient_safe_splits():
    df = _one_to_many_model_frame()

    result = fit_repeated_one_to_many_logistic(
        df,
        n_splits=5,
        test_size=0.30,
        random_state=7,
    )

    assert len(result.split_metrics) == 5
    assert set(result.predictions["split_id"]) == set(range(5))
    assert result.mean_fpr.shape == result.mean_tpr.shape
    assert result.split_metrics["roc_auc"].between(0.0, 1.0).all()
    for split_id in range(5):
        test_patients = set(
            result.predictions.loc[
                result.predictions["split_id"] == split_id,
                "patientId",
            ]
        )
        train_rows = result.split_metrics.loc[
            result.split_metrics["split_id"] == split_id,
            "train_patients",
        ].iloc[0]
        assert len(test_patients) + int(train_rows) == df["patientId"].nunique()


def test_repeated_one_to_many_product_logistic_returns_specimen_predictions():
    df = _one_to_many_model_frame()

    result = fit_repeated_one_to_many_product_logistic(
        df,
        n_splits=4,
        test_size=0.30,
        random_state=11,
        inner_splits=3,
    )

    assert len(result.split_metrics) == 4
    assert set(result.specimen_predictions["set_name"]) == {"train_oof", "test"}
    assert set(result.measurement_predictions["set_name"]) == {"train_oof", "test"}
    assert result.split_metrics["roc_auc"].between(0.0, 1.0).all()
    assert {"threshold_youden", "threshold_target"}.issubset(
        result.threshold_summary.columns
    )


def test_one_to_many_product_comparison_returns_named_summaries():
    df = _one_to_many_model_frame()
    biopsy_df = df[df["patientId"].isin([f"P{idx:02d}" for idx in range(12)])].copy()

    result = fit_repeated_one_to_many_product_logistic_comparison(
        {
            "standard": df,
            "biopsy_only": biopsy_df,
        },
        n_splits=3,
        test_size=0.30,
        random_state=17,
        inner_splits=3,
    )

    assert set(result.results) == {"standard", "biopsy_only"}
    assert set(result.dataset_summary["dataset"]) == {"standard", "biopsy_only"}
    assert set(result.metric_summary["dataset"]) == {"standard", "biopsy_only"}
    assert result.metric_summary["roc_auc_mean"].between(0.0, 1.0).all()


def test_summary_helpers_handle_named_product_results():
    df = _one_to_many_model_frame()
    result = fit_repeated_one_to_many_product_logistic(
        df,
        n_splits=2,
        test_size=0.30,
        random_state=21,
        inner_splits=3,
        aggregation="median",
    )

    dataset_summary = summarize_one_to_many_datasets({"standard": df})
    metric_summary = summarize_one_to_many_product_results({"standard": result})

    assert dataset_summary.loc[0, "dataset"] == "standard"
    assert metric_summary.loc[0, "dataset"] == "standard"
    assert metric_summary.loc[0, "splits"] == 2


def test_aggregate_measurement_scores_by_specimen_uses_mean_score():
    df = _one_to_many_model_frame().head(4).copy()
    df["p_cancer_measurement"] = [0.1, 0.3, 0.8, 1.0]

    out = aggregate_measurement_scores_by_specimen(
        df,
        score_column="p_cancer_measurement",
    )

    assert len(out) == 2
    np.testing.assert_allclose(sorted(out["p_cancer"]), [0.2, 0.9])


def test_aggregate_measurement_scores_by_specimen_rejects_bad_inputs():
    df = _one_to_many_model_frame().head(2).copy()
    df["p_cancer_measurement"] = [0.1, 0.2]

    with pytest.raises(ValueError, match="Unsupported aggregation"):
        aggregate_measurement_scores_by_specimen(
            df,
            score_column="p_cancer_measurement",
            aggregation="max",
        )

    conflicting = pd.concat(
        [
            df,
            df.assign(product_status_group="CANCER"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="has labels"):
        aggregate_measurement_scores_by_specimen(
            conflicting,
            score_column="p_cancer_measurement",
        )


def test_one_to_many_summary_counts_rows_patients_specimens_and_labels():
    df = _one_to_many_model_frame()

    summary = summarize_one_to_many_dataframe(df)

    assert summary.loc[0, "rows"] == 40
    assert summary.loc[0, "patients"] == 20
    assert summary.loc[0, "specimens"] == 20
    assert summary.loc[0, "BENIGN_rows"] == 20
    assert summary.loc[0, "CANCER_rows"] == 20


def test_profile_matrix_rejects_different_profile_lengths():
    df = _one_to_many_model_frame().head(2).copy()
    df.at[df.index[1], "radial_profile_data"] = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="equal length"):
        profile_matrix(df, "radial_profile_data")


def test_profile_matrix_rejects_non_finite_profiles():
    df = _one_to_many_model_frame().head(2).copy()
    profile = np.asarray(df.iloc[0]["radial_profile_data"]).copy()
    profile[0] = np.nan
    df.at[df.index[0], "radial_profile_data"] = profile

    with pytest.raises(ValueError, match="non-finite"):
        profile_matrix(df, "radial_profile_data")


def test_model_matrix_appends_sample_thickness_feature():
    df = _one_to_many_model_frame().head(3).copy()

    x = model_matrix(df, "radial_profile_data", ["sample_thickness_mm"])

    assert x.shape == (3, 21)
    np.testing.assert_allclose(x[:, -1], df["sample_thickness_mm"].to_numpy())


def test_model_matrix_can_use_sample_thickness_without_profile():
    df = _one_to_many_model_frame().head(3).copy()

    x = model_matrix(df, None, ["sample_thickness_mm"])

    assert x.shape == (3, 1)
    np.testing.assert_allclose(x[:, 0], df["sample_thickness_mm"].to_numpy())


def test_compute_binary_thresholds_handles_target_sensitivity():
    thresholds = compute_binary_thresholds(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        target_sensitivity=0.9,
    )

    assert thresholds["target_reached"] is True
    assert thresholds["threshold_youden"] >= 0.0


def test_load_one_to_many_dataframe_rejects_non_dataframe(tmp_path):
    path = tmp_path / "bad.joblib"
    import joblib

    joblib.dump({"not": "a dataframe"}, path)

    with pytest.raises(TypeError, match="Preprocessing joblib"):
        load_one_to_many_dataframe(path)


def test_fusion_feature_table_and_model_comparison_run_on_shared_splits():
    one_to_many = _one_to_many_model_frame()
    one_to_one = _one_to_one_model_frame()

    feature_table = build_fusion_feature_table(one_to_many, one_to_one)
    result = fit_repeated_fusion_logistic_models(
        one_to_many,
        one_to_one,
        n_splits=2,
        test_size=0.30,
        random_state=31,
        inner_splits=3,
        feature_sets={
            "M0": ["logit_p_cancer_one_to_many"],
            "M1": [
                "logit_p_cancer_one_to_many",
                "symmetry_available",
                "symmetry_distance_x_available",
            ],
        },
    )

    assert len(feature_table) == one_to_many["specimenId"].nunique()
    assert set(result.feature_sets) == {"M0", "M1"}
    assert set(result.predictions["model_name"]) == {"M0", "M1"}
    assert set(result.metric_summary["model_name"]) == {"M0", "M1"}
    assert result.feature_table["symmetry_available"].isin([0, 1]).all()


def test_one_to_many_feature_ablation_compares_thickness_only_route():
    df = _one_to_many_model_frame()

    result = fit_repeated_one_to_many_product_feature_ablation_comparison(
        {"standard": df},
        {
            "profile_only": {"use_profile": True, "extra_feature_columns": []},
            "thickness_only": {
                "use_profile": False,
                "extra_feature_columns": ["sample_thickness_mm"],
            },
        },
        n_splits=2,
        test_size=0.30,
        random_state=41,
        inner_splits=3,
    )

    assert set(result.results) == {
        "standard__profile_only",
        "standard__thickness_only",
    }
    assert set(result.metric_summary["feature_route"]) == {
        "profile_only",
        "thickness_only",
    }
    assert result.dataset_summary["extra_feature_columns"].notna().all()


def test_fusion_helpers_expose_expected_feature_set_names():
    assert "M0_one_to_many_only" in default_fusion_feature_sets()
    assert "sample_thickness_mm_mean" not in default_fusion_feature_sets()["M2_plus_quality"]
    assert "A0_age_only" in fusion_ablation_feature_sets()
    assert "T0_thickness_only" in fusion_ablation_feature_sets()


def test_summarize_fusion_results_accepts_precomputed_mean_curves():
    metrics = pd.DataFrame(
        {
            "model_name": ["M0", "M0"],
            "roc_auc": [0.5, 0.7],
            "pr_auc": [0.4, 0.6],
            "sensitivity_target": [1.0, 0.5],
            "specificity_target": [0.2, 0.4],
            "sensitivity_youden": [0.8, 0.9],
            "specificity_youden": [0.7, 0.6],
        }
    )

    out = summarize_fusion_results(metrics, {"M0": {"mean_auc": 0.61}})

    assert out.loc[0, "model_name"] == "M0"
    assert out.loc[0, "splits"] == 2
    assert out.loc[0, "mean_auc_curve"] == 0.61
