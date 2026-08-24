"""Focused synthetic tests for the paired model comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aramina.additive_recalibration import (
    DELTA_LOWER_BOUND,
    RecalibratedJointAdditiveClassifier,
)
from aramina.paired_cohort import (
    construct_common_cohort,
    dataset_context,
    model_columns,
)
from aramina.paired_contract import (
    ADDITIVE_MODEL,
    FPCA30_MODEL,
    MODEL_NAMES,
    RAW100_MODEL,
)
from aramina.paired_evaluation import run_paired_evaluation
from aramina.paired_models import (
    ProfileSpec,
    fit_feature_pair,
)
from aramina.training_evaluation import _fit_split_feature_tables


def _frame(*, npt: int, patients: int = 40, measurements_per_side: int = 8):
    q = np.linspace(2.0, 23.0, npt)
    rows = []
    for patient_index in range(patients):
        cancer = patient_index % 2 == 1
        label = "CANCER" if cancer else "BENIGN"
        for side_index, side in enumerate(("Left", "Right")):
            for measurement_index in range(measurements_per_side):
                profile = (
                    3.0
                    + 0.10 * np.sin(q * (0.31 + patient_index * 0.001))
                    + 0.04 * np.cos(q * 0.75 + side_index * 0.2)
                    + measurement_index * 0.001
                )
                if cancer and side == "Left":
                    profile += 0.06 * np.exp(-((q - 14.0) / 1.3) ** 2)
                rows.append(
                    {
                        "patientId": f"P{patient_index:03d}",
                        "specimenId": f"P{patient_index:03d}_{side}",
                        "side": side,
                        "position": f"P{measurement_index:02d}",
                        "started_at": f"2026-01-{patient_index + 1:02d}T00:00:00",
                        "product_status_group": label,
                        "age": 35 + patient_index,
                        "biopsy": side == "Left",
                        "q_range": q.copy(),
                        "radial_profile_data": profile,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def matched_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _frame(npt=100), _frame(npt=256)


def test_additive_full_model_uses_gates_and_positive_profile_slope():
    profile = np.array([0.04, 0.08, 0.15, 0.24, 0.38, 0.58, 0.72, 0.84, 0.91, 0.96])
    rows = pd.DataFrame(
        {
            "profile_p_cancer_logit_average": profile,
            "age": [31, 35, 38, 43, 47, 52, 57, 63, 68, 73],
            "age_available": [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            "symmetry_available": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            "sk_wasserstein_distance_full_q2": [
                0.1,
                np.nan,
                0.2,
                0.3,
                np.nan,
                0.6,
                0.7,
                np.nan,
                0.9,
                1.0,
            ],
            "sk_weightedrms1": [
                0.1,
                np.nan,
                0.2,
                0.25,
                np.nan,
                0.55,
                0.65,
                np.nan,
                0.8,
                0.9,
            ],
            "sk_weightedrms2": [
                0.15,
                np.nan,
                0.25,
                0.35,
                np.nan,
                0.65,
                0.75,
                np.nan,
                0.95,
                1.05,
            ],
            "sk_mean_peak_value_abs_delta": [
                0.05,
                np.nan,
                0.12,
                0.18,
                np.nan,
                0.32,
                0.42,
                np.nan,
                0.58,
                0.70,
            ],
        }
    )
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    model = RecalibratedJointAdditiveClassifier().fit(rows, labels)

    assert model.recalibration_parameters_["profile_logit_slope"] > 0.0
    assert model.recalibration_parameters_["profile_logit_delta"] > DELTA_LOWER_BOUND
    assert all("available" not in name for name in model.feature_names_)
    assert model.penalty_by_feature_["intercept"] == 0.0
    assert model.predict_proba(rows).shape == (len(rows), 2)


def test_common_cohort_intersects_measurements_and_rejects_metadata_mismatch(
    matched_frames,
):
    raw100, fpca256 = matched_frames
    raw_extra = raw100.iloc[[0]].copy()
    raw_extra["position"] = "raw-only"
    raw_with_extra = pd.concat([raw100, raw_extra], ignore_index=True)

    raw_common, fpca_common, cases = construct_common_cohort(
        raw_with_extra, fpca256
    )

    assert len(raw_common) == len(fpca_common) == len(fpca256)
    assert len(cases) == raw100["patientId"].nunique()
    assert not cases["target_case_id"].duplicated().any()

    changed = fpca256.copy()
    changed.loc[0, "age"] += 1
    with pytest.raises(ValueError, match="differ in label, biopsy, or age"):
        construct_common_cohort(raw100, changed)


def test_fpca30_feature_path_matches_production_helper(matched_frames):
    _, fpca256 = matched_frames
    train_ids = {f"P{index:03d}" for index in range(30)}
    test_ids = set(fpca256["patientId"].astype(str)).difference(train_ids)
    train = fpca256.loc[fpca256["patientId"].isin(train_ids)].copy()
    test = fpca256.loc[fpca256["patientId"].isin(test_ids)].copy()
    model = model_columns()
    context = dataset_context(fpca256)

    paired = fit_feature_pair(
        fpca256,
        context,
        train_ids=train_ids,
        validation_ids=test_ids,
        spec=ProfileSpec(FPCA30_MODEL, 256, "fpca", 30),
        model=model,
        random_state=17,
    )
    production_train, production_test = _fit_split_feature_tables(
        train,
        test,
        profile_column=model["profile_column"],
        label_column=model["label_column"],
        group_column=model["group_column"],
        specimen_column=model["specimen_column"],
        side_column=model["side_column"],
        q_column=model["q_column"],
        age_column=model["age_column"],
        biopsy_column=model["biopsy_column"],
        lr1_row_policy=model["lr1_row_policy"],
        lr1_logreg_c=model["lr1_logreg_c"],
        random_state=17,
    )
    for actual, expected in (
        (paired.train, production_train),
        (paired.validation, production_test),
    ):
        actual = actual.sort_values("target_case_id").reset_index(drop=True)
        expected = expected.sort_values("target_case_id").reset_index(drop=True)
        assert actual["target_case_id"].equals(expected["target_case_id"])
        np.testing.assert_allclose(
            actual["profile_p_cancer_logit_average"],
            expected["profile_p_cancer_logit_average"],
            rtol=0.0,
            atol=1e-12,
        )


def test_paired_evaluation_uses_identical_outer_cases_and_nested_fpca(
    tmp_path: Path,
    matched_frames,
):
    raw100, fpca256 = matched_frames
    result = run_paired_evaluation(
        raw100,
        fpca256,
        tmp_path,
        n_splits=2,
        n_repeats=1,
        inner_lr1_splits=2,
        meta_splits=2,
        random_state=19,
        bootstrap_samples=20,
    )

    expected_outputs = {
        "case_manifest.csv",
        "fold_manifest.csv",
        "fold_metrics.csv",
        "fold_predictions.csv",
        "measurement_manifest.csv",
        "paired_delta_summary.csv",
        "paired_fold_deltas.csv",
        "run_metadata.yaml",
        "summary.csv",
        "threshold_scores.csv",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_outputs
    assert set(result["fold_metrics"]["model_name"]) == set(MODEL_NAMES)
    assert len(result["fold_metrics"]) == 2 * len(MODEL_NAMES)
    assert set(result["summary"]["model_name"]) == set(MODEL_NAMES)
    comparisons = result["paired_delta_summary"][
        ["comparison", "candidate_model", "reference_model"]
    ].drop_duplicates()
    assert set(map(tuple, comparisons.to_numpy())) == {
        ("encoder_effect", FPCA30_MODEL, RAW100_MODEL),
        ("architecture_effect", ADDITIVE_MODEL, FPCA30_MODEL),
        ("total_effect", ADDITIVE_MODEL, RAW100_MODEL),
    }
    metrics = result["fold_metrics"].set_index(["split_id", "model_name"])
    architecture_deltas = result["paired_fold_deltas"].loc[
        result["paired_fold_deltas"]["comparison"].eq("architecture_effect")
    ]
    for delta in architecture_deltas.itertuples(index=False):
        assert delta.delta_roc_auc == pytest.approx(
            metrics.at[(delta.split_id, ADDITIVE_MODEL), "roc_auc"]
            - metrics.at[(delta.split_id, FPCA30_MODEL), "roc_auc"]
        )

    outer = result["fold_manifest"].loc[
        result["fold_manifest"]["level"].eq("outer")
    ]
    predictions = result["fold_predictions"]
    for split_id, split_manifest in outer.groupby("split_id"):
        expected_cases = set(
            split_manifest.loc[
                split_manifest["role"].eq("outer_test"), "target_case_id"
            ]
        )
        test_patients = set(
            split_manifest.loc[
                split_manifest["role"].eq("outer_test"), "patientId"
            ]
        )
        nested_patients = set(
            result["fold_manifest"].loc[
                result["fold_manifest"]["split_id"].eq(split_id)
                & ~result["fold_manifest"]["level"].eq("outer"),
                "patientId",
            ]
        )
        assert test_patients.isdisjoint(nested_patients)
        for model_name in MODEL_NAMES:
            observed = predictions.loc[
                predictions["split_id"].eq(split_id)
                & predictions["model_name"].eq(model_name),
                "target_case_id",
            ]
            assert set(observed) == expected_cases
            assert not observed.duplicated().any()

    additive_metrics = result["fold_metrics"].loc[
        result["fold_metrics"]["model_name"].eq(ADDITIVE_MODEL)
    ]
    assert additive_metrics["model_fit_provenance"].eq(
        "outer_train_patient_safe_fpca30_lr1_oof_additive_meta_fit"
    ).all()
    assert additive_metrics["threshold_provenance"].str.contains(
        "nested_full_chain"
    ).all()
    product_metrics = result["fold_metrics"].loc[
        result["fold_metrics"]["model_name"].isin([RAW100_MODEL, FPCA30_MODEL])
    ]
    assert product_metrics["model_fit_provenance"].eq(
        "outer_train_same_data_lr1_to_lr2"
    ).all()
