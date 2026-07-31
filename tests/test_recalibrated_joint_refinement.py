"""Focused contract tests for nested recalibrated-joint research code."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest


EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1] / "experiments/profile_symmetry_age_refinement"
)
MODEL_PATH = EXPERIMENT_DIR / "recalibrated_joint_model.py"
RUNNER_PATH = EXPERIMENT_DIR / "run_recalibrated_joint_experiment.py"
DATA_PATH = EXPERIMENT_DIR / "recalibrated_joint_data.py"
SELECTOR_PATH = EXPERIMENT_DIR / "select_recalibrated_joint_regularization.py"


def _module(path: Path, name: str):
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _model_module():
    return _module(MODEL_PATH, "recalibrated_joint_model_test")


def _runner_module():
    return _module(RUNNER_PATH, "recalibrated_joint_runner_test")


def _data_module():
    return _module(DATA_PATH, "recalibrated_joint_data_test")


def _selector_module():
    return _module(SELECTOR_PATH, "recalibrated_joint_selector_test")


def _feature_rows() -> tuple[pd.DataFrame, np.ndarray]:
    profile = np.array([0.05, 0.10, 0.18, 0.25, 0.35, 0.55, 0.70, 0.85, 0.92, 0.97])
    rows = pd.DataFrame(
        {
            "profile_p_cancer_logit_average": profile,
            "age": [31, 35, 38, 43, 47, 52, 57, 63, 68, 73],
            "age_available": [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            "symmetry_available": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            "sk_wasserstein_distance_full_q2": [0.1, np.nan, 0.2, 0.3, np.nan, 0.6, 0.7, np.nan, 0.9, 1.0],
            "sk_weightedrms1": [0.1, np.nan, 0.2, 0.25, np.nan, 0.55, 0.65, np.nan, 0.8, 0.9],
            "sk_weightedrms2": [0.15, np.nan, 0.25, 0.35, np.nan, 0.65, 0.75, np.nan, 0.95, 1.05],
            "sk_mean_peak_value_abs_delta": [0.05, np.nan, 0.12, 0.18, np.nan, 0.32, 0.42, np.nan, 0.58, 0.70],
        }
    )
    return rows, np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def _measurements() -> pd.DataFrame:
    rows = []
    q = np.linspace(2.0, 23.0, 24)
    for patient_index in range(16):
        label = "CANCER" if patient_index % 2 else "BENIGN"
        shift = 0.25 if label == "CANCER" else -0.25
        for side, biopsy in (("Left", True), ("Right", False)):
            for measurement_index in range(2):
                rows.append(
                    {
                        "patientId": f"P{patient_index:02d}",
                        "specimenId": f"S{patient_index:02d}_{side}",
                        "measurementId": f"M{patient_index:02d}_{side}_{measurement_index}",
                        "side": side,
                        "biopsy": biopsy,
                        "product_status_group": label,
                        "age": 30 + patient_index,
                        "q_range": q,
                        "radial_profile_data": np.sin(q / 3.0) + shift + measurement_index * 0.01,
                    }
                )
    return pd.DataFrame(rows)


def test_gates_identity_monotonic_slope_and_active_value_validation():
    module = _model_module()
    rows, y = _feature_rows()
    model = module.RecalibratedJointAdditiveClassifier(
        profile_c=0.3,
        age_c=0.3,
        symmetry_c=0.3,
    ).fit(rows, y)
    components = model.prediction_components(rows)

    np.testing.assert_allclose(components.loc[rows.age_available.eq(0), "age_logit_contribution"], 0.0, atol=1e-12)
    np.testing.assert_allclose(components.loc[rows.symmetry_available.eq(0), "symmetry_logit_contribution"], 0.0, atol=1e-12)
    assert all("available" not in name for name in model.feature_names_)
    assert model.penalty_by_feature_["intercept"] == 0.0
    assert model.recalibration_parameters_["profile_logit_slope"] > 0.0
    assert model.recalibration_parameters_["profile_logit_delta"] > module.DELTA_LOWER_BOUND

    with pytest.raises(ValueError, match="age_available requires a finite age"):
        module.RecalibratedJointAdditiveClassifier().fit(rows.assign(age=np.nan), y)
    broken_symmetry = rows.copy()
    broken_symmetry.loc[broken_symmetry.symmetry_available.eq(1), "sk_weightedrms1"] = np.nan
    with pytest.raises(ValueError, match="symmetry_available requires finite SK Core4"):
        module.RecalibratedJointAdditiveClassifier().fit(broken_symmetry, y)


def test_strict_nested_full_chain_manifest_excludes_meta_validation_from_lr1_training():
    data = _data_module()
    frame = _measurements()
    pairs = data.full_chain_meta_pairs(
        frame,
        data.model_columns(),
        lr1_c=0.1,
        meta_splits=2,
        random_state=5,
        outer_split_id=3,
        inner_lr1_splits=2,
    )
    assert len(pairs) == 2
    for pair in pairs:
        manifest = pair.manifest
        validation = set(
            manifest.loc[manifest.role.eq("meta_model_validation"), "patient_id"]
        )
        lr1_train = set(manifest.loc[manifest.role.eq("lr1_fit_train"), "patient_id"])
        meta_train = set(manifest.loc[manifest.role.eq("meta_model_train"), "patient_id"])
        assert validation.isdisjoint(lr1_train)
        assert validation.isdisjoint(meta_train)
        assert set(pair.meta_train_features.patientId.astype(str)).isdisjoint(validation)
        assert set(pair.meta_validation_features.patientId.astype(str)) == validation


def test_runner_outputs_threshold_evidence_and_cli_controls(tmp_path, monkeypatch):
    runner = _runner_module()
    source = tmp_path / "input.joblib"
    joblib.dump({"dataframe": _measurements()}, source)
    payload = runner.run_experiment(
        runner.load_input_dataframe(source),
        tmp_path / "outputs",
        input_path=source,
        outer_splits=2,
        outer_repeats=1,
        inner_lr1_splits=2,
        meta_splits=2,
        candidate_c=(0.03, 0.1, 0.3),
        random_state=7,
    )
    output = tmp_path / "outputs"
    assert {path.name for path in output.iterdir()} == {
        "fold_manifest.csv", "fold_metrics.csv", "paired_fold_deltas.csv",
        "regularization_selection.csv", "split_predictions.csv", "summary.csv",
        "summary.yaml", "train_all_metrics.csv", "threshold_oof_predictions.csv",
    }
    assert payload["controls"]["inner_lr1_crossfit"] == "strictly_nested_within_each_meta_train_fold"
    assert len(payload["reproducibility"]["input_joblib_sha256"]) == 64

    metrics = pd.read_csv(output / "fold_metrics.csv")
    assert {"current_product_exact_legacy", "current_architecture_oof_retrained"}.issubset(set(metrics.model_name))
    assert {"threshold_provenance", "threshold_sample_count", "true_positives", "false_positives"}.issubset(metrics.columns)
    assert (metrics.threshold_sample_count > 0).all()
    assert (metrics.true_positives + metrics.false_negatives == metrics.test_cancer_cases).all()
    assert (metrics.true_negatives + metrics.false_positives == metrics.test_benign_cases).all()
    assert (metrics.test_cancer_cases + metrics.test_benign_cases == metrics.test_target_cases).all()
    assert metrics.loc[metrics.model_name.eq("current_architecture_oof_retrained"), "threshold_provenance"].eq(
        "outer_train_full_chain_lr1_oof_current_lr2_oof_scores"
    ).all()
    threshold_scores = pd.read_csv(output / "threshold_oof_predictions.csv")
    assert {
        "outer_split_id", "model_name", "ablation", "meta_fold_id", "patientId",
        "target_case_id", "label", "p_cancer", "threshold_target",
        "threshold_provenance", "lr1_c", "current_lr2_c", "profile_c", "age_c",
        "symmetry_c",
    }.issubset(threshold_scores.columns)
    assert set(threshold_scores.threshold_score_kind) >= {
        "legacy_fitted_outer_train_scores", "nested_full_chain_oof_scores",
        "training_cohort_fitted_scores", "training_cohort_nested_full_chain_oof_scores",
    }
    outer_evidence = threshold_scores.loc[
        threshold_scores.outer_split_id.ne("train_all")
    ]
    expected_counts = metrics.set_index(["split_id", "model_name", "ablation"])[
        "threshold_sample_count"
    ]
    observed_counts = outer_evidence.groupby(
        ["outer_split_id", "model_name", "ablation"]
    ).size()
    for (split_id, model_name, ablation), expected in expected_counts.items():
        assert observed_counts.loc[(str(split_id), model_name, ablation)] == expected
    assert payload["outputs"]["threshold_oof_predictions"] == "threshold_oof_predictions.csv"
    manifest = pd.read_csv(output / "fold_manifest.csv")
    assert {"outer", "meta", "lr1"}.issubset(set(manifest.level))
    assert (output / "paired_fold_deltas.csv").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner", "--input-joblib", "input.joblib", "--output-dir", "out",
            "--outer-splits", "3", "--outer-repeats", "4", "--inner-lr1-splits", "2",
            "--meta-splits", "3", "--candidate-c", "0.01", "0.1", "1.0",
            "--lr1-c", "0.2", "--current-lr2-c", "0.4", "--random-state", "9",
        ],
    )
    args = runner.parse_args()
    assert (
        args.outer_splits, args.outer_repeats, args.inner_lr1_splits, args.meta_splits,
        args.candidate_c, args.lr1_c, args.current_lr2_c, args.random_state,
    ) == (3, 4, 2, 3, [0.01, 0.1, 1.0], 0.2, 0.4, 9)


def test_selector_records_independent_ablation_selection_and_boundaries(tmp_path):
    selector = _selector_module()
    payload = selector.run_selection(
        _measurements(),
        tmp_path / "selection",
        candidate_c=(0.03, 0.1, 0.3),
        inner_lr1_splits=2,
        meta_splits=2,
        random_state=11,
    )
    selected = payload["selected_regularization_by_ablation"]
    assert set(selected) == {
        "calibrated_profile", "profile_age", "profile_symmetry", "profile_age_symmetry"
    }
    values = pd.read_csv(tmp_path / "selection" / "regularization_selection.csv")
    assert set(values.ablation) == set(selected)
    assert "selected_at_grid_boundary" in values
