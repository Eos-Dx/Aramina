"""Fast contract tests for the research-only staged refinement runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/profile_symmetry_age_refinement/run_experiment.py"
)
STAGED_MODEL_PATH = RUNNER_PATH.with_name("staged_model.py")


def _runner_module():
    spec = importlib.util.spec_from_file_location("refinement_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _staged_model_module():
    spec = importlib.util.spec_from_file_location(
        "refinement_staged_model_test",
        STAGED_MODEL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SyntheticStagedClassifier(BaseEstimator):
    """Small staged model used to test runner contracts, not model behavior."""

    def __init__(self, *, symmetry_c=0.3, age_c=0.3, random_state=42):
        self.symmetry_c = symmetry_c
        self.age_c = age_c
        self.random_state = random_state

    def fit(self, x, y):
        self.profile_model_ = LogisticRegression(C=0.3, max_iter=1000).fit(
            x[["profile_p_cancer_logit_average"]], y
        )
        return self

    def predict_stage_probabilities(self, x):
        profile = self.profile_model_.predict_proba(
            x[["profile_p_cancer_logit_average"]]
        )[:, 1]
        symmetry = np.where(x["symmetry_available"].astype(bool), 0.03, 0.0)
        after_symmetry = np.clip(profile + symmetry, 1e-5, 1.0 - 1e-5)
        age = np.where(x["age_available"].astype(bool), (x["age"] - 50.0) / 1000.0, 0.0)
        final = np.clip(after_symmetry + age, 1e-5, 1.0 - 1e-5)
        return pd.DataFrame(
            {
                "profile_p_cancer": profile,
                "after_symmetry_p_cancer": after_symmetry,
                "final_p_cancer": final,
            },
            index=x.index,
        )

    def predict_proba(self, x):
        score = self.predict_stage_probabilities(x)["final_p_cancer"].to_numpy()
        return np.column_stack([1.0 - score, score])

    def stage_logit_corrections(self, x):
        return pd.DataFrame(
            {"symmetry_logit_correction": 0.0, "age_logit_correction": 0.0},
            index=x.index,
        )


def _synthetic_measurements() -> pd.DataFrame:
    rows = []
    q = np.linspace(2.0, 23.0, 24)
    for patient_index in range(12):
        label = "CANCER" if patient_index % 2 else "BENIGN"
        cancer_shift = 0.20 if label == "CANCER" else -0.20
        for side, biopsy in (("Left", True), ("Right", False)):
            side_shift = 0.03 if side == "Left" else -0.03
            for measurement_index in range(2):
                profile = (
                    np.sin(q / 3.0)
                    + cancer_shift
                    + side_shift
                    + measurement_index * 0.01
                    + patient_index * 0.001
                )
                rows.append(
                    {
                        "patientId": f"P{patient_index:02d}",
                        "specimenId": f"S{patient_index:02d}_{side}",
                        "measurementId": f"M{patient_index:02d}_{side}_{measurement_index}",
                        "side": side,
                        "biopsy": biopsy,
                        "product_status_group": label,
                        "age": 40 + patient_index,
                        "q_range": q,
                        "radial_profile_data": profile,
                    }
                )
    return pd.DataFrame(rows)


def test_runner_writes_patient_safe_stage_metrics_and_outputs(tmp_path, monkeypatch):
    runner = _runner_module()
    monkeypatch.setattr(runner, "_load_staged_classifier_class", lambda: _SyntheticStagedClassifier)
    frame = _synthetic_measurements()
    source = tmp_path / "input.joblib"
    joblib.dump({"dataframe": frame}, source)

    payload = runner.run_experiment(
        runner.load_input_dataframe(source),
        tmp_path / "outputs",
        n_splits=2,
        n_repeats=1,
        random_state=7,
    )

    output = tmp_path / "outputs"
    assert {path.name for path in output.iterdir()} == {
        "fold_metrics.csv",
        "split_predictions.csv",
        "summary.csv",
        "summary.yaml",
        "train_all_metrics.csv",
    }
    assert payload["controls"]["threshold_selection"] == "train_fold_only"
    assert payload["controls"]["n_splits"] == 2

    metrics = pd.read_csv(output / "fold_metrics.csv")
    assert set(metrics["model_name"]) == {
        "current_gated_symmetry_logistic",
        "staged_profile_symmetry_age",
    }
    assert set(metrics.loc[metrics["model_name"] == "staged_profile_symmetry_age", "stage"]) == {
        "profile_p_cancer",
        "after_symmetry_p_cancer",
        "final_p_cancer",
    }
    assert set(metrics["split_id"]) == {0, 1}
    assert np.isfinite(metrics[["roc_auc", "pr_auc", "brier_score", "log_loss"]]).all().all()
    assert (metrics["target_sensitivity"] == 0.95).all()

    predictions = pd.read_csv(output / "split_predictions.csv")
    assert {
        "stage_symmetry_logit_correction",
        "stage_age_logit_correction",
    }.issubset(predictions.columns)
    for split_id, group in predictions.groupby("split_id"):
        test_patients = set(group["patientId"])
        train_patients = set(frame["patientId"]).difference(test_patients)
        assert test_patients.isdisjoint(train_patients)
        assert split_id in {0, 1}

    train_all = pd.read_csv(output / "train_all_metrics.csv")
    assert set(train_all["evaluation_status"]) == {"in_sample_not_independent"}


def test_staged_score_contract_rejects_missing_or_invalid_columns():
    runner = _runner_module()

    class MissingColumns:
        def predict_stage_probabilities(self, x):
            return pd.DataFrame({"profile_p_cancer": [0.5]})

    class InvalidProbability:
        def predict_stage_probabilities(self, x):
            return pd.DataFrame(
                {
                    "profile_p_cancer": [0.5],
                    "after_symmetry_p_cancer": [0.5],
                    "final_p_cancer": [1.2],
                }
            )

    with np.testing.assert_raises_regex(ValueError, "missing stage probability columns"):
        runner._staged_scores(MissingColumns(), pd.DataFrame({"x": [1]}))
    with np.testing.assert_raises_regex(ValueError, "within \\[0, 1\\]"):
        runner._staged_scores(InvalidProbability(), pd.DataFrame({"x": [1]}))


def test_staged_model_missing_blocks_are_exact_identity():
    module = _staged_model_module()
    rows = pd.DataFrame(
        {
            "profile_p_cancer_logit_average": [
                0.15,
                0.20,
                0.25,
                0.35,
                0.55,
                0.65,
                0.75,
                0.85,
            ],
            "symmetry_available": [1, 1, 0, 1, 1, 0, 1, 1],
            "age": [35, 40, 45, 50, 55, 60, 65, 70],
            "age_available": [1, 1, 1, 0, 1, 1, 0, 1],
            "sk_wasserstein_distance_full_q2": [
                0.2,
                0.3,
                np.nan,
                0.4,
                0.8,
                np.nan,
                0.9,
                1.0,
            ],
            "sk_weightedrms1": [
                0.1,
                0.2,
                np.nan,
                0.3,
                0.6,
                np.nan,
                0.7,
                0.8,
            ],
            "sk_weightedrms2": [
                0.15,
                0.25,
                np.nan,
                0.35,
                0.65,
                np.nan,
                0.75,
                0.85,
            ],
            "sk_mean_peak_value_abs_delta": [
                0.05,
                0.10,
                np.nan,
                0.15,
                0.30,
                np.nan,
                0.35,
                0.40,
            ],
        }
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = module.StagedProfileSymmetryAgeClassifier().fit(rows, labels)
    stages = model.predict_stage_probabilities(rows)
    corrections = model.stage_logit_corrections(rows)

    symmetry_missing = rows["symmetry_available"].eq(0)
    age_missing = rows["age_available"].eq(0)
    np.testing.assert_allclose(
        stages.loc[symmetry_missing, "after_symmetry_p_cancer"],
        stages.loc[symmetry_missing, "profile_p_cancer"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        stages.loc[age_missing, "final_p_cancer"],
        stages.loc[age_missing, "after_symmetry_p_cancer"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        corrections.loc[symmetry_missing, "symmetry_logit_correction"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        corrections.loc[age_missing, "age_logit_correction"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isfinite(model.predict_proba(rows)).all()
    assert model.age_feature_names_[-1] == "gated_age_x_incoming_logit"


def test_age_design_can_depend_on_incoming_profile_risk():
    module = _staged_model_module()
    rows = pd.DataFrame(
        {
            "profile_p_cancer_logit_average": [0.2, 0.4, 0.6, 0.8],
            "symmetry_available": [1, 1, 1, 1],
            "age": [35, 45, 55, 65],
            "age_available": [1, 1, 1, 1],
            "sk_wasserstein_distance_full_q2": [0.2, 0.3, 0.4, 0.5],
            "sk_weightedrms1": [0.1, 0.2, 0.3, 0.4],
            "sk_weightedrms2": [0.15, 0.25, 0.35, 0.45],
            "sk_mean_peak_value_abs_delta": [0.05, 0.10, 0.15, 0.20],
        }
    )
    model = module.StagedProfileSymmetryAgeClassifier().fit(
        rows,
        np.array([0, 0, 1, 1]),
    )
    incoming_a = np.array([-1.0, -0.5, 0.5, 1.0])
    incoming_b = np.array([1.0, 0.5, -0.5, -1.0])

    design_a = model._age_design(rows, incoming_a)
    design_b = model._age_design(rows, incoming_b)

    np.testing.assert_allclose(design_a[:, :2], design_b[:, :2])
    assert not np.allclose(design_a[:, 2], design_b[:, 2])
