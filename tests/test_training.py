from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from xrd_preprocessing import save_preprocessing_artifact

from aramis.__main__ import main
from aramis.training import _logit_average_probability, run_training_from_config


def _training_frame() -> pd.DataFrame:
    rows = []
    q = np.linspace(2.0, 23.0, 20)
    for patient_idx in range(20):
        label = "CANCER" if patient_idx % 2 else "BENIGN"
        for measurement_idx in range(2):
            baseline = 1.0 if label == "CANCER" else -1.0
            rows.append(
                {
                    "patientId": f"P{patient_idx:02d}",
                    "specimenId": f"P{patient_idx:02d}_RIGHT",
                    "measurementId": f"P{patient_idx:02d}_M{measurement_idx}",
                    "product_status_group": label,
                    "radial_profile_data": baseline + np.sin(q / 3.0),
                    "q_range": q,
                    "sample_thickness_mm": 8.0 + measurement_idx * 0.1,
                }
            )
    return pd.DataFrame(rows)


def _patient_training_frame() -> pd.DataFrame:
    rows = []
    q = np.linspace(2.0, 23.0, 30)
    for patient_idx in range(30):
        cancer = patient_idx % 3 == 0
        patient_label = "CANCER" if cancer else "BENIGN"
        for side in ("Left", "Right"):
            specimen_id = f"P{patient_idx:02d}_{side}"
            specimen_label = patient_label if side == "Left" else "BENIGN"
            for measurement_idx in range(2):
                shift = 0.8 if specimen_label == "CANCER" else -0.4
                side_shift = 0.2 if side == "Left" and cancer else 0.0
                rows.append(
                    {
                        "patientId": f"P{patient_idx:02d}",
                        "specimenId": specimen_id,
                        "measurementId": f"{specimen_id}_M{measurement_idx}",
                        "side": side,
                        "product_status_group": specimen_label,
                        "radial_profile_data": shift
                        + side_shift
                        + np.sin(q / 3.0)
                        + measurement_idx * 0.01,
                        "q_range": q,
                        "age": 45 + patient_idx,
                        "biopsy": side == "Left",
                    }
                )
    return pd.DataFrame(rows)


def _training_config(input_path: Path, output_path: Path) -> dict:
    return {
        "training": {
            "name": "test_one_to_many_logistic",
            "version": 0.1,
            "branch": "one_to_many",
        },
        "io": {
            "input_dataframe_joblib_path": str(input_path),
            "output_model_joblib_path": str(output_path),
            "prediction_preprocessing_config_path": str(
                Path(__file__).parents[1]
                / "config"
                / "preprocessing"
                / "aramis_prediction_patient_model_input_v0_1.yaml"
            ),
        },
        "model": {
            "type": "logistic_regression",
            "profile_column": "radial_profile_data",
            "label_column": "product_status_group",
            "group_column": "patientId",
            "specimen_column": "specimenId",
            "logreg_c": 1.0,
        },
        "evaluation": {
            "n_splits": 2,
            "test_size": 0.30,
            "random_state": 7,
            "inner_splits": 3,
            "target_sensitivity": 0.95,
            "aggregation": "mean",
        },
    }


def _patient_training_config(
    input_path: Path,
    output_path: Path,
    tmp_path: Path,
    *,
    mode: str = "repeated_stratified_shuffle",
    selected_models: list[str] | None = None,
) -> dict:
    return {
        "training": {
            "name": "test_patient_m0_m1_m2",
            "version": 0.1,
            "branch": "one_to_many",
        },
        "io": {
            "input_dataframe_joblib_path": str(input_path),
            "output_model_joblib_path": str(output_path),
            "output_json_path": str(tmp_path / "summary.json"),
            "output_yaml_path": str(tmp_path / "description.yaml"),
            "prediction_preprocessing_config_path": str(
                Path(__file__).parents[1]
                / "config"
                / "preprocessing"
                / "aramis_prediction_patient_model_input_v0_1.yaml"
            ),
        },
        "model": {
            "type": "patient_m0_m1_m2_logistic_set",
            "profile_column": "radial_profile_data",
            "label_column": "product_status_group",
            "group_column": "patientId",
            "specimen_column": "specimenId",
            "side_column": "side",
            "age_column": "age",
            "biopsy_column": "biopsy",
            "lr1_row_policy": "all_rows",
            "selected_models": selected_models
            or ["M0", "M0Q", "M1", "M1Q", "M2", "M2Q"],
            "logreg_c": 1.0,
        },
        "evaluation": {
            "mode": mode,
            "n_splits": 3,
            "test_size": 0.30,
            "random_state": 7,
            "target_sensitivity": 0.95,
        },
    }


def test_train_cli_writes_one_to_many_model_artifact(tmp_path: Path):
    input_path = tmp_path / "preprocessed.joblib"
    output_path = tmp_path / "model.joblib"
    config_path = tmp_path / "train.yaml"
    save_preprocessing_artifact(
        _training_frame(),
        input_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    config_path.write_text(
        yaml.safe_dump(_training_config(input_path, output_path)),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_path)])
    artifact = joblib.load(output_path)

    assert exit_code == 0
    assert artifact["kind"] == "aramis_training_artifact"
    assert artifact["model_type"] == "one_to_many_logistic_regression"
    assert artifact["training_config_sha256"]
    assert artifact["preprocessing_config_sha256"]
    assert artifact["training_preprocessing_config_sha256"]
    assert artifact["training_preprocessing_config_text"]
    assert artifact["prediction_preprocessing_config_sha256"]
    assert artifact["prediction_preprocessing_config_text"]
    assert artifact["input_dataframe_joblib_sha256"]
    assert artifact["metadata"]["branch"] == "one_to_many"
    assert artifact["metric_summary"]["roc_auc_mean"].between(0.0, 1.0).all()
    scores = artifact["model"].predict_proba(
        np.vstack(_training_frame()["radial_profile_data"])
    )
    assert scores.shape == (40, 2)


def test_train_cli_writes_patient_m0_m1_m2_artifact(tmp_path: Path):
    input_path = tmp_path / "preprocessed.joblib"
    output_path = tmp_path / "patient_model.joblib"
    config_path = tmp_path / "train_patient.yaml"
    save_preprocessing_artifact(
        _patient_training_frame(),
        input_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    config_path.write_text(
        yaml.safe_dump(_patient_training_config(input_path, output_path, tmp_path)),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_path)])
    artifact = joblib.load(output_path)

    assert exit_code == 0
    assert artifact["kind"] == "aramis_training_artifact"
    assert artifact["model_type"] == "patient_m0_m1_m2_logistic_set"
    assert set(artifact["models"]) == {"M0", "M0Q", "M1", "M1Q", "M2", "M2Q"}
    assert artifact["feature_schema"]["M0Q"]["feature_columns"] == [
        "profile_p_cancer_logit_average",
        "profile_p_cancer_n_measurements",
        "target_measurements",
        "contralateral_measurements",
        "min_measurements_per_breast",
        "target_measurements_ok",
        "contralateral_measurements_ok",
        "paired_measurements_ok",
    ]
    assert artifact["feature_schema"]["M1"]["feature_columns"] == [
        "profile_p_cancer_logit_average",
        "symmetry_available",
        "sk_meanrms1",
        "sk_weightedrms1",
        "sk_sigma_target1",
        "sk_sigma_contralateral1",
        "sk_mahalanobis1",
        "sk_meanrms2",
        "sk_weightedrms2",
        "sk_sigma_target2",
        "sk_sigma_contralateral2",
        "sk_mahalanobis2",
        "sk_peak14_intensity",
        "sk_mean_peak_value",
        "sk_wasserstein_distance_mu_tc",
        "sk_cosine_distance_full_q2",
        "sk_wasserstein_distance_full_q2",
    ]
    assert artifact["feature_schema"]["M1Q"]["feature_columns"][-7:] == [
        "profile_p_cancer_n_measurements",
        "target_measurements",
        "contralateral_measurements",
        "min_measurements_per_breast",
        "target_measurements_ok",
        "contralateral_measurements_ok",
        "paired_measurements_ok",
    ]
    assert artifact["warnings"]
    assert any("reliability" in warning for warning in artifact["warnings"])
    assert artifact["training_config_yaml"]
    assert artifact["training_config_text"]
    assert artifact["training_config_sha256"]
    assert artifact["training_preprocessing_config_sha256"]
    assert artifact["training_preprocessing_config_text"]
    assert artifact["prediction_preprocessing_config_sha256"]
    assert artifact["prediction_preprocessing_config_text"]
    assert artifact["metric_summary"]["model_name"].tolist() == [
        "M0",
        "M0Q",
        "M1",
        "M1Q",
        "M2",
        "M2Q",
    ]
    assert "balanced_accuracy_target_mean" in artifact["metric_summary"].columns
    assert "ppv_target_mean" in artifact["metric_summary"].columns
    assert "npv_target_mean" in artifact["metric_summary"].columns
    assert artifact["models"]["M0"]["feature_columns"] == [
        "profile_p_cancer_logit_average"
    ]
    assert "profile_p_cancer_logit_average" in artifact["feature_table"].columns
    assert "profile_p_cancer_probability_mean" in artifact["feature_table"].columns
    assert "profile_p_cancer_n_measurements" in artifact["feature_table"].columns
    assert set(artifact["feature_table"]["inferred_target_side"]) == {"Left"}
    assert set(artifact["feature_table"]["inferred_contralateral_side"]) == {"Right"}
    assert "target_side" not in artifact["feature_table"].columns
    assert "contralateral_side" not in artifact["feature_table"].columns
    assert set(artifact["feature_table"]["profile_p_cancer_n_measurements"]) == {2}
    assert set(artifact["feature_table"]["min_measurements_per_breast"]) == {2}
    assert set(artifact["feature_table"]["paired_measurements_ok"]) == {0}
    assert set(artifact["feature_table"]["result_reliability"]) == {"medium"}
    assert "target_within_cosine_distance_mean" in artifact["feature_table"].columns
    assert (
        "contralateral_within_cosine_distance_mean"
        in artifact["feature_table"].columns
    )
    assert "between_breasts_cosine_distance_mean" in artifact["feature_table"].columns
    assert "symmetry_cosine_score" in artifact["feature_table"].columns
    assert "sk_meanrms1" in artifact["feature_table"].columns
    assert "sk_cosine_distance_full_q2" in artifact["feature_table"].columns
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "description.yaml").exists()


def test_logit_average_probability_preserves_consistent_evidence():
    scores = np.array([0.95, 0.95, 0.50])

    assert float(np.mean(scores)) == pytest.approx(0.80)
    assert _logit_average_probability(scores) == pytest.approx(0.877, abs=0.001)


@pytest.mark.parametrize("mode", ["all_on_all", "loovm", "stratified_kfold"])
def test_patient_training_evaluation_modes(tmp_path: Path, mode: str):
    input_path = tmp_path / f"{mode}.joblib"
    output_path = tmp_path / f"{mode}_model.joblib"
    config_path = tmp_path / f"{mode}.yaml"
    save_preprocessing_artifact(
        _patient_training_frame(),
        input_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    config = _patient_training_config(
        input_path,
        output_path,
        tmp_path,
        mode=mode,
        selected_models=["M0"],
    )
    if mode == "stratified_kfold":
        config["evaluation"]["n_splits"] = 3
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(["train", "--config", str(config_path)])
    artifact = joblib.load(output_path)

    assert exit_code == 0
    assert set(artifact["models"]) == {"M0"}
    assert artifact["metric_summary"]["evaluation_mode"].tolist() == [mode]
    assert artifact["metric_summary"]["roc_auc_mean"].between(0.0, 1.0).all()


def test_train_rejects_unknown_branch(tmp_path: Path):
    input_path = tmp_path / "preprocessed.joblib"
    output_path = tmp_path / "model.joblib"
    config_path = tmp_path / "train.yaml"
    save_preprocessing_artifact(_training_frame(), input_path)
    config = _training_config(input_path, output_path)
    config["training"]["branch"] = "one_to_one"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported training branch"):
        run_training_from_config(config_path)


def test_train_accepts_in_memory_dataframe_override(tmp_path: Path):
    input_path = tmp_path / "bad_preprocessed.joblib"
    output_path = tmp_path / "model.joblib"
    config_path = tmp_path / "train.yaml"
    save_preprocessing_artifact(
        pd.DataFrame({"not_training_data": [1]}),
        input_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="bad: true\n",
        metadata={"branch": "one_to_many"},
    )
    config_path.write_text(
        yaml.safe_dump(_patient_training_config(input_path, output_path, tmp_path)),
        encoding="utf-8",
    )

    artifact = run_training_from_config(
        config_path,
        dataframe=_patient_training_frame(),
        preprocessing_artifact={
            "preprocessing_config_sha256": "in-memory-test",
            "metadata": {"branch": "one_to_many"},
        },
    )

    assert artifact["model_type"] == "patient_m0_m1_m2_logistic_set"
    assert artifact["preprocessing_config_sha256"] == "in-memory-test"


def test_run_workflow_yaml_runs_training_from_referenced_configs(tmp_path: Path):
    input_path = tmp_path / "preprocessed.joblib"
    output_path = tmp_path / "model.joblib"
    preprocessing_config_path = tmp_path / "preprocess.yaml"
    training_config_path = tmp_path / "train.yaml"
    workflow_config_path = tmp_path / "workflow.yaml"
    save_preprocessing_artifact(
        _patient_training_frame(),
        input_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    preprocessing_config_path.write_text(
        yaml.safe_dump(
                {
                    "aramis_preprocessing": {"branch": "one_to_many"},
                    "raw_data": {"source": "gfrm", "allowed_sources": ["gfrm"]},
                    "metadata": {"output_columns": ["radial_profile_data"]},
                    "filters": {},
                    "labels": {},
                    "integration": {},
                    "snr": {},
                    "normalization": {},
                    "profile_gate": {},
                    "branch_settings": {},
                    "io": {"output_joblib_path": str(input_path)},
                }
            ),
        encoding="utf-8",
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _patient_training_config(
                input_path,
                output_path,
                tmp_path,
                mode="stratified_kfold",
                selected_models=["M0"],
            )
        ),
        encoding="utf-8",
    )
    workflow_config_path.write_text(
        yaml.safe_dump(
            {
                "workflow": {
                    "name": "test_workflow",
                    "run_preprocessing": False,
                    "run_training": True,
                    "validate_io_match": True,
                },
                "preprocessing": {"config_path": str(preprocessing_config_path)},
                "training": {"config_path": str(training_config_path)},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["run", "--config", str(workflow_config_path)])
    artifact = joblib.load(output_path)

    assert exit_code == 0
    assert artifact["model_type"] == "patient_m0_m1_m2_logistic_set"
    assert artifact["metric_summary"]["model_name"].tolist() == ["M0"]
