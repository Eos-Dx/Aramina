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
            "lr1_logreg_c": 0.1,
            "lr2_logreg_c": 0.1,
        },
        "evaluation": {
            "mode": mode,
            "n_splits": 3,
            "test_size": 0.30,
            "random_state": 7,
            "target_sensitivity": 0.95,
        },
    }


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
    ]
    assert artifact["feature_schema"]["M1"]["feature_columns"] == [
        "profile_p_cancer_logit_average",
        "sk_wasserstein_distance_full_q2",
        "sk_weightedrms1",
        "sk_weightedrms2",
        "sk_mean_peak_value_abs_delta",
    ]
    assert artifact["feature_schema"]["M1Q"]["feature_columns"][-1:] == [
        "profile_p_cancer_n_measurements",
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


def test_train_keeps_patients_without_paired_breasts(tmp_path: Path):
    input_path = tmp_path / "preprocessed.joblib"
    output_path = tmp_path / "patient_model.joblib"
    config_path = tmp_path / "train_patient.yaml"
    frame = _patient_training_frame()
    frame = frame[
        ~((frame["patientId"] == "P01") & (frame["side"] == "Right"))
    ].copy()
    save_preprocessing_artifact(
        frame,
        input_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    config = _patient_training_config(
        input_path,
        output_path,
        tmp_path,
        mode="all_on_all",
        selected_models=["M2Q"],
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["train", "--config", str(config_path)]) == 0
    artifact = joblib.load(output_path)

    assert "symmetry_available" not in artifact["models"]["M2Q"]["feature_columns"]
    assert len(artifact["feature_table"]) == 30
    unpaired = artifact["feature_table"].query("patientId == 'P01'").iloc[0]
    assert unpaired["symmetry_available"] == 0
    assert unpaired["result_reliability"] == "low"
    assert unpaired["sk_wasserstein_distance_full_q2"] == 0.0
    assert unpaired["sk_mean_peak_value_abs_delta"] == 0.0
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


def test_patient_training_repeated_stratified_kfold(tmp_path: Path):
    input_path = tmp_path / "repeated_kfold.joblib"
    output_path = tmp_path / "repeated_kfold_model.joblib"
    config_path = tmp_path / "repeated_kfold.yaml"
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
        mode="stratified_kfold",
        selected_models=["M0"],
    )
    config["evaluation"]["n_splits"] = 3
    config["evaluation"]["n_repeats"] = 2
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(["train", "--config", str(config_path)])
    artifact = joblib.load(output_path)

    assert exit_code == 0
    assert len(artifact["split_metrics"]) == 6
    assert artifact["metric_summary"]["splits"].tolist() == [6]


def test_train_rejects_unknown_branch(tmp_path: Path):
    input_path = tmp_path / "preprocessed.joblib"
    output_path = tmp_path / "model.joblib"
    config_path = tmp_path / "train.yaml"
    save_preprocessing_artifact(_patient_training_frame(), input_path)
    config = _patient_training_config(input_path, output_path, tmp_path)
    config["training"]["branch"] = "one_to_one"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported training branch"):
        run_training_from_config(config_path)


def test_train_rejects_legacy_measurement_level_model_type(tmp_path: Path):
    input_path = tmp_path / "preprocessed.joblib"
    output_path = tmp_path / "model.joblib"
    config_path = tmp_path / "train.yaml"
    save_preprocessing_artifact(
        _patient_training_frame(),
        input_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    config = _patient_training_config(input_path, output_path, tmp_path)
    config["model"]["type"] = "logistic_regression"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported training model.type"):
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
