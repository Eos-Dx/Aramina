from __future__ import annotations

from pathlib import Path

import joblib
import pytest
import yaml

from aramina.prediction import run_prediction_from_config


ROOT = Path(__file__).parents[1]
RETRAINED_CANDIDATE_MODEL = (
    ROOT
    / "models"
    / "aramina_target_breast_risk_0_2_13-beta_f5e4a04cad11"
    / "model.joblib"
)
DVC_CANDIDATE_MODEL = (
    ROOT
    / "models"
    / "aramina_target_breast_risk_0_2_14-beta_98526329f40d"
    / "model.joblib"
)


def test_retrained_candidate_records_pinned_xrd_release_lineage():
    artifact = joblib.load(RETRAINED_CANDIDATE_MODEL)

    assert artifact["model_identity"]["version"] == "0.2.13-beta"
    assert artifact["reproducibility"]["source_code"]["xrd_preprocessing"] == {
        "git_commit": "88dcaa277c5a0d4be2ab637bc5827a14bd106bea",
        "requested_revision": "88dcaa277c5a0d4be2ab637bc5827a14bd106bea",
        "url": "https://github.com/Eos-Dx/XRD-preprocessing.git",
        "version": "0.1.9b0",
    }


def test_dvc_candidate_records_complete_source_data_lineage():
    artifact = joblib.load(DVC_CANDIDATE_MODEL)

    assert artifact["model_identity"]["version"] == "0.2.14-beta"
    reproducibility = artifact["reproducibility"]
    assert reproducibility["source_code"]["aramina"] == {
        "version": "0.2.14b0",
        "git_sha": "f402662f56a7fd2e6215c7067a4fc81448f1c339",
    }
    assert reproducibility["source_code"]["xrd_preprocessing"]["git_commit"] == (
        "88dcaa277c5a0d4be2ab637bc5827a14bd106bea"
    )
    data_version = reproducibility["source_h5"]["data_version"]
    assert data_version["contract"] == "aramina_dvc_input_v0_1"
    assert data_version["hash"] == "46e199e316e95969731d61d8ab4b2c52"
    assert data_version["input_h5_sha256"] == (
        "d2d61e83850b282c3d2479ea436deed821c4488b96983252d294f3d56ee3f1f9"
    )
    assert (
        "contract: aramina_preprocessing_and_training_config_v0_3"
        in reproducibility["configs"]["preprocess_train_yaml"]
    )


def test_dvc_candidate_scores_external_one_patient_h5(tmp_path: Path):
    source = ROOT / "examples/prediction/configs/config_predict_cancer_example.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["io"]["input_h5_path"] = str(ROOT / config["io"]["input_h5_path"])
    config["io"]["input_model_joblib_path"] = str(DVC_CANDIDATE_MODEL)
    config["io"]["output_folder"] = str(tmp_path / "reports")
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    reports = run_prediction_from_config(config_path)

    assert reports["internal_report"]["model"]["version"] == "0.2.14-beta"
    assert reports["external_report"]["risk_probability"] == pytest.approx(0.86939)
    assert len(list((tmp_path / "reports").glob("*_internal_report.yaml"))) == 1
    assert len(list((tmp_path / "reports").glob("*_external_report.yaml"))) == 1
