from __future__ import annotations

from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from xrd_preprocessing import save_preprocessing_artifact

from aramis.prediction import (
    _validate_h5_container_contract,
    _validate_prediction_config,
    run_prediction_from_config,
)
from aramis.training import run_training_from_config
from aramis.training_config import PRODUCT_MODEL_NAME

from .synthetic_aramis_h5 import write_v0_3_one_patient_h5


PREDICTION_EXAMPLE_ROOT = Path(__file__).parents[1] / "examples" / "prediction_h5"
FINAL_EXAMPLE_MODEL = (
    Path(__file__).parents[1]
    / "examples"
    / "prediction_models"
    / "aramis_m2q_t100_0_2_7_beta.joblib"
)


def test_tracked_prediction_examples_use_final_m2q_artifact():
    configs = sorted(PREDICTION_EXAMPLE_ROOT.glob("*_predict.yaml"))
    assert [path.name for path in configs] == [
        "atypical_predict.yaml",
        "benign_predict.yaml",
        "cancer_predict.yaml",
    ]

    for config_path in configs:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        root = Path(__file__).parents[1]
        h5_path = root / config["io"]["input_h5_path"]
        model_path = (root / config["io"]["input_model_joblib_path"]).resolve()

        assert model_path == FINAL_EXAMPLE_MODEL.resolve()
        with h5py.File(h5_path, "r") as h5:
            assert h5.attrs["schema_version"] == "0.3"
            assert h5.attrs["format"] == "xrd-session"
            assert h5.attrs["fixture_patient_id"] == config["patient"]["patient_id"]
            sets = h5["session/sets"]
            sides = {str(group.attrs["side"]).casefold() for group in sets.values()}
            assert sides == {"left", "right"}


def _patient_frame() -> pd.DataFrame:
    rows = []
    q = np.linspace(2.0, 23.0, 100)
    for patient_idx in range(18):
        cancer = patient_idx % 3 == 0
        patient_label = "CANCER" if cancer else "BENIGN"
        for side in ("Left", "Right"):
            specimen_id = f"P{patient_idx:02d}_{side}"
            specimen_label = patient_label if side == "Left" else "BENIGN"
            for measurement_idx in range(3):
                shift = 0.8 if specimen_label == "CANCER" else -0.4
                rows.append(
                    {
                        "patientId": f"P{patient_idx:02d}",
                        "specimenId": specimen_id,
                        "measurementId": f"{specimen_id}_M{measurement_idx}",
                        "side": side,
                        "product_status_group": specimen_label,
                        "radial_profile_data": shift
                        + np.sin(q / 3.0)
                        + measurement_idx * 0.01,
                        "q_range": q,
                        "age": 45 + patient_idx,
                        "biopsy": side == "Left",
                    }
                )
    return pd.DataFrame(rows)


def _training_config(input_path: Path, output_folder: Path) -> dict:
    return {
        "contract": "aramis_training_config_v0_2",
        "model": {
            "name": PRODUCT_MODEL_NAME,
            "version": "0.1-beta",
            "created_by": "test",
            "clinical_stage": "research draft",
            "intended_use": "Synthetic decision-support test.",
        },
        "run": {"evaluation": True, "train_on_all": True},
        "input": {"dataframe_joblib_path": str(input_path)},
        "output": {"folder": str(output_folder)},
        "evaluation": {
            "method": "repeated_stratified_kfold",
            "folds": 5,
            "repeats": 20,
            "random_seed": 42,
        },
    }


def _prediction_config(
    dataframe_path: Path,
    model_path: Path,
    output_folder: Path,
    *,
    patient_id: str = "P00",
    target_side: str = "Left",
) -> dict:
    return {
        "run": {
            "analysis_author": "Test Author",
            "prediction_comment": "synthetic test",
            "synthetic_test_mode": True,
        },
        "io": {
            "input_dataframe_joblib_path": str(dataframe_path),
            "input_model_joblib_path": str(model_path),
            "output_folder": str(output_folder),
        },
        "patient": {"patient_id": patient_id, "target_side": target_side},
    }


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("prediction_model")
    dataframe_path = root / "training.joblib"
    config_path = root / "train.yaml"
    save_preprocessing_artifact(
        _patient_frame(),
        dataframe_path,
        preprocessing_config_text="pipeline:\n  steps:\n  - name: test\n",
        metadata={"input_h5_sha256": "test-h5"},
    )
    config_path.write_text(
        yaml.safe_dump(_training_config(dataframe_path, root / "runs")),
        encoding="utf-8",
    )
    result = run_training_from_config(config_path)
    return Path(result["model_path"]), dataframe_path


def test_prediction_contract_rejects_unknown_nested_fields(tmp_path: Path):
    config = _prediction_config(
        tmp_path / "data.joblib",
        tmp_path / "model.joblib",
        tmp_path / "outputs",
    )
    config["io"]["output_json_path"] = "forbidden.json"

    with pytest.raises(ValueError, match="Unknown prediction io fields"):
        _validate_prediction_config(config, tmp_path / "predict.yaml")


def test_predict_writes_external_and_internal_reports(tmp_path: Path, trained_model):
    model_path, dataframe_path = trained_model
    config_path = tmp_path / "predict.yaml"
    output_folder = tmp_path / "outputs"
    config_path.write_text(
        yaml.safe_dump(_prediction_config(dataframe_path, model_path, output_folder)),
        encoding="utf-8",
    )

    reports = run_prediction_from_config(config_path)
    external = reports["external_report"]
    internal = reports["internal_report"]

    assert external["output_type"] == "aramis_external_report"
    assert external["suggested_class"] in {"BENIGN", "CANCER"}
    assert "p_cancer" not in external
    performance = external["method_performance"]
    assert performance["evaluation_available"] is True
    assert performance["evaluation_method"] == "repeated_stratified_kfold"
    assert performance["folds"] == 5
    assert performance["repeats"] == 20
    assert 0.0 <= performance["sensitivity"] <= 1.0
    assert 0.0 <= performance["specificity"] <= 1.0
    assert external["reliability"] in {"low", "medium", "high"}
    target = internal["breast_predictions"]["target"]
    contralateral = internal["breast_predictions"]["contralateral"]
    assert 0.0 <= target["final_prediction"]["p_cancer"] <= 1.0
    assert "decision_threshold" in target["final_prediction"]
    assert "threshold" not in target["final_prediction"]
    assert target["azimuthal_integration_target_profile"]["p_cancer"] is not None
    assert contralateral["available"] is True
    assert set(target["final_prediction"]["score_percentiles"]) == {
        "all_training_patients",
        "benign_training_patients",
        "cancer_training_patients",
    }
    assert external["prediction_comment"] == "synthetic test"
    assert internal["prediction_comment"] == "synthetic test"
    assert internal["scan_metadata"]["patient_id"] == "P00"
    assert "patient_id" not in internal
    assert "prediction_config" not in internal
    assert internal["model"]["artifact_sha256"]
    assert len(list(output_folder.glob("*_external_report.yaml"))) == 1
    assert len(list(output_folder.glob("*_internal_report.yaml"))) == 1


def test_predict_target_side_controls_profile_evidence(tmp_path: Path, trained_model):
    model_path, dataframe_path = trained_model
    left_config = tmp_path / "left.yaml"
    right_config = tmp_path / "right.yaml"
    left_config.write_text(
        yaml.safe_dump(
            _prediction_config(
                dataframe_path,
                model_path,
                tmp_path / "left",
                target_side="Left",
            )
        ),
        encoding="utf-8",
    )
    right_config.write_text(
        yaml.safe_dump(
            _prediction_config(
                dataframe_path,
                model_path,
                tmp_path / "right",
                target_side="Right",
            )
        ),
        encoding="utf-8",
    )

    left = run_prediction_from_config(left_config)["internal_report"]
    right = run_prediction_from_config(right_config)["internal_report"]

    assert (
        left["breast_predictions"]["target"]["azimuthal_integration_target_profile"]["p_cancer"]
        != right["breast_predictions"]["target"]["azimuthal_integration_target_profile"]["p_cancer"]
    )


def test_predict_without_contralateral_uses_unavailable_symmetry(
    tmp_path: Path,
    trained_model,
):
    model_path, training_dataframe_path = trained_model
    frame = joblib.load(training_dataframe_path)["dataframe"]
    frame = frame[
        ~((frame["patientId"] == "P00") & (frame["side"] == "Right"))
    ].copy()
    dataframe_path = tmp_path / "unpaired.joblib"
    save_preprocessing_artifact(
        frame,
        dataframe_path,
        preprocessing_config_text="pipeline:\n  steps:\n  - name: test\n",
        metadata={"input_h5_sha256": "test-h5"},
    )
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _prediction_config(dataframe_path, model_path, tmp_path / "outputs")
        ),
        encoding="utf-8",
    )

    report = run_prediction_from_config(config_path)["internal_report"]

    target = report["breast_predictions"]["target"]
    assert target["symmetry"]["available"] is False
    assert target["symmetry"]["status"] == "not_available"
    assert target["model_execution"]["scoring_path"] == (
        "profile_age_with_neutral_symmetry_gate"
    )
    assert target["final_prediction"]["reliability"]["level"] == "low"
    contralateral = report["breast_predictions"]["contralateral"]
    assert contralateral["available"] is False
    assert contralateral["side"] == "unknown"
    assert contralateral["final_prediction"]["p_cancer"] == "unknown"


def test_h5_contract_requires_one_matching_patient(tmp_path: Path, trained_model):
    model_path, _ = trained_model
    artifact = joblib.load(model_path)
    h5_path = tmp_path / "patient.h5"
    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="PX01",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=20,
    )

    _validate_h5_container_contract(
        artifact, h5_path, expected_patient_id="PX01"
    )
    with pytest.raises(ValueError, match="does not match H5 patientId"):
        _validate_h5_container_contract(
            artifact, h5_path, expected_patient_id="WRONG"
        )

    with h5py.File(h5_path, "a") as h5:
        h5["session/sets/set_006_sample_main"].attrs["patientId"] = "PX02"
    with pytest.raises(ValueError, match="exactly one patient"):
        _validate_h5_container_contract(
            artifact, h5_path, expected_patient_id="PX01"
        )


def test_h5_contract_rejects_schema_mismatch(tmp_path: Path, trained_model):
    model_path, _ = trained_model
    artifact = joblib.load(model_path)
    h5_path = tmp_path / "patient.h5"
    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="PX01",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=21,
    )
    with h5py.File(h5_path, "a") as h5:
        h5.attrs["schema_version"] = "0.4"

    with pytest.raises(ValueError, match="schema_version does not match"):
        _validate_h5_container_contract(
            artifact, h5_path, expected_patient_id="PX01"
        )
