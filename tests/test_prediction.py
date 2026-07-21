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
    _metadata_value,
    _validate_h5_container_contract,
    _validate_prediction_config,
    run_prediction_from_config,
)
from aramis.prediction_contract import _config_path
from aramis.prediction_scoring import _tissue_risk_assessment
from aramis.training import run_training_from_config
from aramis.training_config import PRODUCT_MODEL_NAME

from .synthetic_aramis_h5 import write_v0_3_one_patient_h5


PREDICTION_EXAMPLE_ROOT = Path(__file__).parents[1] / "examples" / "prediction" / "configs"
FINAL_EXAMPLE_MODEL = (
    Path(__file__).parents[1]
    / "models"
    / "aramis_target_breast_risk_0_2_9-beta_2479efef4979"
    / "model.joblib"
)


def test_tracked_prediction_examples_use_final_product_artifact():
    configs = sorted(PREDICTION_EXAMPLE_ROOT.glob("config_predict_*_example.yaml"))
    assert [path.name for path in configs] == [
        "config_predict_atypical_example.yaml",
        "config_predict_benign_example.yaml",
        "config_predict_cancer_example.yaml",
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


def test_prediction_relative_paths_resolve_from_configuration_root(tmp_path: Path):
    project_root = tmp_path / "aramis"
    config_path = project_root / "config" / "prediction" / "example.yaml"
    config_path.parent.mkdir(parents=True)
    config = {"io": {"input_h5_path": "examples/prediction_h5/example.h5"}}

    assert _config_path(config, config_path, section="io", key="input_h5_path") == (
        project_root / "examples" / "prediction_h5" / "example.h5"
    )


def test_prediction_example_paths_resolve_from_project_root(tmp_path: Path):
    project_root = tmp_path / "aramis"
    config_path = project_root / "examples" / "prediction" / "configs" / "example.yaml"
    config_path.parent.mkdir(parents=True)
    (project_root / "pyproject.toml").touch()
    config = {"io": {"input_model_joblib_path": "models/example/model.joblib"}}

    assert _config_path(
        config,
        config_path,
        section="io",
        key="input_model_joblib_path",
    ) == (project_root / "models" / "example" / "model.joblib")


@pytest.mark.parametrize(
    ("config_name", "target_p_cancer", "contralateral_p_cancer"),
    [
        ("config_predict_atypical_example.yaml", 0.45023, 0.79691),
        ("config_predict_benign_example.yaml", 0.34270, 0.25711),
        ("config_predict_cancer_example.yaml", 0.84062, 0.76857),
    ],
)
def test_frozen_model_examples_keep_stable_scores(
    tmp_path: Path,
    config_name: str,
    target_p_cancer: float,
    contralateral_p_cancer: float,
):
    """Guard frozen product behavior while modules are reorganised."""
    source = PREDICTION_EXAMPLE_ROOT / config_name
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    project_root = Path(__file__).parents[1]
    config["io"]["input_h5_path"] = str(project_root / config["io"]["input_h5_path"])
    config["io"]["input_model_joblib_path"] = str(
        project_root / config["io"]["input_model_joblib_path"]
    )
    config["io"]["output_folder"] = str(tmp_path / "reports")
    config_path = tmp_path / config_name
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    report = run_prediction_from_config(config_path)["internal_report"]
    predictions = report["breast_predictions"]
    assert predictions["target"]["final_prediction"]["p_cancer"] == pytest.approx(
        target_p_cancer,
        abs=1e-5,
    )
    contralateral_profile = predictions["contralateral"][
        "azimuthal_integration_contralateral_profile"
    ]
    assert 0.0 <= contralateral_profile["p_cancer"] <= 1.0
    assert predictions["contralateral"]["final_prediction"]["p_cancer"] == pytest.approx(
        contralateral_p_cancer,
        abs=1e-5,
    )
    for side in ("target", "contralateral"):
        tra = predictions[side]["final_prediction"]["tissue_risk_assessment"]
        assert 0.0 <= tra["index"] <= 100.0
        assert tra["level"] in {"TRA 1", "TRA 2", "TRA 3", "TRA 4", "TRA 5"}


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.05, "TRA 1"),
        (0.10, "TRA 2"),
        (0.50, "TRA 3"),
        (0.70, "TRA 4"),
        (0.95, "TRA 5"),
    ],
)
def test_tra_uses_frozen_percentile_reference(score: float, expected: str):
    model_info = {
        "prediction_reference_scores": {
            "final_prediction": {"all_target_cases": [0.1, 0.3, 0.5, 0.7, 0.9]}
        }
    }
    tra = _tissue_risk_assessment(model_info, score)
    assert tra["level"] == expected
    assert 0.0 <= tra["index"] <= 100.0


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
        "contract": "aramis_training_config_v0_3",
        "model": {
            "name": PRODUCT_MODEL_NAME,
            "version": "0.1-beta",
            "model_author": "test",
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


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda config: config.pop("patient"), "Missing prediction config sections"),
        (
            lambda config: config["io"].pop("input_dataframe_joblib_path"),
            "Set exactly one input",
        ),
        (
            lambda config: config["io"].update(input_h5_path="patient.h5"),
            "Set exactly one input",
        ),
        (
            lambda config: config["run"].update(synthetic_test_mode="true"),
            "synthetic_test_mode must be boolean",
        ),
        (
            lambda config: config["run"].update(analysis_author=42),
            "analysis_author must be a string",
        ),
        (
            lambda config: config["patient"].update(patient_id="  "),
            "Missing patient.patient_id",
        ),
        (
            lambda config: config["patient"].update(target_side="centre"),
            "target_side must be left or right",
        ),
    ],
)
def test_prediction_contract_rejects_invalid_required_values(tmp_path: Path, mutate, error: str):
    config = _prediction_config(
        tmp_path / "data.joblib",
        tmp_path / "model.joblib",
        tmp_path / "outputs",
    )
    mutate(config)

    with pytest.raises((TypeError, ValueError), match=error):
        _validate_prediction_config(config, tmp_path / "predict.yaml")


def test_optional_scan_metadata_blanks_are_reported_as_unknown():
    assert _metadata_value({"operator_id": "  "}, "operator_id") == "unknown"
    assert _metadata_value({"operator_id": None}, "operator_id") == "unknown"
    assert _metadata_value({"operator_id": "OPT-001"}, "operator_id") == "OPT-001"


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
    assert 0.0 <= external["risk_probability"] <= 1.0
    assert 0.0 <= external["decision_threshold"] <= 1.0
    assert "suggested_class" not in external
    assert "p_cancer" not in external
    assert external["risk_level"] in {"low", "high"}
    assert "tissue_risk_assessment" not in external
    metrics = external["model_metrics"]
    assert metrics["metric_scope"] == "in_sample_not_independent"
    assert 0.0 <= metrics["sensitivity"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0
    assert internal["model_metrics"] == metrics
    assert "method_performance" not in external
    assert "method_performance" not in internal
    assert external["reliability"] in {"low", "medium", "high"}
    target = internal["breast_predictions"]["target"]
    contralateral = internal["breast_predictions"]["contralateral"]
    decision = internal["decision_threshold"]
    assert 0.0 <= target["final_prediction"]["p_cancer"] <= 1.0
    assert target["final_prediction"]["tissue_risk_assessment"]["level"] in {
        "TRA 1",
        "TRA 2",
        "TRA 3",
        "TRA 4",
        "TRA 5",
    }
    assert decision["applies_to"] == [
        "target.final_prediction",
        "contralateral.final_prediction",
    ]
    assert 0.0 <= decision["threshold"] <= 1.0
    assert "decision_threshold" not in target["final_prediction"]
    assert target["azimuthal_integration_target_profile"]["p_cancer"] is not None
    assert contralateral["available"] is True
    assert 0.0 <= contralateral["final_prediction"]["p_cancer"] <= 1.0
    assert contralateral["final_prediction"]["suggested_class"] in {
        "BENIGN",
        "CANCER",
    }
    assert contralateral["final_prediction"]["tissue_risk_assessment"][
        "level"
    ] in {"TRA 1", "TRA 2", "TRA 3", "TRA 4", "TRA 5"}
    assert contralateral["symmetry"]["available"] is False
    assert contralateral["reliability"]["level"] == "low"
    assert contralateral["model_execution"]["scoring_path"] == (
        "profile_age_with_neutral_symmetry_gate"
    )
    assert 0.0 <= contralateral[
        "azimuthal_integration_contralateral_profile"
    ]["p_cancer"] <= 1.0
    assert set(target["final_prediction"]["score_percentiles"]) == {
        "reference_score",
        "reference_population",
        "all_training_target_cases",
        "benign_training_target_cases",
        "cancer_training_target_cases",
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
        left["breast_predictions"]["target"]["azimuthal_integration_target_profile"][
            "p_cancer"
        ]
        != right["breast_predictions"]["target"][
            "azimuthal_integration_target_profile"
        ]["p_cancer"]
    )


def test_predict_without_contralateral_uses_unavailable_symmetry(
    tmp_path: Path,
    trained_model,
):
    model_path, training_dataframe_path = trained_model
    frame = joblib.load(training_dataframe_path)["dataframe"]
    frame = frame[~((frame["patientId"] == "P00") & (frame["side"] == "Right"))].copy()
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
    assert target["reliability"]["level"] == "low"
    contralateral = report["breast_predictions"]["contralateral"]
    assert contralateral["available"] is False
    assert contralateral["side"] == "unknown"
    assert contralateral["azimuthal_integration_contralateral_profile"][
        "p_cancer"
    ] == "unknown"


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

    _validate_h5_container_contract(artifact, h5_path, expected_patient_id="PX01")
    with pytest.raises(ValueError, match="does not match H5 patientId"):
        _validate_h5_container_contract(artifact, h5_path, expected_patient_id="WRONG")

    with h5py.File(h5_path, "a") as h5:
        h5["session/sets/set_006_sample_main"].attrs["patientId"] = "PX02"
    with pytest.raises(ValueError, match="exactly one patient"):
        _validate_h5_container_contract(artifact, h5_path, expected_patient_id="PX01")


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
        _validate_h5_container_contract(artifact, h5_path, expected_patient_id="PX01")


@pytest.mark.parametrize(
    ("missing", "error"),
    [
        ("format", "format does not match"),
        ("session", "missing /session group"),
        ("sets", "missing /session/sets group"),
    ],
)
def test_h5_contract_rejects_missing_required_structure(
    tmp_path: Path,
    trained_model,
    missing: str,
    error: str,
):
    model_path, _ = trained_model
    artifact = joblib.load(model_path)
    h5_path = tmp_path / "patient.h5"
    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="PX01",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=22,
    )
    with h5py.File(h5_path, "a") as h5:
        if missing == "format":
            del h5.attrs["format"]
        elif missing == "session":
            del h5["session"]
        else:
            del h5["session/sets"]

    with pytest.raises(ValueError, match=error):
        _validate_h5_container_contract(artifact, h5_path, expected_patient_id="PX01")


def test_h5_contract_rejects_absent_patient_id(tmp_path: Path, trained_model):
    model_path, _ = trained_model
    artifact = joblib.load(model_path)
    h5_path = tmp_path / "patient.h5"
    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="PX01",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=23,
    )
    with h5py.File(h5_path, "a") as h5:
        for group in h5["session/sets"].values():
            del group.attrs["patientId"]
        del h5["session/sample/patient_name"]

    with pytest.raises(ValueError, match="Prediction H5 contains no patientId values"):
        _validate_h5_container_contract(artifact, h5_path, expected_patient_id="PX01")
