from __future__ import annotations

from pathlib import Path

import h5py
import joblib
import pandas as pd
import pytest
import yaml

from aramina.prediction import (
    _metadata_value,
    _validate_h5_container_contract,
    _validate_prediction_config,
    run_prediction_from_config,
)
from aramina.patient_features import (
    build_patient_prediction_feature_row,
    prediction_metadata_from_target_rows,
)
from aramina.prediction_contract import _config_path
from aramina.prediction_scoring import _tissue_risk_assessment
from aramina.target_breast_model import GatedSymmetryLogistic
from aramina.training_config import PRODUCT_MODEL_NAME
from aramina.tra_policy import TRA_POLICY_CONTRACT, derive_tra_policy

from .prediction_fixtures import prediction_config as _prediction_config
from .prediction_fixtures import train_model
from .synthetic_aramina_h5 import write_v0_3_one_patient_h5
from .artifact_helpers import save_training_preprocessing_artifact


PREDICTION_EXAMPLE_ROOT = Path(__file__).parents[1] / "examples" / "prediction" / "configs"
PREDICTION_REPORT_EXAMPLE_ROOT = (
    Path(__file__).parents[1] / "contracts" / "prediction" / "examples"
)
FINAL_EXAMPLE_MODEL = (
    Path(__file__).parents[1]
    / "models"
    / "aramina_target_breast_risk_0_2_12-beta_9bb911189af6"
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


def test_frozen_product_artifact_keeps_binary_class_definition():
    """Keep report labels tied to the executable frozen model artifact."""
    artifact = joblib.load(FINAL_EXAMPLE_MODEL)
    model_info = artifact["models"][PRODUCT_MODEL_NAME]

    assert model_info["class_definition"] == {
        "reference_class": "BENIGN",
        "target_class": "CANCER",
    }
    assert isinstance(model_info["final_model"], GatedSymmetryLogistic)


@pytest.mark.parametrize(
    ("filename", "expected_p_cancer"),
    [
        ("config_predict_atypical_example.yaml", 0.61924),
        ("config_predict_benign_example.yaml", 0.33870),
        ("config_predict_cancer_example.yaml", 0.86939),
    ],
)
def test_tracked_prediction_fixtures_remain_compatible_with_frozen_artifact(
    tmp_path: Path,
    filename: str,
    expected_p_cancer: float,
):
    root = Path(__file__).parents[1]
    source = PREDICTION_EXAMPLE_ROOT / filename
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["io"]["input_h5_path"] = str(root / config["io"]["input_h5_path"])
    config["io"]["input_model_joblib_path"] = str(
        root / config["io"]["input_model_joblib_path"]
    )
    expected_model_id = Path(config["io"]["input_model_joblib_path"]).parent.name
    config["io"]["output_folder"] = str(tmp_path / source.stem)
    request_path = tmp_path / filename
    request_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    reports = run_prediction_from_config(request_path)

    assert reports["external_report"]["risk_probability"] == pytest.approx(
        expected_p_cancer, abs=1e-5
    )
    assert reports["internal_report"]["model"]["id"] == expected_model_id
    assert reports["internal_report"]["breast_predictions"]["target"][
        "model_execution"
    ]["scoring_path"] == "azimuthal_integration_age_with_symmetry"


def test_prediction_relative_paths_resolve_from_configuration_root(tmp_path: Path):
    project_root = tmp_path / "aramina"
    config_path = project_root / "config" / "prediction" / "example.yaml"
    config_path.parent.mkdir(parents=True)
    (project_root / "pyproject.toml").touch()
    config = {"io": {"input_h5_path": "examples/prediction_h5/example.h5"}}

    assert _config_path(config, config_path, section="io", key="input_h5_path") == (
        project_root / "examples" / "prediction_h5" / "example.h5"
    )


def test_external_prediction_paths_resolve_from_config_directory(tmp_path: Path):
    config_path = tmp_path / "external_request" / "predict.yaml"
    config_path.parent.mkdir(parents=True)
    config = {"io": {"input_h5_path": "inputs/one_patient.h5"}}

    assert _config_path(config, config_path, section="io", key="input_h5_path") == (
        config_path.parent / "inputs" / "one_patient.h5"
    )


def test_prediction_example_paths_resolve_from_project_root(tmp_path: Path):
    project_root = tmp_path / "aramina"
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
        ("config_predict_atypical_example.yaml", 0.61924, 0.81365),
        ("config_predict_benign_example.yaml", 0.33870, 0.25089),
        ("config_predict_cancer_example.yaml", 0.86939, 0.78536),
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
        "azimuthal_integration_profile"
    ]
    assert 0.0 <= contralateral_profile["p_cancer"] <= 1.0
    assert predictions["contralateral"]["final_prediction"]["p_cancer"] == pytest.approx(
        contralateral_p_cancer,
        abs=1e-5,
    )
    for side in ("target", "contralateral"):
        final = predictions[side]["final_prediction"]
        assert final["level"] in {"TRA 1", "TRA 2", "TRA 3", "TRA 4", "TRA 5"}
        assert "tissue_risk_assessment" not in final


def test_report_contract_examples_match_generated_report_schema(tmp_path: Path):
    """Keep tracked report examples aligned with the frozen model output."""
    source = PREDICTION_EXAMPLE_ROOT / "config_predict_cancer_example.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    project_root = Path(__file__).parents[1]
    config["io"]["input_h5_path"] = str(project_root / config["io"]["input_h5_path"])
    config["io"]["input_model_joblib_path"] = str(
        project_root / config["io"]["input_model_joblib_path"]
    )
    config["io"]["output_folder"] = str(tmp_path / "reports")
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    reports = run_prediction_from_config(config_path)
    for report_name, filename in (
        ("internal_report", "internal_report.yaml"),
        ("external_report", "external_report.yaml"),
    ):
        example = yaml.safe_load(
            (PREDICTION_REPORT_EXAMPLE_ROOT / filename).read_text(encoding="utf-8")
        )
        assert _mapping_key_paths(example) == _mapping_key_paths(reports[report_name])


def _mapping_key_paths(value: object, path: str = "") -> set[str]:
    if not isinstance(value, dict):
        return set()
    paths = set()
    for key, nested in value.items():
        current = f"{path}.{key}" if path else str(key)
        paths.add(current)
        paths.update(_mapping_key_paths(nested, current))
    return paths


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.10, "TRA 1"),
        (0.20, "TRA 2"),
        (0.25, "TRA 3"),
        (0.40, "TRA 4"),
        (0.70, "TRA 5"),
    ],
)
def test_tra_is_threshold_centred(score: float, expected: str):
    policy = {
        "contract": TRA_POLICY_CONTRACT,
        "decision_threshold": 0.25,
        "logit_margin_boundaries": {
            "tra_1_to_2": -0.5,
            "tra_2_to_3": 0.0,
            "tra_3_to_4": 0.5,
            "tra_4_to_5": 1.5,
        },
    }
    tra = _tissue_risk_assessment({"tissue_risk_assessment": policy}, score)
    assert tra == {"level": expected}


def test_tra_policy_is_derived_from_patient_safe_oof_predictions():
    rows = []
    for case_index in range(5):
        for split_index in range(4):
            rows.append(
                {
                    "target_case_id": f"P{case_index}",
                    "p_cancer": 0.25,
                    "y_pred_target": split_index % 2,
                }
            )
    policy = derive_tra_policy(pd.DataFrame(rows), decision_threshold=0.25)

    assert policy["contract"] == TRA_POLICY_CONTRACT
    assert policy["decision_threshold"] == pytest.approx(0.25)
    assert policy["calibration"]["method"] == "patient_safe_oof_decision_stability"
    assert policy["calibration"]["target_cases"] == 5
    assert policy["logit_margin_boundaries"] == {
        "tra_1_to_2": -0.1,
        "tra_2_to_3": 0.0,
        "tra_3_to_4": 0.1,
        "tra_4_to_5": 0.3,
    }
    assert [item["decision_support_class"] for item in policy["levels"]] == [
        "BENIGN",
        "BENIGN",
        "CANCER",
        "CANCER",
        "CANCER",
    ]
    assert [item["requires_radiologist_review"] for item in policy["levels"]] == [
        False,
        False,
        True,
        True,
        True,
    ]


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    return train_model(tmp_path_factory)


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


def test_target_side_metadata_drops_conflicting_values():
    metadata = prediction_metadata_from_target_rows(
        pd.DataFrame(
            {
                "operator_id": ["OPT-001", "OPT-002"],
                "hardware_version": ["human-1", "human-1"],
                "scan_date_time": ["", None],
            }
        )
    )

    assert metadata == {"hardware_version": "human-1"}


def test_target_side_metadata_reaches_prediction_feature_row(tmp_path: Path, trained_model):
    model_path, dataframe_path = trained_model
    frame = joblib.load(dataframe_path)["dataframe"].copy()
    left = frame["side"] == "Left"
    frame.loc[left, "operator_id"] = "OPT-TARGET"
    frame.loc[~left, "operator_id"] = "OPT-CONTRALATERAL"
    frame.loc[left, "scan_date_time"] = "2026-07-22T10:15:00+02:00"

    model_info = joblib.load(model_path)["models"][PRODUCT_MODEL_NAME]
    feature_row = build_patient_prediction_feature_row(
        frame,
        model_info,
        patient_id="P00",
        target_side="Left",
    ).iloc[0]

    assert feature_row["operator_id"] == "OPT-TARGET"
    assert feature_row["scan_date_time"] == "2026-07-22T10:15:00+02:00"


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

    assert external["output_type"] == "aramina_external_report"
    assert external["report_version"] == "0.6"
    assert internal["report_version"] == "0.9"
    assert internal["reference_doc"] == (
        "./docs/modeling/internal_clinical_report_content_v0_9.md"
    )
    assert 0.0 <= external["risk_probability"] <= 1.0
    assert 0.0 <= external["decision_threshold"] <= 1.0
    assert "suggested_class" not in external
    assert "p_cancer" not in external
    assert external["biopsy_required"] is True
    assert external["target_class_risk_level"] == "high"
    assert "tissue_risk_assessment" not in external
    metrics = external["model_metrics"]
    assert metrics["dataset"] == "train_on_all_target_breast_cases"
    assert metrics["validation"] == "not_performed"
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
    assert target["final_prediction"]["reference_class"] == "BENIGN"
    assert target["final_prediction"]["target_class"] == "CANCER"
    assert "suggested_class" not in target["final_prediction"]
    assert target["final_prediction"]["target_class_risk_level"] == "high"
    assert target["final_prediction"]["biopsy_required"] is True
    assert target["final_prediction"]["level"] in {
        "TRA 1",
        "TRA 2",
        "TRA 3",
        "TRA 4",
        "TRA 5",
    }
    assert decision["applies_to"] == ["target.final_prediction"]
    assert 0.0 <= decision["threshold"] <= 1.0
    assert "decision_threshold" not in target["final_prediction"]
    assert set(target["azimuthal_integration_profile"]) == {
        "p_cancer",
        "per_measurement_p_cancer",
    }
    assert target["azimuthal_integration_profile"]["p_cancer"] is not None
    assert target["symmetry"] == {"available": True}
    assert target["reliability"] == {
        "level": "high",
        "reason": "at least 2 valid measurements per breast; symmetry refinement applied",
    }
    assert target["model_execution"] == {
        "scoring_path": "azimuthal_integration_age_with_symmetry"
    }
    assert contralateral["available"] is True
    assert 0.0 <= contralateral["final_prediction"]["p_cancer"] <= 1.0
    assert "suggested_class" not in contralateral["final_prediction"]
    assert contralateral["final_prediction"]["reference_class"] == "BENIGN"
    assert contralateral["final_prediction"]["target_class"] == "CANCER"
    assert "suggested_class" not in contralateral["final_prediction"]
    assert "biopsy_required" not in contralateral["final_prediction"]
    assert contralateral["final_prediction"]["level"] in {
        "TRA 1",
        "TRA 2",
        "TRA 3",
        "TRA 4",
        "TRA 5",
    }
    assert contralateral["symmetry"]["available"] is False
    assert contralateral["symmetry"] == {"available": False}
    assert contralateral["reliability"]["level"] == "low"
    assert contralateral["model_execution"]["scoring_path"] == (
        "azimuthal_integration_age"
    )
    assert 0.0 <= contralateral[
        "azimuthal_integration_profile"
    ]["p_cancer"] <= 1.0
    assert set(contralateral["azimuthal_integration_profile"]) == {
        "p_cancer",
        "per_measurement_p_cancer",
    }
    assert set(target["final_prediction"]["score_percentiles"]) == {
        "reference_population",
        "all",
        "reference_class",
        "target_class",
    }
    assert target["final_prediction"]["score_percentiles"][
        "reference_population"
    ] == "train_on_all_target-breast_cases"
    assert external["prediction_comment"] == "synthetic test"
    assert internal["prediction_comment"] == "synthetic test"
    assert internal["scan_metadata"]["patient_id"] == "P00"
    assert "patient_id" not in internal
    assert "prediction_config" not in internal
    assert internal["model"]["artifact_sha256"]
    assert len(list(output_folder.glob("*_external_report.yaml"))) == 1
    assert len(list(output_folder.glob("*_internal_report.yaml"))) == 1
    assert not list(output_folder.glob("*_external_report.json"))
    assert not list(output_folder.glob("*_internal_report.json"))


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
        left["breast_predictions"]["target"]["azimuthal_integration_profile"][
            "p_cancer"
        ]
        != right["breast_predictions"]["target"][
            "azimuthal_integration_profile"
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
    save_training_preprocessing_artifact(
        frame,
        dataframe_path,
        input_h5_sha256="test-h5",
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
    assert target["model_execution"]["scoring_path"] == "azimuthal_integration_age"
    assert target["reliability"]["level"] == "medium"
    contralateral = report["breast_predictions"]["contralateral"]
    assert contralateral["available"] is False
    assert contralateral["side"] == "unknown"
    assert contralateral["azimuthal_integration_profile"][
        "p_cancer"
    ] == "unknown"
    assert set(contralateral["azimuthal_integration_profile"]) == {
        "p_cancer",
        "per_measurement_p_cancer",
    }


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
