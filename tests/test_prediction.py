from __future__ import annotations

from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from xrd_preprocessing import load_preprocessing_config
from xrd_preprocessing import save_preprocessing_artifact

from aramis.__main__ import main
from .synthetic_aramis_h5 import write_v0_3_one_patient_h5


def _prediction_output_paths(
    output_folder: Path,
    *,
    prediction_name: str,
    patient_id: str,
) -> dict[str, Path]:
    stem = f"{prediction_name}_{patient_id}"
    return {
        "dataframe": output_folder / f"{stem}_prediction_dataframe.joblib",
        "external_json": output_folder / f"{stem}_external_report.json",
        "external_yaml": output_folder / f"{stem}_external_report.yaml",
        "internal_json": output_folder / f"{stem}_internal_report.json",
        "internal_yaml": output_folder / f"{stem}_internal_report.yaml",
    }


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


def _training_config(
    input_path: Path,
    output_path: Path,
    prediction_preprocessing_config_path: Path | None = None,
    selected_models: list[str] | None = None,
) -> dict:
    if prediction_preprocessing_config_path is None:
        prediction_preprocessing_config_path = (
            Path(__file__).parents[1]
            / "config"
            / "preprocessing"
            / "aramis_prediction_patient_model_input_v0_1.yaml"
        )
    io = {
        "input_dataframe_joblib_path": str(input_path),
        "output_model_joblib_path": str(output_path),
        "prediction_preprocessing_config_path": str(
            prediction_preprocessing_config_path
        ),
    }
    return {
        "training": {
            "name": "test_predict_train",
            "version": 0.1,
            "branch": "one_to_many",
        },
        "io": io,
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
            "selected_models": selected_models or ["M1Q"],
            "logreg_c": 1.0,
        },
        "evaluation": {
            "mode": "stratified_kfold",
            "n_splits": 3,
            "test_size": 0.30,
            "random_state": 7,
            "target_sensitivity": 0.95,
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
        "prediction": {
            "name": "test_predict",
            "version": 0.1,
            "author": "Test Author",
            "clinical_stage": "research draft",
        },
        "io": {
            "input_dataframe_joblib_path": str(dataframe_path),
            "input_model_joblib_path": str(model_path),
            "output_folder": str(output_folder),
        },
        "reporting": {
            "external_report": {
                "version": "0.1",
                "reference_doc": "../../docs/modeling/prediction_pipeline_v0_1.md",
            },
            "internal_report": {
                "version": "0.1",
                "reference_doc": "../../docs/modeling/internal_clinical_report_content_v0_1.md",
            },
        },
        "patient": {"patient_id": patient_id, "target_side": target_side},
        "model": {"model_id": "test_predict_train", "selected_model": "M1Q"},
        "decision": {"threshold_key": "threshold_target"},
    }


def _h5_prediction_config(
    h5_path: Path,
    model_path: Path,
    output_folder: Path,
    *,
    patient_id: str = "P1",
    target_side: str = "Left",
) -> dict:
    return {
        "prediction": {
            "name": "test_predict_from_h5",
            "version": 0.1,
            "author": "Test Author",
            "clinical_stage": "research draft",
        },
        "io": {
            "input_h5_path": str(h5_path),
            "input_model_joblib_path": str(model_path),
            "output_folder": str(output_folder),
        },
        "reporting": {
            "external_report": {
                "version": "0.1",
                "reference_doc": "../../docs/modeling/prediction_pipeline_v0_1.md",
            },
            "internal_report": {
                "version": "0.1",
                "reference_doc": "../../docs/modeling/internal_clinical_report_content_v0_1.md",
            },
        },
        "patient": {"patient_id": patient_id, "target_side": target_side},
        "container": {
            "schema_version": "0.3",
            "format": "xrd-session",
            "max_patients": 1,
        },
        "model": {"model_id": "test_predict_train", "selected_model": "M1Q"},
        "decision": {"threshold_key": "threshold_target"},
    }


def _valid_prediction_config(
    tmp_path: Path,
    *,
    input_h5: bool = False,
) -> dict:
    io = {
        "input_model_joblib_path": str(tmp_path / "model.joblib"),
        "output_folder": str(tmp_path / "outputs"),
    }
    if input_h5:
        io["input_h5_path"] = str(tmp_path / "patient.h5")
    else:
        io["input_dataframe_joblib_path"] = str(tmp_path / "prediction.joblib")
    config = {
        "prediction": {
            "name": "test_predict",
            "version": 0.1,
            "author": "Test Author",
        },
        "io": io,
        "reporting": {
            "external_report": {
                "version": "0.1",
                "reference_doc": "../../docs/modeling/prediction_pipeline_v0_1.md",
            },
            "internal_report": {
                "version": "0.1",
                "reference_doc": "../../docs/modeling/internal_clinical_report_content_v0_1.md",
            },
        },
        "patient": {"patient_id": "P1", "target_side": "Left"},
        "model": {"model_id": "test_predict_train", "selected_model": "M1Q"},
        "decision": {"threshold_key": "threshold_target"},
    }
    if input_h5:
        config["container"] = {
            "schema_version": "0.3",
            "format": "xrd-session",
            "max_patients": 1,
        }
    return config


def _v0_3_prediction_preprocessing_config(path: Path) -> None:
    preprocessing_config = load_preprocessing_config(
        Path(__file__).parents[1]
        / "config"
        / "preprocessing"
        / "aramis_prediction_patient_model_input_v0_1.yaml"
    )
    preprocessing_config["raw_data"]["source"] = "raw"
    preprocessing_config["raw_data"]["allowed_sources"] = ["gfrm", "raw"]
    preprocessing_config["snr"]["min_snr_db"] = -100.0
    preprocessing_config["profile_gate"]["min_value"] = -1_000_000.0
    path.write_text(yaml.safe_dump(preprocessing_config), encoding="utf-8")


def test_predict_cli_writes_decision_support_report(tmp_path: Path):
    dataframe_path = tmp_path / "preprocessed.joblib"
    model_path = tmp_path / "model.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict.yaml"
    output_folder = tmp_path / "prediction_outputs"
    output_paths = _prediction_output_paths(
        output_folder,
        prediction_name="test_predict",
        patient_id="P00",
    )
    save_preprocessing_artifact(
        _patient_frame(),
        dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(_training_config(dataframe_path, model_path)),
        encoding="utf-8",
    )
    prediction_config_path.write_text(
        yaml.safe_dump(
            _prediction_config(
                dataframe_path,
                model_path,
                output_folder,
            )
        ),
        encoding="utf-8",
    )

    assert main(["train", "--config", str(training_config_path)]) == 0
    assert main(["predict", "--config", str(prediction_config_path)]) == 0
    report = yaml.safe_load(output_paths["external_yaml"].read_text(encoding="utf-8"))
    internal_report = yaml.safe_load(
        output_paths["internal_yaml"].read_text(encoding="utf-8")
    )
    model_artifact = joblib.load(model_path)

    assert output_paths["external_json"].exists()
    assert output_paths["internal_json"].exists()
    assert report["kind"] == "aramis_external_prediction_report"
    assert report["author"] == "Test Author"
    assert report["patient_id"] == "P00"
    assert report["target_side"] == "Left"
    assert report["model_id"] == "test_predict_train"
    assert report["model_name"] == "M1Q"
    assert 0.0 <= report["p_cancer"] <= 1.0
    assert report["suggested_class"] in {"BENIGN", "CANCER"}
    assert report["reliability"] == "high"
    assert report["requires_radiologist_review"] is True
    assert report["provenance"]["training_config_sha256"] == model_artifact[
        "training_config_sha256"
    ]
    assert internal_report["kind"] == "aramis_internal_clinical_report"
    assert internal_report["version"] == "0.1"
    assert internal_report["features"]["azimuthal_integration"][
        "target_profile_model"
    ]["available"]
    assert internal_report["features"]["azimuthal_integration"][
        "contralateral_profile_model"
    ]["available"]
    assert internal_report["intermediate_models"]["lr1_profile_model"]["steps"]


def test_predict_rejects_wrong_model_id(tmp_path: Path):
    dataframe_path = tmp_path / "preprocessed.joblib"
    model_path = tmp_path / "model.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict.yaml"
    output_folder = tmp_path / "prediction_outputs"
    save_preprocessing_artifact(
        _patient_frame(),
        dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(_training_config(dataframe_path, model_path)),
        encoding="utf-8",
    )
    config = _prediction_config(
        dataframe_path,
        model_path,
        output_folder,
    )
    config["model"]["model_id"] = "wrong_model_id"
    prediction_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["train", "--config", str(training_config_path)]) == 0
    with pytest.raises(ValueError, match="model_id does not match"):
        main(["predict", "--config", str(prediction_config_path)])


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda config: config.pop("prediction"), "Missing prediction config sections"),
        (
            lambda config: config["io"].pop("output_folder"),
            "Missing io.output_folder",
        ),
        (
            lambda config: config["reporting"]["internal_report"].pop("version"),
            "Missing reporting.internal_report.version",
        ),
        (lambda config: config["patient"].pop("patient_id"), "Missing patient.patient_id"),
        (lambda config: config["patient"].pop("target_side"), "Missing patient.target_side"),
        (lambda config: config["model"].pop("model_id"), "Missing model.model_id"),
        (lambda config: config["model"].pop("selected_model"), "Missing model.selected_model"),
        (
            lambda config: config["io"].pop("input_dataframe_joblib_path"),
            "Missing io.input_dataframe_joblib_path",
        ),
    ],
)
def test_predict_rejects_invalid_prediction_yaml(tmp_path: Path, mutate, error: str):
    config_path = tmp_path / "predict.yaml"
    config = _valid_prediction_config(tmp_path)
    mutate(config)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        main(["predict", "--config", str(config_path)])


def test_predict_rejects_non_mapping_yaml(tmp_path: Path):
    config_path = tmp_path / "predict.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Prediction config must be a mapping"):
        main(["predict", "--config", str(config_path)])


def test_predict_rejects_invalid_h5_prediction_yaml(tmp_path: Path):
    config_path = tmp_path / "predict.yaml"
    config = _valid_prediction_config(tmp_path, input_h5=True)
    config.pop("container")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing container section"):
        main(["predict", "--config", str(config_path)])


def test_predict_rejects_invalid_preprocessing_override_yaml(tmp_path: Path):
    config_path = tmp_path / "predict.yaml"
    config = _valid_prediction_config(tmp_path, input_h5=True)
    config["preprocessing"] = {}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing preprocessing.config_path"):
        main(["predict", "--config", str(config_path)])


def test_predict_target_side_controls_profile_score(tmp_path: Path):
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    prediction_dataframe_path = tmp_path / "prediction_preprocessed.joblib"
    model_path = tmp_path / "model.joblib"
    training_config_path = tmp_path / "train.yaml"
    left_config_path = tmp_path / "predict_left.yaml"
    right_config_path = tmp_path / "predict_right.yaml"
    left_output_folder = tmp_path / "left_outputs"
    right_output_folder = tmp_path / "right_outputs"
    q = np.linspace(2.0, 23.0, 100)
    rows = []
    for side, shift in (("Left", 1.2), ("Right", -1.2)):
        for measurement_idx in range(3):
            rows.append(
                {
                    "patientId": "PX_TARGET",
                    "specimenId": f"PX_TARGET_{side}",
                    "measurementId": f"PX_TARGET_{side}_{measurement_idx}",
                    "side": side,
                    "product_status_group": "BENIGN",
                    "radial_profile_data": shift + np.sin(q / 3.0),
                    "q_range": q,
                    "age": 55,
                    "biopsy": side == "Left",
                }
            )

    save_preprocessing_artifact(
        _patient_frame(),
        training_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    save_preprocessing_artifact(
        pd.DataFrame(rows),
        prediction_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _training_config(
                training_dataframe_path,
                model_path,
                selected_models=["M0"],
            )
        ),
        encoding="utf-8",
    )
    left_config = _prediction_config(
        prediction_dataframe_path,
        model_path,
        left_output_folder,
        patient_id="PX_TARGET",
        target_side="Left",
    )
    left_config["model"]["selected_model"] = "M0"
    right_config = _prediction_config(
        prediction_dataframe_path,
        model_path,
        right_output_folder,
        patient_id="PX_TARGET",
        target_side="Right",
    )
    right_config["model"]["selected_model"] = "M0"
    left_config_path.write_text(yaml.safe_dump(left_config), encoding="utf-8")
    right_config_path.write_text(yaml.safe_dump(right_config), encoding="utf-8")

    assert main(["train", "--config", str(training_config_path)]) == 0
    assert main(["predict", "--config", str(left_config_path)]) == 0
    assert main(["predict", "--config", str(right_config_path)]) == 0
    left_paths = _prediction_output_paths(
        left_output_folder,
        prediction_name="test_predict",
        patient_id="PX_TARGET",
    )
    right_paths = _prediction_output_paths(
        right_output_folder,
        prediction_name="test_predict",
        patient_id="PX_TARGET",
    )
    left_report = yaml.safe_load(left_paths["external_yaml"].read_text(encoding="utf-8"))
    right_report = yaml.safe_load(right_paths["external_yaml"].read_text(encoding="utf-8"))

    assert left_report["target_side"] == "Left"
    assert right_report["target_side"] == "Right"
    assert left_report["p_cancer"] > right_report["p_cancer"]


def test_predict_cli_can_preprocess_one_patient_h5_before_scoring(tmp_path: Path):
    h5_path = tmp_path / "patient.h5"
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    model_path = tmp_path / "model.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict_from_h5.yaml"
    preprocessing_config_path = tmp_path / "prediction_preprocessing.yaml"
    output_folder = tmp_path / "prediction_outputs"
    output_paths = _prediction_output_paths(
        output_folder,
        prediction_name="test_predict_from_h5",
        patient_id="P1",
    )

    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="P1",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=20,
    )
    _v0_3_prediction_preprocessing_config(preprocessing_config_path)

    save_preprocessing_artifact(
        _patient_frame(),
        training_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _training_config(
                training_dataframe_path,
                model_path,
                preprocessing_config_path,
            )
        ),
        encoding="utf-8",
    )
    prediction_config = _h5_prediction_config(
        h5_path,
        model_path,
        output_folder,
    )
    prediction_config["preprocessing"] = {"config_path": str(preprocessing_config_path)}
    prediction_config_path.write_text(yaml.safe_dump(prediction_config), encoding="utf-8")

    assert main(["train", "--config", str(training_config_path)]) == 0
    assert main(["predict", "--config", str(prediction_config_path)]) == 0
    report = yaml.safe_load(output_paths["external_yaml"].read_text(encoding="utf-8"))

    assert output_paths["dataframe"].exists()
    assert report["patient_id"] == "P1"
    assert report["target_side"] == "Left"
    assert "input_h5_sha256" in report["provenance"]
    assert report["provenance"]["prediction_preprocessing_config_path"]
    assert report["provenance"]["prediction_preprocessing_config_sha256"]
    assert output_paths["internal_yaml"].exists()


def test_predict_rejects_h5_without_embedded_prediction_preprocessing(tmp_path: Path):
    h5_path = tmp_path / "patient.h5"
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    model_path = tmp_path / "model.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict_from_h5.yaml"
    preprocessing_config_path = tmp_path / "prediction_preprocessing.yaml"
    output_folder = tmp_path / "prediction_outputs"

    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="P1",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=30,
    )
    _v0_3_prediction_preprocessing_config(preprocessing_config_path)
    save_preprocessing_artifact(
        _patient_frame(),
        training_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _training_config(
                training_dataframe_path,
                model_path,
                preprocessing_config_path,
            )
        ),
        encoding="utf-8",
    )
    assert main(["train", "--config", str(training_config_path)]) == 0
    artifact = joblib.load(model_path)
    artifact["prediction_preprocessing_config"] = None
    joblib.dump(artifact, model_path)
    prediction_config_path.write_text(
        yaml.safe_dump(
            _h5_prediction_config(
                h5_path,
                model_path,
                output_folder,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no prediction_preprocessing_config"):
        main(["predict", "--config", str(prediction_config_path)])


def test_predict_cli_can_score_three_v0_3_one_patient_h5_containers(tmp_path: Path):
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    model_path = tmp_path / "model.joblib"
    training_config_path = tmp_path / "train.yaml"
    preprocessing_config_path = tmp_path / "prediction_preprocessing_v0_3.yaml"
    _v0_3_prediction_preprocessing_config(preprocessing_config_path)

    save_preprocessing_artifact(
        _patient_frame(),
        training_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _training_config(
                training_dataframe_path,
                model_path,
                preprocessing_config_path,
            )
        ),
        encoding="utf-8",
    )
    assert main(["train", "--config", str(training_config_path)]) == 0

    patients = [
        ("PX01", "BENIGN", "BENIGN", "Left"),
        ("PX02", "CANCER", "BENIGN", "Left"),
        ("PX03", "BENIGN", "CANCER", "Right"),
    ]
    reports = []
    for index, (patient_id, left_status, right_status, target_side) in enumerate(
        patients,
        start=1,
    ):
        h5_path = tmp_path / f"{patient_id}.h5"
        prediction_config_path = tmp_path / f"{patient_id}_predict.yaml"
        output_folder = tmp_path / f"{patient_id}_outputs"
        output_paths = _prediction_output_paths(
            output_folder,
            prediction_name="test_predict_from_h5",
            patient_id=patient_id,
        )
        write_v0_3_one_patient_h5(
            h5_path,
            patient_id=patient_id,
            left_status=left_status,
            right_status=right_status,
            target_side=target_side,
            seed=100 + index,
        )
        prediction_config_path.write_text(
            yaml.safe_dump(
                _h5_prediction_config(
                    h5_path,
                    model_path,
                    output_folder,
                    patient_id=patient_id,
                    target_side=target_side,
                )
            ),
            encoding="utf-8",
        )

        assert main(["predict", "--config", str(prediction_config_path)]) == 0
        report = yaml.safe_load(output_paths["external_yaml"].read_text(encoding="utf-8"))
        internal_report = yaml.safe_load(
            output_paths["internal_yaml"].read_text(encoding="utf-8")
        )
        prediction_artifact = joblib.load(output_paths["dataframe"])
        prediction_df = prediction_artifact["dataframe"]
        reports.append(report)

        assert prediction_df["patientId"].nunique() == 1
        assert prediction_df["patientId"].iloc[0] == patient_id
        assert prediction_df["specimenId"].nunique() == 2
        assert len(prediction_df) == 6
        assert set(prediction_df["side"]) == {"Left", "Right"}
        assert set(prediction_df["measurement_data_source"]) == {"container_raw_data"}
        assert report["patient_id"] == patient_id
        assert report["target_side"] == target_side
        assert report["model_id"] == "test_predict_train"
        assert report["model_name"] == "M1Q"
        assert report["provenance"]["input_h5_sha256"]
        assert report["provenance"]["prediction_preprocessing_config_sha256"]
        assert internal_report["features"]["azimuthal_integration"][
            "contralateral_profile_model"
        ]["available"]

    assert {report["patient_id"] for report in reports} == {"PX01", "PX02", "PX03"}


def test_predict_rejects_h5_patient_id_mismatch(tmp_path: Path):
    h5_path = tmp_path / "patient.h5"
    model_path = tmp_path / "model.joblib"
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict.yaml"
    preprocessing_config_path = tmp_path / "prediction_preprocessing_v0_3.yaml"
    output_folder = tmp_path / "prediction_outputs"

    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="PX_REAL",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=150,
    )
    _v0_3_prediction_preprocessing_config(preprocessing_config_path)
    save_preprocessing_artifact(
        _patient_frame(),
        training_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _training_config(
                training_dataframe_path,
                model_path,
                preprocessing_config_path,
            )
        ),
        encoding="utf-8",
    )
    assert main(["train", "--config", str(training_config_path)]) == 0
    prediction_config_path.write_text(
        yaml.safe_dump(
            _h5_prediction_config(
                h5_path,
                model_path,
                output_folder,
                patient_id="PX_WRONG",
                target_side="Left",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="patient_id does not match"):
        main(["predict", "--config", str(prediction_config_path)])


def test_predict_rejects_h5_format_mismatch(tmp_path: Path):
    h5_path = tmp_path / "patient.h5"
    model_path = tmp_path / "model.joblib"
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict.yaml"
    preprocessing_config_path = tmp_path / "prediction_preprocessing_v0_3.yaml"
    output_folder = tmp_path / "prediction_outputs"

    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="PX_FMT",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=151,
    )
    _v0_3_prediction_preprocessing_config(preprocessing_config_path)
    save_preprocessing_artifact(
        _patient_frame(),
        training_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _training_config(
                training_dataframe_path,
                model_path,
                preprocessing_config_path,
            )
        ),
        encoding="utf-8",
    )
    assert main(["train", "--config", str(training_config_path)]) == 0
    config = _h5_prediction_config(
        h5_path,
        model_path,
        output_folder,
        patient_id="PX_FMT",
        target_side="Left",
    )
    config["container"]["format"] = "wrong-format"
    prediction_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="format does not match"):
        main(["predict", "--config", str(prediction_config_path)])


def test_predict_rejects_unsupported_h5_schema_version(tmp_path: Path):
    h5_path = tmp_path / "patient.h5"
    model_path = tmp_path / "model.joblib"
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict.yaml"
    preprocessing_config_path = tmp_path / "prediction_preprocessing_v0_3.yaml"
    output_folder = tmp_path / "prediction_outputs"

    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="PX_SCHEMA",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=152,
    )
    with h5py.File(h5_path, "a") as h5:
        h5.attrs["schema_version"] = "0.4"
    _v0_3_prediction_preprocessing_config(preprocessing_config_path)
    save_preprocessing_artifact(
        _patient_frame(),
        training_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _training_config(
                training_dataframe_path,
                model_path,
                preprocessing_config_path,
            )
        ),
        encoding="utf-8",
    )
    assert main(["train", "--config", str(training_config_path)]) == 0
    config = _h5_prediction_config(
        h5_path,
        model_path,
        output_folder,
        patient_id="PX_SCHEMA",
        target_side="Left",
    )
    config["container"]["schema_version"] = "0.4"
    prediction_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported prediction H5 schema_version"):
        main(["predict", "--config", str(prediction_config_path)])


def test_predict_rejects_h5_schema_version_mismatch(tmp_path: Path):
    h5_path = tmp_path / "patient.h5"
    model_path = tmp_path / "model.joblib"
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict.yaml"
    preprocessing_config_path = tmp_path / "prediction_preprocessing_v0_3.yaml"
    output_folder = tmp_path / "prediction_outputs"

    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="PX99",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=199,
    )
    _v0_3_prediction_preprocessing_config(preprocessing_config_path)
    save_preprocessing_artifact(
        _patient_frame(),
        training_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _training_config(
                training_dataframe_path,
                model_path,
                preprocessing_config_path,
            )
        ),
        encoding="utf-8",
    )
    assert main(["train", "--config", str(training_config_path)]) == 0

    config = _h5_prediction_config(
        h5_path,
        model_path,
        output_folder,
        patient_id="PX99",
        target_side="Left",
    )
    config["container"]["schema_version"] = "0.2"
    prediction_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version does not match"):
        main(["predict", "--config", str(prediction_config_path)])


def test_predict_rejects_more_than_one_patient_in_h5(tmp_path: Path):
    h5_path = tmp_path / "two_patients.h5"
    model_path = tmp_path / "model.joblib"
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict.yaml"
    preprocessing_config_path = tmp_path / "prediction_preprocessing_v0_3.yaml"
    output_folder = tmp_path / "prediction_outputs"

    write_v0_3_one_patient_h5(
        h5_path,
        patient_id="PX98",
        left_status="BENIGN",
        right_status="CANCER",
        target_side="Left",
        seed=198,
    )
    with h5py.File(h5_path, "a") as h5:
        h5["session/sets/set_006_sample_main"].attrs["patientId"] = "PX_OTHER"
    _v0_3_prediction_preprocessing_config(preprocessing_config_path)
    save_preprocessing_artifact(
        _patient_frame(),
        training_dataframe_path,
        preprocessing_config={"aramis_preprocessing": {"branch": "one_to_many"}},
        preprocessing_config_text="aramis_preprocessing:\n  branch: one_to_many\n",
        metadata={"branch": "one_to_many"},
    )
    training_config_path.write_text(
        yaml.safe_dump(
            _training_config(
                training_dataframe_path,
                model_path,
                preprocessing_config_path,
            )
        ),
        encoding="utf-8",
    )
    assert main(["train", "--config", str(training_config_path)]) == 0
    prediction_config_path.write_text(
        yaml.safe_dump(
            _h5_prediction_config(
                h5_path,
                model_path,
                output_folder,
                patient_id="PX98",
                target_side="Left",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires exactly one patient"):
        main(["predict", "--config", str(prediction_config_path)])
