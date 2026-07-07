from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from xrd_preprocessing import load_preprocessing_config
from xrd_preprocessing import save_preprocessing_artifact

from aramis.__main__ import main
from .synthetic_aramis_h5 import write_known_synthetic_h5


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
) -> dict:
    io = {
        "input_dataframe_joblib_path": str(input_path),
        "output_model_joblib_path": str(output_path),
    }
    if prediction_preprocessing_config_path is not None:
        io["prediction_preprocessing_config_path"] = str(
            prediction_preprocessing_config_path
        )
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
            "selected_models": ["M1Q"],
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
    output_json_path: Path,
    output_yaml_path: Path,
) -> dict:
    return {
        "prediction": {
            "name": "test_predict",
            "version": 0.1,
            "clinical_stage": "research draft",
        },
        "io": {
            "input_dataframe_joblib_path": str(dataframe_path),
            "input_model_joblib_path": str(model_path),
            "output_json_path": str(output_json_path),
            "output_yaml_path": str(output_yaml_path),
        },
        "patient": {"patient_id": "P00", "target_side": "Left"},
        "model": {"selected_model": "M1Q"},
        "decision": {"threshold_key": "threshold_target"},
    }


def _h5_prediction_config(
    h5_path: Path,
    dataframe_path: Path,
    model_path: Path,
    output_json_path: Path,
    output_yaml_path: Path,
) -> dict:
    return {
        "prediction": {
            "name": "test_predict_from_h5",
            "version": 0.1,
            "clinical_stage": "research draft",
        },
        "io": {
            "input_h5_path": str(h5_path),
            "output_dataframe_joblib_path": str(dataframe_path),
            "input_model_joblib_path": str(model_path),
            "output_json_path": str(output_json_path),
            "output_yaml_path": str(output_yaml_path),
        },
        "patient": {"patient_id": "P1", "target_side": "Left"},
        "model": {"selected_model": "M1Q"},
        "decision": {"threshold_key": "threshold_target"},
    }


def test_predict_cli_writes_decision_support_report(tmp_path: Path):
    dataframe_path = tmp_path / "preprocessed.joblib"
    model_path = tmp_path / "model.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict.yaml"
    output_json_path = tmp_path / "report.json"
    output_yaml_path = tmp_path / "report.yaml"
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
                output_json_path,
                output_yaml_path,
            )
        ),
        encoding="utf-8",
    )

    assert main(["train", "--config", str(training_config_path)]) == 0
    assert main(["predict", "--config", str(prediction_config_path)]) == 0
    report = yaml.safe_load(output_yaml_path.read_text(encoding="utf-8"))
    model_artifact = joblib.load(model_path)

    assert output_json_path.exists()
    assert report["kind"] == "aramis_prediction_report"
    assert report["patient_id"] == "P00"
    assert report["target_side"] == "Left"
    assert report["model_name"] == "M1Q"
    assert 0.0 <= report["p_cancer"] <= 1.0
    assert report["suggested_class"] in {"BENIGN", "CANCER"}
    assert report["reliability"] == "high"
    assert report["requires_radiologist_review"] is True
    assert report["provenance"]["training_config_sha256"] == model_artifact[
        "training_config_sha256"
    ]


def test_predict_cli_can_preprocess_one_patient_h5_before_scoring(tmp_path: Path):
    h5_path = tmp_path / "patient.h5"
    training_dataframe_path = tmp_path / "training_preprocessed.joblib"
    prediction_dataframe_path = tmp_path / "prediction_preprocessed.joblib"
    model_path = tmp_path / "model.joblib"
    training_config_path = tmp_path / "train.yaml"
    prediction_config_path = tmp_path / "predict_from_h5.yaml"
    preprocessing_config_path = tmp_path / "prediction_preprocessing.yaml"
    output_json_path = tmp_path / "report.json"
    output_yaml_path = tmp_path / "report.yaml"

    write_known_synthetic_h5(h5_path)
    preprocessing_config = load_preprocessing_config(
        Path(__file__).parents[1]
        / "config"
        / "preprocessing"
        / "aramis_prediction_patient_model_input_v0_1.yaml"
    )
    preprocessing_config["raw_data"]["source"] = "npy"
    preprocessing_config["raw_data"]["allowed_sources"] = ["gfrm", "npy"]
    preprocessing_config["raw_data"]["h5_dataset_candidates"]["npy"] = ["processed/data"]
    preprocessing_config["pipeline"]["steps"][0] = {
        "name": "h5_blob_to_df",
        "transformer": "H5BlobDataFrameTransformer",
        "params": {
            "source": {"$ref": "raw_data.source"},
            "dataset_candidates": {"$ref": "raw_data.h5_dataset_candidates.npy"},
        },
    }
    preprocessing_config["snr"]["min_snr_db"] = -100.0
    preprocessing_config["profile_gate"]["min_value"] = -1_000_000.0
    preprocessing_config_path.write_text(
        yaml.safe_dump(preprocessing_config),
        encoding="utf-8",
    )

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
    prediction_config_path.write_text(
        yaml.safe_dump(
            _h5_prediction_config(
                h5_path,
                prediction_dataframe_path,
                model_path,
                output_json_path,
                output_yaml_path,
            )
        ),
        encoding="utf-8",
    )

    assert main(["train", "--config", str(training_config_path)]) == 0
    assert main(["predict", "--config", str(prediction_config_path)]) == 0
    report = yaml.safe_load(output_yaml_path.read_text(encoding="utf-8"))

    assert prediction_dataframe_path.exists()
    assert report["patient_id"] == "P1"
    assert report["target_side"] == "Left"
    assert "input_h5_sha256" in report["provenance"]
    assert report["provenance"]["prediction_preprocessing_config_sha256"]
