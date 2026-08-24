from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from aramina import mlflow_artifacts
from aramina.mlflow_artifacts import (
    MLFLOW_REQUIRED_ARTIFACTS,
    _dataset_fingerprint,
    _measurement_manifests,
    write_mlflow_product_artifacts,
)
from aramina.pipelines import run_preprocessing_pipeline

from .synthetic_aramina_h5 import load_synthetic_config, write_known_synthetic_h5


def test_product_mlflow_artifact_set_is_complete_and_patient_safe(
    tmp_path: Path,
    monkeypatch,
):
    h5_path = tmp_path / "input.h5"
    preprocessing_config_path = tmp_path / "preprocessing.yaml"
    dataframe_joblib = tmp_path / "preprocessed.joblib"
    config = load_synthetic_config("biopsy_patients")
    config["io"] = {
        "input_h5_path": str(h5_path),
        "output_joblib_path": str(dataframe_joblib),
    }
    config["raw_data"]["h5_dataset_candidates"]["npy"] = ["processed/data"]
    preprocessing_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_known_synthetic_h5(h5_path)
    run_preprocessing_pipeline(
        h5_path,
        preprocessing_config_path,
        output_joblib_path=dataframe_joblib,
    )
    preprocessing_artifact = joblib.load(dataframe_joblib)
    resolved_config = yaml.safe_load(
        preprocessing_artifact["preprocessing_config_yaml"]
    )
    resolved_config["aramina_preprocessing"] = {
        "name": "aramina_biopsy_patients_model_input",
        "version": "0.2",
        "clinical_stage": "research draft",
    }
    data_version = {
        "contract": "aramina_dvc_input_v0_1",
        "system": "dvc",
        "dataset_id": "synthetic_h5",
        "dvc_version": "3.67.1",
        "pointer_path": "data/input.h5.dvc",
        "output_path": "input.h5",
        "hash_algorithm": "md5",
        "hash": "c" * 32,
        "size_bytes": h5_path.stat().st_size,
        "input_h5_sha256": preprocessing_artifact["metadata"]["input_h5_sha256"],
    }
    resolved_config["data_version"] = {
        key: data_version[key]
        for key in (
            "contract",
            "system",
            "dataset_id",
            "dvc_version",
            "pointer_path",
        )
    }
    preprocessing_artifact["metadata"]["data_version"] = data_version
    pointer_path = tmp_path / "data" / "input.h5.dvc"
    pointer_path.parent.mkdir()
    pointer_path.write_text(
        yaml.safe_dump(
            {
                "outs": [
                    {
                        "md5": data_version["hash"],
                        "size": data_version["size_bytes"],
                        "hash": "md5",
                        "path": "input.h5",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    preprocessing_artifact["preprocessing_config_yaml"] = yaml.safe_dump(
        resolved_config, sort_keys=False
    )
    dataframe = preprocessing_artifact["dataframe"]
    candidates = dataframe[
        ["patientId", "specimenId", "side", "position", "started_at"]
    ].copy()
    candidates["session_uid"] = [f"session-{index}" for index in range(len(candidates))]
    candidates["set_path"] = [f"/set/{index}" for index in range(len(candidates))]
    dropped_candidate = candidates.iloc[0].copy()
    dropped_candidate["position"] = "P9"
    dropped_candidate["set_path"] = "/set/dropped"
    candidates = pd.concat(
        [candidates, pd.DataFrame([dropped_candidate])], ignore_index=True
    )
    monkeypatch.setattr(
        mlflow_artifacts,
        "list_h5_measurement_sets",
        lambda *_args, **_kwargs: candidates,
    )

    training_folder = tmp_path / "workflow" / "training" / "run"
    training_folder.mkdir(parents=True)
    (training_folder / "model.joblib").write_bytes(b"test-model")
    predictions = _evaluation_predictions(dataframe)
    predictions.to_csv(training_folder / "evaluation_predictions.csv", index=False)
    _evaluation_splits(dataframe).to_csv(
        training_folder / "evaluation_splits.csv", index=False
    )
    (training_folder / "evaluation.yaml").write_text(
        yaml.safe_dump(
            {
                "output_type": "aramina_evaluation_artifact",
                "metric_summary": [{"roc_auc_mean": 0.7}],
            }
        ),
        encoding="utf-8",
    )
    training_artifact = _training_artifact(
        training_folder=training_folder,
        input_h5_sha256=preprocessing_artifact["metadata"]["input_h5_sha256"],
        data_version=data_version,
    )

    result = write_mlflow_product_artifacts(
        run_folder=tmp_path / "workflow",
        preprocessing_artifact=preprocessing_artifact,
        training_artifact=training_artifact,
        preprocessing_config_path=preprocessing_config_path,
        preprocess_train_contract="aramina_preprocessing_and_training_config_v0_2",
    )

    root = result["artifact_directory"]
    assert all((root / filename).is_file() for filename in MLFLOW_REQUIRED_ARTIFACTS)
    selected = pd.read_csv(root / "selected_measurement_ids.csv")
    dropped = pd.read_csv(root / "dropped_measurements.csv")
    split = pd.read_csv(root / "train_test_split.csv")
    assert len(selected) == len(dataframe)
    assert selected["measurement_id"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert not dropped.empty
    assert set(dropped["drop_reason"]) == {"not_selected_by_resolved_product_pipeline"}
    assert not (
        split.groupby(["split_id", "patient_id"])["partition"].nunique() > 1
    ).any()
    assert set(split["partition"]) == {"train", "test"}
    assert result["tags"]["product"] == "aramina"
    assert result["tags"]["dvc_dataset_id"] == "synthetic_h5"
    assert result["tags"]["dvc_data_hash"] == "c" * 32
    assert result["tags"]["dvc_version"] == "3.67.1"
    assert (
        root.joinpath("dvc_data_pointer.dvc").read_text(encoding="utf-8")
        == pointer_path.read_text(encoding="utf-8")
    )
    assert json.loads(root.joinpath("data_version.json").read_text(encoding="utf-8"))[
        "input_h5_sha256"
    ] == preprocessing_artifact["metadata"]["input_h5_sha256"]
    assert (
        result["tags"]["input_h5_checksum"]
        == preprocessing_artifact["metadata"]["input_h5_sha256"]
    )
    assert len(result["manifest"]["dataset_fingerprint"]) == 64
    feature_schema = json.loads(
        (root / "feature_schema.json").read_text(encoding="utf-8")
    )
    assert (
        _dataset_fingerprint(
            dataframe=dataframe.sample(frac=1.0, random_state=3),
            selected_measurements=selected.sample(frac=1.0, random_state=4),
            feature_schema=feature_schema,
        )
        == result["manifest"]["dataset_fingerprint"]
    )
    assert result["metrics"]["held_out.roc_auc.mean"] == 0.7
    assert result["metrics"]["final_fit.sensitivity"] == 0.96
    assert result["params"]["profile_encoder.type"] == "raw_radial_profile"
    assert result["params"]["profile_encoder.input_q_bins"] == 100
    assert result["params"]["profile_encoder.output_dimensions"] == 100
    assert "profile_encoder.components" not in result["params"]
    manifest = json.loads((root / "mlflow_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"] == "aramina_mlflow_product_run_v0_2"


def test_measurement_manifest_preserves_duplicate_clinical_keys(
    tmp_path: Path,
    monkeypatch,
):
    selected_rows = pd.DataFrame(
        [
            {
                "patientId": "P01",
                "specimenId": "P01_LEFT_P1",
                "side": "LEFT",
                "position": "P1",
                "started_at": "2026-01-02T03:04:05",
                "biopsy": True,
                "product_status_group": "CANCER",
            },
            {
                "patientId": "P01",
                "specimenId": "P01_LEFT_P1",
                "side": "LEFT",
                "position": "P1",
                "started_at": "2026-01-02T03:04:05",
                "biopsy": True,
                "product_status_group": "CANCER",
            },
        ]
    )
    candidates = selected_rows.copy()
    candidates["session_uid"] = ["session-1", "session-1"]
    candidates["set_path"] = ["/sample/set_001", "/sample/set_002"]
    monkeypatch.setattr(
        mlflow_artifacts,
        "list_h5_measurement_sets",
        lambda *_args, **_kwargs: candidates,
    )

    selected, dropped = _measurement_manifests(
        dataframe=selected_rows,
        preprocessing_config={"io": {"input_h5_path": str(tmp_path / "input.h5")}},
        preprocessing_config_path=tmp_path / "preprocessing.yaml",
    )

    assert selected["measurement_id"].nunique() == 2
    assert set(selected["set_path"]) == {"/sample/set_001", "/sample/set_002"}
    assert dropped.empty


def _evaluation_predictions(dataframe: pd.DataFrame) -> pd.DataFrame:
    patients = sorted(dataframe["patientId"].astype(str).unique())
    rows = []
    for patient_index, patient_id in enumerate(patients):
        split_id = patient_index % 2
        rows.append(
            {
                "target_case_id": f"{patient_id}::LEFT",
                "patientId": patient_id,
                "label": split_id % 2,
                "label_name": "CANCER" if split_id % 2 else "BENIGN",
                "model_name": "aramina_target_breast_risk",
                "split_id": split_id,
                "evaluation_mode": "stratified_kfold",
                "p_cancer": 0.6,
                "model_route": "default",
                "threshold_youden": 0.5,
                "threshold_target": 0.25,
                "y_pred_target": 1,
            }
        )
    return pd.DataFrame(rows)


def _evaluation_splits(dataframe: pd.DataFrame) -> pd.DataFrame:
    patients = sorted(dataframe["patientId"].astype(str).unique())
    return pd.DataFrame(
        [
            {
                "split_id": split_id,
                "repeat_id": 0,
                "fold_id": split_id,
                "patientId": patient_id,
                "partition": ("test" if patient_index % 2 == split_id else "train"),
            }
            for split_id in range(2)
            for patient_index, patient_id in enumerate(patients)
        ]
    )


def _training_artifact(
    *,
    training_folder: Path,
    input_h5_sha256: str,
    data_version: dict[str, object],
) -> dict[str, object]:
    return {
        "run_folder": str(training_folder),
        "model_type": "m2q_gated_target_case",
        "model_identity": {
            "name": "aramina_target_breast_risk",
            "version": "0.2.13-beta",
            "clinical_stage": "research draft",
        },
        "models": {
            "aramina_target_breast_risk": {
                "profile_encoder": {
                    "type": "raw_radial_profile",
                    "input_q_bins": 100,
                    "output_dimensions": 100,
                },
                "thresholds": {"threshold_target": 0.25},
            }
        },
        "feature_schema": {"features": ["profile", "age", "symmetry"]},
        "model_columns": {"profile_column": "radial_profile_data"},
        "model_definition_yaml": yaml.safe_dump(
            {
                "class_definition": {
                    "reference_class": "BENIGN",
                    "target_class": "CANCER",
                }
            }
        ),
        "model_performance": {
            "held_out_metrics": {
                "roc_auc": {"mean": 0.7, "std": 0.05},
                "sensitivity": {"mean": 0.8, "std": 0.1},
                "specificity": {"mean": 0.4, "std": 0.1},
            }
        },
        "final_fit_training_metrics": {
            "sensitivity": 0.96,
            "specificity": 0.5,
            "true_positives": 10,
        },
        "evaluation": {
            "protocol": {
                "method": "repeated_stratified_kfold",
                "folds": 2,
                "repeats": 1,
                "random_seed": 42,
            }
        },
        "reproducibility": {
            "source_h5": {
                "filename": "input.h5",
                "sha256": input_h5_sha256,
                "data_version": data_version,
            },
            "source_code": {
                "aramina": {"git_sha": "a" * 40},
                "xrd_preprocessing": {"git_commit": "b" * 40},
            },
        },
    }
