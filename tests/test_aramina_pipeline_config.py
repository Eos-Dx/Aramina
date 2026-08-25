from __future__ import annotations

import joblib
from pathlib import Path
import pandas as pd
import pytest
import yaml
from xrd_preprocessing import (
    build_pipeline_from_config,
    build_pipeline_steps_from_config,
    load_preprocessing_config,
    load_preprocessing_dataframe,
)
from xrd_preprocessing.transformers import H5BlobDataFrameTransformer, KeepColumnsTransformer

from aramina.__main__ import main
from aramina.pipelines import (
    AraminaPreprocessingPipeline,
    _config_path,
    run_preprocessing_artifact_from_config,
)
from aramina.prediction_contract import _validate_prediction_config
from aramina.preprocessing_contract import validate_aramina_preprocessing_config
from aramina.training_config import load_training_config
from aramina.workflows import _load_preprocess_train_config

from .synthetic_aramina_h5 import load_synthetic_config, write_known_synthetic_h5


def test_shipped_product_yaml_contracts_build_or_validate():
    root = Path(__file__).parents[1]
    expected_steps = {
        "config_preprocessing_biopsy_patients_v0_2.yaml": 19,
        "config_preprocessing_prediction_patient_v0_2.yaml": 16,
    }
    for filename, count in expected_steps.items():
        config = load_preprocessing_config(root / "config" / "preprocessing" / filename)
        assert len(build_pipeline_steps_from_config(config)) == count
        validate_aramina_preprocessing_config(config)

    load_training_config(
        root / "config" / "training" / "config_training_target_breast_risk_v0_4.yaml"
    )
    _load_preprocess_train_config(
        root
        / "config"
        / "preprocessing_and_training"
        / "config_preprocess_and_train_target_breast_risk_v0_3.yaml"
    )
    for path in sorted((root / "config" / "prediction").glob("*.yaml")):
        _validate_prediction_config(
            yaml.safe_load(path.read_text(encoding="utf-8")), path
        )


def test_legacy_preprocessing_is_prediction_only_under_current_code():
    root = Path(__file__).parents[1] / "config" / "preprocessing"
    legacy_prediction = load_preprocessing_config(
        root / "config_preprocessing_prediction_patient_v0_1.yaml"
    )
    validate_aramina_preprocessing_config(legacy_prediction)

    legacy_training = load_preprocessing_config(
        root / "config_preprocessing_biopsy_patients_v0_1.yaml"
    )
    with pytest.raises(ValueError, match="preprocessing contract"):
        validate_aramina_preprocessing_config(legacy_training)
    for path in sorted(
        (root / "examples" / "prediction" / "configs").glob("*.yaml")
    ):
        _validate_prediction_config(
            yaml.safe_load(path.read_text(encoding="utf-8")), path
        )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda config: config["snr"].update(min_snr_db=20.0),
            "SNR threshold=18.0",
        ),
        (
            lambda config: config["product_filter"].update(require_biopsy_patient=False),
            "training biopsy-patient filter=True",
        ),
        (
            lambda config: config["metadata"].update(output_columns=[]),
            "output columns are missing",
        ),
        (
            lambda config: config.pop("data_version"),
            "requires mapping data_version",
        ),
    ],
)
def test_aramina_product_preprocessing_contract_rejects_policy_changes(
    mutate,
    error: str,
):
    config = load_preprocessing_config(
        Path(__file__).parents[1]
        / "config"
        / "preprocessing"
        / "config_preprocessing_biopsy_patients_v0_2.yaml"
    )
    mutate(config)

    with pytest.raises(ValueError, match=error):
        validate_aramina_preprocessing_config(config)


def test_yaml_pipeline_steps_build_registered_transformers():
    config = load_synthetic_config()

    steps = build_pipeline_steps_from_config(config)
    names = [name for name, _ in steps]

    assert names[0] == "h5_blob_to_df"
    assert isinstance(steps[0][1], H5BlobDataFrameTransformer)
    assert names[-1] == "keep_columns"
    assert isinstance(steps[-1][1], KeepColumnsTransformer)
    assert steps[-1][1].columns == tuple(config["metadata"]["output_columns"])


def test_yaml_pipeline_can_emit_sklearn_step_progress():
    pipeline = build_pipeline_from_config(load_synthetic_config(), verbose=True)

    assert pipeline.verbose is True


def test_yaml_pipeline_rejects_missing_or_unknown_transformer():
    config = load_synthetic_config()
    config["pipeline"]["steps"][0]["transformer"] = "NoSuchTransformer"

    with pytest.raises(ValueError, match="Unknown pipeline transformer"):
        build_pipeline_steps_from_config(config)

    config = load_synthetic_config()
    del config["pipeline"]["steps"][0]["transformer"]

    with pytest.raises(ValueError, match="missing transformer"):
        build_pipeline_steps_from_config(config)


def test_yaml_pipeline_skips_disabled_steps():
    config = load_synthetic_config()
    step = next(
        step
        for step in config["pipeline"]["steps"]
        if step["name"] == "product_columns"
    )
    step["enabled"] = False

    names = [name for name, _ in build_pipeline_steps_from_config(config)]

    assert "product_columns" not in names


def test_output_columns_are_mandatory(tmp_path):
    h5_path = tmp_path / "known_synthetic_aramina.h5"
    config = load_synthetic_config()
    config["metadata"]["output_columns"] = []
    write_known_synthetic_h5(h5_path)

    pipeline = AraminaPreprocessingPipeline(config=config)

    with pytest.raises(ValueError, match="requires metadata.output_columns"):
        pipeline.fit_transform(h5_path)


def test_config_path_resolves_absolute_relative_and_missing(tmp_path):
    config = {"io": {"input_h5_path": "data/input.h5", "output_joblib_path": ""}}
    config_path = tmp_path / "config" / "preprocess.yaml"
    config_path.parent.mkdir()

    assert _config_path(config, config_path, "input_h5_path") == (
        config_path.parent / "data" / "input.h5"
    ).resolve()
    with pytest.raises(ValueError, match="Missing io.output_joblib_path"):
        _config_path(config, config_path, "output_joblib_path")


def test_preprocess_cli_reads_input_and_output_from_yaml(tmp_path):
    h5_path = tmp_path / "known_synthetic_aramina.h5"
    output_path = tmp_path / "out" / "model_input.joblib"
    config_path = tmp_path / "preprocess.yaml"
    config = load_synthetic_config()
    config["io"] = {
        "input_h5_path": str(h5_path),
        "output_joblib_path": str(output_path),
    }
    config["raw_data"]["h5_dataset_candidates"]["npy"] = ["processed/data"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_known_synthetic_h5(h5_path)

    exit_code = main(["preprocess", "--config", str(config_path)])

    assert exit_code == 0
    assert output_path.exists()
    artifact = joblib.load(output_path)
    df = load_preprocessing_dataframe(output_path)
    assert isinstance(artifact, dict)
    assert isinstance(artifact["dataframe"], pd.DataFrame)
    resolved = yaml.safe_load(artifact["preprocessing_config_yaml"])
    assert resolved["pipeline"]["steps"]
    assert "extends" not in resolved
    assert len(artifact["metadata"]["input_h5_sha256"]) == 64
    assert artifact["metadata"]["aramina_version"]
    assert artifact["metadata"]["aramina_git_sha"]
    assert set(df["product_status_group"]) == {"BENIGN", "CANCER"}


def test_preprocess_cli_can_write_minimal_output_columns(tmp_path):
    h5_path = tmp_path / "known_synthetic_aramina.h5"
    output_path = tmp_path / "out" / "minimal.joblib"
    config_path = tmp_path / "preprocess_minimal.yaml"
    output_columns = [
        "patientId",
        "specimenId",
        "q_range",
        "radial_profile_data_raw",
        "radial_profile_data",
    ]
    config = load_synthetic_config()
    config["io"] = {
        "input_h5_path": str(h5_path),
        "output_joblib_path": str(output_path),
    }
    config["metadata"]["output_columns"] = output_columns
    config["normalization"]["save_initial_data"] = True
    config["raw_data"]["h5_dataset_candidates"]["npy"] = ["processed/data"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_known_synthetic_h5(h5_path)

    exit_code = main(["preprocess", "--config", str(config_path)])

    assert exit_code == 0
    df = load_preprocessing_dataframe(output_path)
    assert df.columns.tolist() == output_columns
    assert not df["radial_profile_data_raw"].equals(df["radial_profile_data"])


def test_preprocessing_artifact_runner_uses_configured_h5_and_output(tmp_path):
    h5_path = tmp_path / "known_synthetic_aramina.h5"
    output_path = tmp_path / "out" / "artifact.joblib"
    config_path = tmp_path / "preprocess.yaml"
    config = load_synthetic_config()
    config["io"] = {
        "input_h5_path": str(h5_path),
        "output_joblib_path": str(output_path),
    }
    config["raw_data"]["h5_dataset_candidates"]["npy"] = ["processed/data"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_known_synthetic_h5(h5_path)

    artifact = run_preprocessing_artifact_from_config(config_path)

    assert output_path.exists()
    assert artifact["dataframe"].equals(load_preprocessing_dataframe(output_path))


def test_preprocess_train_contract_rejects_unknown_fields(tmp_path):
    config_path = tmp_path / "preprocess_train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "contract": "aramina_preprocessing_and_training_config_v0_3",
                "preprocessing_and_training": {
                    "name": "test",
                    "run_author": "tester",
                    "output_folder": "outputs",
                    "mode": "legacy",
                },
                "preprocessing_config_path": "preprocess.yaml",
                "training_config_path": "train.yaml",
                "mlflow": {
                    "enabled": False,
                    "tracking_uri": "outputs/mlflow",
                    "experiment_name": "test",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown preprocessing-and-training fields"):
        _load_preprocess_train_config(config_path)
