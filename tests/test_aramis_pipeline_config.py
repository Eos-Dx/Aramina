from __future__ import annotations

import joblib
import pandas as pd
import pytest
import yaml
from xrd_preprocessing import build_pipeline_steps_from_config, load_preprocessing_dataframe
from xrd_preprocessing.transformers import H5BlobDataFrameTransformer, KeepColumnsTransformer

from aramis.__main__ import main
from aramis.pipelines import (
    AramisPreprocessingPipeline,
    _config_path,
    run_preprocessing_artifact_from_config,
)
from aramis.workflows import _load_workflow_config

from .synthetic_aramis_h5 import load_synthetic_config, write_known_synthetic_h5


def test_yaml_pipeline_steps_build_registered_transformers():
    config = load_synthetic_config()

    steps = build_pipeline_steps_from_config(config)
    names = [name for name, _ in steps]

    assert names[0] == "h5_blob_to_df"
    assert isinstance(steps[0][1], H5BlobDataFrameTransformer)
    assert names[-1] == "keep_columns"
    assert isinstance(steps[-1][1], KeepColumnsTransformer)
    assert steps[-1][1].columns == tuple(config["metadata"]["output_columns"])


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
    h5_path = tmp_path / "known_synthetic_aramis.h5"
    config = load_synthetic_config()
    config["metadata"]["output_columns"] = []
    write_known_synthetic_h5(h5_path)

    pipeline = AramisPreprocessingPipeline(config=config)

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
    h5_path = tmp_path / "known_synthetic_aramis.h5"
    output_path = tmp_path / "out" / "model_input.joblib"
    config_path = tmp_path / "preprocess.yaml"
    config = load_synthetic_config()
    config["io"] = {
        "input_h5_path": "known_synthetic_aramis.h5",
        "output_joblib_path": "out/model_input.joblib",
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
    assert artifact["metadata"]["aramis_version"]
    assert artifact["metadata"]["aramis_git_sha"]
    assert set(df["product_status_group"]) == {"BENIGN", "CANCER"}


def test_preprocess_cli_can_write_minimal_output_columns(tmp_path):
    h5_path = tmp_path / "known_synthetic_aramis.h5"
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
        "input_h5_path": "known_synthetic_aramis.h5",
        "output_joblib_path": "out/minimal.joblib",
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
    h5_path = tmp_path / "known_synthetic_aramis.h5"
    output_path = tmp_path / "out" / "artifact.joblib"
    config_path = tmp_path / "preprocess.yaml"
    config = load_synthetic_config()
    config["io"] = {
        "input_h5_path": h5_path.name,
        "output_joblib_path": "out/artifact.joblib",
    }
    config["raw_data"]["h5_dataset_candidates"]["npy"] = ["processed/data"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_known_synthetic_h5(h5_path)

    artifact = run_preprocessing_artifact_from_config(config_path)

    assert output_path.exists()
    assert artifact["dataframe"].equals(load_preprocessing_dataframe(output_path))


def test_workflow_contract_rejects_unknown_fields(tmp_path):
    workflow_config_path = tmp_path / "workflow.yaml"
    workflow_config_path.write_text(
        yaml.safe_dump(
            {
                "contract": "aramis_preprocess_train_workflow_v0_1",
                "workflow": {
                    "name": "test",
                    "created_by": "tester",
                    "created_at": "2026-07-14",
                    "output_folder": "outputs",
                    "mode": "legacy",
                },
                "preprocessing_config_path": "preprocess.yaml",
                "training_config_path": "train.yaml",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown workflow fields"):
        _load_workflow_config(workflow_config_path)
