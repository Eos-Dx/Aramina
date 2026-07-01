from __future__ import annotations

import joblib
import pandas as pd
import pytest
import yaml
from xrd_preprocessing import build_pipeline_steps_from_config, load_preprocessing_dataframe
from xrd_preprocessing.transformers import H5BlobDataFrameTransformer, KeepColumnsTransformer

from aramis.__main__ import main
from aramis.pipelines import (
    AramisOneToManyPreprocessingPipeline,
    _config_path,
)

from .synthetic_aramis_h5 import load_synthetic_config, write_known_synthetic_h5


def test_yaml_pipeline_steps_build_registered_transformers():
    config = load_synthetic_config("one_to_many_benign_cancer")

    steps = build_pipeline_steps_from_config(config)
    names = [name for name, _ in steps]

    assert names[0] == "h5_blob_to_df"
    assert isinstance(steps[0][1], H5BlobDataFrameTransformer)
    assert names[-1] == "keep_columns"
    assert isinstance(steps[-1][1], KeepColumnsTransformer)
    assert steps[-1][1].columns == tuple(config["metadata"]["output_columns"])


def test_yaml_pipeline_rejects_missing_or_unknown_transformer():
    config = load_synthetic_config("one_to_many_benign_cancer")
    config["pipeline"]["steps"][0]["transformer"] = "NoSuchTransformer"

    with pytest.raises(ValueError, match="Unknown pipeline transformer"):
        build_pipeline_steps_from_config(config)

    config = load_synthetic_config("one_to_many_benign_cancer")
    del config["pipeline"]["steps"][0]["transformer"]

    with pytest.raises(ValueError, match="missing transformer"):
        build_pipeline_steps_from_config(config)


def test_yaml_pipeline_skips_disabled_steps():
    config = load_synthetic_config("one_to_many_benign_cancer")
    config["pipeline"]["steps"][1]["enabled"] = False

    names = [name for name, _ in build_pipeline_steps_from_config(config)]

    assert "product_columns" not in names


def test_output_columns_are_mandatory(tmp_path):
    h5_path = tmp_path / "known_synthetic_aramis.h5"
    config = load_synthetic_config("one_to_many_benign_cancer")
    config["metadata"]["output_columns"] = []
    write_known_synthetic_h5(h5_path)

    pipeline = AramisOneToManyPreprocessingPipeline(config=config)

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
    output_path = tmp_path / "out" / "one_to_many.joblib"
    config_path = tmp_path / "preprocess.yaml"
    config = load_synthetic_config("one_to_many_benign_cancer")
    config["io"] = {
        "input_h5_path": "known_synthetic_aramis.h5",
        "output_joblib_path": "out/one_to_many.joblib",
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
    assert artifact["preprocessing_config"]["aramis_preprocessing"]["branch"] == "one_to_many"
    assert artifact["preprocessing_config_text"]
    assert artifact["preprocessing_config_sha256"]
    assert artifact["metadata"]["branch"] == "one_to_many"
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
    config = load_synthetic_config("one_to_many_benign_cancer")
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
