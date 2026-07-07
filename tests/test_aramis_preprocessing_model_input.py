from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from xrd_preprocessing import load_preprocessing_dataframe

from aramis.pipelines import AramisPreprocessingPipeline, run_preprocessing_pipeline

from .synthetic_aramis_h5 import (
    PAYLOAD_COLUMNS,
    load_synthetic_config,
    write_known_synthetic_h5,
)


def test_all_patients_model_input_preprocessing_contract(tmp_path: Path):
    h5_path = tmp_path / "known_synthetic_aramis.h5"
    config_path = tmp_path / "aramis_all_patients_model_input_v0_1.yaml"
    config = load_synthetic_config("all_patients")
    config["raw_data"]["h5_dataset_candidates"]["npy"] = ["processed/data"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_known_synthetic_h5(h5_path)

    pipeline = AramisPreprocessingPipeline(config=config_path)
    df = pipeline.fit_transform(h5_path)

    assert pipeline.fit(h5_path) is pipeline
    assert set(df.columns) == set(config["metadata"]["output_columns"])
    assert PAYLOAD_COLUMNS.isdisjoint(df.columns)
    assert len(df) == 6
    assert set(df["patientId"]) == {"P1", "P2", "P3", "P4"}
    assert set(df["specimenId"]) == {
        "P1_LEFT",
        "P1_RIGHT",
        "P2_LEFT",
        "P3_LEFT",
        "P3_RIGHT",
        "P4_RIGHT",
    }
    assert set(df["product_status_group"]) == {"BENIGN", "CANCER"}
    assert set(df[df["specimen_status"] == "NORMAL"]["product_status_group"]) == {
        "BENIGN"
    }
    assert "P4_LEFT" not in set(df["specimenId"])
    assert "P5" not in set(df["patientId"])
    assert set(df["measurement_data_source"]) == {"npy:processed/data"}


def test_biopsy_patients_model_input_preprocessing_writes_joblib(tmp_path: Path):
    h5_path = tmp_path / "known_synthetic_aramis.h5"
    config_path = tmp_path / "aramis_biopsy_patients_model_input_v0_1.yaml"
    joblib_path = tmp_path / "aramis_biopsy_patients_model_input_v0_1.joblib"
    config = load_synthetic_config("biopsy_patients")
    config["raw_data"]["h5_dataset_candidates"]["npy"] = ["processed/data"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_known_synthetic_h5(h5_path)

    df = run_preprocessing_pipeline(
        h5_path,
        config_path,
        output_joblib_path=joblib_path,
    )
    loaded = joblib.load(joblib_path)
    loaded_df = load_preprocessing_dataframe(joblib_path)

    assert loaded["kind"] == "xrd_preprocessing_dataframe"
    assert loaded["preprocessing_config"]["aramis_preprocessing"]["branch"] == "one_to_many"
    assert loaded["preprocessing_config_text"]
    assert loaded["preprocessing_config_sha256"]
    assert loaded["metadata"]["branch"] == "one_to_many"
    assert len(loaded["metadata"]["input_h5_sha256"]) == 64
    assert loaded["metadata"]["aramis_version"]
    assert loaded["metadata"]["aramis_git_sha"]
    pd.testing.assert_frame_equal(df, loaded_df)
    assert set(df.columns) == set(config["metadata"]["output_columns"])
    assert PAYLOAD_COLUMNS.isdisjoint(df.columns)
    assert len(df) == 5
    assert set(df["patientId"]) == {"P1", "P3", "P4"}
    assert set(df["specimenId"]) == {
        "P1_LEFT",
        "P1_RIGHT",
        "P3_LEFT",
        "P3_RIGHT",
        "P4_RIGHT",
    }
    assert set(df["product_status_group"]) == {"BENIGN", "CANCER"}
    assert set(df[df["specimen_status"] == "NORMAL"]["product_status_group"]) == {
        "BENIGN"
    }
