from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from xrd_preprocessing import load_preprocessing_dataframe

from aramina.pipelines import run_preprocessing_pipeline

from .synthetic_aramina_h5 import (
    PAYLOAD_COLUMNS,
    load_synthetic_config,
    write_known_synthetic_h5,
)


def test_biopsy_patients_model_input_preprocessing_writes_joblib(tmp_path: Path):
    h5_path = tmp_path / "known_synthetic_aramina.h5"
    config_path = tmp_path / "aramina_biopsy_patients_model_input_v0_2.yaml"
    joblib_path = tmp_path / "aramina_biopsy_patients_model_input_v0_2.joblib"
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
    resolved_config = yaml.safe_load(loaded["preprocessing_config_yaml"])
    assert resolved_config["product_filter"]["require_biopsy_patient"] is True
    assert resolved_config["labels"]["product_column_builder"]["benign_values"] == [
        "BENIGN",
        "NORMAL",
    ]
    assert len(loaded["metadata"]["input_h5_sha256"]) == 64
    assert loaded["metadata"]["aramina_version"]
    assert loaded["metadata"]["aramina_git_sha"]
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
