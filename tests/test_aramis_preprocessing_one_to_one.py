from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from xrd_preprocessing import load_preprocessing_dataframe

from aramis.pipelines import (
    AramisOneToOnePreprocessingPipeline,
    run_one_to_one_preprocessing_pipeline,
)

from .synthetic_aramis_h5 import (
    ONE_TO_ONE_OUTPUT_COLUMNS,
    assert_common_output_contract,
    load_synthetic_config,
    write_known_synthetic_h5,
)


def test_one_to_one_pipeline_dataframe_and_joblib_contract(tmp_path: Path):
    h5_path = tmp_path / "known_synthetic_aramis.h5"
    config_path = tmp_path / "aramis_one_to_one_max_v0_1.yaml"
    config = load_synthetic_config("one_to_one")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_known_synthetic_h5(h5_path)

    pipeline = AramisOneToOnePreprocessingPipeline(config=config_path)
    df = pipeline.fit_transform(h5_path)

    assert pipeline.fit(h5_path) is pipeline
    assert set(df.columns) == ONE_TO_ONE_OUTPUT_COLUMNS
    assert_common_output_contract(df)
    assert len(df) == 4
    assert set(df["patientId"]) == {"P1", "P3"}
    assert set(df["specimenId"]) == {"P1_LEFT", "P1_RIGHT", "P3_LEFT", "P3_RIGHT"}
    assert set(df["product_status_group"]) == {"BENIGN", "CANCER", "NORMAL"}
    assert set(df["one_to_one_pair_type"]) == {
        "BENIGN__CANCER",
        "CANCER__NORMAL",
    }
    assert set(df["patient_valid_specimen_count"]) == {2}
    assert "P2" not in set(df["patientId"])
    assert "P4" not in set(df["patientId"])
    assert "P5" not in set(df["patientId"])
    assert set(df["measurement_data_source"]) == {"npy:raw/data"}


def test_one_to_one_biopsy_keeps_contralateral_non_biopsy_side(tmp_path: Path):
    h5_path = tmp_path / "known_synthetic_aramis.h5"
    config_path = tmp_path / "aramis_one_to_one_biopsy_max_v0_1.yaml"
    config = load_synthetic_config("one_to_one_biopsy")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_known_synthetic_h5(h5_path)

    df = AramisOneToOnePreprocessingPipeline(config=config_path).fit_transform(h5_path)

    assert set(df.columns) == ONE_TO_ONE_OUTPUT_COLUMNS
    assert_common_output_contract(df)
    assert "P3_LEFT" in set(df["specimenId"])
    assert set(df[df["patientId"] == "P3"]["biopsy"]) == {False, True}
    assert set(df[df["patientId"] == "P3"]["product_status_group"]) == {
        "CANCER",
        "NORMAL",
    }


def test_one_to_one_wrapper_writes_joblib(tmp_path: Path):
    h5_path = tmp_path / "known_synthetic_aramis.h5"
    config_path = tmp_path / "aramis_one_to_one_max_v0_1.yaml"
    joblib_path = tmp_path / "aramis_one_to_one_dataframe.joblib"
    config = load_synthetic_config("one_to_one")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_known_synthetic_h5(h5_path)

    df = run_one_to_one_preprocessing_pipeline(
        h5_path,
        config_path,
        output_joblib_path=joblib_path,
    )
    loaded = joblib.load(joblib_path)
    loaded_df = load_preprocessing_dataframe(joblib_path)

    assert loaded["kind"] == "xrd_preprocessing_dataframe"
    assert loaded["preprocessing_config"]["aramis_preprocessing"]["branch"] == "one_to_one"
    assert loaded["preprocessing_config_text"]
    assert loaded["preprocessing_config_sha256"]
    assert loaded["metadata"]["branch"] == "one_to_one"
    assert len(loaded["metadata"]["input_h5_sha256"]) == 64
    assert loaded["metadata"]["aramis_version"]
    assert loaded["metadata"]["aramis_git_sha"]
    pd.testing.assert_frame_equal(df, loaded_df)
