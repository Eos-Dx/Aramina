from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml

from aramina.prediction import run_prediction_from_config
from aramina.prediction_scoring import _prediction_columns, _side_prediction
from aramina.training_config import PRODUCT_MODEL_NAME


ROOT = Path(__file__).parents[1]
EXPECTED_PATH = ROOT / "tests" / "data" / "golden_cancer_prediction_v0_2.yaml"
REQUEST_PATH = (
    ROOT
    / "examples"
    / "prediction"
    / "configs"
    / "config_predict_cancer_example.yaml"
)


def test_frozen_model_golden_h5_pipeline_parity(tmp_path: Path):
    """Protect H5-to-p_cancer behavior without changing numeric model policy."""
    expected = yaml.safe_load(EXPECTED_PATH.read_text(encoding="utf-8"))
    request = yaml.safe_load(REQUEST_PATH.read_text(encoding="utf-8"))
    h5_path = ROOT / request["io"]["input_h5_path"]
    model_path = ROOT / request["io"]["input_model_joblib_path"]
    request["io"]["input_h5_path"] = str(h5_path)
    request["io"]["input_model_joblib_path"] = str(model_path)
    request["io"]["output_folder"] = str(tmp_path / "prediction")
    request_path = tmp_path / "predict.yaml"
    request_path.write_text(
        yaml.safe_dump(request, sort_keys=False),
        encoding="utf-8",
    )

    reports = run_prediction_from_config(request_path)
    dataframe_paths = sorted(
        (tmp_path / "prediction").glob("*_prediction_dataframe.joblib")
    )
    assert len(dataframe_paths) == 1
    preprocessing_artifact = joblib.load(dataframe_paths[0])
    dataframe = preprocessing_artifact["dataframe"]

    assert _file_sha256(h5_path) == expected["h5_sha256"]
    assert preprocessing_artifact["version"] == "0.2"
    assert len(preprocessing_artifact["pipeline_fingerprint"]) == 64
    assert len(dataframe) == len(expected["retained_rows"])
    for row, reference in zip(
        dataframe.itertuples(index=False),
        expected["retained_rows"],
        strict=True,
    ):
        assert str(row.side).casefold() == reference["side"]
        assert row.position == reference["position"]
        assert float(row.snr_db) == pytest.approx(reference["snr_db"], abs=1e-8)
        assert float(np.sum(row.radial_profile_data)) == pytest.approx(
            reference["profile_sum"],
            abs=1e-8,
        )
        q = np.asarray(row.q_range, dtype=float)
        profile = np.asarray(row.radial_profile_data, dtype=float)
        normalizer = np.median(profile[(q >= 6.7) & (q <= 7.1)])
        assert normalizer == pytest.approx(1.0, abs=1e-10)

    q = np.asarray(dataframe.iloc[0]["q_range"], dtype=float)
    assert q.size == expected["q_grid"]["length"]
    assert float(q[0]) == pytest.approx(expected["q_grid"]["first"], abs=1e-10)
    assert float(q[-1]) == pytest.approx(expected["q_grid"]["last"], abs=1e-10)

    model_artifact = joblib.load(model_path)
    model_info = model_artifact["models"][PRODUCT_MODEL_NAME]
    prediction = _side_prediction(
        dataframe,
        model_info,
        patient_id=expected["patient_id"],
        target_side=expected["target_side"],
        columns=_prediction_columns(model_artifact),
        model_name=PRODUCT_MODEL_NAME,
        threshold_key="threshold_target",
    )
    feature_row = prediction["feature_row"]
    assert prediction["xrd_profile"]["measurement_p_cancer"] == pytest.approx(
        expected["target_measurement_p_cancer"],
        abs=1e-10,
    )
    assert feature_row["profile_p_cancer_logit_average"] == pytest.approx(
        expected["profile_p_cancer_logit_average"],
        abs=1e-10,
    )
    for column, reference in expected["core4"].items():
        assert feature_row[column] == pytest.approx(reference, abs=1e-10)
    assert feature_row["symmetry_available"] == expected["symmetry_available"]
    assert prediction["p_cancer"] == pytest.approx(
        expected["final_p_cancer"],
        abs=1e-10,
    )
    assert reports["internal_report"]["breast_predictions"]["target"][
        "final_prediction"
    ]["p_cancer"] == pytest.approx(expected["final_p_cancer"], abs=1e-5)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
