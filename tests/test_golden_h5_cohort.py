from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import joblib
import pytest
import yaml

from aramina.pipelines import run_preprocessing_pipeline
from aramina.prediction_scoring import _prediction_columns, _side_prediction
from aramina.training_config import PRODUCT_MODEL_NAME


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "tests" / "data" / "golden_prediction_cohort_v0_1.yaml"
MODEL_PATH = (
    ROOT
    / "models"
    / "aramina_target_breast_risk_0_2_12-beta_9bb911189af6"
    / "model.joblib"
)


def test_frozen_model_ten_patient_golden_cohort():
    """Protect prediction parity across bilateral and neutral-symmetry routes."""
    expected = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = expected["cases"]
    assert len(cases) == 10
    assert len({case["patient_id"] for case in cases}) == 10
    assert {case["symmetry_available"] for case in cases} == {0, 1}

    model_artifact = joblib.load(MODEL_PATH)
    preprocessing_config = yaml.safe_load(
        model_artifact["prediction_preprocessing_yaml"]
    )
    model_info = model_artifact["models"][PRODUCT_MODEL_NAME]
    columns = _prediction_columns(model_artifact)

    _validate_source_hashes(expected["sources"])
    archive_path = (
        ROOT / expected["sources"]["five_patient_archive"]["path"]
    ).resolve()
    archive_dataframe = run_preprocessing_pipeline(
        archive_path,
        preprocessing_config,
        allow_legacy_product_config=True,
    )

    for case in cases:
        source_path = case.get("source_path")
        if source_path:
            dataframe = run_preprocessing_pipeline(
                (ROOT / source_path).resolve(),
                preprocessing_config,
                allow_legacy_product_config=True,
            )
        else:
            dataframe = archive_dataframe.loc[
                archive_dataframe["patientId"].eq(case["patient_id"])
            ].copy()
        prediction = _side_prediction(
            dataframe,
            model_info,
            patient_id=case["patient_id"],
            target_side=case["target_side"],
            columns=columns,
            model_name=PRODUCT_MODEL_NAME,
            threshold_key="threshold_target",
        )
        feature_row = prediction["feature_row"]

        assert len(dataframe) == case["retained_rows"]
        assert (
            len(prediction["xrd_profile"]["measurement_p_cancer"])
            == case["target_measurements"]
        )
        assert int(feature_row["symmetry_available"]) == case["symmetry_available"]
        assert feature_row["profile_p_cancer_logit_average"] == pytest.approx(
            case["profile_p_cancer_logit_average"],
            abs=1e-10,
        )
        assert prediction["p_cancer"] == pytest.approx(
            case["p_cancer"],
            abs=1e-10,
        )

def _validate_source_hashes(sources: dict) -> None:
    for relative_path, expected_hash in sources["examples"]["sha256"].items():
        assert _file_sha256(ROOT / relative_path) == expected_hash
    archive = sources["five_patient_archive"]
    assert _file_sha256(ROOT / archive["path"]) == archive["sha256"]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
