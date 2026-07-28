from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib

import pandas as pd
import pytest
import yaml
from xrd_preprocessing import (
    build_preprocessing_artifact,
    load_preprocessing_config,
)

from aramina import preprocessing_lineage
from aramina.preprocessing_contract import validate_aramina_preprocessing_config


ROOT = Path(__file__).parents[1]
TRAINING_CONFIG = (
    ROOT
    / "config"
    / "preprocessing"
    / "config_preprocessing_biopsy_patients_v0_2.yaml"
)
PREDICTION_CONFIG = (
    ROOT
    / "config"
    / "preprocessing"
    / "config_preprocessing_prediction_patient_v0_2.yaml"
)
XRD_IDENTITY = {
    "version": "0.1.8b0",
    "requested_revision": "a" * 40,
    "git_commit": "a" * 40,
}
XRD_COMMIT = "18ddac4be429e612ac82f8e81605d98399acee02"


@pytest.fixture(autouse=True)
def fixed_xrd_identity(monkeypatch):
    monkeypatch.setattr(
        preprocessing_lineage,
        "xrd_runtime_identity",
        lambda: dict(XRD_IDENTITY),
    )


def _config(path: Path) -> dict:
    return load_preprocessing_config(path)


def _artifact(config: dict) -> dict:
    lineage = preprocessing_lineage.build_preprocessing_lineage(config)
    return build_preprocessing_artifact(
        pd.DataFrame({"patientId": ["P01"]}),
        preprocessing_config=config,
        preprocessing_config_text=yaml.safe_dump(config, sort_keys=False),
        metadata={
            "input_h5_sha256": "abc",
            "aramina_preprocessing_lineage": lineage,
        },
    )


def test_product_marker_requires_contract_route_and_version():
    config = _config(TRAINING_CONFIG)
    del config["aramina_preprocessing"]["route"]

    with pytest.raises(ValueError, match="product route='training'"):
        validate_aramina_preprocessing_config(config)


def test_new_training_requires_v02_training_route_and_exact_lineage():
    config = _config(TRAINING_CONFIG)
    artifact = _artifact(config)

    lineage = preprocessing_lineage.require_training_preprocessing_artifact(artifact)

    assert lineage["product"]["route"] == "training"
    assert lineage["xrd_preprocessing"] == {
        "release_tag": config["xrd_preprocessing"]["release_tag"],
        **XRD_IDENTITY,
    }
    assert len(lineage["pipeline_fingerprint"]) == 64


def test_product_lineage_rejects_declared_xrd_version_mismatch(monkeypatch):
    config = _config(TRAINING_CONFIG)
    monkeypatch.setattr(
        preprocessing_lineage,
        "xrd_runtime_identity",
        lambda: {**XRD_IDENTITY, "version": "0.1.7b0"},
    )

    with pytest.raises(ValueError, match="package version differs"):
        preprocessing_lineage.build_preprocessing_lineage(config)


def test_repository_and_prediction_image_pin_same_full_xrd_commit():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency = next(
        item
        for item in pyproject["project"]["dependencies"]
        if item.startswith("xrd-preprocessing @")
    )
    dockerfile = (
        ROOT / "packaging" / "prediction_api_bundle" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert dependency.endswith(f"@{XRD_COMMIT}")
    assert f"XRD-preprocessing.git@{XRD_COMMIT}" in dockerfile


def test_new_training_rejects_legacy_v01_artifact():
    artifact = {
        "kind": "xrd_preprocessing_dataframe",
        "version": "0.1",
        "dataframe": pd.DataFrame({"patientId": ["P01"]}),
        "preprocessing_config_yaml": None,
        "metadata": {},
    }

    with pytest.raises(ValueError, match="legacy v0.1 is read-only"):
        preprocessing_lineage.require_training_preprocessing_artifact(artifact)


def test_training_rejects_tampered_product_lineage():
    artifact = _artifact(_config(TRAINING_CONFIG))
    artifact["metadata"]["aramina_preprocessing_lineage"][
        "pipeline_fingerprint"
    ] = "0" * 64

    with pytest.raises(ValueError, match="lineage differs"):
        preprocessing_lineage.require_training_preprocessing_artifact(artifact)


def test_prediction_accepts_frozen_legacy_model_config_read_only():
    frozen = yaml.safe_load(
        (
            ROOT
            / "models"
            / "aramina_target_breast_risk_0_2_12-beta_9bb911189af6"
            / "prediction_preprocessing_config.yaml"
        ).read_text(encoding="utf-8")
    )

    preprocessing_lineage.validate_prediction_preprocessing_compatibility(
        {},
        frozen,
    )


def test_new_prediction_requires_exact_model_held_fingerprint():
    config = _config(PREDICTION_CONFIG)
    identity = preprocessing_lineage.build_preprocessing_lineage(config)
    model_artifact = {"prediction_preprocessing_lineage": identity}

    preprocessing_lineage.validate_prediction_preprocessing_compatibility(
        model_artifact,
        config,
    )

    changed = deepcopy(config)
    changed["snr"]["min_snr_db"] = 20.0
    with pytest.raises(ValueError):
        preprocessing_lineage.validate_prediction_preprocessing_compatibility(
            model_artifact,
            changed,
        )
