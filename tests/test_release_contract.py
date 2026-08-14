from __future__ import annotations

from pathlib import Path
import re
import tomllib

import yaml


ROOT = Path(__file__).parents[1]
XRD_COMMIT = "88dcaa277c5a0d4be2ab637bc5827a14bd106bea"
XRD_VERSION = "0.1.9b0"
XRD_RELEASE_TAG = "v0.1.9-beta"
SOURCE_MODEL_VERSION = "0.3.1-beta"
OPERATIONAL_MODEL_VERSION = "0.2.12-beta"


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _yaml(path: str) -> dict:
    value = yaml.safe_load(_text(path))
    assert isinstance(value, dict)
    return value


def _xrd_dependency(pyproject: dict) -> str:
    dependencies = pyproject["project"]["dependencies"]
    return next(item for item in dependencies if item.startswith("xrd-preprocessing @"))


def test_source_package_model_and_xrd_release_identity_are_consistent():
    pyproject = tomllib.loads(_text("pyproject.toml"))
    package_version = pyproject["project"]["version"]
    assert package_version == "0.3.1b0"
    assert package_version.replace("b0", "-beta") == SOURCE_MODEL_VERSION
    assert _xrd_dependency(pyproject).endswith(f"@{XRD_COMMIT}")

    policy = _yaml("config/preprocessing/shared/aramina_policy_v0_1.yaml")
    assert policy["xrd_preprocessing"] == {
        "version": XRD_VERSION,
        "release_tag": XRD_RELEASE_TAG,
        "git_commit": XRD_COMMIT,
    }

    training_config = _yaml(
        "config/training/config_training_target_breast_risk_v0_1.yaml"
    )
    assert training_config["model"]["version"] == SOURCE_MODEL_VERSION

    for path in (
        "config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml",
        "config/preprocessing/config_preprocessing_prediction_patient_v0_2.yaml",
    ):
        assert "./config/preprocessing/shared/aramina_policy_v0_1.yaml" in _text(path)


def test_prediction_bundle_pins_the_same_xrd_release():
    dockerfile = _text("packaging/prediction_api_bundle/Dockerfile")
    bundle_script = _text("packaging/prediction_api_bundle/make_bundle.sh")

    assert f"XRD-preprocessing.git@{XRD_COMMIT}" in dockerfile
    assert f'XRD_RELEASE_TAG="{XRD_RELEASE_TAG}"' in bundle_script
    assert f'XRD_COMMIT="{XRD_COMMIT}"' in bundle_script


def test_reproducible_bundle_passes_xrd_identity_into_docker():
    dockerfile = _text("packaging/reproducible_training_bundle/assets/Dockerfile")
    bundle_script = _text("packaging/reproducible_training_bundle/make_bundle.sh")

    assert "ARG XRD_PREPROCESSING_GIT_COMMIT" in dockerfile
    assert "ARG XRD_PREPROCESSING_REQUESTED_REVISION" in dockerfile
    assert "XRD_PREPROCESSING_GIT_COMMIT=${XRD_PREPROCESSING_GIT_COMMIT}" in dockerfile
    assert "XRD_PREPROCESSING_REQUESTED_REVISION=${XRD_PREPROCESSING_REQUESTED_REVISION}" in dockerfile
    assert 'XRD_PREPROCESSING_GIT_COMMIT=${XRD_COMMIT}' in bundle_script
    assert 'XRD_PREPROCESSING_REQUESTED_REVISION=${XRD_COMMIT}' in bundle_script


def test_operational_artifact_and_examples_are_explicitly_separate_from_source_model():
    readme = _text("README.md")
    bundle_script = _text("packaging/reproducible_training_bundle/make_bundle.sh")

    assert f"source model definition: {SOURCE_MODEL_VERSION}" in readme
    assert f"preserved executable artifacts: {OPERATIONAL_MODEL_VERSION}" in readme
    assert f"No `{SOURCE_MODEL_VERSION}` joblib is tracked" in readme
    assert f'MODEL_VERSION="{OPERATIONAL_MODEL_VERSION}"' in bundle_script


def test_model_and_xrd_versions_use_expected_release_formats():
    model_version = _yaml("config/training/config_training_target_breast_risk_v0_1.yaml")[
        "model"
    ]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+-beta", model_version)
    assert re.fullmatch(r"\d+\.\d+\.\d+b0", XRD_VERSION)
    assert re.fullmatch(r"v\d+\.\d+\.\d+-beta", XRD_RELEASE_TAG)
