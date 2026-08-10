"""Configuration contract for the research-only FPCA256 experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONTRACT = "aramina_fpca256_profile_encoder_experiment_v0_1"
REQUIRED_COMPONENTS = [4, 5, 6, 7]
EXPECTED_INTEGRATION_METHOD = ["bbox", "csr", "cython"]
EXPECTED_PYFAI_VERSION = "2026.5.0"
EXPECTED_INTEGRATION_METHOD_SOURCE = "pyfai_integrate1d_default"
EXPECTED_BASE_MAIN_GIT_SHA = "394a34640441d33ebb994cd93107ac4447707461"
EXPECTED_SOURCE_H5_SHA256 = (
    "d2d61e83850b282c3d2479ea436deed821c4488b96983252d294f3d56ee3f1f9"
)
EXPECTED_BASE_PREPROCESSING_CONFIG_SHA256 = (
    "2913d1dfada1596bc12afec69ddb69c217af577f4cae10243931760900d01d3b"
)
EXPECTED_ARTIFACT_LINEAGE = {
    "common_npt100": {
        "sha256": "1e33f4a5447993c40d4496dedc955a884cf8738447bb39d2b11eae0163bd4eff",
        "pipeline_fingerprint": (
            "fc0a58dd54a85348985b8e16201a65b9152baed22e3a6be65ecaeaffaf47d95d"
        ),
    },
    "common_npt256": {
        "sha256": "195d3150dedc3847ca967abbbd37d49b89a4dd68a0aab13b3e9c2f20f5e739ae",
        "pipeline_fingerprint": (
            "9436335af89fc7c9457ca614e5e4f17797de23c7cf14d771f4b20e253728968e"
        ),
    },
    "full_npt256": {
        "sha256": "2eb7b46c6f42d284d996bd1d9b6d0a3d8df3ed372fb7fe0b9346bcbd87479504",
        "pipeline_fingerprint": (
            "9436335af89fc7c9457ca614e5e4f17797de23c7cf14d771f4b20e253728968e"
        ),
    },
}
CONTROLLED_MODEL_VALUES = {
    "profile_column": "radial_profile_data",
    "label_column": "product_status_group",
    "group_column": "patientId",
    "specimen_column": "specimenId",
    "side_column": "side",
    "q_column": "q_range",
    "age_column": "age",
    "biopsy_column": "biopsy",
    "lr1_row_policy": "biopsy_only",
    "lr1_logreg_c": 0.1,
    "lr2_logreg_c": 0.3,
    "class_weight": "balanced",
    "profile_aggregation": "logit_average",
    "lr2_architecture": "age_plus_gated_sk_core4",
}


def load_experiment_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and validate one experiment YAML."""
    source = Path(path).expanduser().resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    validate_experiment_config(config)
    return config, source


def validate_experiment_config(config: Any) -> None:
    """Reject missing fields and controlled-variable changes."""
    if not isinstance(config, dict):
        raise TypeError("Experiment config must be a mapping.")
    _exact_keys(
        config,
        required={
            "contract",
            "clinical_stage",
            "lineage",
            "cohorts",
            "preprocessing",
            "model",
            "evaluation",
            "output",
        },
        where="experiment config",
    )
    if config["contract"] != CONTRACT:
        raise ValueError(f"Unsupported experiment contract: {config['contract']!r}")
    if config["clinical_stage"] != "research_only":
        raise ValueError("clinical_stage must be 'research_only'.")

    _validate_lineage(config["lineage"])
    _validate_cohorts(config["cohorts"])
    _validate_preprocessing(config["preprocessing"])
    _validate_model(config["model"])
    _validate_evaluation(config["evaluation"])
    _exact_keys(config["output"], required={"folder"}, where="output")
    _nonempty_string(config["output"]["folder"], "output.folder")


def resolve_path(value: str | Path, source: Path) -> Path:
    """Resolve experiment paths relative to the YAML file."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def _validate_cohorts(cohorts: Any) -> None:
    _exact_keys(cohorts, required={"common", "full_npt256"}, where="cohorts")
    common = cohorts["common"]
    _exact_keys(
        common,
        required={
            "enabled",
            "npt100_artifact",
            "npt256_artifact",
            "expected_rows",
            "expected_patients",
            "expected_target_cases",
        },
        where="cohorts.common",
    )
    full = cohorts["full_npt256"]
    _exact_keys(
        full,
        required={
            "enabled",
            "npt256_artifact",
            "expected_rows",
            "expected_patients",
            "expected_target_cases",
        },
        where="cohorts.full_npt256",
    )
    for name, section in (("common", common), ("full_npt256", full)):
        if not isinstance(section["enabled"], bool):
            raise TypeError(f"cohorts.{name}.enabled must be boolean.")
        for key in ("expected_rows", "expected_patients", "expected_target_cases"):
            _positive_int(section[key], f"cohorts.{name}.{key}")
    _validate_artifact_pin(
        common["npt100_artifact"],
        where="cohorts.common.npt100_artifact",
        expected_npt=100,
        expected_variant="npt100_bbox",
        expected_lineage=EXPECTED_ARTIFACT_LINEAGE["common_npt100"],
    )
    _validate_artifact_pin(
        common["npt256_artifact"],
        where="cohorts.common.npt256_artifact",
        expected_npt=256,
        expected_variant="npt256_bbox",
        expected_lineage=EXPECTED_ARTIFACT_LINEAGE["common_npt256"],
    )
    _validate_artifact_pin(
        full["npt256_artifact"],
        where="cohorts.full_npt256.npt256_artifact",
        expected_npt=256,
        expected_variant="npt256_bbox",
        expected_lineage=EXPECTED_ARTIFACT_LINEAGE["full_npt256"],
    )


def _validate_lineage(lineage: Any) -> None:
    _exact_keys(
        lineage,
        required={
            "aramina_base_main_git_sha",
            "source_h5_sha256",
            "base_preprocessing_config_sha256",
            "integration_method",
            "pyfai_version",
            "integration_method_source",
        },
        where="lineage",
    )
    _git_sha_string(
        lineage["aramina_base_main_git_sha"],
        "lineage.aramina_base_main_git_sha",
    )
    for key in ("source_h5_sha256", "base_preprocessing_config_sha256"):
        _sha256_string(lineage[key], f"lineage.{key}")
    expected_values = {
        "aramina_base_main_git_sha": EXPECTED_BASE_MAIN_GIT_SHA,
        "source_h5_sha256": EXPECTED_SOURCE_H5_SHA256,
        "base_preprocessing_config_sha256": (
            EXPECTED_BASE_PREPROCESSING_CONFIG_SHA256
        ),
    }
    for key, expected in expected_values.items():
        if lineage[key] != expected:
            raise ValueError(f"lineage.{key} must be pinned to {expected}.")
    if lineage["integration_method"] != EXPECTED_INTEGRATION_METHOD:
        raise ValueError(
            f"lineage.integration_method must be {EXPECTED_INTEGRATION_METHOD}."
        )
    if lineage["pyfai_version"] != EXPECTED_PYFAI_VERSION:
        raise ValueError(
            f"lineage.pyfai_version must be pinned to {EXPECTED_PYFAI_VERSION}."
        )
    if (
        lineage["integration_method_source"]
        != EXPECTED_INTEGRATION_METHOD_SOURCE
    ):
        raise ValueError(
            "lineage.integration_method_source must be pinned to "
            f"{EXPECTED_INTEGRATION_METHOD_SOURCE}."
        )


def _validate_artifact_pin(
    artifact: Any,
    *,
    where: str,
    expected_npt: int,
    expected_variant: str,
    expected_lineage: dict[str, str],
) -> None:
    _exact_keys(
        artifact,
        required={
            "path",
            "sha256",
            "pipeline_fingerprint",
            "input_h5_sha256",
            "integration_variant",
            "integration_npt",
            "integration_method",
        },
        where=where,
    )
    _nonempty_string(artifact["path"], f"{where}.path")
    for key in ("sha256", "pipeline_fingerprint", "input_h5_sha256"):
        _sha256_string(artifact[key], f"{where}.{key}")
    for key, expected in expected_lineage.items():
        if artifact[key] != expected:
            raise ValueError(f"{where}.{key} must be pinned to {expected}.")
    if artifact["input_h5_sha256"] != EXPECTED_SOURCE_H5_SHA256:
        raise ValueError(
            f"{where}.input_h5_sha256 must be pinned to "
            f"{EXPECTED_SOURCE_H5_SHA256}."
        )
    if artifact["integration_variant"] != expected_variant:
        raise ValueError(f"{where}.integration_variant must be {expected_variant!r}.")
    if artifact["integration_npt"] != expected_npt:
        raise ValueError(f"{where}.integration_npt must be {expected_npt}.")
    if artifact["integration_method"] != EXPECTED_INTEGRATION_METHOD:
        raise ValueError(
            f"{where}.integration_method must be {EXPECTED_INTEGRATION_METHOD}."
        )


def _validate_preprocessing(preprocessing: Any) -> None:
    _exact_keys(
        preprocessing,
        required={
            "base_config_path",
            "generated_npt256_artifact_path",
            "integration_npt",
        },
        where="preprocessing",
    )
    _nonempty_string(preprocessing["base_config_path"], "preprocessing base path")
    _nonempty_string(
        preprocessing["generated_npt256_artifact_path"],
        "preprocessing generated artifact path",
    )
    if preprocessing["integration_npt"] != 256:
        raise ValueError("Research preprocessing integration_npt must be 256.")


def _validate_model(model: Any) -> None:
    required = {*CONTROLLED_MODEL_VALUES, "fpca_components", "raw_baselines"}
    _exact_keys(model, required=required, where="model")
    for key, expected in CONTROLLED_MODEL_VALUES.items():
        if model[key] != expected:
            raise ValueError(
                f"Controlled model field {key!r} must be {expected!r}; "
                f"received {model[key]!r}."
            )
    if model["fpca_components"] != REQUIRED_COMPONENTS:
        raise ValueError(f"model.fpca_components must be {REQUIRED_COMPONENTS}.")
    if model["raw_baselines"] != [100, 256]:
        raise ValueError("model.raw_baselines must be [100, 256].")


def _validate_evaluation(evaluation: Any) -> None:
    _exact_keys(
        evaluation,
        required={
            "method",
            "folds",
            "repeats",
            "random_seed",
            "target_sensitivity",
        },
        where="evaluation",
    )
    if evaluation["method"] != "repeated_stratified_kfold":
        raise ValueError("evaluation.method must be repeated_stratified_kfold.")
    _int_at_least(evaluation["folds"], 2, "evaluation.folds")
    _int_at_least(evaluation["repeats"], 1, "evaluation.repeats")
    _int_at_least(evaluation["random_seed"], 0, "evaluation.random_seed")
    if float(evaluation["target_sensitivity"]) != 0.95:
        raise ValueError("evaluation.target_sensitivity must be 0.95.")


def _exact_keys(value: Any, *, required: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a mapping.")
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"Missing {where} fields: {missing}")
    unknown = sorted(set(value).difference(required))
    if unknown:
        raise ValueError(f"Unknown {where} fields: {unknown}")


def _nonempty_string(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string.")


def _sha256_string(value: Any, where: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256 hex digest.")


def _git_sha_string(value: Any, where: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a full lowercase Git SHA.")


def _positive_int(value: Any, where: str) -> None:
    _int_at_least(value, 1, where)


def _int_at_least(value: Any, minimum: int, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}.")
