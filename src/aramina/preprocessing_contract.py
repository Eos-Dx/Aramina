"""Fixed Aramina product preprocessing policy layered on XRD YAML validation."""

from __future__ import annotations

from typing import Any


ARAMINA_PREPROCESSING_CONTRACT = "aramina_product_preprocessing_v0_2"
LEGACY_ARAMINA_PREPROCESSING_CONTRACT = "aramina_product_preprocessing_v0_1"
_INTEGRATION_NPT_BY_VERSION = {
    "0.1": 100,
    "0.2": 256,
}
_ROUTES = {
    "aramina_biopsy_patients_model_input": "training",
    "aramina_prediction_patient_model_input": "prediction",
}
_STEP_NAMES = (
    "poni_geometry",
    "select_h5_sessions",
    "h5_to_df",
    "product_columns",
    "position_filter",
    "sample_thickness_filter",
    "calibrant_thickness_filter",
    "biopsy_filter",
    "patient_biopsy_filter",
    "specimen_status_filter",
    "product_status_filter",
    "paired_patient_filter",
    "faulty_pixels",
    "interpolation_q_range",
    "azimuthal_integration",
    "snr",
    "snr_filter",
    "specimen_validity",
    "normalization",
    "profile_gate",
    "keep_columns",
)
_COMMON_OUTPUT_COLUMNS = {
    "patientId",
    "specimenId",
    "side",
    "age",
    "q_range",
    "radial_profile_data",
    "snr_db",
    "specimen_measurement_count",
}


def validate_aramina_preprocessing_config(
    config: dict[str, Any],
    *,
    allow_legacy: bool = False,
) -> None:
    """Reject a resolved product YAML that changes the fixed Aramina data route."""
    product = config.get("aramina_preprocessing")
    if not isinstance(product, dict):
        raise ValueError("Aramina preprocessing config requires aramina_preprocessing.")
    name = _nonempty_string(product, "name", "aramina_preprocessing")
    if name not in _ROUTES:
        raise ValueError(f"Unknown Aramina preprocessing route: {name!r}")
    version = str(_nonempty_scalar(product, "version", "aramina_preprocessing"))
    if version not in _INTEGRATION_NPT_BY_VERSION:
        raise ValueError(
            "Unsupported Aramina preprocessing version: "
            f"{version!r}; expected one of {sorted(_INTEGRATION_NPT_BY_VERSION)}."
        )
    route = _ROUTES[name]
    contract = product.get("contract")
    if version == "0.2":
        _equal(contract, ARAMINA_PREPROCESSING_CONTRACT, "preprocessing contract")
        _equal(product.get("route"), route, "preprocessing product route")
    elif not (
        allow_legacy
        and contract in {None, LEGACY_ARAMINA_PREPROCESSING_CONTRACT}
    ):
        raise ValueError(
            "Legacy Aramina preprocessing config is read-only and requires "
            "allow_legacy=True."
        )
    _nonempty_string(product, "clinical_stage", "aramina_preprocessing")
    _nonempty_string(config.get("io"), "input_h5_path", "io")
    _nonempty_string(config.get("io"), "output_joblib_path", "io")
    _require_common_policy(
        config,
        expected_npt=_INTEGRATION_NPT_BY_VERSION[version],
    )
    _require_pipeline_order(config)
    _require_output_columns(config, route=route)
    if route == "training":
        _require_training_route(config)
    else:
        _require_prediction_route(config)


def validate_if_aramina_product_config(
    config: dict[str, Any],
    *,
    allow_legacy: bool = False,
) -> None:
    """Validate product YAMLs while preserving generic XRD pipeline use in tests."""
    if "aramina_preprocessing" in config:
        validate_aramina_preprocessing_config(config, allow_legacy=allow_legacy)


def _require_common_policy(config: dict[str, Any], *, expected_npt: int) -> None:
    raw_data = _mapping(config, "raw_data")
    if raw_data.get("source") != "gfrm" or raw_data.get("allowed_sources") != ["gfrm"]:
        raise ValueError("Aramina preprocessing requires GFRM as the only raw-data source.")

    filters = _mapping(config, "filters")
    _equal(filters.get("measurement_positions"), ["P1", "P2", "P3"], "measurement positions")
    _equal(filters.get("required_q_max_nm_inv"), 23.0, "required PONI q max")
    _equal(filters.get("require_sample_thickness_mm"), True, "sample thickness requirement")
    _equal(filters.get("require_calibrant_thickness_mm"), True, "calibrant thickness requirement")
    _equal(filters.get("calibrant_thickness_range_mm"), [2.0, 40.0], "calibrant thickness range")

    integration = _mapping(config, "integration")
    _equal(integration.get("npt"), expected_npt, "integration npt")
    _equal(integration.get("q_range_nm_inv"), [2.0, 23.0], "integration q range")
    _equal(integration.get("error_model"), "poisson", "integration error model")

    snr = _mapping(config, "snr")
    _equal(snr.get("method"), "poisson", "SNR method")
    _equal(snr.get("min_snr_db"), 18.0, "SNR threshold")

    normalization = _mapping(config, "normalization")
    _equal(normalization.get("mode"), "value", "normalization mode")
    _equal(normalization.get("statistic"), "median", "normalization statistic")
    _equal(normalization.get("q_range_nm_inv"), [6.7, 7.1], "normalization q range")
    profile_gate = _mapping(config, "profile_gate")
    _equal(profile_gate.get("q_nm_inv"), 14.0, "profile gate q")
    _equal(profile_gate.get("min_value"), 2.0, "profile gate threshold")


def _require_pipeline_order(config: dict[str, Any]) -> None:
    pipeline = _mapping(config, "pipeline")
    steps = pipeline.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Aramina preprocessing requires pipeline.steps.")
    names = [step.get("name") for step in steps if isinstance(step, dict)]
    if names != list(_STEP_NAMES):
        raise ValueError("Aramina preprocessing pipeline steps differ from the approved product order.")


def _require_output_columns(config: dict[str, Any], *, route: str) -> None:
    metadata = _mapping(config, "metadata")
    columns = metadata.get("output_columns")
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise ValueError("Aramina preprocessing metadata.output_columns must be a string list.")
    if len(columns) != len(set(columns)):
        raise ValueError("Aramina preprocessing metadata.output_columns contains duplicates.")
    missing = sorted(_COMMON_OUTPUT_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Aramina preprocessing output columns are missing: {missing}")
    expected_errors = "raise" if route == "training" else "ignore"
    _equal(metadata.get("keep_columns_errors"), expected_errors, "keep-columns error policy")


def _require_training_route(config: dict[str, Any]) -> None:
    filters = _mapping(config, "filters")
    quality = _mapping(filters, "quality_exclusions")
    _equal(quality.get("enabled"), True, "training quality exclusions")
    product_filter = _mapping(config, "product_filter")
    _equal(product_filter.get("require_biopsy_patient"), True, "training biopsy-patient filter")
    _equal(product_filter.get("require_biopsy_rows"), False, "training biopsy-row filter")
    _equal(product_filter.get("product_status_group_keep"), ["BENIGN", "CANCER"], "training labels")
    labels = _mapping(config, "labels")
    builder = _mapping(labels, "product_column_builder")
    if "NORMAL" not in builder.get("benign_values", []):
        raise ValueError("Training preprocessing must map NORMAL to BENIGN.")


def _require_prediction_route(config: dict[str, Any]) -> None:
    filters = _mapping(config, "filters")
    _equal(_mapping(filters, "date_filter").get("enabled"), False, "prediction date filter")
    _equal(_mapping(filters, "quality_exclusions").get("enabled"), False, "prediction quality exclusions")
    product_filter = _mapping(config, "product_filter")
    for key in (
        "filter_by_specimen_status",
        "filter_by_product_status_group",
        "require_biopsy_rows",
        "require_biopsy_patient",
        "require_patient_pair",
    ):
        _equal(product_filter.get(key), False, f"prediction product filter {key}")


def _mapping(value: dict[str, Any] | None, key: str) -> dict[str, Any]:
    child = value.get(key) if isinstance(value, dict) else None
    if not isinstance(child, dict):
        raise ValueError(f"Aramina preprocessing requires mapping {key}.")
    return child


def _nonempty_string(value: dict[str, Any] | None, key: str, where: str) -> str:
    item = value.get(key) if isinstance(value, dict) else None
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Aramina preprocessing requires non-empty {where}.{key}.")
    return item


def _nonempty_scalar(value: dict[str, Any] | None, key: str, where: str) -> Any:
    item = value.get(key) if isinstance(value, dict) else None
    if item is None or item == "" or (isinstance(item, str) and not item.strip()):
        raise ValueError(f"Aramina preprocessing requires non-empty {where}.{key}.")
    if isinstance(item, bool | list | dict):
        raise ValueError(f"Aramina preprocessing requires scalar {where}.{key}.")
    return item


def _equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"Aramina preprocessing requires {name}={expected!r}; got {actual!r}.")
