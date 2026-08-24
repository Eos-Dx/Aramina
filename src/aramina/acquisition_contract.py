"""Strict validation for the versioned Aramina acquisition record."""

from __future__ import annotations

from math import isclose
from pathlib import Path
from typing import Any

import yaml


ACQUISITION_PROTOCOL_CONTRACT = "aramina_acquisition_protocol_v0_1"
ACQUISITION_SCHEMA_VERSION = "0.1"
POINT_COUNT_VARIANTS = {6: (0, 0), 9: (3, 0), 12: (3, 3)}
_SIDES = {"target", "contralateral"}
_POINT_ROLES = {"standardized_middle_plane", "central"}


def load_acquisition_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and validate one Aramina acquisition YAML record."""
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    record = yaml.safe_load(text)
    validate_acquisition_protocol(record)
    return record, text


def validate_acquisition_protocol(config: Any) -> None:
    """Reject records that cannot be traced or reproduced."""
    _exact_keys(config, {"contract", "schema_version", "record"}, "root")
    _equal(config["contract"], ACQUISITION_PROTOCOL_CONTRACT, "contract")
    _equal(config["schema_version"], ACQUISITION_SCHEMA_VERSION, "schema_version")
    record = _mapping(config, "record", "root")
    _exact_keys(
        record,
        {
            "record_id",
            "patient_id",
            "breast_profile",
            "geometry",
            "points",
            "thickness",
            "dose",
            "traceability",
            "timing",
            "longitudinal",
        },
        "record",
    )
    for key in ("record_id", "patient_id"):
        _nonempty_string(record, key, f"record.{key}")
    breast_widths = _validate_breast_profile(record["breast_profile"])
    variant, alpha = _validate_geometry(record["geometry"])
    point_ids = _validate_points(record["points"], variant, alpha, breast_widths)
    _validate_thickness(record["thickness"])
    _validate_dose(record["dose"], point_ids)
    _validate_traceability(record["traceability"])
    _validate_timing(record["timing"])
    _validate_longitudinal(record["longitudinal"])


def _validate_breast_profile(value: Any) -> dict[str, float]:
    profile = value
    _exact_keys(
        profile,
        {"captured", "stored", "storage_ref", "target", "contralateral"},
        "record.breast_profile",
    )
    _equal(profile["captured"], True, "record.breast_profile.captured")
    _equal(profile["stored"], True, "record.breast_profile.stored")
    _nonempty_string(profile, "storage_ref", "record.breast_profile.storage_ref")
    widths: dict[str, float] = {}
    for side in _SIDES:
        breast = _mapping(profile, side, "record.breast_profile")
        _exact_keys(
            breast,
            {"laterality", "profile_id", "width_mm", "thickness_mm"},
            f"record.breast_profile.{side}",
        )
        _nonempty_string(breast, "laterality", f"record.breast_profile.{side}.laterality")
        _nonempty_string(breast, "profile_id", f"record.breast_profile.{side}.profile_id")
        _positive_number(breast["width_mm"], f"record.breast_profile.{side}.width_mm")
        _positive_number(
            breast["thickness_mm"], f"record.breast_profile.{side}.thickness_mm"
        )
        widths[side] = float(breast["width_mm"])
    return widths


def _validate_geometry(value: Any) -> tuple[int, float]:
    geometry = value
    _exact_keys(
        geometry,
        {"coordinate_system", "central_point", "middle_plane", "point_count_variant"},
        "record.geometry",
    )
    _nonempty_string(geometry, "coordinate_system", "record.geometry.coordinate_system")
    central = _mapping(geometry, "central_point", "record.geometry")
    _exact_keys(
        central,
        {"method", "machine_guided", "normalized_width_fraction", "point_ids"},
        "record.geometry.central_point",
    )
    _equal(central["machine_guided"], True, "record.geometry.central_point.machine_guided")
    _equal(
        central["normalized_width_fraction"],
        0.5,
        "record.geometry.central_point.normalized_width_fraction",
    )
    _nonempty_string(central, "method", "record.geometry.central_point.method")
    point_ids = _mapping(central, "point_ids", "record.geometry.central_point")
    _exact_keys(point_ids, _SIDES, "record.geometry.central_point.point_ids")
    for side in _SIDES:
        _nonempty_string(point_ids, side, f"record.geometry.central_point.point_ids.{side}")
    middle_plane = _mapping(geometry, "middle_plane", "record.geometry")
    _exact_keys(
        middle_plane,
        {
            "definition",
            "spacing_rule",
            "width_field",
            "operator_fixed_distance_allowed",
            "alpha",
            "alpha_validation_status",
        },
        "record.geometry.middle_plane",
    )
    _equal(
        middle_plane["spacing_rule"],
        "normalized_breast_width",
        "record.geometry.middle_plane.spacing_rule",
    )
    _equal(
        middle_plane["width_field"],
        "breast_profile.width_mm",
        "record.geometry.middle_plane.width_field",
    )
    _equal(
        middle_plane["operator_fixed_distance_allowed"],
        False,
        "record.geometry.middle_plane.operator_fixed_distance_allowed",
    )
    _nonempty_string(middle_plane, "definition", "record.geometry.middle_plane.definition")
    alpha = middle_plane["alpha"]
    _number(alpha, "record.geometry.middle_plane.alpha")
    if not 0 < alpha < 0.5:
        raise ValueError("record.geometry.middle_plane.alpha must satisfy 0 < alpha < 0.5.")
    _equal(
        middle_plane["alpha_validation_status"],
        "pending_hardware_validation",
        "record.geometry.middle_plane.alpha_validation_status",
    )
    variant = geometry["point_count_variant"]
    if isinstance(variant, bool) or not isinstance(variant, int):
        raise TypeError("record.geometry.point_count_variant must be an integer.")
    if variant not in POINT_COUNT_VARIANTS:
        raise ValueError(
            "record.geometry.point_count_variant must be one of 6, 9, or 12."
        )
    return variant, float(alpha)


def _validate_points(
    value: Any, variant: int, alpha: float, breast_widths: dict[str, float]
) -> set[str]:
    points = value
    _exact_keys(
        points,
        {"standardized_target", "standardized_contralateral", "lesion_local"},
        "record.points",
    )
    ids: set[str] = set()
    for side in _SIDES:
        entries = points[f"standardized_{side}"]
        if not isinstance(entries, list) or len(entries) != 3:
            raise ValueError(
                f"record.points.standardized_{side} must contain exactly 3 points."
            )
        central_count = 0
        expected_fractions = (0.5 - alpha, 0.5, 0.5 + alpha)
        for point, expected_fraction in zip(entries, expected_fractions, strict=True):
            point_id = _validate_point(point, side, "standardized_middle_plane")
            if point_id in ids:
                raise ValueError(f"Duplicate acquisition point_id: {point_id!r}.")
            ids.add(point_id)
            if point["role"] == "central":
                central_count += 1
                _equal(point["normalized_width_fraction"], 0.5, f"{point_id}.fraction")
            if not isclose(
                point["normalized_width_fraction"], expected_fraction, rel_tol=0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"{point_id}.fraction must match protocol alpha={alpha}"
                )
            expected_x = (expected_fraction - 0.5) * breast_widths[side]
            actual_x = point["coordinates_mm"]["x"]
            if not isclose(actual_x, expected_x, rel_tol=0, abs_tol=1e-6):
                raise ValueError(
                    f"{point_id}.coordinates_mm.x must equal alpha-scaled breast-width offset."
                )
        if central_count != 1:
            raise ValueError(f"standardized_{side} requires exactly one central point.")

    local = _mapping(points, "lesion_local", "record.points")
    _exact_keys(local, {"target", "mirrored_contralateral"}, "record.points.lesion_local")
    target_points = local["target"]
    mirrored_points = local["mirrored_contralateral"]
    expected_target, expected_mirrored = POINT_COUNT_VARIANTS[variant]
    if not isinstance(target_points, list) or len(target_points) != expected_target:
        raise ValueError(
            f"record.points.lesion_local.target must contain exactly {expected_target} points for total variant {variant}."
        )
    if not isinstance(mirrored_points, list) or len(mirrored_points) != expected_mirrored:
        raise ValueError(
            f"record.points.lesion_local.mirrored_contralateral must contain exactly {expected_mirrored} points for total variant {variant}."
        )
    target_ids = [
        _validate_point(point, "target", "lesion_local_target") for point in target_points
    ]
    mirrored_ids = [
        _validate_point(point, "contralateral", "lesion_local_mirrored_contralateral")
        for point in mirrored_points
    ]
    local_ids = target_ids + mirrored_ids
    if len(set(local_ids)) != len(local_ids) or ids.intersection(local_ids):
        raise ValueError("Lesion-local point_id values must be unique from grid points and each other.")
    if mirrored_ids and len(target_ids) != len(mirrored_ids):
        raise ValueError("Target and mirrored lesion-local arrays must have equal length.")
    for target_point, mirrored_point in zip(target_points, mirrored_points, strict=False):
        _equal(
            target_point["mirror_pair_id"],
            mirrored_point["mirror_pair_id"],
            "record.points.lesion_local mirror_pair_id",
        )
    ids.update(local_ids)
    if len(ids) != variant:
        raise ValueError(
            f"Selected point-count variant {variant} requires exactly {variant} total patient points; got {len(ids)}."
        )
    return ids


def _validate_point(value: Any, side: str, role: str) -> str:
    point = value
    required = {
        "point_id",
        "side",
        "role",
        "normalized_width_fraction",
        "coordinates_mm",
        "xrd",
        "attenuation_coefficient",
    }
    if role.startswith("lesion_local"):
        required.add("mirror_pair_id")
    _exact_keys(point, required, f"point {side}")
    _equal(point["side"], side, f"{point['point_id']}.side")
    if role == "standardized_middle_plane":
        if point["role"] not in _POINT_ROLES:
            raise ValueError(f"{point['point_id']}.role must be standardized_middle_plane or central.")
    else:
        _equal(point["role"], role, f"{point['point_id']}.role")
    _nonempty_string(point, "point_id", f"point {side}.point_id")
    fraction = point["normalized_width_fraction"]
    if not isinstance(fraction, int | float) or isinstance(fraction, bool) or not 0 <= fraction <= 1:
        raise ValueError(f"{point['point_id']}.normalized_width_fraction must be between 0 and 1.")
    coordinates = _mapping(point, "coordinates_mm", point["point_id"])
    _exact_keys(coordinates, {"x", "y", "z"}, f"{point['point_id']}.coordinates_mm")
    for axis in ("x", "y", "z"):
        _number(coordinates[axis], f"{point['point_id']}.coordinates_mm.{axis}")
    _validate_measurements(point, point["point_id"])
    if role.startswith("lesion_local"):
        _nonempty_string(point, "mirror_pair_id", f"{point['point_id']}.mirror_pair_id")
    return point["point_id"]


def _validate_measurements(point: dict[str, Any], point_id: str) -> None:
    xrd = _mapping(point, "xrd", point_id)
    _exact_keys(xrd, {"acquired", "measurement_id", "raw_data_ref"}, f"{point_id}.xrd")
    _equal(xrd["acquired"], True, f"{point_id}.xrd.acquired")
    _nonempty_string(xrd, "measurement_id", f"{point_id}.xrd.measurement_id")
    _nonempty_string(xrd, "raw_data_ref", f"{point_id}.xrd.raw_data_ref")
    attenuation = _mapping(point, "attenuation_coefficient", point_id)
    _exact_keys(
        attenuation,
        {"value_cm_inv", "uncertainty_cm_inv", "unit", "method", "source"},
        f"{point_id}.attenuation_coefficient",
    )
    _positive_number(attenuation["value_cm_inv"], f"{point_id}.attenuation_coefficient.value_cm_inv")
    _number(attenuation["uncertainty_cm_inv"], f"{point_id}.attenuation_coefficient.uncertainty_cm_inv")
    if attenuation["uncertainty_cm_inv"] < 0:
        raise ValueError(f"{point_id}.attenuation_coefficient uncertainty must be non-negative.")
    _equal(attenuation["unit"], "cm^-1", f"{point_id}.attenuation_coefficient.unit")
    for key in ("method", "source"):
        _nonempty_string(attenuation, key, f"{point_id}.attenuation_coefficient.{key}")


def _validate_thickness(value: Any) -> None:
    thickness = value
    _exact_keys(thickness, {"preferred_max_mm", "measured_mm", "above_limit"}, "record.thickness")
    _equal(thickness["preferred_max_mm"], 50.0, "record.thickness.preferred_max_mm")
    _positive_number(thickness["measured_mm"], "record.thickness.measured_mm")
    above = _mapping(thickness, "above_limit", "record.thickness")
    _exact_keys(above, {"present", "handling", "reason"}, "record.thickness.above_limit")
    is_above = thickness["measured_mm"] > thickness["preferred_max_mm"]
    _equal(above["present"], is_above, "record.thickness.above_limit.present")
    if is_above:
        _equal(above["handling"], "qualified_review_and_deviation_record", "record.thickness.above_limit.handling")
        _nonempty_string(above, "reason", "record.thickness.above_limit.reason")
    elif above["handling"] != "none_within_preferred_limit" or above["reason"] is not None:
        raise ValueError("Within-limit thickness requires no above-limit handling or reason.")


def _validate_dose(value: Any, point_ids: set[str]) -> None:
    dose = value
    _exact_keys(dose, {"per_point", "cumulative", "control"}, "record.dose")
    per_point = dose["per_point"]
    if not isinstance(per_point, list) or {entry.get("point_id") for entry in per_point} != point_ids:
        raise ValueError("record.dose.per_point must contain exactly one entry for every acquisition point.")
    for entry in per_point:
        _exact_keys(
            entry,
            {"point_id", "planned_dose_mgy", "delivered_dose_mgy", "maximum_dose_mgy", "within_limit"},
            "record.dose.per_point entry",
        )
        _nonempty_string(entry, "point_id", "record.dose.per_point.point_id")
        for key in ("planned_dose_mgy", "delivered_dose_mgy", "maximum_dose_mgy"):
            _positive_number(entry[key], f"record.dose.per_point.{key}")
        _equal(
            entry["within_limit"],
            entry["delivered_dose_mgy"] <= entry["maximum_dose_mgy"],
            f"record.dose.per_point.{entry['point_id']}.within_limit",
        )
    cumulative = _mapping(dose, "cumulative", "record.dose")
    _exact_keys(
        cumulative,
        {"planned_dose_mgy", "delivered_dose_mgy", "maximum_dose_mgy", "within_limit"},
        "record.dose.cumulative",
    )
    for key in ("planned_dose_mgy", "delivered_dose_mgy", "maximum_dose_mgy"):
        _positive_number(cumulative[key], f"record.dose.cumulative.{key}")
    _equal(
        cumulative["within_limit"],
        cumulative["delivered_dose_mgy"] <= cumulative["maximum_dose_mgy"],
        "record.dose.cumulative.within_limit",
    )
    control = _mapping(dose, "control", "record.dose")
    _exact_keys(control, {"per_point_limit_enforced", "cumulative_limit_enforced", "stop_on_exceedance"}, "record.dose.control")
    for key in ("per_point_limit_enforced", "cumulative_limit_enforced", "stop_on_exceedance"):
        _equal(control[key], True, f"record.dose.control.{key}")


def _validate_traceability(value: Any) -> None:
    traceability = value
    _exact_keys(traceability, {"operator", "hardware", "session"}, "record.traceability")
    for section, keys in {
        "operator": {"operator_id", "role"},
        "hardware": {"instrument_id", "model", "serial_number", "firmware_version", "calibration_id"},
        "session": {"session_id", "started_at", "ended_at", "software_version", "protocol_version"},
    }.items():
        child = _mapping(traceability, section, "record.traceability")
        _exact_keys(child, keys, f"record.traceability.{section}")
        for key in keys:
            _nonempty_string(child, key, f"record.traceability.{section}.{key}")
    _equal(
        traceability["session"]["protocol_version"],
        ACQUISITION_PROTOCOL_CONTRACT,
        "record.traceability.session.protocol_version",
    )


def _validate_timing(value: Any) -> None:
    timing = value
    _exact_keys(
        timing,
        {"visit_type", "acquired_at", "biopsy_at", "hours_before_biopsy", "timing_deviation"},
        "record.timing",
    )
    _equal(timing["visit_type"], "baseline_pre_biopsy", "record.timing.visit_type")
    for key in ("acquired_at", "biopsy_at"):
        _nonempty_string(timing, key, f"record.timing.{key}")
    _number(timing["hours_before_biopsy"], "record.timing.hours_before_biopsy")
    if timing["hours_before_biopsy"] < 0:
        raise ValueError("record.timing.hours_before_biopsy must be non-negative.")
    deviation = _mapping(timing, "timing_deviation", "record.timing")
    _exact_keys(deviation, {"present", "reason"}, "record.timing.timing_deviation")
    if deviation["present"]:
        _nonempty_string(deviation, "reason", "record.timing.timing_deviation.reason")
    elif deviation["reason"] is not None:
        raise ValueError("Absent timing deviation requires a null reason.")


def _validate_longitudinal(value: Any) -> None:
    longitudinal = value
    _exact_keys(
        longitudinal,
        {"optional_future_visits", "baseline_endpoint_uses_only_pre_biopsy", "visits"},
        "record.longitudinal",
    )
    _equal(longitudinal["optional_future_visits"], True, "record.longitudinal.optional_future_visits")
    _equal(
        longitudinal["baseline_endpoint_uses_only_pre_biopsy"],
        True,
        "record.longitudinal.baseline_endpoint_uses_only_pre_biopsy",
    )
    if not isinstance(longitudinal["visits"], list):
        raise TypeError("record.longitudinal.visits must be a list.")


def _mapping(value: Any, key: str, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get(key), dict):
        raise TypeError(f"{where}.{key} must be a mapping.")
    return value[key]


def _exact_keys(value: Any, required: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a mapping.")
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"Missing {where} fields: {missing}")
    unknown = sorted(set(value).difference(required))
    if unknown:
        raise ValueError(f"Unknown {where} fields: {unknown}")


def _nonempty_string(section: dict[str, Any], key: str, where: str) -> None:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string.")


def _number(value: Any, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{where} must be numeric.")


def _positive_number(value: Any, where: str) -> None:
    _number(value, where)
    if value <= 0:
        raise ValueError(f"{where} must be greater than zero.")


def _equal(actual: Any, expected: Any, where: str) -> None:
    if actual != expected:
        raise ValueError(f"{where} must be {expected!r}; got {actual!r}.")
