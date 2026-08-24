from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from aramina.acquisition_contract import (
    ACQUISITION_PROTOCOL_CONTRACT,
    load_acquisition_protocol,
    validate_acquisition_protocol,
)


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples/acquisition/aramina_acquisition_protocol_v0_1.yaml"
SCHEMA = ROOT / "config/acquisition/schema/aramina_acquisition_protocol_v0_1.yaml"


def _example() -> dict:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def test_versioned_acquisition_example_loads_and_schema_documents_variants():
    record, text = load_acquisition_protocol(EXAMPLE)
    schema = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))

    assert record["contract"] == ACQUISITION_PROTOCOL_CONTRACT
    assert record["record"]["geometry"]["point_count_variant"] == 12
    assert len(record["record"]["points"]["standardized_target"]) == 3
    assert len(record["record"]["points"]["standardized_contralateral"]) == 3
    assert len(record["record"]["points"]["lesion_local"]["target"]) == 3
    assert len(record["record"]["points"]["lesion_local"]["mirrored_contralateral"]) == 3
    assert len(record["record"]["dose"]["per_point"]) == 12
    assert "point_count_variants" in schema["record"]["geometry"]
    assert [item["point_count"] for item in schema["record"]["geometry"]["point_count_variants"]] == [6, 9, 12]
    assert "normalized_breast_width" in text


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda config: config["record"]["geometry"]["middle_plane"].update(
                operator_fixed_distance_allowed=True
            ),
            "operator_fixed_distance_allowed",
        ),
        (
            lambda config: config["record"]["geometry"]["middle_plane"].update(
                orientation_reference="operator_selected_axis"
            ),
            "orientation_reference",
        ),
        (
            lambda config: config["record"]["geometry"]["central_point"][
                "point_ids"
            ].update(target="WRONG-CENTRAL-ID"),
            "central_point.point_ids.target",
        ),
        (
            lambda config: config["record"]["geometry"].update(point_count_variant=7),
            "one of 6, 9, or 12",
        ),
        (
            lambda config: config["record"]["geometry"]["middle_plane"].update(alpha=0.0),
            "0 < alpha < 0.5",
        ),
        (
            lambda config: config["record"]["points"]["standardized_target"][2].update(
                normalized_width_fraction=0.71
            ),
            "must match protocol alpha",
        ),
        (
            lambda config: config["record"]["points"]["lesion_local"].pop(
                "mirrored_contralateral"
            ),
            "mirrored_contralateral",
        ),
        (
            lambda config: config["record"]["thickness"].update(
                measured_mm=51.0,
                above_limit={"present": True, "handling": "none_within_preferred_limit", "reason": None},
            ),
            "qualified_review_and_deviation_record",
        ),
        (
            lambda config: config["record"]["dose"]["cumulative"].update(
                delivered_dose_mgy=8.0
            ),
            "within_limit",
        ),
    ],
)
def test_acquisition_validator_rejects_non_reproducible_or_uncontrolled_records(
    mutate, error: str
):
    config = deepcopy(_example())
    mutate(config)

    with pytest.raises((TypeError, ValueError), match=error):
        validate_acquisition_protocol(config)


def test_above_preferred_thickness_requires_review_and_reason():
    config = deepcopy(_example())
    config["record"]["thickness"].update(
        measured_mm=51.0,
        above_limit={
            "present": True,
            "handling": "qualified_review_and_deviation_record",
            "reason": "profile compression exceeded preferred research limit",
        },
    )

    validate_acquisition_protocol(config)


@pytest.mark.parametrize(
    ("variant", "target_local_count", "mirrored_local_count"),
    [(6, 0, 0), (9, 3, 0), (12, 3, 3)],
)
def test_point_count_variants_are_total_patient_points(
    variant: int, target_local_count: int, mirrored_local_count: int
):
    config = deepcopy(_example())
    points = config["record"]["points"]
    points["lesion_local"]["target"] = points["lesion_local"]["target"][:target_local_count]
    points["lesion_local"]["mirrored_contralateral"] = points["lesion_local"][
        "mirrored_contralateral"
    ][:mirrored_local_count]
    config["record"]["geometry"]["point_count_variant"] = variant
    point_ids = {
        point["point_id"]
        for side in ("standardized_target", "standardized_contralateral")
        for point in points[side]
    }
    point_ids.update(
        point["point_id"]
        for side in ("target", "mirrored_contralateral")
        for point in points["lesion_local"][side]
    )
    config["record"]["dose"]["per_point"] = [
        entry
        for entry in config["record"]["dose"]["per_point"]
        if entry["point_id"] in point_ids
    ]
    config["record"]["dose"]["cumulative"].update(
        planned_dose_mgy=float(variant) / 2,
        delivered_dose_mgy=float(variant) / 2 - 0.15,
        maximum_dose_mgy=float(variant) / 2,
    )

    validate_acquisition_protocol(config)
