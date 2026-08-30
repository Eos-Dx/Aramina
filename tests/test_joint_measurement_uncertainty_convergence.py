from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aramina.experiments.joint_measurement_uncertainty import (
    Scenario,
    _load_config,
    _monte_carlo_design,
    summarize_nested_axis_changes,
    summarize_nested_axis_convergence,
)


ROOT = Path(__file__).parents[1]


def _case_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_case_id": ["P1::LEFT"],
            "patient_id": ["P1"],
            "target_side": ["left"],
            "label": [1],
            "deterministic_p_cancer": [0.6],
            "decision_threshold": [0.5],
        }
    )


def test_nested_monte_carlo_design_separates_geometry_and_output_counts():
    design = _monte_carlo_design(
        {
            "monte_carlo": {
                "design": "nested_geometry_photon",
                "geometry_draws": 1000,
                "photon_replicates_per_geometry": 5,
            },
            "execution": {"global_stage_geometry_draws": 50},
        }
    )

    assert design.geometry_draws == 1000
    assert design.photon_replicates == 5
    assert design.output_draws == 5000
    assert design.output_stage_draws == 250


def test_nested_experiment_declares_at_least_50_photon_draws_per_geometry():
    config = _load_config(
        ROOT
        / "config"
        / "experiments"
        / "config_joint_measurement_uncertainty_nested_v0_1.yaml"
    )

    assert config["monte_carlo"]["photon_replicates_per_geometry"] == 50
    assert config["convergence"]["minimum_photon_replicates_per_geometry"] == 50
    assert config["convergence"]["photon_prefixes"] == [10, 20, 30, 40, 50]


def test_nested_convergence_keeps_geometry_and_photon_axes_separate():
    probabilities = np.arange(12, dtype=float).reshape(1, 1, 12) / 12.0
    scenarios = (Scenario("joint", True, True, True, True),)

    summary = summarize_nested_axis_convergence(
        probabilities,
        _case_table(),
        scenarios=scenarios,
        quantiles=(0.025, 0.5, 0.975),
        geometry_draws=3,
        photon_replicates=4,
        geometry_prefixes=(1, 2),
        photon_prefixes=(2,),
    )

    assert summary["convergence_axis"].tolist() == [
        "geometry",
        "geometry",
        "geometry",
        "photon",
        "photon",
    ]
    assert summary["axis_prefix"].tolist() == [1, 2, 3, 2, 4]
    assert summary["effective_draws"].tolist() == [4, 8, 12, 6, 12]
    changes = summarize_nested_axis_changes(summary)
    assert len(changes) == 3
    assert set(changes["convergence_axis"]) == {"geometry", "photon"}


def test_nested_convergence_rejects_flat_cube_with_wrong_design_shape():
    with pytest.raises(ValueError, match="nested design"):
        summarize_nested_axis_convergence(
            np.zeros((1, 1, 11)),
            _case_table(),
            scenarios=(Scenario("joint", True, True, True, True),),
            quantiles=(0.025, 0.5, 0.975),
            geometry_draws=3,
            photon_replicates=4,
        )
