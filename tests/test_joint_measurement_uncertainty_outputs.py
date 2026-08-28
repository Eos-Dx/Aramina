from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aramina.experiments.joint_measurement_uncertainty import (
    RESULT_CONTRACT,
    Scenario,
    _load_config,
    convergence_draw_prefixes,
    summarize_case_convergence,
    summarize_case_uncertainty,
    summarize_cohort_convergence,
)


ROOT = Path(__file__).parents[1]


def _case_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_case_id": ["P1::LEFT"],
            "patient_id": ["P1"],
            "target_side": ["left"],
            "label": [1],
            "deterministic_p_cancer": [0.3],
            "decision_threshold": [0.25],
        }
    )


def test_summary_reports_threshold_crossing_and_flip_probability():
    probabilities = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)
    summary = summarize_case_uncertainty(
        probabilities,
        _case_frame(),
        scenarios=(Scenario("joint", True, True, True, True),),
        quantiles=(0.025, 0.5, 0.975),
    ).iloc[0]

    assert bool(summary["threshold_crossing"])
    assert summary["scenario_draw_fraction_at_or_above_threshold"] == pytest.approx(
        0.5
    )
    assert summary["scenario_class_flip_fraction"] == pytest.approx(0.5)
    assert "probability_at_or_above_threshold" not in summary.index
    assert "class_flip_probability" not in summary.index


def test_convergence_prefixes_and_case_cohort_artifacts():
    assert RESULT_CONTRACT == "aramina_joint_measurement_uncertainty_results_v0_3"
    assert convergence_draw_prefixes(10) == (10,)
    assert convergence_draw_prefixes(5000) == tuple(range(250, 5001, 250))
    probabilities = np.linspace(0.0, 1.0, 500, dtype=np.float32).reshape(1, 1, 500)
    convergence = summarize_case_convergence(
        probabilities,
        _case_frame(),
        scenarios=(Scenario("joint", True, True, True, True),),
        quantiles=(0.025, 0.5, 0.975),
    )
    cohort = summarize_cohort_convergence(convergence)

    assert convergence["draw_prefix"].tolist() == [250, 500]
    assert convergence["draws"].tolist() == [250, 500]
    assert {
        "p_cancer_p025",
        "p_cancer_p50",
        "p_cancer_p975",
        "interval_width",
        "scenario_draw_fraction_at_or_above_threshold",
        "scenario_class_flip_fraction",
    }.issubset(convergence.columns)
    assert cohort["draw_prefix"].tolist() == [250, 500]
    assert cohort["target_cases"].tolist() == [1, 1]


def test_pilot_and_full_configs_are_valid():
    for filename in (
        "config_joint_measurement_uncertainty_pilot_v0_1.yaml",
        "config_joint_measurement_uncertainty_full_v0_1.yaml",
    ):
        config = _load_config(ROOT / "config/experiments" / filename)
        assert config["experiment"]["model_version"] == "0.2.15-beta"
