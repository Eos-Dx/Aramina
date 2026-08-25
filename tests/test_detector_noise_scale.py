from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aramina.experiments import detector_noise_scale as experiment


ROOT = Path(__file__).resolve().parents[1]


def test_committed_pilot_config_is_bounded_and_valid():
    path = ROOT / "config/experiments/config_detector_noise_scale_pilot_v0_1.yaml"
    config = experiment._load_config(path)

    assert config["targets"]["patient_count"] == 10
    assert config["validation"]["integration_smoke_measurements"] == 10
    assert config["execution"]["workers"] == 4
    assert config["execution"]["score_after_integration"] is True
    assert config["monte_carlo"]["draws"] == 10
    assert config["monte_carlo"]["noise_scales"] == [0.25, 0.5, 1.0, 1.25, 1.5]


def test_committed_full_config_stops_after_four_profile_joblibs():
    path = ROOT / "config/experiments/config_detector_noise_scale_full_v0_1.yaml"
    config = experiment._load_config(path)

    assert config["targets"]["mode"] == "all_training_target_cases"
    assert config["execution"] == {
        "workers": 4,
        "score_after_integration": False,
    }
    assert config["monte_carlo"]["draws"] == 5000
    assert config["monte_carlo"]["convergence_draws"] == [
        100,
        250,
        500,
        1000,
        2000,
        5000,
    ]


def test_noise_scale_config_rejects_duplicate_levels(tmp_path: Path):
    source = ROOT / "config/experiments/config_detector_noise_scale_pilot_v0_1.yaml"
    config = yaml.safe_load(source.read_text())
    config["monte_carlo"]["noise_scales"] = [1.0, 1.0]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="unique"):
        experiment._load_config(path)


def test_balanced_assignments_preserve_every_selected_patient_once():
    dataframe = pd.DataFrame(
        {
            "patientId": ["P1"] * 6 + ["P2"] * 3 + ["P3"] * 5 + ["P4"] * 2,
        }
    )
    cases = pd.DataFrame({"patient_id": ["P1", "P2", "P3", "P4"]})

    assignments = experiment.balanced_patient_assignments(
        dataframe,
        cases,
        worker_count=3,
    )

    flattened = [patient for assignment in assignments for patient in assignment]
    assert sorted(flattened) == ["P1", "P2", "P3", "P4"]
    assert len(flattened) == len(set(flattened))


def test_balanced_case_selection_excludes_cross_class_patient_overlap(monkeypatch):
    cases = pd.DataFrame(
        {
            "target_case_id": ["P1::LEFT", "P1::RIGHT", "P2::LEFT", "P3::LEFT"],
            "patientId": ["P1", "P1", "P2", "P3"],
            "target_side": ["Left", "Right", "Left", "Left"],
            "label": [0, 1, 0, 1],
        }
    )
    monkeypatch.setattr(experiment, "target_breast_cases", lambda *_args, **_kwargs: cases)

    selected = experiment.select_balanced_unique_patient_cases(
        pd.DataFrame(),
        patient_count=2,
    )

    assert selected["patient_id"].nunique() == 2
    assert set(selected["patient_id"]) == {"P1", "P3"}


def test_case_interval_summary_keeps_noise_levels_separate():
    rows = []
    for noise_scale, values in ((0.5, [0.1, 0.2]), (1.0, [0.3, 0.4])):
        for draw_index, value in enumerate(values):
            rows.append(
                {
                    "target_case_id": "P1::left",
                    "patient_id": "P1",
                    "target_side": "left",
                    "target_label": "BENIGN",
                    "label": 0,
                    "noise_scale": noise_scale,
                    "draw_index": draw_index,
                    "p_cancer": value,
                    "decision_threshold": 0.25,
                }
            )

    summary = experiment.summarize_case_intervals(
        pd.DataFrame(rows),
        interval_quantiles=(0.025, 0.5, 0.975),
    )

    assert summary["noise_scale"].tolist() == [0.5, 1.0]
    assert summary["threshold_crossing"].tolist() == [False, False]
    assert np.all(summary["draws"].eq(2))
