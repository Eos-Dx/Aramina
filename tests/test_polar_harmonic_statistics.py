from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aramina.experiments import polar_harmonic_statistics as statistics
from aramina.experiments.polar_harmonic_statistics import (
    PolarHarmonicStatisticsConfig,
    PolarHarmonicStatisticsError,
    analyze_polar_harmonic_runs,
)
from aramina.model_metrics import binary_metric_values


def test_paired_analysis_reports_deterministic_contrasts_and_resolution_stability() -> None:
    predictions, metrics = _synthetic_run_tables()
    config = PolarHarmonicStatisticsConfig(seed=19, bootstrap_iterations=120)

    first = analyze_polar_harmonic_runs(predictions, metrics, config=config)
    second = analyze_polar_harmonic_runs(predictions, metrics, config=config)

    assert set(first) == {
        "fingerprints",
        "paired_split_deltas",
        "bootstrap_confidence_intervals",
        "holm_correction",
        "paired_contrasts",
        "chi_resolution_per_split",
        "chi_resolution_summary",
        "direction_consistency",
    }
    assert len(first["fingerprints"]) == 12
    assert set(first["paired_split_deltas"]["contrast"]) == {
        "A0+A2 minus A0",
        "A0+A2+A4 minus A0+A2",
    }
    assert len(first["bootstrap_confidence_intervals"]) == 24
    assert first["holm_correction"]["holm_adjusted_p_value"].between(0.0, 1.0).all()
    assert set(first["chi_resolution_summary"]["n_chi"]) == {12, 18, 72}
    assert first["chi_resolution_per_split"][
        "threshold_class_disagreement"
    ].gt(0.0).any()
    pd.testing.assert_frame_equal(
        first["bootstrap_confidence_intervals"],
        second["bootstrap_confidence_intervals"],
    )


def test_paired_analysis_fails_closed_for_nonmatching_held_out_cases() -> None:
    predictions, metrics = _synthetic_run_tables()
    changed = predictions.copy()
    index = changed.index[
        (changed["n_chi"] == 18)
        & (changed["mode_set"] == "A0")
        & (changed["split_id"] == 0)
    ][0]
    changed.loc[index, "target_case_id"] = "different-case"

    with pytest.raises(
        PolarHarmonicStatisticsError,
        match="Cohort fingerprint mismatch|Held-out case sets differ",
    ):
        analyze_polar_harmonic_runs(
            changed,
            metrics,
            config=PolarHarmonicStatisticsConfig(bootstrap_iterations=100),
        )


def test_paired_analysis_rejects_nonmatching_metric_grid() -> None:
    predictions, metrics = _synthetic_run_tables()
    missing = metrics.iloc[1:].copy()

    with pytest.raises(
        PolarHarmonicStatisticsError,
        match="fold_metrics variant/split grid",
    ):
        analyze_polar_harmonic_runs(
            predictions,
            missing,
            config=PolarHarmonicStatisticsConfig(bootstrap_iterations=100),
        )


def test_bootstrap_metric_helper_calculates_only_required_metrics() -> None:
    result = statistics._safe_binary_metric_values(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.7, 0.4, 0.9]),
        np.full(4, 0.5),
    )
    assert result == {
        "sensitivity": 0.5,
        "specificity": 0.5,
        "roc_auc": 0.75,
    }


def _synthetic_run_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metric_rows = []
    cases = [
        ("case-1", "patient-1", 0),
        ("case-2", "patient-2", 0),
        ("case-3", "patient-3", 1),
        ("case-4", "patient-4", 1),
    ]
    base_scores = {0: [0.18, 0.42, 0.58, 0.82], 1: [0.25, 0.48, 0.52, 0.74]}
    offsets = {"A0": 0.0, "A0+A2": 0.08, "A0+A2+A4": 0.12}
    for n_chi in (12, 18, 36, 72):
        resolution_offset = {12: -0.04, 18: -0.02, 36: 0.0, 72: 0.02}[n_chi]
        for mode_set, mode_offset in offsets.items():
            for split_id in (0, 1):
                split_offset = 0.01 * split_id
                split_rows = []
                for index, (case_id, patient_id, label) in enumerate(cases):
                    score = base_scores[split_id][index] + mode_offset + resolution_offset
                    row = {
                        "n_chi": n_chi,
                        "mode_set": mode_set,
                        "split_id": split_id,
                        "target_case_id": case_id,
                        "patientId": patient_id,
                        "label": label,
                        "p_cancer": score + split_offset,
                        "threshold": 0.55,
                    }
                    rows.append(row)
                    split_rows.append(row)
                labels = np.asarray([row["label"] for row in split_rows])
                scores = np.asarray([row["p_cancer"] for row in split_rows])
                values = binary_metric_values(labels, scores, np.full(len(labels), 0.55))
                metric_rows.append(
                    {
                        "n_chi": n_chi,
                        "mode_set": mode_set,
                        "split_id": split_id,
                        **{metric: values[metric] for metric in ("sensitivity", "specificity", "roc_auc")},
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(metric_rows)
