from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aramina import __main__ as cli
from aramina.experiments.covariance_uncertainty import LowRankCovarianceModel
from aramina.experiments.measurement_uncertainty import MeasurementUncertaintyError
from aramina.experiments import uncertainty_rank_scan as scan


@pytest.mark.parametrize(
    "name,draws",
    [
        ("config_measurement_uncertainty_rank_scan_pilot_v0_1.yaml", 1000),
        ("config_measurement_uncertainty_rank_scan_v0_1.yaml", 5000),
    ],
)
def test_committed_rank_scan_config_is_valid(name: str, draws: int) -> None:
    root = Path(__file__).resolve().parents[1]
    config = scan.load_rank_scan_config(
        root / "config" / "experiments" / name
    )

    assert [item["rank"] for item in config["rank_scan"]["variants"]] == [
        30,
        50,
        75,
        100,
        100,
    ]
    assert config["monte_carlo"]["draws"] == draws


def test_source_detector_artifacts_round_trip(tmp_path: Path) -> None:
    profile_draws = {
        "first": np.arange(24, dtype=float).reshape(6, 4),
        "second": np.arange(24, 48, dtype=float).reshape(6, 4),
    }
    np.savez_compressed(
        tmp_path / "detector_profile_fit_draws.npz",
        profile_0000=profile_draws["first"],
        profile_0001=profile_draws["second"],
    )
    pd.DataFrame(
        {
            "profile_key": ["first", "second"],
            "npz_key": ["profile_0000", "profile_0001"],
        }
    ).to_csv(tmp_path / "detector_profile_fit_manifest.csv", index=False)
    pd.DataFrame(
        {
            "target_case_id": ["P1::left"],
            "deterministic_p_cancer": [0.4],
            "p_cancer_low": [0.2],
            "p_cancer_high": [0.6],
            "threshold_crossing": [True],
        }
    ).to_csv(
        tmp_path / "detector_measurement_uncertainty_summary.csv", index=False
    )
    (tmp_path / "lineage.json").write_text(
        json.dumps({"data_version": {}, "model": {}}), encoding="utf-8"
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"contract": scan.MEASUREMENT_UNCERTAINTY_CONTRACT}),
        encoding="utf-8",
    )

    source = scan.load_source_detector_artifacts(tmp_path)

    assert set(source.profile_draws) == {"first", "second"}
    np.testing.assert_array_equal(source.profile_draws["first"], profile_draws["first"])
    assert len(source.measurement_manifest) == 2
    assert len(source.checksums) == 5


def test_source_detector_artifacts_reject_missing_array(tmp_path: Path) -> None:
    np.savez_compressed(tmp_path / "detector_profile_fit_draws.npz", other=np.ones((4, 3)))
    pd.DataFrame({"profile_key": ["first"], "npz_key": ["missing"]}).to_csv(
        tmp_path / "detector_profile_fit_manifest.csv", index=False
    )
    pd.DataFrame(
        {
            "target_case_id": ["P1::left"],
            "deterministic_p_cancer": [0.4],
            "p_cancer_low": [0.2],
            "p_cancer_high": [0.6],
            "threshold_crossing": [True],
        }
    ).to_csv(
        tmp_path / "detector_measurement_uncertainty_summary.csv", index=False
    )
    (tmp_path / "lineage.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"contract": scan.MEASUREMENT_UNCERTAINTY_CONTRACT}),
        encoding="utf-8",
    )

    with pytest.raises(MeasurementUncertaintyError, match="missing"):
        scan.load_source_detector_artifacts(tmp_path)


def test_variant_metrics_apply_all_three_gates() -> None:
    model = LowRankCovarianceModel(
        basis=np.eye(2),
        eigenvalues=np.ones(2),
        diagonal=np.zeros(2),
        diagnostics={"selected_rank": 2, "explained_variance_retained": 1.0},
    )
    summaries = pd.DataFrame(
        {
            "p_cancer_low": [0.1, 0.2],
            "p_cancer_high": [0.3, 0.5],
            "threshold_crossing": [False, True],
        }
    )
    comparison = pd.DataFrame(
        {
            "threshold_crossing_agreement": [True, True],
            "covariance_to_detector_width_ratio": [1.0, 1.1],
            "abs_probability_above_threshold_difference": [0.01, 0.02],
        }
    )
    convergence = pd.DataFrame(
        {
            "draws": [1000, 1000, 5000, 5000],
            "abs_delta_low": [np.nan, np.nan, 0.001, 0.002],
            "abs_delta_high": [np.nan, np.nan, 0.002, 0.003],
        }
    )

    metrics = scan.summarize_variant_metrics(
        name="full",
        model=model,
        summaries=summaries,
        comparison=comparison,
        convergence=convergence,
        gates={
            "threshold_crossing_agreement_min": 0.95,
            "interval_width_ratio_min": 0.8,
            "interval_width_ratio_max": 1.25,
            "interval_endpoint_convergence_max": 0.005,
        },
    )

    assert metrics["all_provisional_gates_pass"] is True
    assert metrics["max_interval_endpoint_change"] == pytest.approx(0.003)


def test_rank_scan_cli_reports_result(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "run_uncertainty_rank_scan_from_config",
        lambda *args, **kwargs: {
            "patients_scored": 175,
            "variants": 5,
            "run_folder": tmp_path / "run",
            "metrics_path": tmp_path / "run" / "rank_scan_metrics.csv",
            "mlflow": {"run_id": "run-id", "status": "FINISHED"},
        },
    )

    result = cli.main(
        [
            "experiment-measurement-uncertainty-rank-scan",
            "--config",
            str(tmp_path / "config.yaml"),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "patients_scored=175" in output
    assert "variants=5" in output
