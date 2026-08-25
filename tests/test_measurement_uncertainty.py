from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from aramina.experiments import measurement_uncertainty as uncertainty
from aramina.runtime_identity import file_hashes

from .prediction_fixtures import patient_frame, train_model


def _profile_sigma_frame() -> pd.DataFrame:
    frame = patient_frame()
    raw_profiles = []
    normalized_profiles = []
    sigmas = []
    for profile, q in zip(frame["radial_profile_data"], frame["q_range"], strict=True):
        raw = 6.0 + np.asarray(profile, dtype=float)
        q_values = np.asarray(q, dtype=float)
        scale = float(np.median(raw[(q_values >= 6.7) & (q_values <= 7.1)]))
        raw_profiles.append(raw)
        normalized_profiles.append(raw / scale)
        sigmas.append(np.full(raw.shape, 0.02, dtype=float))
    frame["radial_profile_data"] = normalized_profiles
    frame[uncertainty.RAW_PROFILE_COLUMN] = raw_profiles
    frame[uncertainty.PROFILE_SIGMA_COLUMN] = sigmas
    frame["snr_db"] = np.linspace(18.1, 38.7, len(frame))
    return frame


def _model_artifact(tmp_path_factory) -> dict:
    model_path, _ = train_model(tmp_path_factory)
    return joblib.load(model_path)


@pytest.fixture(scope="module")
def model_artifact(tmp_path_factory) -> dict:
    return _model_artifact(tmp_path_factory)


def test_profile_sigma_scores_are_seeded_and_include_all_patient_measurements(
    model_artifact: dict,
):
    frame = _profile_sigma_frame()
    target = uncertainty.TargetRequest(patient_id="P00", target_side="left")

    first_summary, first_draws = (
        uncertainty.score_profile_sigma_measurement_uncertainty(
            frame,
            model_artifact=model_artifact,
            targets=[target],
            draws=30,
            seed=47,
        )
    )
    second_summary, second_draws = (
        uncertainty.score_profile_sigma_measurement_uncertainty(
            frame,
            model_artifact=model_artifact,
            targets=[target],
            draws=30,
            seed=47,
        )
    )

    pd.testing.assert_frame_equal(first_summary, second_summary)
    pd.testing.assert_frame_equal(first_draws, second_draws)
    summary = first_summary.iloc[0]
    assert summary["measurement_count"] == 6
    assert summary["target_measurements"] == 3
    assert summary["contralateral_measurements"] == 3
    assert len(first_draws) == 30
    assert 0.0 <= summary["probability_above_threshold"] <= 1.0
    assert (
        summary["p_cancer_low"]
        <= summary["p_cancer_median"]
        <= summary["p_cancer_high"]
    )
    assert "pyfai_poisson_per_bin_sigma" in summary["included_uncertainty_sources"]


def test_profile_sigma_fails_closed_for_missing_or_invalid_sigma(model_artifact: dict):
    frame = _profile_sigma_frame().drop(columns=uncertainty.PROFILE_SIGMA_COLUMN)

    with pytest.raises(
        uncertainty.MeasurementUncertaintyError, match="uncertainty columns"
    ):
        uncertainty.score_profile_sigma_measurement_uncertainty(
            frame,
            model_artifact=model_artifact,
            targets=[uncertainty.TargetRequest(patient_id="P00", target_side="left")],
            draws=30,
            seed=47,
        )

    frame = _profile_sigma_frame()
    frame.at[0, uncertainty.PROFILE_SIGMA_COLUMN] = np.zeros(100)
    with pytest.raises(uncertainty.MeasurementUncertaintyError, match="invalid pyFAI"):
        uncertainty.score_profile_sigma_measurement_uncertainty(
            frame,
            model_artifact=model_artifact,
            targets=[uncertainty.TargetRequest(patient_id="P00", target_side="left")],
            draws=30,
            seed=47,
        )


def test_detector_reference_subset_uses_historical_biopsy_targets_and_snr_strata():
    frame = _profile_sigma_frame()
    targets = uncertainty._targets_for_run(
        frame,
        {"targets": {"mode": "all_training_target_cases", "selected": []}},
    )
    subset = uncertainty.select_detector_reference_subset(
        frame,
        targets=targets,
        quantiles=4,
        max_cases_per_stratum=1,
    )

    assert set(subset["target_label"]) == {"BENIGN", "CANCER"}
    assert subset["target_snr_quantile"].between(1, 4).all()
    assert set(subset["calibration_nuisance_included"]) == {False}
    assert set(subset["calibration_nuisance_future_scope"]) == {
        "shared_by_calib_session"
    }
    assert subset["target_case_id"].str.endswith("::left").all()


def test_profile_detector_comparison_preserves_score_and_compares_intervals():
    profile = pd.DataFrame(
        {
            "target_case_id": ["P00::left"],
            "deterministic_p_cancer": [0.3],
            "p_cancer_low": [0.1],
            "p_cancer_high": [0.5],
            "p_cancer_sd": [0.1],
            "probability_above_threshold": [0.7],
            "threshold_crossing": [True],
        }
    )
    detector = profile.copy()
    detector[["p_cancer_low", "p_cancer_high"]] = [0.2, 0.4]
    detector["threshold_crossing"] = False

    comparison = uncertainty.compare_profile_detector_reference(profile, detector)

    assert comparison["profile_to_detector_width_ratio"].iloc[0] == pytest.approx(2.0)
    assert comparison["threshold_crossing_agreement"].tolist() == [False]


def test_profile_convergence_uses_nested_seeded_draw_stream():
    draws = pd.DataFrame(
        {
            "target_case_id": ["P00::left"] * 10,
            "draw_index": np.arange(10),
            "p_cancer": np.linspace(0.1, 1.0, 10),
            "above_threshold": [False, False, True, True, True] * 2,
        }
    )

    convergence = uncertainty.summarize_profile_monte_carlo_convergence(
        draws,
        checkpoints=(5, 10),
        interval_quantiles=(0.025, 0.5, 0.975),
    )

    assert convergence["draws"].tolist() == [5, 10]
    assert np.isnan(convergence["abs_delta_low"].iloc[0])
    assert convergence["abs_delta_low"].iloc[1] > 0.0


def test_full_experiment_logs_dvc_model_lineage_and_mlflow(
    tmp_path: Path,
    model_artifact: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    input_h5 = tmp_path / "input.h5"
    input_h5.write_bytes(b"dvc-verified-test-h5")
    hashes = file_hashes(input_h5, algorithms=("sha256", "md5"))
    pointer = tmp_path / "input.h5.dvc"
    pointer.write_text(
        yaml.safe_dump(
            {
                "outs": [
                    {
                        "md5": hashes["md5"],
                        "size": input_h5.stat().st_size,
                        "hash": "md5",
                        "path": input_h5.name,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data_version = {
        "contract": "aramina_dvc_input_v0_1",
        "system": "dvc",
        "dataset_id": "synthetic_measurement_uncertainty",
        "dvc_version": "3.67.1",
        "pointer_path": pointer.name,
        "output_path": input_h5.name,
        "hash_algorithm": "md5",
        "hash": hashes["md5"],
        "size_bytes": input_h5.stat().st_size,
        "input_h5_sha256": hashes["sha256"],
    }
    artifact = deepcopy(model_artifact)
    artifact["model_identity"] = {
        "name": "aramina_target_breast_risk",
        "version": "0.2.14-beta",
    }
    artifact["reproducibility"] = {
        "source_h5": {"data_version": data_version},
        "source_code": {
            "aramina": {"git_sha": "a" * 40},
            "xrd_preprocessing": {"git_commit": "b" * 40},
        },
    }
    artifact["historical_preprocessing_yaml"] = yaml.safe_dump(
        {
            "io": {"input_h5_path": "placeholder", "output_joblib_path": "placeholder"},
            "data_version": {"pointer_path": pointer.name},
            "normalization": {"save_initial_data": False},
            "metadata": {"output_columns": []},
            "pipeline": {"steps": [{"name": "normalization", "params": {}}]},
        },
        sort_keys=False,
    )
    model_path = tmp_path / "frozen.joblib"
    joblib.dump(artifact, model_path)
    config = {
        "contract": uncertainty.MEASUREMENT_UNCERTAINTY_CONTRACT,
        "experiment": {
            "name": "synthetic_measurement_uncertainty",
            "model_name": "aramina_target_breast_risk",
            "model_version": "0.2.14-beta",
        },
        "input": {
            "input_h5_path": str(input_h5),
            "model_joblib_path": str(model_path),
        },
        "data_version": {
            key: data_version[key]
            for key in (
                "contract",
                "system",
                "dataset_id",
                "dvc_version",
                "pointer_path",
            )
        },
        "targets": {
            "mode": "selected",
            "selected": [{"patient_id": "P00", "target_side": "left"}],
        },
            "profile_monte_carlo": {
                "draws": 30,
                "seed": 47,
                "interval_quantiles": [0.025, 0.5, 0.975],
                "convergence_draws": [10, 20, 30],
                "convergence_tolerance": 0.005,
            },
            "detector_reference": {
                "draws": 10,
                "seed": 48,
                "snr_quantiles": 4,
                "max_cases_per_stratum": 1,
                "calibration_nuisance": {
                    "enabled": False,
                    "exclusion_reason": "covariance_unavailable",
                    "future_scope": "shared_by_calib_session",
                },
                "masked_pixel_policy": (
                    "zero_masked_pixels_then_centered_poisson_on_positive_component"
                ),
            },
        "polar_cake": {
            "n_q": 32,
            "n_chi": 8,
            "parity_max_relative_rmse": 0.05,
        },
        "mlflow": {
            "enabled": True,
            "tracking_uri": (tmp_path / "mlruns").as_uri(),
            "experiment_name": "aramina-measurement-uncertainty-test",
        },
        "output": {"folder": str(tmp_path / "outputs")},
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        uncertainty,
        "run_preprocessing_pipeline",
        lambda *_args, **_kwargs: _profile_sigma_frame(),
    )
    monkeypatch.setattr(
        uncertainty,
        "select_detector_reference_subset",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "patient_id": "P00",
                    "target_side": "left",
                    "target_label": "BENIGN",
                    "target_snr_quantile": 1,
                }
            ]
        ),
    )

    def fake_detector(dataframe, *, model_artifact, targets, **_kwargs):
        return uncertainty.score_profile_sigma_measurement_uncertainty(
            dataframe,
            model_artifact=model_artifact,
            targets=targets,
            draws=10,
            seed=48,
        )

    monkeypatch.setattr(
        uncertainty,
        "score_detector_poisson_uncertainty",
        fake_detector,
    )
    monkeypatch.setattr(
        uncertainty,
        "write_polar_cake_artifacts",
        lambda *_args, **_kwargs: pd.DataFrame(
            [{"parity_pass": True, "relative_rmse": 0.001}]
        ),
    )

    result = uncertainty.run_measurement_uncertainty_from_config(config_path)

    assert result["patients_scored"] == 1
    assert result["mlflow"]["status"] == "FINISHED"
    artifacts = result["run_folder"]
    assert {path.name for path in artifacts.iterdir()}.issuperset(
        uncertainty.REQUIRED_ARTIFACTS
    )
    lineage = json.loads((artifacts / "lineage.json").read_text(encoding="utf-8"))
    assert (
        lineage["model"]["sha256"]
        == file_hashes(model_path, algorithms=("sha256",))["sha256"]
    )
    assert lineage["data_version"]["hash"] == hashes["md5"]
    assert (
        len(pd.read_csv(artifacts / "profile_measurement_uncertainty_draws.csv")) == 30
    )
    assert (
        len(pd.read_csv(artifacts / "detector_measurement_uncertainty_draws.csv")) == 10
    )
