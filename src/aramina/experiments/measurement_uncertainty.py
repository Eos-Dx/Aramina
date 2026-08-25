"""Bounded measurement-uncertainty experiment for frozen Aramina models.

This module is deliberately outside the product prediction route.  It leaves the
deterministic score, model artifact, report contracts, and legacy YAML files
unchanged.  The implemented adapter samples pyFAI's per-bin Poisson sigma after
integration, then reruns the frozen model's profile normalization, LR1
aggregation, and symmetry calculations for every draw.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from ..config_paths import resolve_config_path
from ..data_versioning import DVC_DATA_CONTRACT, verify_dvc_input
from ..mlflow_tracking import MlflowRun
from ..patient_features import build_patient_prediction_feature_row, target_breast_cases
from ..pipelines import run_preprocessing_pipeline
from ..prediction_contract import _model_threshold
from ..prediction_scoring import (
    _prediction_columns,
    _prediction_model_route,
    _score_model,
)
from ..runtime_identity import (
    aramina_git_sha,
    aramina_version,
    file_sha256,
    xrd_preprocessing_git_sha,
)
from ..training_config import PRODUCT_MODEL_NAME
from .detector_uncertainty import (
    MASKED_PIXEL_POLICY,
    iter_detector_poisson_profiles,
    write_polar_cake_artifacts,
)
from .covariance_uncertainty import (
    LowRankCovarianceModel,
    covariance_diagnostics_frame,
    covariance_eigen_spectrum_frame,
    fit_low_rank_covariance,
    normalized_profile_sigma,
    write_low_rank_covariance,
)


MEASUREMENT_UNCERTAINTY_CONTRACT = "aramina_measurement_uncertainty_v0_2"
LEGACY_MEASUREMENT_UNCERTAINTY_CONTRACT = "aramina_measurement_uncertainty_v0_1"
COVARIANCE_ADAPTER = "detector_mc_pooled_correlation_low_rank_v0_2"
FROZEN_MODEL_NAME = "aramina_target_breast_risk"
FROZEN_MODEL_VERSION = "0.2.14-beta"
RAW_PROFILE_COLUMN = "radial_profile_data_raw"
PROFILE_SIGMA_COLUMN = "radial_profile_sigma"
REQUIRED_ARTIFACTS = (
    "effective_experiment_config.yaml",
    "effective_training_preprocessing.yaml",
    "dvc_data_pointer.dvc",
    "detector_reference_subset.csv",
    "covariance_detector_reference_comparison.csv",
    "covariance_diagnostics.csv",
    "covariance_eigen_spectrum.csv",
    "covariance_model.npz",
    "covariance_monte_carlo_convergence.csv",
    "covariance_measurement_uncertainty_draws.csv",
    "covariance_measurement_uncertainty_summary.csv",
    "detector_measurement_uncertainty_draws.csv",
    "detector_measurement_uncertainty_summary.csv",
    "detector_profile_fit_manifest.csv",
    "detector_profile_fit_draws.npz",
    "lineage.json",
    "polar_cake_manifest.csv",
    "run_manifest.json",
)
LEGACY_REQUIRED_ARTIFACTS = (
    "effective_experiment_config.yaml",
    "effective_training_preprocessing.yaml",
    "dvc_data_pointer.dvc",
    "detector_reference_subset.csv",
    "detector_measurement_uncertainty_summary.csv",
    "detector_measurement_uncertainty_draws.csv",
    "lineage.json",
    "polar_cake_manifest.csv",
    "profile_detector_reference_comparison.csv",
    "profile_monte_carlo_convergence.csv",
    "profile_measurement_uncertainty_summary.csv",
    "profile_measurement_uncertainty_draws.csv",
    "run_manifest.json",
)
LEGACY_PROFILE_INCLUDED_SOURCES = ("pyfai_poisson_per_bin_sigma",)
LEGACY_PROFILE_EXCLUDED_SOURCES = (
    "pixel_splitting_cross_bin_covariance",
    "poni_geometry_calibration_uncertainty_shared_by_calib_session",
    "sample_and_calibrant_thickness_uncertainty",
    "detector_gain_and_readout_uncertainty",
    "detector_baseline_uncertainty",
    "faulty_pixel_mask_and_measurement_selection_uncertainty",
    "positioning_and_biological_repeatability",
    "model_parameter_and_training_data_uncertainty",
)
COVARIANCE_INCLUDED_SOURCES = (
    "detector_centered_poisson_empirical_correlation_after_product_integration_and_normalization",
    "measurement_specific_normalized_pyfai_sigma_scaling",
)
COVARIANCE_EXCLUDED_SOURCES = (
    "poni_geometry_calibration_uncertainty_shared_by_calib_session",
    "sample_and_calibrant_thickness_uncertainty",
    "detector_gain_and_readout_uncertainty",
    "detector_baseline_uncertainty",
    "faulty_pixel_mask_and_measurement_selection_uncertainty",
    "positioning_and_biological_repeatability",
    "model_parameter_and_training_data_uncertainty",
)
DETECTOR_INCLUDED_SOURCES = (
    "centered_poisson_perturbation_of_positive_estimated_photon_component",
    "product_faulty_pixel_mask",
    "pyfai_reintegration",
    "product_q_normalization",
)
DETECTOR_EXCLUDED_SOURCES = (
    "poni_geometry_calibration_uncertainty_shared_by_calib_session",
    "sample_and_calibrant_thickness_uncertainty",
    "detector_gain_and_readout_uncertainty",
    "detector_baseline_uncertainty",
    "positioning_and_biological_repeatability",
    "model_parameter_and_training_data_uncertainty",
)


class MeasurementUncertaintyError(ValueError):
    """Raised when the experimental uncertainty path cannot be audited."""


@dataclass(frozen=True)
class TargetRequest:
    """One clinician-selected target breast in the experimental H5 cohort."""

    patient_id: str
    target_side: str

    @property
    def target_case_id(self) -> str:
        return f"{self.patient_id}::{self.target_side}"


@dataclass(frozen=True)
class DetectorReferenceCollection:
    """Exact detector-MC reference scores and normalized profile draws."""

    summaries: pd.DataFrame
    score_draws: pd.DataFrame
    profile_draws: dict[str, np.ndarray]
    measurement_manifest: pd.DataFrame


def run_measurement_uncertainty_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the seeded profile-sigma experiment from an explicit YAML config."""
    path = Path(config_path).expanduser().resolve()
    config = _load_config(path)
    if config["contract"] == LEGACY_MEASUREMENT_UNCERTAINTY_CONTRACT:
        return _run_measurement_uncertainty_v0_1(config, path, verbose=verbose)
    input_config = _mapping(config, "input")
    input_h5_path = _resolve_required_path(input_config, "input_h5_path", path)
    model_path = _resolve_required_path(input_config, "model_joblib_path", path)
    data_version = verify_dvc_input(
        {"data_version": config["data_version"]},
        config_path=path,
        input_h5_path=input_h5_path,
    )
    if data_version is None:
        raise MeasurementUncertaintyError(
            "Measurement uncertainty requires DVC data lineage."
        )

    model_artifact = _load_frozen_model(model_path)
    _verify_model_data_lineage(model_artifact, data_version)
    run_folder = _create_run_folder(config, path)
    effective_preprocessing = _experimental_preprocessing_config(
        model_artifact,
        input_h5_path=input_h5_path,
        output_joblib_path=run_folder / "preprocessed_measurement_uncertainty.joblib",
        data_version=data_version,
    )
    dataframe = run_preprocessing_pipeline(
        input_h5_path,
        effective_preprocessing,
        verbose=verbose,
    )
    targets = _targets_for_run(dataframe, config)
    reference_subset = select_detector_reference_subset(
        dataframe,
        targets=targets,
        quantiles=int(config["detector_reference"]["snr_quantiles"]),
        max_cases_per_stratum=int(
            config["detector_reference"]["max_cases_per_stratum"]
        ),
    )
    reference_targets = [
        TargetRequest(row.patient_id, row.target_side)
        for row in reference_subset.itertuples(index=False)
    ]
    detector_reference = collect_detector_reference_draws(
        dataframe,
        model_artifact=model_artifact,
        targets=reference_targets,
        fit_draws=int(config["covariance_model"]["estimation_draws"]),
        comparison_draws=int(config["detector_reference"]["draws"]),
        seed=int(config["detector_reference"]["seed"]),
        interval_quantiles=tuple(config["covariance_monte_carlo"]["interval_quantiles"]),
    )
    covariance_model = fit_low_rank_covariance(
        detector_reference.profile_draws,
        detector_reference.measurement_manifest,
        explained_variance=float(config["covariance_model"]["explained_variance"]),
        max_rank=int(config["covariance_model"]["max_rank"]),
        minimum_diagonal_variance=float(
            config["covariance_model"]["minimum_diagonal_variance"]
        ),
    )
    covariance_summaries, covariance_draws = score_correlated_covariance_uncertainty(
        dataframe,
        model_artifact=model_artifact,
        targets=targets,
        covariance_model=covariance_model,
        draws=int(config["covariance_monte_carlo"]["draws"]),
        seed=int(config["covariance_monte_carlo"]["seed"]),
        interval_quantiles=tuple(
            config["covariance_monte_carlo"]["interval_quantiles"]
        ),
    )
    convergence = summarize_profile_monte_carlo_convergence(
        covariance_draws,
        checkpoints=tuple(config["covariance_monte_carlo"]["convergence_draws"]),
        interval_quantiles=tuple(
            config["covariance_monte_carlo"]["interval_quantiles"]
        ),
    )
    polar_config = config["polar_cake"]
    polar_manifest = write_polar_cake_artifacts(
        dataframe,
        target_cases=[
            (target.patient_id, target.target_side) for target in reference_targets
        ],
        output_folder=run_folder / "polar_cakes",
        n_q=int(polar_config["n_q"]),
        n_chi=int(polar_config["n_chi"]),
        parity_max_relative_rmse=float(polar_config["parity_max_relative_rmse"]),
    )
    method_comparison = compare_covariance_detector_reference(
        covariance_summaries,
        detector_reference.summaries,
    )

    lineage = _lineage(
        model_artifact=model_artifact,
        model_path=model_path,
        data_version=data_version,
    )
    _write_audit_artifacts(
        run_folder=run_folder,
        config=config,
        effective_preprocessing=effective_preprocessing,
        data_version=data_version,
        config_path=path,
        lineage=lineage,
        covariance_model=covariance_model,
        covariance_summaries=covariance_summaries,
        covariance_draws=covariance_draws,
        detector_reference=reference_subset,
        detector_collection=detector_reference,
        polar_manifest=polar_manifest,
        method_comparison=method_comparison,
        convergence=convergence,
    )
    mlflow = _log_mlflow_run(
        config=config,
        config_path=path,
        run_folder=run_folder,
        lineage=lineage,
        covariance_model=covariance_model,
        covariance_summaries=covariance_summaries,
        detector_summaries=detector_reference.summaries,
        polar_manifest=polar_manifest,
        method_comparison=method_comparison,
        convergence=convergence,
    )
    return {
        "contract": MEASUREMENT_UNCERTAINTY_CONTRACT,
        "run_folder": run_folder,
        "summary_path": run_folder / "covariance_measurement_uncertainty_summary.csv",
        "draws_path": run_folder / "covariance_measurement_uncertainty_draws.csv",
        "patients_scored": int(len(covariance_summaries)),
        "detector_reference_patients": int(len(detector_reference.summaries)),
        "mlflow": mlflow,
    }


def _run_measurement_uncertainty_v0_1(
    config: dict[str, Any],
    config_path: Path,
    *,
    verbose: bool,
) -> dict[str, Any]:
    """Execute the immutable historical diagonal-profile experiment contract."""
    input_config = _mapping(config, "input")
    input_h5_path = _resolve_required_path(input_config, "input_h5_path", config_path)
    model_path = _resolve_required_path(input_config, "model_joblib_path", config_path)
    data_version = verify_dvc_input(
        {"data_version": config["data_version"]},
        config_path=config_path,
        input_h5_path=input_h5_path,
    )
    if data_version is None:
        raise MeasurementUncertaintyError(
            "Measurement uncertainty requires DVC data lineage."
        )
    model_artifact = _load_frozen_model(model_path)
    _verify_model_data_lineage(model_artifact, data_version)
    run_folder = _create_run_folder(config, config_path)
    effective_preprocessing = _experimental_preprocessing_config(
        model_artifact,
        input_h5_path=input_h5_path,
        output_joblib_path=run_folder / "preprocessed_measurement_uncertainty.joblib",
        data_version=data_version,
    )
    dataframe = run_preprocessing_pipeline(
        input_h5_path,
        effective_preprocessing,
        verbose=verbose,
    )
    targets = _targets_for_run(dataframe, config)
    profile_config = config["profile_monte_carlo"]
    profile_summaries, profile_draws = score_profile_sigma_measurement_uncertainty(
        dataframe,
        model_artifact=model_artifact,
        targets=targets,
        draws=int(profile_config["draws"]),
        seed=int(profile_config["seed"]),
        interval_quantiles=tuple(profile_config["interval_quantiles"]),
    )
    convergence = summarize_profile_monte_carlo_convergence(
        profile_draws,
        checkpoints=tuple(profile_config["convergence_draws"]),
        interval_quantiles=tuple(profile_config["interval_quantiles"]),
    )
    reference_subset = select_detector_reference_subset(
        dataframe,
        targets=targets,
        quantiles=int(config["detector_reference"]["snr_quantiles"]),
        max_cases_per_stratum=int(
            config["detector_reference"]["max_cases_per_stratum"]
        ),
    )
    reference_targets = [
        TargetRequest(row.patient_id, row.target_side)
        for row in reference_subset.itertuples(index=False)
    ]
    detector_summaries, detector_draws = score_detector_poisson_uncertainty(
        dataframe,
        model_artifact=model_artifact,
        targets=reference_targets,
        draws=int(config["detector_reference"]["draws"]),
        seed=int(config["detector_reference"]["seed"]),
        interval_quantiles=tuple(profile_config["interval_quantiles"]),
    )
    polar_config = config["polar_cake"]
    polar_manifest = write_polar_cake_artifacts(
        dataframe,
        target_cases=[
            (target.patient_id, target.target_side) for target in reference_targets
        ],
        output_folder=run_folder / "polar_cakes",
        n_q=int(polar_config["n_q"]),
        n_chi=int(polar_config["n_chi"]),
        parity_max_relative_rmse=float(polar_config["parity_max_relative_rmse"]),
    )
    comparison = compare_profile_detector_reference(profile_summaries, detector_summaries)
    lineage = _lineage(
        model_artifact=model_artifact,
        model_path=model_path,
        data_version=data_version,
    )
    _write_legacy_audit_artifacts(
        run_folder=run_folder,
        config=config,
        effective_preprocessing=effective_preprocessing,
        data_version=data_version,
        config_path=config_path,
        lineage=lineage,
        detector_reference=reference_subset,
        profile_summaries=profile_summaries,
        profile_draws=profile_draws,
        detector_summaries=detector_summaries,
        detector_draws=detector_draws,
        polar_manifest=polar_manifest,
        comparison=comparison,
        convergence=convergence,
    )
    mlflow = _log_legacy_mlflow_run(
        config=config,
        config_path=config_path,
        run_folder=run_folder,
        lineage=lineage,
        profile_summaries=profile_summaries,
        detector_summaries=detector_summaries,
        polar_manifest=polar_manifest,
        comparison=comparison,
    )
    return {
        "contract": LEGACY_MEASUREMENT_UNCERTAINTY_CONTRACT,
        "run_folder": run_folder,
        "summary_path": run_folder / "profile_measurement_uncertainty_summary.csv",
        "draws_path": run_folder / "profile_measurement_uncertainty_draws.csv",
        "patients_scored": int(len(profile_summaries)),
        "detector_reference_patients": int(len(detector_summaries)),
        "mlflow": mlflow,
    }


def compare_profile_detector_reference(
    profile_summaries: pd.DataFrame,
    detector_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Pair fast profile intervals with detector-reference intervals."""
    columns = [
        "target_case_id",
        "deterministic_p_cancer",
        "p_cancer_low",
        "p_cancer_high",
        "p_cancer_sd",
        "probability_above_threshold",
        "threshold_crossing",
    ]
    profile = profile_summaries[columns].rename(
        columns={column: f"profile_{column}" for column in columns[1:]}
    )
    detector = detector_summaries[columns].rename(
        columns={column: f"detector_{column}" for column in columns[1:]}
    )
    paired = profile.merge(
        detector,
        on="target_case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(detector_summaries):
        raise MeasurementUncertaintyError(
            "Every detector-reference case must have one profile-level summary."
        )
    if not np.allclose(
        paired["profile_deterministic_p_cancer"],
        paired["detector_deterministic_p_cancer"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise MeasurementUncertaintyError(
            "Profile and detector methods changed deterministic p_cancer."
        )
    paired["profile_interval_width"] = (
        paired["profile_p_cancer_high"] - paired["profile_p_cancer_low"]
    )
    paired["detector_interval_width"] = (
        paired["detector_p_cancer_high"] - paired["detector_p_cancer_low"]
    )
    paired["profile_to_detector_width_ratio"] = paired[
        "profile_interval_width"
    ] / paired["detector_interval_width"].replace(0.0, np.nan)
    paired["threshold_crossing_agreement"] = (
        paired["profile_threshold_crossing"]
        == paired["detector_threshold_crossing"]
    )
    return paired


def compare_covariance_detector_reference(
    covariance_summaries: pd.DataFrame,
    detector_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Compare transferred covariance intervals with held-out detector draws."""
    columns = [
        "target_case_id",
        "deterministic_p_cancer",
        "p_cancer_low",
        "p_cancer_high",
        "p_cancer_sd",
        "probability_above_threshold",
        "threshold_crossing",
    ]
    covariance = covariance_summaries[columns].rename(
        columns={column: f"covariance_{column}" for column in columns[1:]}
    )
    detector = detector_summaries[columns].rename(
        columns={column: f"detector_{column}" for column in columns[1:]}
    )
    paired = covariance.merge(
        detector,
        on="target_case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(detector_summaries):
        raise MeasurementUncertaintyError(
            "Every detector-reference case requires one covariance summary."
        )
    if not np.allclose(
        paired["covariance_deterministic_p_cancer"],
        paired["detector_deterministic_p_cancer"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise MeasurementUncertaintyError(
            "Covariance and detector paths changed deterministic p_cancer."
        )
    paired["covariance_interval_width"] = (
        paired["covariance_p_cancer_high"] - paired["covariance_p_cancer_low"]
    )
    paired["detector_interval_width"] = (
        paired["detector_p_cancer_high"] - paired["detector_p_cancer_low"]
    )
    paired["covariance_to_detector_width_ratio"] = paired[
        "covariance_interval_width"
    ] / paired["detector_interval_width"].replace(0.0, np.nan)
    paired["threshold_crossing_agreement"] = (
        paired["covariance_threshold_crossing"]
        == paired["detector_threshold_crossing"]
    )
    paired["abs_probability_above_threshold_difference"] = np.abs(
        paired["covariance_probability_above_threshold"]
        - paired["detector_probability_above_threshold"]
    )
    return paired


def summarize_profile_monte_carlo_convergence(
    profile_draws: pd.DataFrame,
    *,
    checkpoints: tuple[int, ...],
    interval_quantiles: tuple[float, float, float],
) -> pd.DataFrame:
    """Summarize nested Monte Carlo checkpoints from one seeded draw stream."""
    _validate_quantiles(interval_quantiles)
    if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
        raise MeasurementUncertaintyError(
            "Monte Carlo convergence checkpoints must be unique and increasing."
        )
    available_draws = int(profile_draws["draw_index"].max()) + 1
    if checkpoints[-1] != available_draws:
        raise MeasurementUncertaintyError(
            "Final convergence checkpoint must equal the configured draw count."
        )
    rows: list[dict[str, Any]] = []
    for target_case_id, case_draws in profile_draws.groupby(
        "target_case_id",
        sort=True,
    ):
        ordered = case_draws.sort_values("draw_index")
        if len(ordered) != available_draws:
            raise MeasurementUncertaintyError(
                f"Target case {target_case_id!r} has an incomplete draw stream."
            )
        previous: dict[str, float] | None = None
        for checkpoint in checkpoints:
            selected = ordered.iloc[:checkpoint]
            values = selected["p_cancer"].to_numpy(dtype=float)
            lower, median, upper = (
                float(np.quantile(values, quantile))
                for quantile in interval_quantiles
            )
            current = {
                "p_cancer_low": lower,
                "p_cancer_median": median,
                "p_cancer_high": upper,
                "probability_above_threshold": float(
                    selected["above_threshold"].mean()
                ),
            }
            rows.append(
                {
                    "target_case_id": target_case_id,
                    "draws": checkpoint,
                    **current,
                    "abs_delta_low": (
                        np.nan
                        if previous is None
                        else abs(lower - previous["p_cancer_low"])
                    ),
                    "abs_delta_high": (
                        np.nan
                        if previous is None
                        else abs(upper - previous["p_cancer_high"])
                    ),
                    "abs_delta_probability_above_threshold": (
                        np.nan
                        if previous is None
                        else abs(
                            current["probability_above_threshold"]
                            - previous["probability_above_threshold"]
                        )
                    ),
                }
            )
            previous = current
    return pd.DataFrame(rows)


def score_profile_sigma_measurement_uncertainty(
    dataframe: pd.DataFrame,
    *,
    model_artifact: dict[str, Any],
    targets: Sequence[TargetRequest],
    draws: int,
    seed: int,
    interval_quantiles: tuple[float, float, float] = (0.025, 0.5, 0.975),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score bounded Gaussian profile-sigma draws through frozen patient features."""
    if draws < 10:
        raise MeasurementUncertaintyError("Monte Carlo draws must be at least 10.")
    if not targets:
        raise MeasurementUncertaintyError("At least one target breast is required.")
    _validate_quantiles(interval_quantiles)
    model_info = _frozen_model_info(model_artifact)
    columns = _prediction_columns(model_artifact)
    profile_column = columns["profile_column"]
    _validate_uncertainty_dataframe(dataframe, profile_column=profile_column)

    summary_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        patient_frame = _patient_frame(
            dataframe,
            patient_id=target.patient_id,
            group_column=columns["group_column"],
        )
        _validate_normalization_parity(patient_frame, profile_column=profile_column)
        deterministic = _score_patient_frame(
            patient_frame,
            model_info=model_info,
            model_name=FROZEN_MODEL_NAME,
            target=target,
            columns=columns,
        )
        rng = np.random.default_rng(np.random.SeedSequence((seed, target_index)))
        probabilities: list[float] = []
        routes: set[str | None] = set()
        thresholds: set[float] = set()
        for draw_index, sampled_frame in enumerate(
            _profile_sigma_draws(patient_frame, draws=draws, rng=rng)
        ):
            score = _score_patient_frame(
                sampled_frame,
                model_info=model_info,
                model_name=FROZEN_MODEL_NAME,
                target=target,
                columns=columns,
            )
            probabilities.append(score["p_cancer"])
            routes.add(score["model_route"])
            thresholds.add(score["threshold"])
            draw_rows.append(
                {
                    "target_case_id": target.target_case_id,
                    "patient_id": target.patient_id,
                    "target_side": target.target_side,
                    "draw_index": draw_index,
                    "p_cancer": score["p_cancer"],
                    "threshold": score["threshold"],
                    "above_threshold": bool(score["p_cancer"] >= score["threshold"]),
                    "model_route": score["model_route"] or "single_model",
                }
            )
        if len(routes) != 1 or len(thresholds) != 1:
            raise MeasurementUncertaintyError(
                "Monte Carlo draws changed the frozen model route or threshold; "
                "this adapter does not support mixed decision routes."
            )
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (draws,) or not np.isfinite(values).all():
            raise MeasurementUncertaintyError(
                "Monte Carlo produced non-finite patient scores."
            )
        lower, median, upper = (
            float(np.quantile(values, q)) for q in interval_quantiles
        )
        threshold = next(iter(thresholds))
        summary_rows.append(
            {
                "target_case_id": target.target_case_id,
                "patient_id": target.patient_id,
                "target_side": target.target_side,
                "deterministic_p_cancer": deterministic["p_cancer"],
                "decision_threshold": threshold,
                "draws": draws,
                "seed": seed,
                "measurement_count": int(len(patient_frame)),
                "target_measurements": deterministic["target_measurements"],
                "contralateral_measurements": deterministic[
                    "contralateral_measurements"
                ],
                "symmetry_available": deterministic["symmetry_available"],
                "model_route": deterministic["model_route"] or "single_model",
                "p_cancer_mean": float(np.mean(values)),
                "p_cancer_sd": float(np.std(values, ddof=1)),
                "p_cancer_low": lower,
                "p_cancer_median": median,
                "p_cancer_high": upper,
                "probability_above_threshold": float(np.mean(values >= threshold)),
                "threshold_crossing": bool(lower <= threshold <= upper),
                "included_uncertainty_sources": ";".join(
                    LEGACY_PROFILE_INCLUDED_SOURCES
                ),
                "excluded_uncertainty_sources": ";".join(
                    LEGACY_PROFILE_EXCLUDED_SOURCES
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def score_detector_poisson_uncertainty(
    dataframe: pd.DataFrame,
    *,
    model_artifact: dict[str, Any],
    targets: Sequence[TargetRequest],
    draws: int,
    seed: int,
    interval_quantiles: tuple[float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate detector Poisson draws through frozen 0.2.14 scoring."""
    if draws < 10:
        raise MeasurementUncertaintyError("Detector Monte Carlo requires 10 draws.")
    _validate_quantiles(interval_quantiles)
    model_info = _frozen_model_info(model_artifact)
    columns = _prediction_columns(model_artifact)
    summaries: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        patient_frame = _patient_frame(
            dataframe,
            patient_id=target.patient_id,
            group_column=columns["group_column"],
        )
        deterministic = _score_patient_frame(
            patient_frame,
            model_info=model_info,
            model_name=FROZEN_MODEL_NAME,
            target=target,
            columns=columns,
        )
        probabilities: list[float] = []
        draw_seed = int(
            np.random.SeedSequence((seed, target_index)).generate_state(1)[0]
        )
        iterator = iter_detector_poisson_profiles(
            patient_frame,
            draws=draws,
            random_state=draw_seed,
            normalization_q_range=(6.7, 7.1),
        )
        for draw_index, sampled_frame in enumerate(iterator):
            score = _score_patient_frame(
                sampled_frame,
                model_info=model_info,
                model_name=FROZEN_MODEL_NAME,
                target=target,
                columns=columns,
            )
            if score["threshold"] != deterministic["threshold"]:
                raise MeasurementUncertaintyError(
                    "Detector draws changed the frozen decision threshold."
                )
            probability = float(score["p_cancer"])
            probabilities.append(probability)
            draw_rows.append(
                {
                    "target_case_id": target.target_case_id,
                    "patient_id": target.patient_id,
                    "target_side": target.target_side,
                    "draw_index": draw_index,
                    "p_cancer": probability,
                    "threshold": deterministic["threshold"],
                    "above_threshold": bool(probability >= deterministic["threshold"]),
                    "model_route": score["model_route"] or "single_model",
                }
            )
        values = np.asarray(probabilities, dtype=float)
        lower, median, upper = (
            float(np.quantile(values, quantile)) for quantile in interval_quantiles
        )
        threshold = float(deterministic["threshold"])
        summaries.append(
            {
                "target_case_id": target.target_case_id,
                "patient_id": target.patient_id,
                "target_side": target.target_side,
                "deterministic_p_cancer": deterministic["p_cancer"],
                "decision_threshold": threshold,
                "draws": draws,
                "seed": seed,
                "measurement_count": int(len(patient_frame)),
                "target_measurements": deterministic["target_measurements"],
                "contralateral_measurements": deterministic[
                    "contralateral_measurements"
                ],
                "symmetry_available": deterministic["symmetry_available"],
                "model_route": deterministic["model_route"] or "single_model",
                "p_cancer_mean": float(np.mean(values)),
                "p_cancer_sd": float(np.std(values, ddof=1)),
                "p_cancer_low": lower,
                "p_cancer_median": median,
                "p_cancer_high": upper,
                "probability_above_threshold": float(np.mean(values >= threshold)),
                "threshold_crossing": bool(lower <= threshold <= upper),
                "included_uncertainty_sources": ";".join(DETECTOR_INCLUDED_SOURCES),
                "excluded_uncertainty_sources": ";".join(DETECTOR_EXCLUDED_SOURCES),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(draw_rows)


def collect_detector_reference_draws(
    dataframe: pd.DataFrame,
    *,
    model_artifact: dict[str, Any],
    targets: Sequence[TargetRequest],
    fit_draws: int,
    comparison_draws: int,
    seed: int,
    interval_quantiles: tuple[float, float, float],
) -> DetectorReferenceCollection:
    """Collect exact normalized detector-MC draws for fit and comparison.

    The first seeded block estimates the pooled correlation.  The second block
    is scored as a detector reference, so the fast covariance comparison is not
    evaluated on the same stochastic draws used to fit its correlation shape.
    """
    if fit_draws < 3 or comparison_draws < 10:
        raise MeasurementUncertaintyError(
            "Detector reference requires at least 3 fit and 10 comparison draws."
        )
    _validate_quantiles(interval_quantiles)
    model_info = _frozen_model_info(model_artifact)
    columns = _prediction_columns(model_artifact)
    profile_column = columns["profile_column"]
    _validate_uncertainty_dataframe(dataframe, profile_column=profile_column)
    summaries: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    profile_draws: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        patient_frame = _patient_frame(
            dataframe,
            patient_id=target.patient_id,
            group_column=columns["group_column"],
        )
        _validate_normalization_parity(patient_frame, profile_column=profile_column)
        deterministic = _score_patient_frame(
            patient_frame,
            model_info=model_info,
            model_name=FROZEN_MODEL_NAME,
            target=target,
            columns=columns,
        )
        keys: list[str] = []
        collected: list[list[np.ndarray]] = []
        for measurement_index, (_, row) in enumerate(patient_frame.iterrows()):
            key = f"{target.target_case_id}::measurement_{measurement_index}"
            keys.append(key)
            collected.append([])
            normalized_sigma = normalized_profile_sigma(
                np.asarray(row["q_range"], dtype=float),
                np.asarray(row[RAW_PROFILE_COLUMN], dtype=float),
                np.asarray(row[PROFILE_SIGMA_COLUMN], dtype=float),
            )
            manifest_rows.append(
                {
                    "profile_key": key,
                    "target_case_id": target.target_case_id,
                    "patient_id": target.patient_id,
                    "target_side": target.target_side,
                    "measurement_index": measurement_index,
                    "measured_side": str(row["side"]),
                    "specimen_id": str(row["specimenId"]),
                    "snr_db": float(row["snr_db"]),
                    "normalized_sigma_rms": float(
                        np.sqrt(np.mean(normalized_sigma**2))
                    ),
                }
            )
        draw_seed = int(
            np.random.SeedSequence((seed, target_index)).generate_state(1)[0]
        )
        iterator = iter_detector_poisson_profiles(
            patient_frame,
            draws=fit_draws + comparison_draws,
            random_state=draw_seed,
            normalization_q_range=(6.7, 7.1),
        )
        probabilities: list[float] = []
        routes: set[str | None] = set()
        thresholds: set[float] = set()
        for draw_index, sampled_frame in enumerate(iterator):
            if draw_index < fit_draws:
                for measurement_index, key in enumerate(keys):
                    collected[measurement_index].append(
                        np.asarray(
                            sampled_frame[profile_column].iloc[measurement_index],
                            dtype=float,
                        )
                    )
                continue
            score = _score_patient_frame(
                sampled_frame,
                model_info=model_info,
                model_name=FROZEN_MODEL_NAME,
                target=target,
                columns=columns,
            )
            probability = float(score["p_cancer"])
            probabilities.append(probability)
            routes.add(score["model_route"])
            thresholds.add(float(score["threshold"]))
            score_rows.append(
                {
                    "target_case_id": target.target_case_id,
                    "patient_id": target.patient_id,
                    "target_side": target.target_side,
                    "draw_index": draw_index - fit_draws,
                    "detector_stream_draw_index": draw_index,
                    "p_cancer": probability,
                    "threshold": float(score["threshold"]),
                    "above_threshold": bool(probability >= score["threshold"]),
                    "model_route": score["model_route"] or "single_model",
                }
            )
        for key, samples in zip(keys, collected, strict=True):
            values = np.vstack(samples)
            if values.shape[0] != fit_draws:
                raise MeasurementUncertaintyError(
                    f"Detector covariance fit draws are incomplete for {key!r}."
                )
            profile_draws[key] = values
        if len(routes) != 1 or len(thresholds) != 1:
            raise MeasurementUncertaintyError(
                "Detector reference changed frozen model route or threshold."
            )
        values = np.asarray(probabilities, dtype=float)
        lower, median, upper = (
            float(np.quantile(values, quantile)) for quantile in interval_quantiles
        )
        threshold = next(iter(thresholds))
        summaries.append(
            {
                "target_case_id": target.target_case_id,
                "patient_id": target.patient_id,
                "target_side": target.target_side,
                "deterministic_p_cancer": deterministic["p_cancer"],
                "decision_threshold": threshold,
                "draws": comparison_draws,
                "fit_draws": fit_draws,
                "seed": seed,
                "measurement_count": int(len(patient_frame)),
                "target_measurements": deterministic["target_measurements"],
                "contralateral_measurements": deterministic[
                    "contralateral_measurements"
                ],
                "symmetry_available": deterministic["symmetry_available"],
                "model_route": deterministic["model_route"] or "single_model",
                "p_cancer_mean": float(np.mean(values)),
                "p_cancer_sd": float(np.std(values, ddof=1)),
                "p_cancer_low": lower,
                "p_cancer_median": median,
                "p_cancer_high": upper,
                "probability_above_threshold": float(np.mean(values >= threshold)),
                "threshold_crossing": bool(lower <= threshold <= upper),
                "included_uncertainty_sources": ";".join(
                    DETECTOR_INCLUDED_SOURCES
                ),
                "excluded_uncertainty_sources": ";".join(
                    DETECTOR_EXCLUDED_SOURCES
                ),
            }
        )
    return DetectorReferenceCollection(
        summaries=pd.DataFrame(summaries),
        score_draws=pd.DataFrame(score_rows),
        profile_draws=profile_draws,
        measurement_manifest=pd.DataFrame(manifest_rows),
    )


def score_correlated_covariance_uncertainty(
    dataframe: pd.DataFrame,
    *,
    model_artifact: dict[str, Any],
    targets: Sequence[TargetRequest],
    covariance_model: LowRankCovarianceModel,
    draws: int,
    seed: int,
    interval_quantiles: tuple[float, float, float] = (0.025, 0.5, 0.975),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate transferred low-rank detector-MC covariance through frozen LR1/LR2."""
    if draws < 10:
        raise MeasurementUncertaintyError("Covariance Monte Carlo requires 10 draws.")
    if not targets:
        raise MeasurementUncertaintyError("At least one target breast is required.")
    _validate_quantiles(interval_quantiles)
    model_info = _frozen_model_info(model_artifact)
    columns = _prediction_columns(model_artifact)
    profile_column = columns["profile_column"]
    _validate_uncertainty_dataframe(dataframe, profile_column=profile_column)
    summary_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        patient_frame = _patient_frame(
            dataframe,
            patient_id=target.patient_id,
            group_column=columns["group_column"],
        )
        _validate_normalization_parity(patient_frame, profile_column=profile_column)
        deterministic = _score_patient_frame(
            patient_frame,
            model_info=model_info,
            model_name=FROZEN_MODEL_NAME,
            target=target,
            columns=columns,
        )
        rng = np.random.default_rng(np.random.SeedSequence((seed, target_index)))
        baseline_profiles = [
            np.asarray(value, dtype=float) for value in patient_frame[profile_column]
        ]
        scale_vectors = [
            normalized_profile_sigma(
                np.asarray(row["q_range"], dtype=float),
                np.asarray(row[RAW_PROFILE_COLUMN], dtype=float),
                np.asarray(row[PROFILE_SIGMA_COLUMN], dtype=float),
            )
            for _, row in patient_frame.iterrows()
        ]
        perturbations = [
            covariance_model.sample(draws=draws, sigma_scale=scale, rng=rng)
            for scale in scale_vectors
        ]
        probabilities: list[float] = []
        routes: set[str | None] = set()
        thresholds: set[float] = set()
        for draw_index in range(draws):
            sampled_frame = patient_frame.copy(deep=True)
            sampled_profiles = [
                baseline + perturbation[draw_index]
                for baseline, perturbation in zip(
                    baseline_profiles, perturbations, strict=True
                )
            ]
            if not all(np.isfinite(value).all() for value in sampled_profiles):
                raise MeasurementUncertaintyError("Covariance draw produced non-finite profiles.")
            sampled_frame[profile_column] = sampled_profiles
            score = _score_patient_frame(
                sampled_frame,
                model_info=model_info,
                model_name=FROZEN_MODEL_NAME,
                target=target,
                columns=columns,
            )
            probabilities.append(float(score["p_cancer"]))
            routes.add(score["model_route"])
            thresholds.add(float(score["threshold"]))
            draw_rows.append(
                {
                    "target_case_id": target.target_case_id,
                    "patient_id": target.patient_id,
                    "target_side": target.target_side,
                    "draw_index": draw_index,
                    "p_cancer": float(score["p_cancer"]),
                    "threshold": float(score["threshold"]),
                    "above_threshold": bool(score["p_cancer"] >= score["threshold"]),
                    "model_route": score["model_route"] or "single_model",
                }
            )
        if len(routes) != 1 or len(thresholds) != 1:
            raise MeasurementUncertaintyError(
                "Covariance draws changed frozen model route or threshold."
            )
        values = np.asarray(probabilities, dtype=float)
        lower, median, upper = (
            float(np.quantile(values, quantile)) for quantile in interval_quantiles
        )
        threshold = next(iter(thresholds))
        summary_rows.append(
            {
                "target_case_id": target.target_case_id,
                "patient_id": target.patient_id,
                "target_side": target.target_side,
                "deterministic_p_cancer": deterministic["p_cancer"],
                "decision_threshold": threshold,
                "draws": draws,
                "seed": seed,
                "measurement_count": int(len(patient_frame)),
                "target_measurements": deterministic["target_measurements"],
                "contralateral_measurements": deterministic[
                    "contralateral_measurements"
                ],
                "symmetry_available": deterministic["symmetry_available"],
                "model_route": deterministic["model_route"] or "single_model",
                "p_cancer_mean": float(np.mean(values)),
                "p_cancer_sd": float(np.std(values, ddof=1)),
                "p_cancer_low": lower,
                "p_cancer_median": median,
                "p_cancer_high": upper,
                "probability_above_threshold": float(np.mean(values >= threshold)),
                "threshold_crossing": bool(lower <= threshold <= upper),
                "included_uncertainty_sources": ";".join(COVARIANCE_INCLUDED_SOURCES),
                "excluded_uncertainty_sources": ";".join(COVARIANCE_EXCLUDED_SOURCES),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def select_detector_reference_subset(
    dataframe: pd.DataFrame,
    *,
    targets: Sequence[TargetRequest],
    quantiles: int,
    max_cases_per_stratum: int,
) -> pd.DataFrame:
    """Select deterministic BENIGN/CANCER and SNR detector-reference strata."""
    cases = target_breast_cases(
        dataframe,
        group_column="patientId",
        side_column="side",
        label_column="product_status_group",
        biopsy_column="biopsy",
    ).set_index("target_case_id")
    rows: list[dict[str, Any]] = []
    for target in targets:
        target_id = f"{target.patient_id}::{target.target_side.upper()}"
        if target_id not in cases.index:
            raise MeasurementUncertaintyError(
                f"Historical target case is unavailable: {target_id}."
            )
        patient_rows = dataframe[
            dataframe["patientId"].astype(str) == target.patient_id
        ]
        target_rows = patient_rows[
            patient_rows["side"].map(_side_norm) == target.target_side.upper()
        ]
        snr = pd.to_numeric(target_rows["snr_db"], errors="coerce").dropna()
        if snr.empty:
            continue
        label = int(cases.loc[target_id, "label"])
        rows.append(
            {
                "target_case_id": target.target_case_id,
                "patient_id": target.patient_id,
                "target_side": target.target_side,
                "target_label": "CANCER" if label == 1 else "BENIGN",
                "target_snr_db_median": float(snr.median()),
                "calibration_nuisance_included": False,
                "calibration_nuisance_future_scope": "shared_by_calib_session",
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty or candidates["target_label"].nunique() != 2:
        raise MeasurementUncertaintyError(
            "Detector reference requires BENIGN and CANCER target cases."
        )
    ranks = candidates.groupby("target_label")["target_snr_db_median"].rank(
        method="first",
        pct=True,
    )
    candidates["target_snr_quantile"] = np.minimum(
        np.floor(ranks * quantiles).astype(int) + 1,
        quantiles,
    )
    return (
        candidates.sort_values(
            ["target_label", "target_snr_quantile", "target_case_id"]
        )
        .groupby(
            ["target_label", "target_snr_quantile"],
            group_keys=False,
        )
        .head(max_cases_per_stratum)
        .reset_index(drop=True)
    )


def _score_patient_frame(
    patient_frame: pd.DataFrame,
    *,
    model_info: dict[str, Any],
    model_name: str,
    target: TargetRequest,
    columns: dict[str, str],
) -> dict[str, Any]:
    feature_table = build_patient_prediction_feature_row(
        patient_frame,
        model_info,
        patient_id=target.patient_id,
        target_side=target.target_side,
        **columns,
    )
    route = _prediction_model_route(feature_table, model_info)
    threshold = _model_threshold(model_info, "threshold_target", route)
    return {
        "p_cancer": _score_model(feature_table, model_name, model_info, route),
        "threshold": threshold,
        "model_route": route,
        "target_measurements": int(feature_table["target_measurements"].iloc[0]),
        "contralateral_measurements": int(
            feature_table["contralateral_measurements"].iloc[0]
        ),
        "symmetry_available": int(feature_table["symmetry_available"].iloc[0]),
    }


def _profile_sigma_draws(
    patient_frame: pd.DataFrame,
    *,
    draws: int,
    rng: np.random.Generator,
) -> Iterator[pd.DataFrame]:
    raw = np.vstack(
        [np.asarray(value, dtype=float) for value in patient_frame[RAW_PROFILE_COLUMN]]
    )
    sigma = np.vstack(
        [
            np.asarray(value, dtype=float)
            for value in patient_frame[PROFILE_SIGMA_COLUMN]
        ]
    )
    q = np.vstack(
        [np.asarray(value, dtype=float) for value in patient_frame["q_range"]]
    )
    for _ in range(draws):
        sampled_raw = raw + rng.normal(loc=0.0, scale=sigma, size=raw.shape)
        sampled_profiles = [
            _normalize_profile_with_product_policy(row_q, row_profile)
            for row_q, row_profile in zip(q, sampled_raw, strict=True)
        ]
        out = patient_frame.copy(deep=True)
        out["radial_profile_data"] = sampled_profiles
        yield out


def _normalize_profile_with_product_policy(
    q: np.ndarray, profile: np.ndarray
) -> np.ndarray:
    q_values = np.asarray(q, dtype=float).ravel()
    values = np.asarray(profile, dtype=float).ravel()
    if q_values.shape != values.shape or not np.isfinite(values).all():
        raise MeasurementUncertaintyError(
            "Sampled profile is not finite on the expected q grid."
        )
    band = (q_values >= 6.7) & (q_values <= 7.1)
    if int(np.sum(band)) < 1:
        raise MeasurementUncertaintyError(
            "Normalization q range [6.7, 7.1] has no profile bins."
        )
    scale = float(np.median(values[band]))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise MeasurementUncertaintyError(
            "Monte Carlo draw has an invalid non-positive normalization value."
        )
    return values / scale


def _validate_uncertainty_dataframe(
    dataframe: pd.DataFrame,
    *,
    profile_column: str,
) -> None:
    required = {
        "patientId",
        "specimenId",
        "side",
        "q_range",
        profile_column,
        RAW_PROFILE_COLUMN,
        PROFILE_SIGMA_COLUMN,
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise MeasurementUncertaintyError(
            "Experimental preprocessing did not retain required uncertainty columns: "
            f"{missing}."
        )
    if dataframe.empty:
        raise MeasurementUncertaintyError(
            "Experimental preprocessing returned no measurements."
        )
    for row_index, row in dataframe.iterrows():
        q = np.asarray(row["q_range"], dtype=float).ravel()
        profile = np.asarray(row[profile_column], dtype=float).ravel()
        raw = np.asarray(row[RAW_PROFILE_COLUMN], dtype=float).ravel()
        sigma = np.asarray(row[PROFILE_SIGMA_COLUMN], dtype=float).ravel()
        if (
            q.size < 2
            or q.shape != profile.shape
            or q.shape != raw.shape
            or q.shape != sigma.shape
        ):
            raise MeasurementUncertaintyError(
                f"Measurement row {row_index} has incompatible q/profile/sigma shapes."
            )
        if (
            not np.isfinite(q).all()
            or not np.isfinite(profile).all()
            or not np.isfinite(raw).all()
        ):
            raise MeasurementUncertaintyError(
                f"Measurement row {row_index} has non-finite profile values."
            )
        if (
            not np.isfinite(sigma).all()
            or np.any(sigma < 0.0)
            or not np.any(sigma > 0.0)
        ):
            raise MeasurementUncertaintyError(
                f"Measurement row {row_index} has invalid pyFAI profile sigma."
            )


def _validate_normalization_parity(
    patient_frame: pd.DataFrame,
    *,
    profile_column: str,
) -> None:
    for row_index, row in patient_frame.iterrows():
        expected = np.asarray(row[profile_column], dtype=float).ravel()
        actual = _normalize_profile_with_product_policy(
            np.asarray(row["q_range"], dtype=float),
            np.asarray(row[RAW_PROFILE_COLUMN], dtype=float),
        )
        if not np.allclose(actual, expected, rtol=1e-8, atol=1e-10):
            raise MeasurementUncertaintyError(
                "Experimental raw-profile normalization differs from the frozen product "
                f"profile for measurement row {row_index}."
            )


def _patient_frame(
    dataframe: pd.DataFrame,
    *,
    patient_id: str,
    group_column: str,
) -> pd.DataFrame:
    frame = dataframe[dataframe[group_column].astype(str) == patient_id].copy(deep=True)
    if frame.empty:
        raise MeasurementUncertaintyError(
            f"Requested patient is absent after experimental preprocessing: {patient_id!r}."
        )
    return frame.reset_index(drop=True)


def _load_frozen_model(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Frozen model artifact is unavailable: {model_path}")
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise MeasurementUncertaintyError("Frozen model artifact must be a mapping.")
    identity = artifact.get("model_identity")
    if not isinstance(identity, dict):
        raise MeasurementUncertaintyError(
            "Frozen model artifact has no model_identity."
        )
    if (
        identity.get("name") != FROZEN_MODEL_NAME
        or identity.get("version") != FROZEN_MODEL_VERSION
    ):
        raise MeasurementUncertaintyError(
            "Measurement uncertainty experiment requires frozen "
            f"{FROZEN_MODEL_NAME} {FROZEN_MODEL_VERSION}; got {identity!r}."
        )
    _frozen_model_info(artifact)
    return artifact


def _frozen_model_info(model_artifact: dict[str, Any]) -> dict[str, Any]:
    models = model_artifact.get("models")
    if not isinstance(models, dict):
        raise MeasurementUncertaintyError(
            "Frozen model artifact has no models mapping."
        )
    model_info = models.get(PRODUCT_MODEL_NAME)
    if not isinstance(model_info, dict):
        raise MeasurementUncertaintyError(
            f"Frozen model artifact has no {PRODUCT_MODEL_NAME!r} model."
        )
    if model_info.get("lr1_model") is None or model_info.get("final_model") is None:
        raise MeasurementUncertaintyError(
            "Frozen model artifact is missing fitted LR1 or LR2."
        )
    thresholds = model_info.get("thresholds")
    if not isinstance(thresholds, dict) or "threshold_target" not in thresholds:
        raise MeasurementUncertaintyError(
            "Frozen model artifact has no threshold_target."
        )
    return model_info


def _verify_model_data_lineage(
    model_artifact: dict[str, Any], data_version: dict[str, Any]
) -> None:
    model_data_version = (
        model_artifact.get("reproducibility", {})
        .get("source_h5", {})
        .get("data_version")
    )
    if not isinstance(model_data_version, dict):
        raise MeasurementUncertaintyError(
            "Frozen model artifact has no DVC source-data lineage."
        )
    required = (
        "contract",
        "system",
        "dataset_id",
        "pointer_path",
        "hash_algorithm",
        "hash",
        "size_bytes",
        "input_h5_sha256",
    )
    mismatches = {
        key: {"model": model_data_version.get(key), "input": data_version.get(key)}
        for key in required
        if model_data_version.get(key) != data_version.get(key)
    }
    if mismatches:
        raise MeasurementUncertaintyError(
            f"DVC-verified input H5 does not match frozen model lineage: {mismatches}."
        )


def _experimental_preprocessing_config(
    model_artifact: dict[str, Any],
    *,
    input_h5_path: Path,
    output_joblib_path: Path,
    data_version: dict[str, Any],
) -> dict[str, Any]:
    raw_yaml = model_artifact.get("historical_preprocessing_yaml")
    if not isinstance(raw_yaml, str):
        raise MeasurementUncertaintyError(
            "Frozen model artifact has no historical_preprocessing_yaml."
        )
    config = yaml.safe_load(raw_yaml)
    if not isinstance(config, dict):
        raise MeasurementUncertaintyError(
            "Frozen training preprocessing is not a YAML mapping."
        )
    out = deepcopy(config)
    io = _mapping(out, "io")
    io["input_h5_path"] = str(input_h5_path)
    io["output_joblib_path"] = str(output_joblib_path)
    _mapping(out, "data_version")["pointer_path"] = str(data_version["pointer_path"])
    normalization = _mapping(out, "normalization")
    normalization["save_initial_data"] = True
    metadata = _mapping(out, "metadata")
    columns = metadata.get("output_columns")
    if not isinstance(columns, list) or not all(
        isinstance(value, str) for value in columns
    ):
        raise MeasurementUncertaintyError(
            "Frozen prediction preprocessing has invalid output columns."
        )
    experiment_columns = (
        RAW_PROFILE_COLUMN,
        PROFILE_SIGMA_COLUMN,
        "measurement_data",
        "faulty_pixel_mask",
        "ponifile",
        "interpolation_q_range",
        "calibration_session_uid",
    )
    for column in experiment_columns:
        if column not in columns:
            columns.append(column)
    for step in _mapping(out, "pipeline").get("steps", []):
        if isinstance(step, dict) and step.get("name") == "normalization":
            params = step.setdefault("params", {})
            if not isinstance(params, dict):
                raise MeasurementUncertaintyError(
                    "Frozen normalization step has invalid params."
                )
            params["save_initial_data"] = True
            break
    else:
        raise MeasurementUncertaintyError(
            "Frozen preprocessing has no normalization step."
        )
    return out


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Measurement uncertainty config is unavailable: {path}"
        )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise MeasurementUncertaintyError(
            "Measurement uncertainty config must be a YAML mapping."
        )
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    contract = config.get("contract")
    if contract == LEGACY_MEASUREMENT_UNCERTAINTY_CONTRACT:
        legacy = deepcopy(config)
        legacy["contract"] = MEASUREMENT_UNCERTAINTY_CONTRACT
        legacy["covariance_monte_carlo"] = legacy.pop("profile_monte_carlo")
        legacy["covariance_model"] = {
            "estimation_draws": 3,
            "explained_variance": 0.95,
            "max_rank": 1,
            "minimum_diagonal_variance": 0.0,
            "transfer_assumption": (
                "pooled_detector_mc_correlation_with_measurement_specific_pyfai_sigma"
            ),
            "provisional_gates": {
                "deterministic_parity_atol": 1.0e-12,
                "threshold_crossing_agreement_min": 0.95,
                "interval_width_ratio_min": 0.8,
                "interval_width_ratio_max": 1.25,
                "interval_endpoint_convergence_max": 0.005,
            },
        }
        _validate_config(legacy)
        return
    allowed = {
        "contract",
        "experiment",
        "input",
        "data_version",
        "targets",
        "covariance_monte_carlo",
        "covariance_model",
        "detector_reference",
        "polar_cake",
        "mlflow",
        "output",
    }
    _exact_fields(config, allowed, "experiment config")
    if config["contract"] != MEASUREMENT_UNCERTAINTY_CONTRACT:
        raise MeasurementUncertaintyError(
            f"Unsupported measurement uncertainty contract: {config['contract']!r}."
        )
    experiment = _mapping(config, "experiment")
    if experiment.get("model_name") != FROZEN_MODEL_NAME:
        raise MeasurementUncertaintyError(
            "experiment.model_name must select the frozen Aramina model."
        )
    if experiment.get("model_version") != FROZEN_MODEL_VERSION:
        raise MeasurementUncertaintyError(
            "experiment.model_version must be 0.2.14-beta."
        )
    _required_text(experiment, "name", "experiment")
    input_config = _mapping(config, "input")
    _required_text(input_config, "input_h5_path", "input")
    _required_text(input_config, "model_joblib_path", "input")
    data_version = _mapping(config, "data_version")
    if (
        data_version.get("contract") != DVC_DATA_CONTRACT
        or data_version.get("system") != "dvc"
    ):
        raise MeasurementUncertaintyError(
            "Experiment data_version must use the Aramina DVC contract."
        )
    for key in ("dataset_id", "dvc_version", "pointer_path"):
        _required_text(data_version, key, "data_version")
    if not str(data_version["pointer_path"]).endswith(".dvc"):
        raise MeasurementUncertaintyError("data_version.pointer_path must end in .dvc.")

    targets = _mapping(config, "targets")
    _exact_fields(targets, {"mode", "selected"}, "targets")
    if targets["mode"] not in {"all_training_target_cases", "selected"}:
        raise MeasurementUncertaintyError(
            "targets.mode must be all_training_target_cases or selected."
        )
    selected = targets["selected"]
    if not isinstance(selected, list):
        raise MeasurementUncertaintyError("targets.selected must be a list.")
    if targets["mode"] == "selected" and not selected:
        raise MeasurementUncertaintyError(
            "targets.selected cannot be empty in selected mode."
        )
    for item in selected:
        _target_from_mapping(item)

    profile_mc = _mapping(config, "covariance_monte_carlo")
    _exact_fields(
        profile_mc,
        {
            "draws",
            "seed",
            "interval_quantiles",
            "convergence_draws",
            "convergence_tolerance",
        },
        "covariance_monte_carlo",
    )
    _bounded_integer(
        profile_mc["draws"], 10, 10_000, "covariance_monte_carlo.draws"
    )
    _seed(profile_mc["seed"], "covariance_monte_carlo.seed")
    quantiles = profile_mc["interval_quantiles"]
    if not isinstance(quantiles, list):
        raise MeasurementUncertaintyError(
            "covariance_monte_carlo.interval_quantiles must be a list."
        )
    _validate_quantiles(tuple(quantiles))
    convergence_draws = profile_mc["convergence_draws"]
    valid_convergence_draws = (
        isinstance(convergence_draws, list)
        and bool(convergence_draws)
        and all(
            not isinstance(value, bool) and isinstance(value, int)
            for value in convergence_draws
        )
    )
    if (
        not valid_convergence_draws
        or convergence_draws != sorted(set(convergence_draws))
        or convergence_draws[-1] != profile_mc["draws"]
        or convergence_draws[0] < 10
    ):
        raise MeasurementUncertaintyError(
            "covariance_monte_carlo.convergence_draws must be increasing integers, "
            "start at 10 or more, and end at covariance_monte_carlo.draws."
        )
    tolerance = profile_mc["convergence_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, int | float)
        or not 0.0 < float(tolerance) <= 0.1
    ):
        raise MeasurementUncertaintyError(
            "covariance_monte_carlo.convergence_tolerance must be inside (0, 0.1]."
        )

    covariance_model = _mapping(config, "covariance_model")
    _exact_fields(
        covariance_model,
        {
            "estimation_draws",
            "explained_variance",
            "max_rank",
            "minimum_diagonal_variance",
            "transfer_assumption",
            "provisional_gates",
        },
        "covariance_model",
    )
    _bounded_integer(
        covariance_model["estimation_draws"],
        3,
        1_000,
        "covariance_model.estimation_draws",
    )
    if not 0.5 <= float(covariance_model["explained_variance"]) < 1.0:
        raise MeasurementUncertaintyError(
            "covariance_model.explained_variance must be in [0.5, 1)."
        )
    _bounded_integer(covariance_model["max_rank"], 1, 100, "covariance_model.max_rank")
    minimum_diagonal = covariance_model["minimum_diagonal_variance"]
    if (
        isinstance(minimum_diagonal, bool)
        or not isinstance(minimum_diagonal, int | float)
        or float(minimum_diagonal) < 0.0
    ):
        raise MeasurementUncertaintyError(
            "covariance_model.minimum_diagonal_variance must be nonnegative."
        )
    if covariance_model.get("transfer_assumption") != (
        "pooled_detector_mc_correlation_with_measurement_specific_pyfai_sigma"
    ):
        raise MeasurementUncertaintyError(
            "covariance_model.transfer_assumption must record the approved "
            "heteroscedastic covariance transfer."
        )
    gates = _mapping(covariance_model, "provisional_gates")
    _exact_fields(
        gates,
        {
            "deterministic_parity_atol",
            "threshold_crossing_agreement_min",
            "interval_width_ratio_min",
            "interval_width_ratio_max",
            "interval_endpoint_convergence_max",
        },
        "covariance_model.provisional_gates",
    )
    for key in gates:
        value = gates[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MeasurementUncertaintyError(
                f"covariance_model.provisional_gates.{key} must be numeric."
            )

    detector_reference = _mapping(config, "detector_reference")
    expected_reference = {
        "draws",
        "seed",
        "snr_quantiles",
        "max_cases_per_stratum",
        "calibration_nuisance",
        "masked_pixel_policy",
    }
    _exact_fields(detector_reference, expected_reference, "detector_reference")
    _bounded_integer(detector_reference["draws"], 10, 1_000, "detector_reference.draws")
    _seed(detector_reference["seed"], "detector_reference.seed")
    calibration_nuisance = _mapping(detector_reference, "calibration_nuisance")
    _exact_fields(
        calibration_nuisance,
        {"enabled", "exclusion_reason", "future_scope"},
        "detector_reference.calibration_nuisance",
    )
    if calibration_nuisance.get("enabled") is not False:
        raise MeasurementUncertaintyError(
            "Calibration nuisance sampling must remain disabled without covariance."
        )
    if calibration_nuisance.get("exclusion_reason") != "covariance_unavailable":
        raise MeasurementUncertaintyError(
            "Calibration nuisance exclusion must record covariance_unavailable."
        )
    if calibration_nuisance.get("future_scope") != "shared_by_calib_session":
        raise MeasurementUncertaintyError(
            "Future calibration nuisance scope must be shared_by_calib_session."
        )
    if detector_reference.get("masked_pixel_policy") != MASKED_PIXEL_POLICY:
        raise MeasurementUncertaintyError(
            f"detector_reference.masked_pixel_policy must be {MASKED_PIXEL_POLICY}."
        )
    for key, lower, upper in (
        ("snr_quantiles", 2, 20),
        ("max_cases_per_stratum", 1, 10),
    ):
        value = detector_reference.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not lower <= value <= upper
        ):
            raise MeasurementUncertaintyError(
                f"detector_reference.{key} must be an integer from {lower} to {upper}."
            )

    polar = _mapping(config, "polar_cake")
    _exact_fields(
        polar,
        {"n_q", "n_chi", "parity_max_relative_rmse"},
        "polar_cake",
    )
    _bounded_integer(polar["n_q"], 16, 2_048, "polar_cake.n_q")
    _bounded_integer(polar["n_chi"], 4, 360, "polar_cake.n_chi")
    tolerance = polar["parity_max_relative_rmse"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, int | float)
        or not 0.0 < float(tolerance) <= 1.0
    ):
        raise MeasurementUncertaintyError(
            "polar_cake.parity_max_relative_rmse must be inside (0, 1]."
        )

    mlflow = _mapping(config, "mlflow")
    if mlflow.get("enabled") is not True:
        raise MeasurementUncertaintyError(
            "Measurement uncertainty experiments require mlflow.enabled=true."
        )
    for key in ("tracking_uri", "experiment_name"):
        _required_text(mlflow, key, "mlflow")
    output = _mapping(config, "output")
    _required_text(output, "folder", "output")


def _targets_for_run(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> list[TargetRequest]:
    target_config = config["targets"]
    if target_config["mode"] == "selected":
        return [_target_from_mapping(item) for item in target_config["selected"]]
    cases = target_breast_cases(
        dataframe,
        group_column="patientId",
        side_column="side",
        label_column="product_status_group",
        biopsy_column="biopsy",
    )
    return [
        TargetRequest(
            patient_id=str(row.patientId),
            target_side=str(row.target_side).lower(),
        )
        for row in cases.itertuples(index=False)
    ]


def _target_from_mapping(value: dict[str, Any]) -> TargetRequest:
    if not isinstance(value, dict):
        raise MeasurementUncertaintyError("Each selected target must be a mapping.")
    _exact_fields(value, {"patient_id", "target_side"}, "selected target")
    patient_id = _required_text(value, "patient_id", "target")
    if patient_id.startswith("REPLACE_WITH_"):
        raise MeasurementUncertaintyError(
            "target.patient_id still contains the configuration template placeholder."
        )
    target_side = _required_text(value, "target_side", "target").lower()
    if target_side not in {"left", "right"}:
        raise MeasurementUncertaintyError("target.target_side must be left or right.")
    return TargetRequest(patient_id=patient_id, target_side=target_side)


def _validate_quantiles(quantiles: tuple[float, ...]) -> None:
    if len(quantiles) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in quantiles
    ):
        raise MeasurementUncertaintyError(
            "interval_quantiles must contain three numeric values."
        )
    values = tuple(float(value) for value in quantiles)
    if not 0.0 < values[0] < values[1] < values[2] < 1.0:
        raise MeasurementUncertaintyError(
            "interval_quantiles must be strictly increasing inside (0, 1)."
        )


def _create_run_folder(config: dict[str, Any], config_path: Path) -> Path:
    output_folder = _resolve_required_path(
        _mapping(config, "output"), "folder", config_path
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = output_folder / f"measurement_uncertainty_{timestamp}"
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise MeasurementUncertaintyError(
            f"Experiment run folder already exists: {folder}"
        ) from exc
    return folder


def _write_audit_artifacts(
    *,
    run_folder: Path,
    config: dict[str, Any],
    effective_preprocessing: dict[str, Any],
    data_version: dict[str, Any],
    config_path: Path,
    lineage: dict[str, Any],
    detector_reference: pd.DataFrame,
    detector_collection: DetectorReferenceCollection,
    covariance_model: LowRankCovarianceModel,
    covariance_summaries: pd.DataFrame,
    covariance_draws: pd.DataFrame,
    polar_manifest: pd.DataFrame,
    method_comparison: pd.DataFrame,
    convergence: pd.DataFrame,
) -> None:
    (run_folder / "effective_experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_folder / "effective_training_preprocessing.yaml").write_text(
        yaml.safe_dump(effective_preprocessing, sort_keys=False), encoding="utf-8"
    )
    pointer_path = resolve_config_path(data_version["pointer_path"], config_path)
    (run_folder / "dvc_data_pointer.dvc").write_bytes(pointer_path.read_bytes())
    _write_json(run_folder / "lineage.json", lineage)
    detector_reference.to_csv(run_folder / "detector_reference_subset.csv", index=False)
    covariance_summaries.to_csv(
        run_folder / "covariance_measurement_uncertainty_summary.csv", index=False
    )
    covariance_draws.to_csv(
        run_folder / "covariance_measurement_uncertainty_draws.csv", index=False
    )
    detector_collection.summaries.to_csv(
        run_folder / "detector_measurement_uncertainty_summary.csv", index=False
    )
    detector_collection.score_draws.to_csv(
        run_folder / "detector_measurement_uncertainty_draws.csv", index=False
    )
    detector_manifest = _write_detector_profile_fit_draws(
        run_folder / "detector_profile_fit_draws.npz",
        detector_collection.profile_draws,
        detector_collection.measurement_manifest,
    )
    detector_manifest.to_csv(
        run_folder / "detector_profile_fit_manifest.csv", index=False
    )
    write_low_rank_covariance(
        str(run_folder / "covariance_model.npz"), covariance_model
    )
    covariance_diagnostics_frame(covariance_model).to_csv(
        run_folder / "covariance_diagnostics.csv", index=False
    )
    covariance_eigen_spectrum_frame(covariance_model).to_csv(
        run_folder / "covariance_eigen_spectrum.csv", index=False
    )
    polar_manifest.to_csv(run_folder / "polar_cake_manifest.csv", index=False)
    method_comparison.to_csv(
        run_folder / "covariance_detector_reference_comparison.csv",
        index=False,
    )
    convergence.to_csv(
        run_folder / "covariance_monte_carlo_convergence.csv",
        index=False,
    )
    gates = config["covariance_model"]["provisional_gates"]
    final_convergence = convergence[convergence["draws"] == convergence["draws"].max()]
    endpoint_changes = final_convergence[["abs_delta_low", "abs_delta_high"]]
    median_width_ratio = float(
        method_comparison["covariance_to_detector_width_ratio"].median()
    )
    _write_json(
        run_folder / "run_manifest.json",
        {
            "contract": MEASUREMENT_UNCERTAINTY_CONTRACT,
            "covariance_included_uncertainty_sources": list(
                COVARIANCE_INCLUDED_SOURCES
            ),
            "covariance_excluded_uncertainty_sources": list(
                COVARIANCE_EXCLUDED_SOURCES
            ),
            "detector_included_uncertainty_sources": list(DETECTOR_INCLUDED_SOURCES),
            "detector_excluded_uncertainty_sources": list(DETECTOR_EXCLUDED_SOURCES),
            "covariance_transfer_assumption": config["covariance_model"][
                "transfer_assumption"
            ],
            "covariance_patients_scored": int(len(covariance_summaries)),
            "covariance_draw_rows": int(len(covariance_draws)),
            "detector_patients_scored": int(len(detector_collection.summaries)),
            "detector_draw_rows": int(len(detector_collection.score_draws)),
            "detector_profile_fit_measurements": int(len(detector_manifest)),
            "polar_cakes": int(len(polar_manifest)),
            "covariance_detector_reference_cases": int(len(method_comparison)),
            "covariance_detector_threshold_crossing_agreement": float(
                method_comparison["threshold_crossing_agreement"].mean()
            ),
            "covariance_convergence_checkpoints": sorted(
                int(value) for value in convergence["draws"].unique()
            ),
            "provisional_research_gates": {
                **gates,
                "deterministic_parity_pass": True,
                "threshold_crossing_agreement_pass": bool(
                    method_comparison["threshold_crossing_agreement"].mean()
                    >= gates["threshold_crossing_agreement_min"]
                ),
                "interval_width_ratio_median": median_width_ratio,
                "interval_width_ratio_pass": bool(
                    gates["interval_width_ratio_min"]
                    <= median_width_ratio
                    <= gates["interval_width_ratio_max"]
                ),
                "max_interval_endpoint_change": float(
                    endpoint_changes.max(axis=1).max()
                ),
                "interval_endpoint_convergence_pass": bool(
                    endpoint_changes.max(axis=1).max()
                    <= gates["interval_endpoint_convergence_max"]
                ),
            },
            "required_artifacts": list(REQUIRED_ARTIFACTS),
        },
    )


def _write_legacy_audit_artifacts(
    *,
    run_folder: Path,
    config: dict[str, Any],
    effective_preprocessing: dict[str, Any],
    data_version: dict[str, Any],
    config_path: Path,
    lineage: dict[str, Any],
    detector_reference: pd.DataFrame,
    profile_summaries: pd.DataFrame,
    profile_draws: pd.DataFrame,
    detector_summaries: pd.DataFrame,
    detector_draws: pd.DataFrame,
    polar_manifest: pd.DataFrame,
    comparison: pd.DataFrame,
    convergence: pd.DataFrame,
) -> None:
    """Persist v0.1 artifacts under their original filenames and semantics."""
    (run_folder / "effective_experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_folder / "effective_training_preprocessing.yaml").write_text(
        yaml.safe_dump(effective_preprocessing, sort_keys=False), encoding="utf-8"
    )
    pointer_path = resolve_config_path(data_version["pointer_path"], config_path)
    (run_folder / "dvc_data_pointer.dvc").write_bytes(pointer_path.read_bytes())
    _write_json(run_folder / "lineage.json", lineage)
    detector_reference.to_csv(run_folder / "detector_reference_subset.csv", index=False)
    profile_summaries.to_csv(
        run_folder / "profile_measurement_uncertainty_summary.csv", index=False
    )
    profile_draws.to_csv(
        run_folder / "profile_measurement_uncertainty_draws.csv", index=False
    )
    detector_summaries.to_csv(
        run_folder / "detector_measurement_uncertainty_summary.csv", index=False
    )
    detector_draws.to_csv(
        run_folder / "detector_measurement_uncertainty_draws.csv", index=False
    )
    polar_manifest.to_csv(run_folder / "polar_cake_manifest.csv", index=False)
    comparison.to_csv(
        run_folder / "profile_detector_reference_comparison.csv", index=False
    )
    convergence.to_csv(run_folder / "profile_monte_carlo_convergence.csv", index=False)
    _write_json(
        run_folder / "run_manifest.json",
        {
            "contract": LEGACY_MEASUREMENT_UNCERTAINTY_CONTRACT,
            "profile_included_uncertainty_sources": [
                "pyfai_poisson_per_bin_sigma"
            ],
            "profile_excluded_uncertainty_sources": list(
                LEGACY_PROFILE_EXCLUDED_SOURCES
            ),
            "detector_included_uncertainty_sources": list(DETECTOR_INCLUDED_SOURCES),
            "detector_excluded_uncertainty_sources": list(DETECTOR_EXCLUDED_SOURCES),
            "profile_patients_scored": int(len(profile_summaries)),
            "profile_draw_rows": int(len(profile_draws)),
            "detector_patients_scored": int(len(detector_summaries)),
            "detector_draw_rows": int(len(detector_draws)),
            "required_artifacts": list(LEGACY_REQUIRED_ARTIFACTS),
        },
    )


def _write_detector_profile_fit_draws(
    path: Path,
    profile_draws: dict[str, np.ndarray],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Persist detector-MC fit draws without object arrays or pickle."""
    output = manifest.copy(deep=True)
    arrays: dict[str, np.ndarray] = {}
    npz_keys: list[str] = []
    for index, key in enumerate(output["profile_key"].astype(str)):
        npz_key = f"profile_{index:04d}"
        arrays[npz_key] = np.asarray(profile_draws[key], dtype=float)
        npz_keys.append(npz_key)
    np.savez_compressed(path, **arrays)
    output["npz_key"] = npz_keys
    return output


def _log_legacy_mlflow_run(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_folder: Path,
    lineage: dict[str, Any],
    profile_summaries: pd.DataFrame,
    detector_summaries: pd.DataFrame,
    polar_manifest: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Log a v0.1 run without changing its historical artifacts or adapter."""
    tracking_uri = _tracking_uri(str(config["mlflow"]["tracking_uri"]), config_path)
    params = {
        "experiment": config["experiment"],
        "profile_monte_carlo": config["profile_monte_carlo"],
        "detector_reference": config["detector_reference"],
        "polar_cake": config["polar_cake"],
    }
    tags = {
        "product": "aramina",
        "clinical_stage": "research_draft",
        "input_h5_checksum": lineage["data_version"]["input_h5_sha256"],
        "dvc": lineage["data_version"],
        "source_code": lineage["source_code"],
        "legacy_contract": LEGACY_MEASUREMENT_UNCERTAINTY_CONTRACT,
    }
    metrics = {
        "profile.patients_scored": float(len(profile_summaries)),
        "profile.mean_interval_width": float(
            (
                profile_summaries["p_cancer_high"]
                - profile_summaries["p_cancer_low"]
            ).mean()
        ),
        "detector.patients_scored": float(len(detector_summaries)),
        "polar.parity_pass_fraction": float(polar_manifest["parity_pass"].mean()),
        "reference.threshold_crossing_agreement_fraction": float(
            comparison["threshold_crossing_agreement"].mean()
        ),
    }
    run_name = f"{config['experiment']['name']}_{run_folder.name.rsplit('_', 1)[-1]}"
    with MlflowRun(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=str(config["mlflow"]["experiment_name"]),
        run_name=run_name,
        params=params,
        tags=tags,
    ) as run:
        run.log_metrics(metrics)
        run.log_artifact_directory(
            run_folder,
            required_files=LEGACY_REQUIRED_ARTIFACTS,
            artifact_path="measurement_uncertainty",
        )
        run_id = run.run_id
    return {
        "enabled": True,
        "run_id": run_id,
        "status": run.status,
        "tracking_uri": tracking_uri,
    }


def _log_mlflow_run(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_folder: Path,
    lineage: dict[str, Any],
    covariance_model: LowRankCovarianceModel,
    covariance_summaries: pd.DataFrame,
    detector_summaries: pd.DataFrame,
    polar_manifest: pd.DataFrame,
    method_comparison: pd.DataFrame,
    convergence: pd.DataFrame,
) -> dict[str, Any]:
    profile_mc = config["covariance_monte_carlo"]
    params = {
        "experiment": config["experiment"],
        "covariance_monte_carlo": {
            "adapter": COVARIANCE_ADAPTER,
            "draws": profile_mc["draws"],
            "seed": profile_mc["seed"],
            "interval_quantiles": ",".join(
                str(value) for value in profile_mc["interval_quantiles"]
            ),
            "convergence_draws": ",".join(
                str(value) for value in profile_mc["convergence_draws"]
            ),
            "convergence_tolerance": profile_mc["convergence_tolerance"],
        },
        "covariance_model": config["covariance_model"],
        "detector_reference": config["detector_reference"],
        "polar_cake": config["polar_cake"],
        "model": {
            "name": lineage["model"]["identity"]["name"],
            "version": lineage["model"]["identity"]["version"],
            "artifact_sha256": lineage["model"]["sha256"],
        },
    }
    tags = {
        "product": "aramina",
        "clinical_stage": "research_draft",
        "intended_use_id": "aramina_target_breast_biopsy_decision_support_v0_1",
        "input_h5_id": lineage["data_version"]["output_path"],
        "input_h5_checksum": lineage["data_version"]["input_h5_sha256"],
        "dataset_fingerprint": lineage["data_version"]["input_h5_sha256"],
        "dvc": lineage["data_version"],
        "source_code": lineage["source_code"],
        "model": {
            "artifact_sha256": lineage["model"]["sha256"],
            "training_aramina_git_sha": lineage["model"]["training_aramina_git_sha"],
        },
    }
    metric_values = {
        "covariance.patients_scored": float(len(covariance_summaries)),
        "covariance.mean_interval_width": float(
            (
                covariance_summaries["p_cancer_high"]
                - covariance_summaries["p_cancer_low"]
            ).mean()
        ),
        "covariance.threshold_crossing_patients": float(
            covariance_summaries["threshold_crossing"].sum()
        ),
        "detector.patients_scored": float(len(detector_summaries)),
        "detector.mean_interval_width": float(
            (
                detector_summaries["p_cancer_high"] - detector_summaries["p_cancer_low"]
            ).mean()
        ),
        "detector.threshold_crossing_patients": float(
            detector_summaries["threshold_crossing"].sum()
        ),
        "polar.cakes": float(len(polar_manifest)),
        "polar.parity_pass_fraction": float(polar_manifest["parity_pass"].mean()),
        "polar.mean_relative_rmse": float(polar_manifest["relative_rmse"].mean()),
        "reference.median_covariance_to_detector_width_ratio": float(
            method_comparison["covariance_to_detector_width_ratio"].median()
        ),
        "reference.threshold_crossing_agreement_fraction": float(
            method_comparison["threshold_crossing_agreement"].mean()
        ),
    }
    final_convergence = convergence[
        convergence["draws"] == convergence["draws"].max()
    ]
    endpoint_changes = final_convergence[["abs_delta_low", "abs_delta_high"]]
    metric_values.update(
        {
            "convergence.max_abs_interval_endpoint_change": float(
                endpoint_changes.max(axis=1).max()
            ),
            "convergence.median_abs_interval_endpoint_change": float(
                endpoint_changes.max(axis=1).median()
            ),
            "convergence.max_abs_threshold_probability_change": float(
                final_convergence["abs_delta_probability_above_threshold"].max()
            ),
        }
    )
    metric_values.update(
        {
            f"covariance.{key}": float(value)
            for key, value in covariance_model.diagnostics.items()
            if isinstance(value, int | float)
        }
    )
    mlflow_config = config["mlflow"]
    tracking_uri = _tracking_uri(str(mlflow_config["tracking_uri"]), config_path)
    run_name = f"{config['experiment']['name']}_{run_folder.name.rsplit('_', 1)[-1]}"
    with MlflowRun(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=str(mlflow_config["experiment_name"]),
        run_name=run_name,
        params=params,
        tags=tags,
    ) as run:
        run.log_metrics(metric_values)
        run.log_artifact_directory(
            run_folder,
            required_files=REQUIRED_ARTIFACTS,
            artifact_path="measurement_uncertainty",
        )
        run_id = run.run_id
    return {
        "enabled": True,
        "run_id": run_id,
        "status": run.status,
        "tracking_uri": tracking_uri,
    }


def _lineage(
    *,
    model_artifact: dict[str, Any],
    model_path: Path,
    data_version: dict[str, Any],
) -> dict[str, Any]:
    model_reproducibility = model_artifact.get("reproducibility", {})
    training_aramina_sha = (
        model_reproducibility.get("source_code", {}).get("aramina", {}).get("git_sha")
    )
    training_xrd_sha = (
        model_reproducibility.get("source_code", {})
        .get("xrd_preprocessing", {})
        .get("git_commit")
    )
    if not isinstance(training_aramina_sha, str) or len(training_aramina_sha) != 40:
        raise MeasurementUncertaintyError(
            "Frozen model has no valid training Aramina Git SHA."
        )
    if not isinstance(training_xrd_sha, str) or len(training_xrd_sha) != 40:
        raise MeasurementUncertaintyError(
            "Frozen model has no valid training XRD-preprocessing Git SHA."
        )
    return {
        "data_version": data_version,
        "model": {
            "path": str(model_path),
            "sha256": file_sha256(model_path),
            "identity": model_artifact["model_identity"],
            "training_aramina_git_sha": training_aramina_sha,
            "training_xrd_preprocessing_git_sha": training_xrd_sha,
        },
        "source_code": {
            "aramina": {"version": aramina_version(), "git_sha": aramina_git_sha()},
            "xrd_preprocessing": {"git_sha": xrd_preprocessing_git_sha()},
        },
    }


def _resolve_required_path(
    mapping: dict[str, Any], key: str, config_path: Path
) -> Path:
    return resolve_config_path(_required_text(mapping, key, "config"), config_path)


def _tracking_uri(value: str, config_path: Path) -> str:
    prefix = "sqlite:///"
    if not value.startswith(prefix):
        return value
    database_path = value.removeprefix(prefix)
    if database_path.startswith("/"):
        return value
    return f"{prefix}{resolve_config_path(database_path, config_path)}"


def _side_norm(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text.startswith("left") or text == "l":
        return "LEFT"
    if text.startswith("right") or text == "r":
        return "RIGHT"
    return None


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    child = value.get(key)
    if not isinstance(child, dict):
        raise MeasurementUncertaintyError(f"{key} must be a mapping.")
    return child


def _exact_fields(
    mapping: dict[str, Any],
    expected: set[str],
    where: str,
) -> None:
    missing = sorted(expected.difference(mapping))
    unknown = sorted(set(mapping).difference(expected))
    if missing or unknown:
        raise MeasurementUncertaintyError(
            f"{where} fields invalid; missing={missing}, unknown={unknown}."
        )


def _bounded_integer(value: Any, lower: int, upper: int, where: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= upper
    ):
        raise MeasurementUncertaintyError(
            f"{where} must be an integer from {lower} to {upper}."
        )
    return value


def _seed(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise MeasurementUncertaintyError(
            f"{where} must be a non-negative 64-bit integer."
        )
    return value


def _required_text(mapping: dict[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MeasurementUncertaintyError(f"{where}.{key} must be a non-empty string.")
    return value.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
