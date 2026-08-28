"""Direct detector Monte Carlo with bounded geometry sensitivity scenarios."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from ..data_versioning import verify_dvc_input
from ..mlflow_tracking import MlflowRun
from ..pipelines import run_preprocessing_pipeline
from ..runtime_identity import file_sha256
from .detector_noise_scale import (
    detector_integration_parity_check,
    select_balanced_unique_patient_cases,
    select_target_cases,
)
from .detector_uncertainty import (
    MASK_COLUMN,
    RAW_FRAME_COLUMN,
    PreparedDetectorIntegration,
    _centered_poisson_observation,
    normalize_profile,
)
from .measurement_uncertainty import (
    FROZEN_MODEL_NAME,
    MeasurementUncertaintyError,
    TargetRequest,
    _experimental_preprocessing_config,
    _frozen_model_info,
    _score_patient_frame,
    _verify_model_data_lineage,
)
from .vectorized_frozen_scorer import score_frozen_aramina_0_2_15_cube


CONTRACT = "aramina_joint_measurement_uncertainty_v0_1"
RESULT_CONTRACT = "aramina_joint_measurement_uncertainty_results_v0_1"
COMPONENTS = ("photon", "thickness", "beam_center", "detector_distance")


@dataclass(frozen=True)
class Scenario:
    """One explicitly named subset of uncertainty sources."""

    name: str
    photon: bool
    thickness: bool
    beam_center: bool
    detector_distance: bool
    beam_center_scale: float = 1.0
    detector_distance_scale: float = 1.0

    @property
    def geometry_enabled(self) -> bool:
        return self.thickness or self.beam_center or self.detector_distance


@dataclass(frozen=True)
class NuisanceDraws:
    """Common random numbers with explicit measurement/session scope."""

    thickness_delta_mm: np.ndarray
    beam_center_row_delta_px: np.ndarray
    beam_center_col_delta_px: np.ndarray
    detector_distance_delta_mm: np.ndarray
    photon_measurement_seeds: np.ndarray


def run_joint_measurement_uncertainty_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run bounded component and joint sensitivity scenarios on a frozen model."""
    started = perf_counter()
    path = Path(config_path).expanduser().resolve()
    config = _load_config(path)
    input_h5_path = _resolve_path(config["input"]["input_h5_path"], path)
    model_path = _resolve_path(config["input"]["model_joblib_path"], path)
    data_version = verify_dvc_input(
        {"data_version": config["data_version"]},
        config_path=path,
        input_h5_path=input_h5_path,
    )
    if data_version is None:
        raise MeasurementUncertaintyError("Joint uncertainty requires DVC lineage.")
    model_artifact = _load_model(
        model_path,
        expected_version=str(config["experiment"]["model_version"]),
    )
    _verify_model_data_lineage(model_artifact, data_version)
    model_info = _frozen_model_info(model_artifact)
    run_folder = _create_run_folder(config, path)
    effective_preprocessing = _experimental_preprocessing_config(
        model_artifact,
        input_h5_path=input_h5_path,
        output_joblib_path=run_folder / "preprocessed_joint_uncertainty.joblib",
        data_version=data_version,
    )
    cached_frame_value = config["input"].get(
        "preprocessed_detector_frame_joblib_path"
    )
    if cached_frame_value:
        dataframe = joblib.load(_resolve_path(cached_frame_value, path))
        if not isinstance(dataframe, pd.DataFrame):
            raise MeasurementUncertaintyError(
                "Cached detector frame must be a pandas DataFrame."
            )
    else:
        dataframe = run_preprocessing_pipeline(
            input_h5_path,
            effective_preprocessing,
            verbose=verbose,
        )
    selected_cases = _select_cases(dataframe, config["targets"])
    patient_ids = selected_cases["patient_id"].astype(str).unique().tolist()
    selected_frame = dataframe[
        dataframe["patientId"].astype(str).isin(patient_ids)
    ].reset_index(drop=True)
    joblib.dump(
        selected_frame,
        run_folder / "selected_detector_frame.joblib",
        compress=0,
    )
    parity = detector_integration_parity_check(
        selected_frame,
        measurement_count=int(config["validation"]["parity_measurements"]),
        tolerance=float(config["validation"]["pyfai_parity_tolerance"]),
    )
    if not bool(parity["parity_pass"].all()):
        raise MeasurementUncertaintyError(
            "Prepared pyFAI integration does not reproduce product profiles."
        )

    scenarios = tuple(_scenario(value) for value in config["scenarios"])
    draws = int(config["monte_carlo"]["draws"])
    quantiles = tuple(float(value) for value in config["monte_carlo"]["quantiles"])
    probability_path = run_folder / "p_cancer_probability_cube.npy"
    probabilities = np.lib.format.open_memmap(
        probability_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected_cases), len(scenarios), draws),
    )
    case_index = {
        str(case_id): index
        for index, case_id in enumerate(selected_cases["target_case_id"])
    }
    deterministic = np.full(len(selected_cases), np.nan, dtype=float)
    thresholds = np.full(len(selected_cases), np.nan, dtype=float)
    metal_parity_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []

    for patient_id in patient_ids:
        patient_frame = selected_frame[
            selected_frame["patientId"].astype(str).eq(patient_id)
        ].reset_index(drop=True)
        patient_cases = selected_cases[
            selected_cases["patient_id"].astype(str).eq(patient_id)
        ].reset_index(drop=True)
        nuisance = sample_nuisance_draws(
            patient_frame,
            draws=draws,
            seed=int(config["monte_carlo"]["seed"]),
            thickness_config=config["nuisance"]["sample_thickness"],
            beam_center_config=config["nuisance"]["beam_center"],
            detector_distance_config=config["nuisance"]["detector_distance"],
        )
        baseline_scores = _score_cases(
            patient_frame,
            patient_cases,
            model_info=model_info,
        )
        for case_id, score in baseline_scores.items():
            index = case_index[case_id]
            deterministic[index] = score["p_cancer"]
            thresholds[index] = score["threshold"]

        for scenario_index, scenario in enumerate(scenarios):
            patient_probabilities, patient_parity, patient_geometry = (
                _run_patient_scenario(
                    patient_frame,
                    patient_cases,
                    model_artifact=model_artifact,
                    model_info=model_info,
                    scenario=scenario,
                    nuisance=nuisance,
                    draws=draws,
                    draw_chunk_size=int(config["execution"]["draw_chunk_size"]),
                    profile_batch_size=int(
                        config["execution"]["profile_batch_size"]
                    ),
                    geometry_audit_draws=int(
                        config["execution"]["geometry_audit_draws"]
                    ),
                    normalization_q_range=tuple(
                        float(value)
                        for value in config["integration"]["normalization_q_range"]
                    ),
                    metal_parity_tolerance=float(
                        config["validation"]["metal_parity_tolerance"]
                    ),
                    metal_p_cancer_parity_tolerance=float(
                        config["validation"][
                            "metal_p_cancer_parity_tolerance"
                        ]
                    ),
                    random_seed=int(config["monte_carlo"]["seed"]),
                )
            )
            metal_parity_rows.extend(patient_parity)
            geometry_rows.extend(patient_geometry)
            for case_id, values in patient_probabilities.items():
                probabilities[case_index[case_id], scenario_index] = values
        probabilities.flush()

    case_table = selected_cases.copy()
    case_table["deterministic_p_cancer"] = deterministic
    case_table["decision_threshold"] = thresholds
    summaries = summarize_case_uncertainty(
        probabilities,
        case_table,
        scenarios=scenarios,
        quantiles=quantiles,
    )
    metadata_qc = thickness_metadata_audit(selected_frame)
    artifacts = _write_artifacts(
        run_folder,
        config=config,
        config_path=path,
        effective_preprocessing=effective_preprocessing,
        data_version=data_version,
        model_path=model_path,
        selected_cases=case_table,
        summaries=summaries,
        parity=parity,
        metal_parity=pd.DataFrame(metal_parity_rows),
        geometry_draws=pd.DataFrame(geometry_rows),
        metadata_qc=metadata_qc,
        scenarios=scenarios,
        elapsed_seconds=perf_counter() - started,
    )
    mlflow = _log_mlflow(
        run_folder,
        config=config,
        config_path=path,
        summaries=summaries,
        manifest=artifacts["manifest"],
    )
    return {
        "run_folder": str(run_folder),
        "summary_path": str(run_folder / "case_uncertainty_summary.csv"),
        "probability_path": str(probability_path),
        "patients": len(patient_ids),
        "target_cases": len(selected_cases),
        "mlflow": mlflow,
        "manifest": artifacts["manifest"],
    }


def sample_nuisance_draws(
    patient_frame: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    thickness_config: dict[str, Any],
    beam_center_config: dict[str, Any],
    detector_distance_config: dict[str, Any],
) -> NuisanceDraws:
    """Sample bounded effects while preserving visit and calibration scopes."""
    measurements = len(patient_frame)
    if draws < 1 or measurements < 1:
        raise ValueError("draws and measurements must be positive.")
    patient_ids = patient_frame["patientId"].astype(str).unique()
    if len(patient_ids) != 1:
        raise ValueError("Nuisance sampling requires exactly one patient.")
    patient_id = str(patient_ids[0])
    thickness = pd.to_numeric(
        patient_frame["sample_thickness_mm"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(thickness).all() or np.any(thickness <= 0.0):
        raise ValueError("Sample thickness must be finite and positive.")
    threshold = float(thickness_config["thin_max_mm"])
    bounds = np.where(
        thickness <= threshold,
        float(thickness_config["thin_half_width_mm"]),
        float(thickness_config["thick_half_width_mm"]),
    )
    correlation = str(thickness_config["correlation"])
    thickness_rng = _keyed_rng(seed, "thickness", patient_id)
    if correlation == "visit_shared":
        thickness_unit = thickness_rng.uniform(-1.0, 1.0, size=(draws, 1))
    elif correlation == "measurement_independent":
        thickness_unit = np.column_stack(
            [
                _keyed_rng(seed, "thickness", patient_id, _measurement_key(row)).uniform(
                    -1.0, 1.0, size=draws
                )
                for _, row in patient_frame.iterrows()
            ]
        )
    else:
        raise ValueError("Thickness correlation must be visit_shared or independent.")
    thickness_delta = np.broadcast_to(
        thickness_unit, (draws, measurements)
    ).copy() * bounds

    session_ids = patient_frame["calibration_session_uid"].astype(str).to_numpy()
    sessions, session_index = np.unique(session_ids, return_inverse=True)
    session_count = len(sessions)
    radius = float(beam_center_config["radius_px"])
    row_by_session = np.empty((draws, session_count), dtype=float)
    col_by_session = np.empty((draws, session_count), dtype=float)
    distance_by_session = np.empty((draws, session_count), dtype=float)
    for index, session_id in enumerate(sessions):
        geometry_rng = _keyed_rng(seed, "geometry", str(session_id))
        radial = radius * np.sqrt(geometry_rng.uniform(size=draws))
        angle = geometry_rng.uniform(0.0, 2.0 * np.pi, size=draws)
        row_by_session[:, index] = radial * np.sin(angle)
        col_by_session[:, index] = radial * np.cos(angle)
        distance_by_session[:, index] = geometry_rng.uniform(
            -float(detector_distance_config["half_width_mm"]),
            float(detector_distance_config["half_width_mm"]),
            size=draws,
        )
    photon_seeds = np.column_stack(
        [
            _keyed_rng(seed, "photon", _measurement_key(row)).integers(
                0,
                np.iinfo(np.uint64).max,
                size=draws,
                dtype=np.uint64,
            )
            for _, row in patient_frame.iterrows()
        ]
    )
    return NuisanceDraws(
        thickness_delta_mm=thickness_delta,
        beam_center_row_delta_px=row_by_session[:, session_index],
        beam_center_col_delta_px=col_by_session[:, session_index],
        detector_distance_delta_mm=distance_by_session[:, session_index],
        photon_measurement_seeds=photon_seeds,
    )


def effective_detector_distance_m(
    poni_distance_m: float,
    sample_thickness_mm: float,
    calibrant_thickness_mm: float,
    *,
    sample_thickness_delta_mm: float = 0.0,
    detector_distance_delta_mm: float = 0.0,
) -> float:
    """Apply product thickness correction plus bounded PONI-distance error."""
    return float(poni_distance_m) + float(detector_distance_delta_mm) * 1e-3 - 0.5 * (
        float(sample_thickness_mm)
        + float(sample_thickness_delta_mm)
        - float(calibrant_thickness_mm)
    ) * 1e-3


def summarize_case_uncertainty(
    probabilities: np.ndarray,
    case_table: pd.DataFrame,
    *,
    scenarios: tuple[Scenario, ...],
    quantiles: tuple[float, float, float],
) -> pd.DataFrame:
    """Summarize scenario quantiles and threshold stability per target case."""
    values = np.asarray(probabilities)
    if values.shape[:2] != (len(case_table), len(scenarios)):
        raise ValueError("Probability cube does not match cases and scenarios.")
    rows: list[dict[str, Any]] = []
    for case_index, case in case_table.reset_index(drop=True).iterrows():
        threshold = float(case["decision_threshold"])
        deterministic = float(case["deterministic_p_cancer"])
        baseline_class = deterministic >= threshold
        for scenario_index, scenario in enumerate(scenarios):
            draws = values[case_index, scenario_index].astype(float)
            lower, median, upper = np.quantile(draws, quantiles)
            rows.append(
                {
                    "target_case_id": case["target_case_id"],
                    "patient_id": case["patient_id"],
                    "target_side": case["target_side"],
                    "label": int(case["label"]),
                    "scenario": scenario.name,
                    "draws": len(draws),
                    "deterministic_p_cancer": deterministic,
                    "decision_threshold": threshold,
                    "p_cancer_p025": lower,
                    "p_cancer_p50": median,
                    "p_cancer_p975": upper,
                    "interval_width": upper - lower,
                    "probability_at_or_above_threshold": float(
                        np.mean(draws >= threshold)
                    ),
                    "class_flip_probability": float(
                        np.mean((draws >= threshold) != baseline_class)
                    ),
                    "threshold_crossing": bool(lower < threshold <= upper),
                }
            )
    return pd.DataFrame(rows)


def thickness_metadata_audit(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return missing and high thickness values without silent correction."""
    thickness = pd.to_numeric(dataframe["sample_thickness_mm"], errors="coerce")
    status = np.where(
        thickness.isna(),
        "missing_excluded_by_product_gate",
        np.where(thickness > 100.0, "review_above_100_mm", "within_audit_range"),
    )
    columns = [
        column
        for column in (
            "patientId",
            "specimenId",
            "side",
            "position",
            "sample_thickness_mm",
            "calibrant_thickness_mm",
            "calibration_session_uid",
            "measurement_data_source",
        )
        if column in dataframe
    ]
    out = dataframe[columns].copy()
    out["thickness_metadata_qc_status"] = status
    return out[out["thickness_metadata_qc_status"] != "within_audit_range"].copy()


def _run_patient_scenario(
    patient_frame: pd.DataFrame,
    patient_cases: pd.DataFrame,
    *,
    model_artifact: dict[str, Any],
    model_info: dict[str, Any],
    scenario: Scenario,
    nuisance: NuisanceDraws,
    draws: int,
    draw_chunk_size: int,
    profile_batch_size: int,
    geometry_audit_draws: int,
    normalization_q_range: tuple[float, float],
    metal_parity_tolerance: float,
    metal_p_cancer_parity_tolerance: float,
    random_seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    case_values = {
        str(case_id): np.empty(draws, dtype=np.float32)
        for case_id in patient_cases["target_case_id"]
    }
    parity_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for start in range(0, draws, draw_chunk_size):
        stop = min(draws, start + draw_chunk_size)
        profiles, metal_nominal, expected, q_values, parity, geometry = (
            _metal_profile_chunk(
            patient_frame,
            scenario=scenario,
            nuisance=nuisance,
            start=start,
            stop=stop,
            profile_batch_size=profile_batch_size,
            geometry_audit_draws=geometry_audit_draws,
            normalization_q_range=normalization_q_range,
            parity_tolerance=metal_parity_tolerance,
            random_seed=random_seed,
        )
        )
        parity_rows.extend(parity)
        geometry_rows.extend(geometry)
        score_kwargs = {
            "patient_manifest": patient_frame,
            "q_grid": q_values,
            "target_manifest": patient_cases,
            "model_artifact": model_artifact,
        }
        scores = score_frozen_aramina_0_2_15_cube(profiles, **score_kwargs)
        metal_scores = score_frozen_aramina_0_2_15_cube(
            metal_nominal, **score_kwargs
        )
        expected_scores = score_frozen_aramina_0_2_15_cube(expected, **score_kwargs)
        maximum_score_error = float(
            np.max(np.abs(metal_scores.p_cancer - expected_scores.p_cancer))
        )
        for target_index, case_id in enumerate(scores.target_case_ids):
            case_values[case_id][start:stop] = scores.p_cancer[:, target_index]
        parity[-1]["maximum_absolute_p_cancer_error"] = maximum_score_error
        parity[-1]["p_cancer_parity_tolerance"] = (
            metal_p_cancer_parity_tolerance
        )
        parity[-1]["p_cancer_parity_pass"] = bool(
            maximum_score_error <= metal_p_cancer_parity_tolerance
        )
        if maximum_score_error > metal_p_cancer_parity_tolerance:
            raise MeasurementUncertaintyError(
                "Metal p_cancer parity exceeds configured tolerance: "
                f"{maximum_score_error:.8g} > "
                f"{metal_p_cancer_parity_tolerance:.8g}."
            )
    return case_values, parity_rows, geometry_rows


def _metal_profile_chunk(
    patient_frame: pd.DataFrame,
    *,
    scenario: Scenario,
    nuisance: NuisanceDraws,
    start: int,
    stop: int,
    profile_batch_size: int,
    geometry_audit_draws: int,
    normalization_q_range: tuple[float, float],
    parity_tolerance: float,
    random_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    try:
        from xrdanalysis.direct_monte_carlo import prepare_native_plan
        from xrdanalysis.direct_monte_carlo_metal import prepare_metal_plan
        from xrdanalysis.direct_monte_carlo_metal_session import (
            GroupedPersistentMetalMonteCarlo,
        )
    except ImportError as error:
        raise RuntimeError(
            "Joint uncertainty requires xrd-analysis direct-MC sources."
        ) from error

    images: list[np.ndarray] = []
    plans: list[Any] = []
    q_rows: list[np.ndarray] = []
    expected_rows: list[np.ndarray] = []
    measurement_seeds: list[int] = []
    geometry_rows: list[dict[str, Any]] = []
    measurements = len(patient_frame)
    for draw_index in range(start, stop):
        for measurement_index, (_, row) in enumerate(patient_frame.iterrows()):
            image = _centered_poisson_observation(
                row[RAW_FRAME_COLUMN], row[MASK_COLUMN]
            )
            thickness_delta = (
                float(nuisance.thickness_delta_mm[draw_index, measurement_index])
                if scenario.thickness
                else 0.0
            )
            row_delta = (
                float(
                    nuisance.beam_center_row_delta_px[
                        draw_index, measurement_index
                    ]
                )
                * scenario.beam_center_scale
                if scenario.beam_center
                else 0.0
            )
            col_delta = (
                float(
                    nuisance.beam_center_col_delta_px[
                        draw_index, measurement_index
                    ]
                )
                * scenario.beam_center_scale
                if scenario.beam_center
                else 0.0
            )
            distance_delta = (
                float(
                    nuisance.detector_distance_delta_mm[
                        draw_index, measurement_index
                    ]
                )
                * scenario.detector_distance_scale
                if scenario.detector_distance
                else 0.0
            )
            context, geometry = _perturbed_context(
                row,
                thickness_delta_mm=thickness_delta,
                center_row_delta_px=row_delta,
                center_col_delta_px=col_delta,
                distance_delta_mm=distance_delta,
            )
            baseline = context.integrator.integrate1d(
                image,
                context.npt,
                radial_range=context.radial_range,
                azimuth_range=context.azimuth_range,
                mask=context.mask,
                error_model="poisson",
            )
            q = np.asarray(baseline.radial, dtype=float)
            intensity = np.asarray(baseline.intensity, dtype=float)
            native_plan = prepare_native_plan(
                context.integrator,
                image.shape,
                normalization_denominators=baseline.sum_normalization,
                q_grid=q,
                q_normalization_band=normalization_q_range,
            )
            images.append(image)
            plans.append(prepare_metal_plan(native_plan))
            q_rows.append(q)
            expected_rows.append(
                normalize_profile(q, intensity, q_range=normalization_q_range)
            )
            measurement_seeds.append(
                int(nuisance.photon_measurement_seeds[draw_index, measurement_index])
            )
            if draw_index < geometry_audit_draws:
                geometry_rows.append(
                    {
                        "patient_id": str(row["patientId"]),
                        "specimen_id": str(row["specimenId"]),
                        "scenario": scenario.name,
                        "draw_index": draw_index,
                        "measurement_index": measurement_index,
                        **geometry,
                    }
                )

    with GroupedPersistentMetalMonteCarlo(
        plans,
        images,
        measurement_seeds=measurement_seeds,
        scale_capacity=1,
        profile_batch_size=profile_batch_size,
    ) as session:
        integrated = session.integrate()
        metal_nominal = integrated.copy()
        parity_errors = np.max(
            np.abs(integrated - np.asarray(expected_rows)), axis=1
        )
        if float(np.max(parity_errors)) > parity_tolerance:
            raise MeasurementUncertaintyError(
                "Metal integration parity exceeds configured tolerance: "
                f"{float(np.max(parity_errors)):.8g} > {parity_tolerance:.8g}."
            )
        if scenario.photon:
            integrated = session.run((1.0,), 1, seed=random_seed)[0, 0]

    parity_rows = [
        {
            "patient_id": str(patient_frame["patientId"].iloc[0]),
            "scenario": scenario.name,
            "draw_start": start,
            "draw_stop": stop,
            "maximum_absolute_error": float(np.max(parity_errors)),
            "parity_tolerance": parity_tolerance,
            "distinct_integration_plans": int(session.group_count),
            "parity_pass": bool(float(np.max(parity_errors)) <= parity_tolerance),
        }
    ]
    q_cube = np.asarray(q_rows).reshape(stop - start, measurements, -1)
    if not np.allclose(q_cube, q_cube[0], rtol=0.0, atol=1e-12):
        raise MeasurementUncertaintyError(
            "Perturbed geometry changed the fixed product q grid."
        )
    profile_cube = np.asarray(integrated).reshape(stop - start, measurements, -1)
    metal_nominal_cube = np.asarray(metal_nominal).reshape(
        stop - start, measurements, -1
    )
    expected_cube = np.asarray(expected_rows).reshape(
        stop - start, measurements, -1
    )
    return (
        profile_cube,
        metal_nominal_cube,
        expected_cube,
        q_cube[0],
        parity_rows,
        geometry_rows,
    )


def _perturbed_context(
    row: pd.Series,
    *,
    thickness_delta_mm: float,
    center_row_delta_px: float,
    center_col_delta_px: float,
    distance_delta_mm: float,
) -> tuple[PreparedDetectorIntegration, dict[str, float]]:
    from xrd_preprocessing.azimuthal import (
        _coerce_integration_mask,
        _integrator_from_poni_text,
    )

    image = np.asarray(row[RAW_FRAME_COLUMN])
    integrator = copy.deepcopy(_integrator_from_poni_text(str(row["ponifile"])))
    base_distance = float(integrator.dist)
    effective_distance = effective_detector_distance_m(
        base_distance,
        float(row["sample_thickness_mm"]),
        float(row["calibrant_thickness_mm"]),
        sample_thickness_delta_mm=thickness_delta_mm,
        detector_distance_delta_mm=distance_delta_mm,
    )
    if effective_distance <= 0.0:
        raise MeasurementUncertaintyError("Perturbed detector distance is non-positive.")
    pixel1 = float(integrator.detector.pixel1)
    pixel2 = float(integrator.detector.pixel2)
    poni1 = float(integrator.poni1) + center_row_delta_px * pixel1
    poni2 = float(integrator.poni2) + center_col_delta_px * pixel2
    integrator.set_dist(effective_distance)
    integrator.set_poni1(poni1)
    integrator.set_poni2(poni2)
    mask = _coerce_integration_mask(row[MASK_COLUMN], image.shape).astype(bool)
    return (
        PreparedDetectorIntegration(
            integrator=integrator,
            mask=mask,
            radial_range=row.get("interpolation_q_range"),
            azimuth_range=row.get("azimuthal_range"),
            npt=100,
        ),
        {
            "sample_thickness_delta_mm": thickness_delta_mm,
            "beam_center_row_delta_px": center_row_delta_px,
            "beam_center_col_delta_px": center_col_delta_px,
            "detector_distance_delta_mm": distance_delta_mm,
            "effective_detector_distance_m": effective_distance,
            "poni1_m": poni1,
            "poni2_m": poni2,
            "pixel1_m": pixel1,
            "pixel2_m": pixel2,
        },
    )


def _score_cases(
    patient_frame: pd.DataFrame,
    patient_cases: pd.DataFrame,
    *,
    model_info: dict[str, Any],
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for case in patient_cases.itertuples(index=False):
        target = TargetRequest(str(case.patient_id), str(case.target_side))
        score = _score_patient_frame(
            patient_frame,
            model_info=model_info,
            model_name=FROZEN_MODEL_NAME,
            target=target,
            columns={
                "profile_column": "radial_profile_data",
                "group_column": "patientId",
                "specimen_column": "specimenId",
                "side_column": "side",
                "q_column": "q_range",
                "age_column": "age",
            },
        )
        scores[str(case.target_case_id)] = {
            "p_cancer": float(score["p_cancer"]),
            "threshold": float(score["threshold"]),
        }
    return scores


def _load_model(path: Path, *, expected_version: str) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise MeasurementUncertaintyError("Model artifact must be a mapping.")
    identity = artifact.get("model_identity", {})
    if identity.get("name") != FROZEN_MODEL_NAME:
        raise MeasurementUncertaintyError("Unexpected model name.")
    if identity.get("version") != expected_version:
        raise MeasurementUncertaintyError(
            f"Expected model {expected_version}, got {identity.get('version')}."
        )
    return artifact


def _select_cases(dataframe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    mode = str(config["mode"])
    if mode == "all_training_target_cases":
        return select_target_cases(dataframe, {"mode": mode})
    if mode == "balanced_unique_patient_pilot":
        return select_balanced_unique_patient_cases(
            dataframe,
            patient_count=int(config["patient_count"]),
        )
    raise ValueError(f"Unsupported target mode: {mode}.")


def _scenario(value: dict[str, Any]) -> Scenario:
    required = {"name", *COMPONENTS}
    optional = {"beam_center_scale", "detector_distance_scale"}
    if not required.issubset(value) or not set(value).issubset(required | optional):
        raise ValueError(
            f"Scenario fields require {sorted(required)} and allow {sorted(optional)}."
        )
    beam_center_scale = float(value.get("beam_center_scale", 1.0))
    detector_distance_scale = float(
        value.get("detector_distance_scale", 1.0)
    )
    if not 0.0 <= beam_center_scale <= 1.0:
        raise ValueError("beam_center_scale must be inside [0, 1].")
    if not 0.0 <= detector_distance_scale <= 1.0:
        raise ValueError("detector_distance_scale must be inside [0, 1].")
    return Scenario(
        name=str(value["name"]),
        photon=bool(value["photon"]),
        thickness=bool(value["thickness"]),
        beam_center=bool(value["beam_center"]),
        detector_distance=bool(value["detector_distance"]),
        beam_center_scale=beam_center_scale,
        detector_distance_scale=detector_distance_scale,
    )


def _keyed_rng(seed: int, *parts: str) -> np.random.Generator:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    words = np.frombuffer(digest, dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(words))


def _measurement_key(row: pd.Series) -> str:
    values = (
        row.get("measurement_data_source"),
        row.get("specimenId"),
        row.get("side"),
        row.get("position"),
        row.get("started_at"),
    )
    return "\x1f".join("" if pd.isna(value) else str(value) for value in values)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("contract") != CONTRACT:
        raise ValueError(f"Config must use {CONTRACT}.")
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("At least one uncertainty scenario is required.")
    parsed = [_scenario(value) for value in scenarios]
    if len({value.name for value in parsed}) != len(parsed):
        raise ValueError("Scenario names must be unique.")
    quantiles = config["monte_carlo"]["quantiles"]
    if len(quantiles) != 3 or not 0 < quantiles[0] < quantiles[1] < quantiles[2] < 1:
        raise ValueError("Monte Carlo quantiles must be ordered inside (0, 1).")
    return config


def _resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parents[2] / path).resolve()


def _create_run_folder(config: dict[str, Any], config_path: Path) -> Path:
    root = _resolve_path(config["output"]["folder"], config_path)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"joint_measurement_uncertainty_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _write_artifacts(
    run_folder: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    effective_preprocessing: dict[str, Any],
    data_version: dict[str, Any],
    model_path: Path,
    selected_cases: pd.DataFrame,
    summaries: pd.DataFrame,
    parity: pd.DataFrame,
    metal_parity: pd.DataFrame,
    geometry_draws: pd.DataFrame,
    metadata_qc: pd.DataFrame,
    scenarios: tuple[Scenario, ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    shutil.copy2(config_path, run_folder / "effective_experiment_config.yaml")
    (run_folder / "effective_training_preprocessing.yaml").write_text(
        yaml.safe_dump(effective_preprocessing, sort_keys=False), encoding="utf-8"
    )
    pointer = _resolve_path(config["data_version"]["pointer_path"], config_path)
    shutil.copy2(pointer, run_folder / "dvc_data_pointer.dvc")
    selected_cases.to_csv(run_folder / "selected_cases.csv", index=False)
    summaries.to_csv(run_folder / "case_uncertainty_summary.csv", index=False)
    parity.to_csv(run_folder / "pyfai_parity.csv", index=False)
    metal_parity.to_csv(run_folder / "metal_parity.csv", index=False)
    geometry_draws.to_csv(run_folder / "geometry_draws.csv", index=False)
    metadata_qc.to_csv(run_folder / "thickness_metadata_qc.csv", index=False)
    lineage = {
        "input_h5_sha256": data_version["input_h5_sha256"],
        "input_h5_dvc_hash": data_version["hash"],
        "model_sha256": file_sha256(model_path),
        "model_path": str(model_path),
        "xrd_analysis_git_sha": config["backend"]["xrd_analysis_git_sha"],
    }
    (run_folder / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "contract": RESULT_CONTRACT,
        "status": "complete",
        "patients": int(selected_cases["patient_id"].nunique()),
        "target_cases": len(selected_cases),
        "draws": int(config["monte_carlo"]["draws"]),
        "scenarios": [value.name for value in scenarios],
        "probability_values": int(
            len(selected_cases)
            * len(scenarios)
            * int(config["monte_carlo"]["draws"])
        ),
        "interval_interpretation": "bounded_scenario_quantiles_not_clinical_ci",
        "elapsed_seconds": elapsed_seconds,
        "lineage": lineage,
    }
    (run_folder / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"manifest": manifest}


def _log_mlflow(
    run_folder: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    summaries: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    tracking_uri = str(config["mlflow"]["tracking_uri"])
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith(
        "sqlite:////"
    ):
        tracking_uri = f"sqlite:///{_resolve_path(tracking_uri[10:], config_path)}"
    run = MlflowRun(
        enabled=bool(config["mlflow"]["enabled"]),
        tracking_uri=tracking_uri,
        experiment_name=str(config["mlflow"]["experiment_name"]),
        run_name=f"joint-measurement-uncertainty-{run_folder.name}",
        params={
            "contract": CONTRACT,
            "model_version": config["experiment"]["model_version"],
            "patients": manifest["patients"],
            "target_cases": manifest["target_cases"],
            "draws": manifest["draws"],
            "scenarios": len(manifest["scenarios"]),
        },
        tags={
            "product": "aramina",
            "clinical_stage": "research draft",
            "uncertainty_scope": "photon_thickness_poni_geometry",
            "interval_interpretation": manifest["interval_interpretation"],
        },
    )
    required = [
        "effective_experiment_config.yaml",
        "effective_training_preprocessing.yaml",
        "dvc_data_pointer.dvc",
        "selected_cases.csv",
        "case_uncertainty_summary.csv",
        "pyfai_parity.csv",
        "metal_parity.csv",
        "geometry_draws.csv",
        "thickness_metadata_qc.csv",
        "lineage.json",
        "run_manifest.json",
        "p_cancer_probability_cube.npy",
    ]
    with run:
        for step, scenario in enumerate(manifest["scenarios"]):
            subset = summaries[summaries["scenario"].eq(scenario)]
            run.log_metrics(
                {
                    "median_interval_width": float(subset["interval_width"].median()),
                    "threshold_crossing_fraction": float(
                        subset["threshold_crossing"].mean()
                    ),
                    "median_class_flip_probability": float(
                        subset["class_flip_probability"].median()
                    ),
                },
                step=step,
            )
        run.log_artifact_directory(run_folder, required_files=required)
    return {
        "enabled": run.enabled,
        "run_id": run.run_id,
        "status": run.status,
        "tracking_uri": tracking_uri,
    }


__all__ = [
    "CONTRACT",
    "NuisanceDraws",
    "Scenario",
    "effective_detector_distance_m",
    "run_joint_measurement_uncertainty_from_config",
    "sample_nuisance_draws",
    "summarize_case_uncertainty",
    "thickness_metadata_audit",
]
