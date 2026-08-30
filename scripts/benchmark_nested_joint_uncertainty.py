"""Benchmark geometry-aware nested Metal photon sampling."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from aramina.experiments.joint_measurement_uncertainty import (
    _frozen_model_info,
    _load_config,
    _load_model,
    _metal_profile_chunk,
    _prepare_patient_metal_context,
    _profile_parity_metrics,
    _score_cases,
    _scenario,
    sample_cohort_nuisance_draws,
    summarize_nested_axis_changes,
    summarize_nested_axis_convergence,
)
from aramina.experiments.vectorized_frozen_scorer import (
    score_frozen_aramina_0_2_15_cube,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "experiments"
    / "config_joint_measurement_uncertainty_nested_v0_1.yaml"
)
DEFAULT_CACHE = (
    ROOT
    / "examples"
    / "outputs"
    / "experiments"
    / "joint_measurement_uncertainty"
    / "joint_measurement_uncertainty_20260829T184329Z"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--geometry-draws", type=int, default=1)
    parser.add_argument("--photon-replicates", type=int, default=50)
    parser.add_argument("--geometry-audit-draws", type=int, default=1)
    parser.add_argument("--scenario", default="joint_10px_10mm")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.resolve().parents[2] / path).resolve()


def main() -> None:
    args = _arguments()
    if (
        args.patients < 1
        or args.geometry_draws < 1
        or args.photon_replicates < 1
        or not 1 <= args.geometry_audit_draws <= args.geometry_draws
    ):
        raise ValueError("Patient and Monte Carlo counts must be positive.")
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    cache_folder = args.cache_folder.expanduser().resolve()
    frame = joblib.load(cache_folder / "selected_detector_frame.joblib")
    cases = joblib.load(cache_folder / "selected_cases_checkpoint.joblib")
    if not isinstance(frame, pd.DataFrame) or not isinstance(cases, pd.DataFrame):
        raise TypeError("Cached detector frame and selected cases must be DataFrames.")

    patient_ids = cases["patient_id"].astype(str).drop_duplicates().tolist()
    patient_ids = patient_ids[: args.patients]
    if len(patient_ids) != args.patients:
        raise ValueError(f"Requested {args.patients} patients; found {len(patient_ids)}.")
    cohort_frame = frame[
        frame["patientId"].astype(str).isin(patient_ids)
    ].reset_index(drop=True)
    scenario_values = {str(value["name"]): value for value in config["scenarios"]}
    if args.scenario not in scenario_values:
        raise ValueError(f"Unknown scenario: {args.scenario}.")
    scenario = _scenario(scenario_values[args.scenario])
    if not scenario.photon:
        raise ValueError("Prepared Metal benchmark requires a photon scenario.")

    model_artifact = _load_model(
        _resolve(config["input"]["model_joblib_path"], config_path),
        expected_version=str(config["experiment"]["model_version"]),
    )
    model_info = _frozen_model_info(model_artifact)
    normalization_q_range = tuple(
        float(value) for value in config["integration"]["normalization_q_range"]
    )
    cohort_nuisance = sample_cohort_nuisance_draws(
        cohort_frame,
        draws=args.geometry_draws,
        seed=args.seed,
        thickness_config=config["nuisance"]["sample_thickness"],
        beam_center_config=config["nuisance"]["beam_center"],
        detector_distance_config=config["nuisance"]["detector_distance"],
    )

    elapsed = 0.0
    profile_maxima: list[float] = []
    profile_p99: list[float] = []
    score_maxima: list[float] = []
    decision_agreement: list[bool] = []
    selected_cases = cases[
        cases["patient_id"].astype(str).isin(patient_ids)
    ].reset_index(drop=True)
    case_index = {
        str(case_id): index
        for index, case_id in enumerate(selected_cases["target_case_id"])
    }
    probability_cube = np.full(
        (
            len(selected_cases),
            1,
            args.geometry_draws * args.photon_replicates,
        ),
        np.nan,
        dtype=np.float32,
    )
    deterministic = np.full(len(selected_cases), np.nan, dtype=float)
    thresholds = np.full(len(selected_cases), np.nan, dtype=float)
    target_cases = 0
    detector_measurements = 0
    for patient_id in patient_ids:
        patient_frame = cohort_frame[
            cohort_frame["patientId"].astype(str).eq(patient_id)
        ].reset_index(drop=True)
        patient_cases = cases[
            cases["patient_id"].astype(str).eq(patient_id)
        ].reset_index(drop=True)
        nuisance = cohort_nuisance.for_frame(patient_frame)
        with _prepare_patient_metal_context(
            patient_frame,
            nuisance=nuisance,
            draw_chunk_size=args.geometry_draws,
            profile_batch_size=int(config["execution"]["profile_batch_size"]),
            normalization_q_range=normalization_q_range,
        ) as context:
            started = perf_counter()
            profiles, metal_audit, expected_audit, q_grid, _ = _metal_profile_chunk(
                patient_frame,
                metal_context=context,
                scenario=scenario,
                nuisance=nuisance,
                start=0,
                stop=args.geometry_draws,
                audit_draw_start=0,
                geometry_audit_draws=args.geometry_audit_draws,
                normalization_q_range=normalization_q_range,
                random_seed=args.seed,
                photon_replicates=args.photon_replicates,
            )
            elapsed += perf_counter() - started
        metrics = _profile_parity_metrics(
            metal_audit,
            expected_audit,
            q_grid,
            draw_start=0,
            maximum_tolerance=1.0,
            p99_tolerance=1.0,
        )
        profile_maxima.append(float(metrics["maximum_absolute_error"]))
        profile_p99.append(float(metrics["p99_absolute_error"]))
        score_kwargs = {
            "patient_manifest": patient_frame,
            "q_grid": q_grid,
            "target_manifest": patient_cases,
            "model_artifact": model_artifact,
        }
        metal_scores = score_frozen_aramina_0_2_15_cube(
            metal_audit, **score_kwargs
        )
        expected_scores = score_frozen_aramina_0_2_15_cube(
            expected_audit, **score_kwargs
        )
        score_maxima.append(
            float(np.max(np.abs(metal_scores.p_cancer - expected_scores.p_cancer)))
        )
        decision_agreement.append(
            bool(
                np.array_equal(
                    metal_scores.p_cancer >= metal_scores.threshold,
                    expected_scores.p_cancer >= expected_scores.threshold,
                )
            )
        )
        full_scores = score_frozen_aramina_0_2_15_cube(
            profiles, **score_kwargs
        )
        baseline_scores = _score_cases(
            patient_frame,
            patient_cases,
            model_info=model_info,
        )
        for target_index, case_id in enumerate(full_scores.target_case_ids):
            output_index = case_index[str(case_id)]
            probability_cube[output_index, 0] = full_scores.p_cancer[
                :, target_index
            ]
            deterministic[output_index] = baseline_scores[str(case_id)]["p_cancer"]
            thresholds[output_index] = baseline_scores[str(case_id)]["threshold"]
        target_cases += len(patient_cases)
        detector_measurements += len(patient_frame)

    photon_profiles = (
        detector_measurements * args.geometry_draws * args.photon_replicates
    )
    if not np.isfinite(probability_cube).all():
        raise RuntimeError("Benchmark did not score every selected target case.")
    case_table = selected_cases.copy()
    case_table["deterministic_p_cancer"] = deterministic
    case_table["decision_threshold"] = thresholds
    photon_prefixes = tuple(
        value
        for value in (
            10,
            20,
            30,
            40,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            1000,
            2000,
        )
        if value <= args.photon_replicates
    )
    nested_convergence = summarize_nested_axis_convergence(
        probability_cube,
        case_table,
        scenarios=(scenario,),
        quantiles=tuple(float(value) for value in config["monte_carlo"]["quantiles"]),
        geometry_draws=args.geometry_draws,
        photon_replicates=args.photon_replicates,
        geometry_prefixes=(args.geometry_draws,),
        photon_prefixes=photon_prefixes,
    )
    nested_changes = summarize_nested_axis_changes(nested_convergence)
    result = {
        "contract": "aramina_geometry_aware_metal_benchmark_v0_4",
        "patients": len(patient_ids),
        "target_cases": target_cases,
        "detector_measurements": detector_measurements,
        "scenario": scenario.name,
        "geometry_scope": "cohort_aligned_by_poni_file",
        "geometry_draws": args.geometry_draws,
        "geometry_audit_draws": args.geometry_audit_draws,
        "photon_replicates_per_geometry": args.photon_replicates,
        "photon_profiles": photon_profiles,
        "elapsed_seconds": elapsed,
        "photon_profiles_per_second": photon_profiles / elapsed,
        "profile_absolute_error_max": max(profile_maxima),
        "profile_absolute_error_p99_max": max(profile_p99),
        "p_cancer_absolute_error_max": max(score_maxima),
        "decision_class_agreement": bool(all(decision_agreement)),
        "photon_prefixes": list(photon_prefixes),
        "created_at": datetime.now(UTC).isoformat(),
    }
    output = args.output or (
        ROOT
        / "examples"
        / "outputs"
        / "benchmarks"
        / "prepared_geometry_metal_10_patients.json"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    nested_convergence.to_csv(
        output.with_name(f"{output.stem}_nested_axis_convergence.csv"),
        index=False,
    )
    nested_changes.to_csv(
        output.with_name(f"{output.stem}_nested_axis_changes.csv"),
        index=False,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
