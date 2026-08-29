"""Benchmark flattened and nested geometry/photon Metal sampling."""

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
    _load_config,
    _load_model,
    _prepare_patient_metal_context,
    _scenario,
    _scenario_geometry_arrays,
    sample_nuisance_draws,
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
    / "joint_measurement_uncertainty_20260829T073337Z"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--geometry-draws", type=int, default=50)
    parser.add_argument("--photon-replicates", type=int, default=5)
    parser.add_argument("--scenario", default="joint_10px_10mm")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.resolve().parents[2] / path).resolve()


def _score(
    profiles: np.ndarray,
    *,
    patient_frame: pd.DataFrame,
    patient_cases: pd.DataFrame,
    q_grid: np.ndarray,
    model_artifact: dict,
):
    return score_frozen_aramina_0_2_15_cube(
        profiles,
        patient_manifest=patient_frame,
        q_grid=q_grid,
        target_manifest=patient_cases,
        model_artifact=model_artifact,
    )


def main() -> None:
    args = _arguments()
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
    scenario_values = {
        str(value["name"]): value for value in config["scenarios"]
    }
    if args.scenario not in scenario_values:
        raise ValueError(f"Unknown scenario: {args.scenario}.")
    scenario = _scenario(scenario_values[args.scenario])
    model_path = _resolve(config["input"]["model_joblib_path"], config_path)
    model_artifact = _load_model(
        model_path,
        expected_version=str(config["experiment"]["model_version"]),
    )
    normalization_q_range = tuple(
        float(value) for value in config["integration"]["normalization_q_range"]
    )
    output_draws = args.geometry_draws * args.photon_replicates

    direct_seconds = 0.0
    nested_seconds = 0.0
    profile_differences: list[np.ndarray] = []
    probability_differences: list[np.ndarray] = []
    quantile_differences: list[np.ndarray] = []
    direct_decisions: list[np.ndarray] = []
    nested_decisions: list[np.ndarray] = []
    target_case_count = 0

    for patient_id in patient_ids:
        patient_frame = frame[
            frame["patientId"].astype(str).eq(patient_id)
        ].reset_index(drop=True)
        patient_cases = cases[
            cases["patient_id"].astype(str).eq(patient_id)
        ].reset_index(drop=True)
        nuisance = sample_nuisance_draws(
            patient_frame,
            draws=args.geometry_draws,
            seed=args.seed,
            thickness_config=config["nuisance"]["sample_thickness"],
            beam_center_config=config["nuisance"]["beam_center"],
            detector_distance_config=config["nuisance"]["detector_distance"],
        )
        with _prepare_patient_metal_context(
            patient_frame,
            nuisance=nuisance,
            draw_chunk_size=args.geometry_draws,
            profile_batch_size=int(config["execution"]["profile_batch_size"]),
            normalization_q_range=normalization_q_range,
        ) as context:
            distance, poni1, poni2 = _scenario_geometry_arrays(
                context,
                scenario,
                nuisance,
                start=0,
                stop=args.geometry_draws,
            )
            context.session.integrate(
                1,
                effective_distance_m=distance[:1],
                poni1_m=poni1[:1],
                poni2_m=poni2[:1],
                draw_chunk_size=1,
            )

            started = perf_counter()
            direct_profiles = context.session.run(
                (1.0,),
                output_draws,
                seed=args.seed,
                draw_offset=0,
                draw_chunk_size=output_draws,
                effective_distance_m=np.repeat(
                    distance, args.photon_replicates, axis=0
                ),
                poni1_m=np.repeat(poni1, args.photon_replicates, axis=0),
                poni2_m=np.repeat(poni2, args.photon_replicates, axis=0),
            )[0]
            direct_seconds += perf_counter() - started

            started = perf_counter()
            nested_profiles = context.session.run_nested(
                (1.0,),
                args.geometry_draws,
                args.photon_replicates,
                seed=args.seed,
                geometry_draw_offset=0,
                photon_draw_offset=0,
                geometry_chunk_size=args.geometry_draws,
                effective_distance_m=distance,
                poni1_m=poni1,
                poni2_m=poni2,
            )[0].reshape(direct_profiles.shape)
            nested_seconds += perf_counter() - started

            direct_scores = _score(
                direct_profiles,
                patient_frame=patient_frame,
                patient_cases=patient_cases,
                q_grid=context.q_grid,
                model_artifact=model_artifact,
            )
            nested_scores = _score(
                nested_profiles,
                patient_frame=patient_frame,
                patient_cases=patient_cases,
                q_grid=context.q_grid,
                model_artifact=model_artifact,
            )

        profile_differences.append(
            np.abs(direct_profiles.astype(float) - nested_profiles.astype(float))
        )
        probability_differences.append(
            np.abs(direct_scores.p_cancer - nested_scores.p_cancer)
        )
        quantile_differences.append(
            np.abs(
                np.quantile(direct_scores.p_cancer, (0.025, 0.5, 0.975), axis=0)
                - np.quantile(
                    nested_scores.p_cancer,
                    (0.025, 0.5, 0.975),
                    axis=0,
                )
            )
        )
        direct_decisions.append(direct_scores.p_cancer >= direct_scores.threshold)
        nested_decisions.append(nested_scores.p_cancer >= nested_scores.threshold)
        target_case_count += len(direct_scores.target_case_ids)

    profile_delta = np.concatenate([value.ravel() for value in profile_differences])
    probability_delta = np.concatenate(
        [value.ravel() for value in probability_differences]
    )
    quantile_delta = np.concatenate([value.ravel() for value in quantile_differences])
    direct_class = np.concatenate([value.ravel() for value in direct_decisions])
    nested_class = np.concatenate([value.ravel() for value in nested_decisions])
    result = {
        "contract": "aramina_nested_joint_uncertainty_benchmark_v0_1",
        "patients": len(patient_ids),
        "target_cases": target_case_count,
        "scenario": scenario.name,
        "geometry_draws": args.geometry_draws,
        "photon_replicates_per_geometry": args.photon_replicates,
        "output_draws_per_case": output_draws,
        "direct_seconds": direct_seconds,
        "nested_seconds": nested_seconds,
        "speedup": direct_seconds / nested_seconds,
        "profile_absolute_error_max": float(profile_delta.max()),
        "profile_absolute_error_p99": float(np.quantile(profile_delta, 0.99)),
        "p_cancer_absolute_error_max": float(probability_delta.max()),
        "p_cancer_absolute_error_p99": float(np.quantile(probability_delta, 0.99)),
        "interval_quantile_absolute_error_max": float(quantile_delta.max()),
        "decision_draw_agreement": float(np.mean(direct_class == nested_class)),
        "created_at": datetime.now(UTC).isoformat(),
    }
    output = args.output
    if output is None:
        output = (
            ROOT
            / "examples"
            / "outputs"
            / "benchmarks"
            / "nested_joint_uncertainty_10_patients.json"
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**result, "output": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
