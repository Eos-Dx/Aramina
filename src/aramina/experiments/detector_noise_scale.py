"""Parallel detector-level photon-noise scale experiment."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import UTC, datetime
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from ..config_paths import resolve_config_path
from ..data_versioning import verify_dvc_input
from ..mlflow_tracking import MlflowRun
from ..patient_features import target_breast_cases
from ..pipelines import run_preprocessing_pipeline
from ..prediction_scoring import _prediction_columns
from ..runtime_identity import file_sha256
from .detector_uncertainty import (
    _centered_poisson_observation,
    integrate_scaled_detector_profile_cube,
    normalize_profile,
    prepare_detector_integration,
)
from .measurement_uncertainty import (
    FROZEN_MODEL_NAME,
    MeasurementUncertaintyError,
    TargetRequest,
    _experimental_preprocessing_config,
    _frozen_model_info,
    _lineage,
    _load_frozen_model,
    _score_patient_frame,
    _verify_model_data_lineage,
)


DETECTOR_NOISE_SCALE_CONTRACT = "aramina_detector_noise_scale_v0_1"
RESULT_CONTRACT = "aramina_detector_noise_scale_results_v0_1"


def run_detector_noise_scale_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a bounded parallel pilot or full detector-noise experiment."""
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
        raise MeasurementUncertaintyError(
            "Detector-noise experiment requires DVC data lineage."
        )
    model_artifact = _load_frozen_model(model_path)
    _verify_model_data_lineage(model_artifact, data_version)
    run_folder = _create_run_folder(config, path)
    effective_preprocessing = _experimental_preprocessing_config(
        model_artifact,
        input_h5_path=input_h5_path,
        output_joblib_path=run_folder / "preprocessed_detector_noise_scale.joblib",
        data_version=data_version,
    )
    started = perf_counter()
    dataframe = run_preprocessing_pipeline(
        input_h5_path,
        effective_preprocessing,
        verbose=verbose,
    )
    selected_cases = select_target_cases(dataframe, config["targets"])
    selected_patient_ids = selected_cases["patient_id"].tolist()
    selected_frame = dataframe[
        dataframe["patientId"].astype(str).isin(selected_patient_ids)
    ].copy()
    parity = detector_integration_parity_check(
        selected_frame,
        measurement_count=int(
            config["validation"]["integration_smoke_measurements"]
        ),
        tolerance=float(config["validation"]["parity_tolerance"]),
    )
    if not bool(parity["parity_pass"].all()):
        raise MeasurementUncertaintyError(
            "Cached detector integration does not match product preprocessing."
        )

    worker_count = int(config["execution"]["workers"])
    assignments = balanced_patient_assignments(
        selected_frame,
        selected_cases,
        worker_count=worker_count,
    )
    shard_folder = run_folder / "worker_inputs"
    result_folder = run_folder / "integrated_profiles"
    shard_folder.mkdir()
    result_folder.mkdir()
    worker_specs: list[dict[str, Any]] = []
    for worker_index, patient_ids in enumerate(assignments):
        worker_cases = selected_cases[
            selected_cases["patient_id"].isin(patient_ids)
        ].reset_index(drop=True)
        worker_frame = selected_frame[
            selected_frame["patientId"].astype(str).isin(patient_ids)
        ].reset_index(drop=True)
        shard_path = shard_folder / f"worker_{worker_index:02d}_input.joblib"
        result_path = (
            result_folder / f"worker_{worker_index:02d}_integrated_profiles.joblib"
        )
        joblib.dump(
            {"dataframe": worker_frame, "cases": worker_cases},
            shard_path,
            compress=0,
        )
        worker_specs.append(
            {
                "worker_index": worker_index,
                "input_path": str(shard_path),
                "result_path": str(result_path),
                "checkpoint_folder": str(
                    result_folder / f"worker_{worker_index:02d}_checkpoints"
                ),
                "model_path": str(model_path),
                "draws": int(config["monte_carlo"]["draws"]),
                "noise_scales": [
                    float(value) for value in config["monte_carlo"]["noise_scales"]
                ],
                "seed": int(config["monte_carlo"]["seed"]),
            }
        )

    context = mp.get_context("spawn")
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        futures = {pool.submit(_integrate_worker, spec): spec for spec in worker_specs}
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda item: int(item["worker_index"]))
    for spec in worker_specs:
        Path(spec["input_path"]).unlink(missing_ok=True)
    shard_folder.rmdir()

    integrated_paths = [Path(item["result_path"]) for item in completed]
    integrated_index = [
        {
            "worker_index": int(item["worker_index"]),
            "path": str(result_path),
            "size_bytes": result_path.stat().st_size,
            "sha256": file_sha256(result_path),
        }
        for item, result_path in zip(completed, integrated_paths, strict=True)
    ]
    score_after_integration = bool(config["execution"]["score_after_integration"])
    if score_after_integration:
        draws = score_integrated_profile_joblibs(
            integrated_paths,
            model_path=model_path,
        )
        summaries = summarize_case_intervals(
            draws,
            interval_quantiles=tuple(config["monte_carlo"]["interval_quantiles"]),
        )
        metrics = summarize_metric_distributions(draws)
    else:
        draws = pd.DataFrame()
        summaries = pd.DataFrame()
        metrics = pd.DataFrame()
    aggregate = {
        "contract": RESULT_CONTRACT,
        "config": config,
        "selected_cases": selected_cases,
        "parity": parity,
        "draws": draws,
        "case_summaries": summaries,
        "metric_distributions": metrics,
        "worker_results": completed,
        "integrated_profile_index": integrated_index,
    }
    aggregate_path = run_folder / "detector_noise_scale_results.joblib"
    joblib.dump(aggregate, aggregate_path, compress=3)
    (run_folder / "integrated_profile_index.json").write_text(
        json.dumps(integrated_index, indent=2, sort_keys=True), encoding="utf-8"
    )
    selected_cases.to_csv(run_folder / "selected_cases.csv", index=False)
    parity.to_csv(run_folder / "integration_parity.csv", index=False)
    if score_after_integration:
        summaries.to_csv(run_folder / "case_uncertainty_summary.csv", index=False)
        metrics.to_csv(run_folder / "metric_uncertainty_summary.csv", index=False)
    (run_folder / "effective_experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_folder / "effective_training_preprocessing.yaml").write_text(
        yaml.safe_dump(effective_preprocessing, sort_keys=False), encoding="utf-8"
    )
    pointer_path = resolve_config_path(data_version["pointer_path"], path)
    (run_folder / "dvc_data_pointer.dvc").write_bytes(pointer_path.read_bytes())
    lineage = _lineage(
        model_artifact=model_artifact,
        model_path=model_path,
        data_version=data_version,
    )
    (run_folder / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True), encoding="utf-8"
    )
    elapsed = perf_counter() - started
    manifest = {
        "contract": DETECTOR_NOISE_SCALE_CONTRACT,
        "status": "complete",
        "patients": int(selected_cases["patient_id"].nunique()),
        "target_cases": int(len(selected_cases)),
        "measurements": int(len(selected_frame)),
        "draws_per_noise_scale": int(config["monte_carlo"]["draws"]),
        "noise_scales": config["monte_carlo"]["noise_scales"],
        "workers": worker_count,
        "calibration_sessions": int(
            selected_frame["calibration_session_uid"].astype(str).nunique()
        ),
        "integrated_profile_joblibs": len(completed),
        "integrated_profile_bytes": int(
            sum(item["size_bytes"] for item in integrated_index)
        ),
        "score_after_integration": score_after_integration,
        "elapsed_seconds": elapsed,
    }
    (run_folder / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    mlflow_result = _log_mlflow(
        config=config,
        config_path=path,
        run_folder=run_folder,
        manifest=manifest,
        summaries=summaries,
        metrics=metrics,
    )
    return {
        "run_folder": run_folder,
        "aggregate_path": aggregate_path,
        "manifest": manifest,
        "mlflow": mlflow_result,
    }


def select_target_cases(
    dataframe: pd.DataFrame,
    target_config: dict[str, Any],
) -> pd.DataFrame:
    """Select either the bounded pilot or every historical training target case."""
    mode = target_config.get("mode")
    if mode == "balanced_unique_patient_pilot":
        return select_balanced_unique_patient_cases(
            dataframe,
            patient_count=int(target_config["patient_count"]),
        )
    if mode != "all_training_target_cases":
        raise ValueError(f"Unsupported targets.mode: {mode!r}.")
    cases = target_breast_cases(
        dataframe,
        group_column="patientId",
        side_column="side",
        label_column="product_status_group",
        biopsy_column="biopsy",
    ).rename(columns={"patientId": "patient_id"})
    return cases.assign(
        target_side=cases["target_side"].astype(str).str.lower(),
        target_label=np.where(cases["label"].eq(1), "CANCER", "BENIGN"),
    ).sort_values("target_case_id").reset_index(drop=True)


def select_balanced_unique_patient_cases(
    dataframe: pd.DataFrame,
    *,
    patient_count: int,
) -> pd.DataFrame:
    """Select a deterministic class-balanced pilot with unique patients."""
    if patient_count < 2 or patient_count % 2:
        raise ValueError("pilot.patients must be an even integer of at least 2.")
    cases = target_breast_cases(
        dataframe,
        group_column="patientId",
        side_column="side",
        label_column="product_status_group",
        biopsy_column="biopsy",
    ).rename(columns={"patientId": "patient_id"})
    selected: list[pd.DataFrame] = []
    selected_patient_ids: set[str] = set()
    per_class = patient_count // 2
    for label in (0, 1):
        candidates = (
            cases[
                cases["label"].eq(label)
                & ~cases["patient_id"].astype(str).isin(selected_patient_ids)
            ]
            .sort_values("target_case_id")
            .drop_duplicates("patient_id")
            .head(per_class)
        )
        if len(candidates) != per_class:
            raise ValueError(f"Not enough unique patients for label {label}.")
        selected.append(candidates)
        selected_patient_ids.update(candidates["patient_id"].astype(str))
    out = pd.concat(selected, ignore_index=True).sort_values("target_case_id")
    if out["patient_id"].astype(str).nunique() != patient_count:
        raise RuntimeError("Balanced pilot selection did not preserve unique patients.")
    return out.assign(
        target_side=out["target_side"].astype(str).str.lower(),
        target_label=np.where(out["label"].eq(1), "CANCER", "BENIGN"),
    ).reset_index(drop=True)


def balanced_patient_assignments(
    dataframe: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    worker_count: int,
) -> list[list[str]]:
    """Greedily balance workers by measurement count."""
    if worker_count < 1:
        raise ValueError("execution.workers must be positive.")
    counts = (
        dataframe.groupby(dataframe["patientId"].astype(str)).size().to_dict()
    )
    patient_ids = cases["patient_id"].astype(str).drop_duplicates().tolist()
    assignments = [[] for _ in range(min(worker_count, len(patient_ids)))]
    loads = [0] * len(assignments)
    for patient_id in sorted(patient_ids, key=lambda value: (-counts[value], value)):
        worker = int(np.argmin(loads))
        assignments[worker].append(patient_id)
        loads[worker] += int(counts[patient_id])
    return assignments


def detector_integration_parity_check(
    dataframe: pd.DataFrame,
    *,
    measurement_count: int,
    tolerance: float,
) -> pd.DataFrame:
    """Compare cached contexts with frozen preprocessed profiles."""
    rows: list[dict[str, Any]] = []
    for row_index, row in dataframe.head(measurement_count).iterrows():
        observed = _centered_poisson_observation(
            row["measurement_data"], row["faulty_pixel_mask"]
        )
        context = prepare_detector_integration(row)
        q, intensity = context.integrate(observed)
        actual = normalize_profile(q, intensity, q_range=(6.7, 7.1))
        expected = np.asarray(row["radial_profile_data"], dtype=float)
        maximum = float(np.max(np.abs(actual - expected)))
        rows.append(
            {
                "row_index": int(row_index),
                "patient_id": str(row["patientId"]),
                "calibration_session_uid": str(row["calibration_session_uid"]),
                "max_abs_difference": maximum,
                "parity_tolerance": float(tolerance),
                "parity_pass": bool(maximum <= tolerance),
            }
        )
    if len(rows) != measurement_count:
        raise ValueError("Pilot dataframe has fewer rows than requested parity checks.")
    return pd.DataFrame(rows)


def summarize_case_intervals(
    draws: pd.DataFrame,
    *,
    interval_quantiles: tuple[float, float, float],
) -> pd.DataFrame:
    """Summarize patient score intervals for each photon-noise scale."""
    rows: list[dict[str, Any]] = []
    keys = ["target_case_id", "patient_id", "target_side", "target_label", "label"]
    for values, group in draws.groupby([*keys, "noise_scale"], sort=True):
        *case_values, noise_scale = values
        probabilities = group.sort_values("draw_index")["p_cancer"].to_numpy()
        low, median, high = np.quantile(probabilities, interval_quantiles)
        threshold = float(group["decision_threshold"].iloc[0])
        rows.append(
            {
                **dict(zip(keys, case_values, strict=True)),
                "noise_scale": float(noise_scale),
                "draws": int(len(probabilities)),
                "decision_threshold": threshold,
                "p_cancer_mean": float(np.mean(probabilities)),
                "p_cancer_sd": float(np.std(probabilities, ddof=1)),
                "p_cancer_low": float(low),
                "p_cancer_median": float(median),
                "p_cancer_high": float(high),
                "interval_width": float(high - low),
                "probability_above_threshold": float(
                    np.mean(probabilities >= threshold)
                ),
                "threshold_crossing": bool(low <= threshold <= high),
            }
        )
    return pd.DataFrame(rows)


def summarize_metric_distributions(draws: pd.DataFrame) -> pd.DataFrame:
    """Summarize pilot sensitivity and specificity across nested draw indices."""
    records: list[dict[str, float]] = []
    for (noise_scale, draw_index), group in draws.groupby(
        ["noise_scale", "draw_index"], sort=True
    ):
        predicted = group["p_cancer"].to_numpy() >= group[
            "decision_threshold"
        ].to_numpy()
        labels = group["label"].to_numpy(dtype=int)
        cancer = labels == 1
        benign = labels == 0
        records.append(
            {
                "noise_scale": float(noise_scale),
                "draw_index": int(draw_index),
                "sensitivity": float(np.mean(predicted[cancer])),
                "specificity": float(np.mean(~predicted[benign])),
            }
        )
    per_draw = pd.DataFrame(records)
    rows: list[dict[str, float]] = []
    for noise_scale, group in per_draw.groupby("noise_scale", sort=True):
        row: dict[str, float] = {"noise_scale": float(noise_scale)}
        for metric in ("sensitivity", "specificity"):
            values = group[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_low"] = float(np.quantile(values, 0.025))
            row[f"{metric}_high"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _integrate_worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Integrate one static patient shard and persist only 1D profile draws."""
    started = perf_counter()
    payload = joblib.load(spec["input_path"])
    dataframe = payload["dataframe"]
    cases = payload["cases"]
    patients: dict[str, dict[str, Any]] = {}
    checkpoint_folder = Path(spec["checkpoint_folder"])
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    for patient_number, (patient_id, patient_cases) in enumerate(
        cases.groupby("patient_id", sort=True)
    ):
        checkpoint_path = checkpoint_folder / f"{patient_id}.joblib"
        if checkpoint_path.is_file():
            patients[str(patient_id)] = joblib.load(checkpoint_path)
            continue
        patient_frame = dataframe[
            dataframe["patientId"].astype(str) == str(patient_id)
        ].reset_index(drop=True)
        random_states = [
            int(
                np.random.SeedSequence(
                    (spec["seed"], spec["worker_index"], patient_number, scale_index)
                ).generate_state(1)[0]
            )
            for scale_index in range(len(spec["noise_scales"]))
        ]
        q_values, profiles = integrate_scaled_detector_profile_cube(
            patient_frame,
            draws=int(spec["draws"]),
            noise_scales=spec["noise_scales"],
            random_states=random_states,
            normalization_q_range=(6.7, 7.1),
        )
        heavy_columns = {
            "measurement_data",
            "faulty_pixel_mask",
            "ponifile",
            "radial_profile_data_raw",
            "radial_profile_sigma",
        }
        metadata = patient_frame.drop(
            columns=[column for column in heavy_columns if column in patient_frame]
        ).copy()
        patient_result = {
            "metadata": metadata,
            "cases": patient_cases.reset_index(drop=True),
            "q_values": q_values,
            "profiles": profiles,
        }
        joblib.dump(patient_result, checkpoint_path, compress=0)
        patients[str(patient_id)] = patient_result
    result_path = Path(spec["result_path"])
    result = {
        "contract": RESULT_CONTRACT,
        "worker_index": int(spec["worker_index"]),
        "noise_scales": np.asarray(spec["noise_scales"], dtype=float),
        "draws": int(spec["draws"]),
        "patients_data": patients,
        "elapsed_seconds": perf_counter() - started,
        "patients": int(cases["patient_id"].nunique()),
        "target_cases": int(len(cases)),
        "measurements": int(len(dataframe)),
    }
    joblib.dump(result, result_path, compress=0)
    for checkpoint_path in checkpoint_folder.glob("*.joblib"):
        checkpoint_path.unlink()
    checkpoint_folder.rmdir()
    return {
        "worker_index": result["worker_index"],
        "result_path": str(result_path),
        "elapsed_seconds": result["elapsed_seconds"],
        "patients": result["patients"],
        "target_cases": result["target_cases"],
        "measurements": result["measurements"],
    }


def score_integrated_profile_joblibs(
    paths: list[Path],
    *,
    model_path: Path,
) -> pd.DataFrame:
    """Score persisted 1D profile draws through the frozen product model."""
    model_artifact = _load_frozen_model(model_path)
    model_info = _frozen_model_info(model_artifact)
    columns = _prediction_columns(model_artifact)
    records: list[dict[str, Any]] = []
    for path in paths:
        integrated = joblib.load(path)
        noise_scales = np.asarray(integrated["noise_scales"], dtype=float)
        for patient_id, patient_data in integrated["patients_data"].items():
            metadata = patient_data["metadata"].reset_index(drop=True)
            cases = patient_data["cases"]
            q_values = np.asarray(patient_data["q_values"], dtype=float)
            profiles = np.asarray(patient_data["profiles"], dtype=float)
            baseline_frame = metadata.copy()
            baseline_frame["q_range"] = [
                np.asarray(value, dtype=float) for value in metadata["q_range"]
            ]
            baseline_frame["radial_profile_data"] = [
                np.asarray(value, dtype=float)
                for value in metadata["radial_profile_data"]
            ]
            targets = [
                TargetRequest(str(patient_id), str(row.target_side))
                for row in cases.itertuples(index=False)
            ]
            deterministic = {
                target.target_case_id: _score_patient_frame(
                    baseline_frame,
                    model_info=model_info,
                    model_name=FROZEN_MODEL_NAME,
                    target=target,
                    columns=columns,
                )
                for target in targets
            }
            labels = {
                f"{patient_id}::{str(row.target_side)}": int(row.label)
                for row in cases.itertuples(index=False)
            }
            for scale_index, noise_scale in enumerate(noise_scales):
                for draw_index in range(int(integrated["draws"])):
                    sampled_frame = metadata.copy()
                    sampled_frame["q_range"] = [
                        q_values[index] for index in range(len(metadata))
                    ]
                    sampled_frame["radial_profile_data"] = [
                        profiles[scale_index, draw_index, index]
                        for index in range(len(metadata))
                    ]
                    for target in targets:
                        score = _score_patient_frame(
                            sampled_frame,
                            model_info=model_info,
                            model_name=FROZEN_MODEL_NAME,
                            target=target,
                            columns=columns,
                        )
                        baseline = deterministic[target.target_case_id]
                        if score["threshold"] != baseline["threshold"]:
                            raise MeasurementUncertaintyError(
                                "Integrated profile draw changed frozen threshold."
                            )
                        label = labels[target.target_case_id]
                        records.append(
                            {
                                "target_case_id": target.target_case_id,
                                "patient_id": target.patient_id,
                                "target_side": target.target_side,
                                "target_label": (
                                    "CANCER" if label == 1 else "BENIGN"
                                ),
                                "label": label,
                                "noise_scale": float(noise_scale),
                                "draw_index": int(draw_index),
                                "p_cancer": float(score["p_cancer"]),
                                "decision_threshold": float(score["threshold"]),
                                "deterministic_p_cancer": float(
                                    baseline["p_cancer"]
                                ),
                                "model_route": (
                                    score["model_route"] or "single_model"
                                ),
                            }
                        )
                    del sampled_frame
    return pd.DataFrame(records)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("contract") != DETECTOR_NOISE_SCALE_CONTRACT:
        raise ValueError(f"Expected contract {DETECTOR_NOISE_SCALE_CONTRACT!r}.")
    required = {
        "input",
        "data_version",
        "targets",
        "validation",
        "execution",
        "monte_carlo",
        "mlflow",
        "output",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Detector-noise config is missing sections: {missing}.")
    draws = config["monte_carlo"].get("draws")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("monte_carlo.draws must be an integer of at least 2.")
    scales = config["monte_carlo"].get("noise_scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("monte_carlo.noise_scales must be a non-empty list.")
    normalized = [float(value) for value in scales]
    if any(not np.isfinite(value) or value <= 0.0 for value in normalized):
        raise ValueError("Every noise scale must be finite and positive.")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Noise scales must be unique.")
    mode = config["targets"].get("mode")
    if mode not in {"balanced_unique_patient_pilot", "all_training_target_cases"}:
        raise ValueError("Unsupported targets.mode.")
    if mode == "balanced_unique_patient_pilot":
        patient_count = config["targets"].get("patient_count")
        if (
            isinstance(patient_count, bool)
            or not isinstance(patient_count, int)
            or patient_count < 2
            or patient_count % 2
        ):
            raise ValueError("targets.patient_count must be an even integer >= 2.")
    workers = config["execution"].get("workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("execution.workers must be a positive integer.")
    if not isinstance(config["execution"].get("score_after_integration"), bool):
        raise ValueError("execution.score_after_integration must be boolean.")
    quantiles = tuple(config["monte_carlo"].get("interval_quantiles", []))
    if len(quantiles) != 3 or not 0 < quantiles[0] < quantiles[1] < quantiles[2] < 1:
        raise ValueError("interval_quantiles must contain three increasing values.")
    convergence = config["monte_carlo"].get("convergence_draws")
    if convergence is not None:
        if (
            not isinstance(convergence, list)
            or not convergence
            or convergence != sorted(set(convergence))
            or convergence[-1] != draws
        ):
            raise ValueError(
                "convergence_draws must be unique, increasing, and end at draws."
            )
    return deepcopy(config)


def _create_run_folder(config: dict[str, Any], config_path: Path) -> Path:
    root = _resolve_path(config["output"]["folder"], config_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"detector_noise_scale_{timestamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _resolve_path(value: str, config_path: Path) -> Path:
    return resolve_config_path(str(value), config_path)


def _log_mlflow(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_folder: Path,
    manifest: dict[str, Any],
    summaries: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, Any]:
    mlflow = config["mlflow"]
    tracking_uri = str(mlflow["tracking_uri"])
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        database = tracking_uri.removeprefix("sqlite:///")
        tracking_uri = f"sqlite:///{_resolve_path(database, config_path)}"
    run = MlflowRun(
        enabled=bool(mlflow["enabled"]),
        tracking_uri=tracking_uri,
        experiment_name=str(mlflow["experiment_name"]),
        run_name=f"detector-noise-scale-{run_folder.name}",
        params={
            "contract": DETECTOR_NOISE_SCALE_CONTRACT,
            "patients": manifest["patients"],
            "measurements": manifest["measurements"],
            "draws": manifest["draws_per_noise_scale"],
            "workers": manifest["workers"],
            "noise_scales": ",".join(map(str, manifest["noise_scales"])),
        },
        tags={
            "product": "aramina",
            "clinical_stage": "research draft",
            "uncertainty_scope": "photon_statistics_only",
        },
    )
    required = [
        "effective_experiment_config.yaml",
        "effective_training_preprocessing.yaml",
        "dvc_data_pointer.dvc",
        "selected_cases.csv",
        "integration_parity.csv",
        "integrated_profile_index.json",
        "detector_noise_scale_results.joblib",
        "lineage.json",
        "run_manifest.json",
    ]
    if not summaries.empty:
        required.extend(
            ["case_uncertainty_summary.csv", "metric_uncertainty_summary.csv"]
        )
    audit_folder = run_folder / "mlflow_audit"
    audit_folder.mkdir(exist_ok=True)
    for relative in required:
        shutil.copy2(run_folder / relative, audit_folder / relative)
    with run:
        if not metrics.empty:
            for row in metrics.itertuples(index=False):
                run.log_metrics(
                    {
                        "sensitivity_mean": row.sensitivity_mean,
                        "specificity_mean": row.specificity_mean,
                        "median_interval_width": float(
                            summaries.loc[
                                summaries["noise_scale"].eq(row.noise_scale),
                                "interval_width",
                            ].median()
                        ),
                    },
                    step=int(round(float(row.noise_scale) * 100)),
                )
        run.log_artifact_directory(audit_folder, required_files=required)
    return {
        "enabled": run.enabled,
        "run_id": run.run_id,
        "status": run.status,
        "tracking_uri": tracking_uri,
    }
