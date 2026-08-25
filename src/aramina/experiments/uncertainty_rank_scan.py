"""Rank and shrinkage comparison for photon-statistical uncertainty.

This research-only runner reuses one completed detector Monte Carlo reference
run. It changes neither the frozen product score nor report contracts. Every
covariance variant is propagated through the same LR1, aggregation, symmetry,
and LR2 path and compared with the same held-out detector draws.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..config_paths import resolve_config_path
from ..data_versioning import DVC_DATA_CONTRACT, verify_dvc_input
from ..mlflow_tracking import MlflowRun
from ..pipelines import run_preprocessing_pipeline
from ..runtime_identity import file_sha256
from .covariance_uncertainty import (
    LowRankCovarianceModel,
    covariance_diagnostics_frame,
    covariance_eigen_spectrum_frame,
    fit_full_shrinkage_covariance,
    fit_low_rank_covariance,
    write_low_rank_covariance,
)
from .measurement_uncertainty import (
    FROZEN_MODEL_NAME,
    FROZEN_MODEL_VERSION,
    MEASUREMENT_UNCERTAINTY_CONTRACT,
    MeasurementUncertaintyError,
    _experimental_preprocessing_config,
    _lineage,
    _load_frozen_model,
    _targets_for_run,
    _tracking_uri,
    _validate_quantiles,
    _verify_model_data_lineage,
    compare_covariance_detector_reference,
    score_correlated_covariance_uncertainty,
    summarize_profile_monte_carlo_convergence,
)


RANK_SCAN_CONTRACT = "aramina_measurement_uncertainty_rank_scan_v0_1"
EMPIRICAL_ESTIMATOR = "pooled_empirical_correlation_eigendecomposition"
SHRINKAGE_ESTIMATOR = "pooled_standardized_residual_ledoit_wolf"
BASE_REQUIRED_ARTIFACTS = (
    "effective_experiment_config.yaml",
    "effective_training_preprocessing.yaml",
    "dvc_data_pointer.dvc",
    "lineage.json",
    "source_artifact_checksums.json",
    "source_detector_measurement_uncertainty_summary.csv",
    "rank_scan_metrics.csv",
    "rank_scan_case_comparison.csv",
    "rank_scan_covariance_diagnostics.csv",
    "rank_scan_eigen_spectrum.csv",
    "rank_scan_monte_carlo_convergence.csv",
    "rank_scan_measurement_uncertainty_summary.csv",
    "run_manifest.json",
)


@dataclass(frozen=True)
class SourceDetectorArtifacts:
    """Validated detector-MC artifacts reused by one rank scan."""

    profile_draws: dict[str, np.ndarray]
    measurement_manifest: pd.DataFrame
    detector_summaries: pd.DataFrame
    lineage: dict[str, Any]
    run_manifest: dict[str, Any]
    checksums: dict[str, str]


def run_uncertainty_rank_scan_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run fixed-rank and full-shrinkage covariance comparisons."""
    path = Path(config_path).expanduser().resolve()
    config = load_rank_scan_config(path)
    input_h5 = _resolve_path(config["input"]["input_h5_path"], path)
    model_path = _resolve_path(config["input"]["model_joblib_path"], path)
    source_folder = _resolve_path(config["source_run"]["folder"], path)
    data_version = verify_dvc_input(
        {"data_version": config["data_version"]},
        config_path=path,
        input_h5_path=input_h5,
    )
    if data_version is None:
        raise MeasurementUncertaintyError("Rank scan requires DVC data lineage.")
    model_artifact = _load_frozen_model(model_path)
    _verify_model_data_lineage(model_artifact, data_version)
    source = load_source_detector_artifacts(source_folder)
    lineage = _lineage(
        model_artifact=model_artifact,
        model_path=model_path,
        data_version=data_version,
    )
    _validate_source_lineage(source, lineage)

    run_folder = _create_run_folder(config, path)
    scratch_path = run_folder / "preprocessed_rank_scan.joblib"
    effective_preprocessing = _experimental_preprocessing_config(
        model_artifact,
        input_h5_path=input_h5,
        output_joblib_path=scratch_path,
        data_version=data_version,
    )
    dataframe = run_preprocessing_pipeline(
        input_h5,
        effective_preprocessing,
        verbose=verbose,
    )
    if scratch_path.exists():
        scratch_path.unlink()
    targets = _targets_for_run(dataframe, config)
    source = _select_source_reference_cases(source, targets)

    draws = int(config["monte_carlo"]["draws"])
    seed = int(config["monte_carlo"]["seed"])
    quantiles = tuple(config["monte_carlo"]["interval_quantiles"])
    checkpoints = tuple(config["monte_carlo"]["convergence_draws"])
    gates = config["provisional_gates"]
    model_folder = run_folder / "covariance_models"
    draw_folder = run_folder / "draws"
    model_folder.mkdir()
    draw_folder.mkdir()

    summary_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    convergence_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    spectrum_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    dynamic_artifacts: list[str] = []

    for variant in config["rank_scan"]["variants"]:
        name = str(variant["name"])
        covariance_model = _fit_variant(
            variant,
            source,
            minimum_diagonal_variance=float(
                config["rank_scan"]["minimum_diagonal_variance"]
            ),
        )
        summaries, variant_draws = score_correlated_covariance_uncertainty(
            dataframe,
            model_artifact=model_artifact,
            targets=targets,
            covariance_model=covariance_model,
            draws=draws,
            seed=seed,
            interval_quantiles=quantiles,
        )
        convergence = summarize_profile_monte_carlo_convergence(
            variant_draws,
            checkpoints=checkpoints,
            interval_quantiles=quantiles,
        )
        comparison = compare_covariance_detector_reference(
            summaries,
            source.detector_summaries,
        )
        for frame in (summaries, variant_draws, convergence, comparison):
            frame.insert(0, "variant", name)
        diagnostics = covariance_diagnostics_frame(covariance_model)
        diagnostics.insert(0, "variant", name)
        spectrum = covariance_eigen_spectrum_frame(covariance_model)
        spectrum.insert(0, "variant", name)

        model_relative = f"covariance_models/{name}.npz"
        draws_relative = f"draws/{name}.parquet"
        write_low_rank_covariance(str(run_folder / model_relative), covariance_model)
        variant_draws.to_parquet(run_folder / draws_relative, index=False)
        dynamic_artifacts.extend((model_relative, draws_relative))
        summary_frames.append(summaries)
        comparison_frames.append(comparison)
        convergence_frames.append(convergence)
        diagnostic_frames.append(diagnostics)
        spectrum_frames.append(spectrum)
        metric_rows.append(
            summarize_variant_metrics(
                name=name,
                model=covariance_model,
                summaries=summaries,
                comparison=comparison,
                convergence=convergence,
                gates=gates,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    summaries = pd.concat(summary_frames, ignore_index=True)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    convergence = pd.concat(convergence_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    spectrum = pd.concat(spectrum_frames, ignore_index=True)
    required_artifacts = (*BASE_REQUIRED_ARTIFACTS, *dynamic_artifacts)
    lineage["source_uncertainty_run"] = {
        "folder": str(source_folder),
        "mlflow_run_id": config["source_run"]["mlflow_run_id"],
        "checksums": source.checksums,
    }
    _write_rank_scan_artifacts(
        run_folder=run_folder,
        config=config,
        config_path=path,
        effective_preprocessing=effective_preprocessing,
        data_version=data_version,
        lineage=lineage,
        source=source,
        metrics=metrics,
        summaries=summaries,
        comparisons=comparisons,
        convergence=convergence,
        diagnostics=diagnostics,
        spectrum=spectrum,
        required_artifacts=required_artifacts,
    )
    mlflow = _log_rank_scan_mlflow(
        config=config,
        config_path=path,
        run_folder=run_folder,
        lineage=lineage,
        metrics=metrics,
        required_artifacts=required_artifacts,
    )
    return {
        "contract": RANK_SCAN_CONTRACT,
        "run_folder": run_folder,
        "metrics_path": run_folder / "rank_scan_metrics.csv",
        "variants": int(len(metrics)),
        "patients_scored": int(summaries["target_case_id"].nunique()),
        "mlflow": mlflow,
    }


def load_source_detector_artifacts(folder: str | Path) -> SourceDetectorArtifacts:
    """Load and validate one completed v0.2 detector-reference run."""
    root = Path(folder).expanduser().resolve()
    required = {
        "detector_profile_fit_draws.npz",
        "detector_profile_fit_manifest.csv",
        "detector_measurement_uncertainty_summary.csv",
        "lineage.json",
        "run_manifest.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise MeasurementUncertaintyError(
            f"Source uncertainty run is incomplete; missing={missing}."
        )
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    lineage = json.loads((root / "lineage.json").read_text(encoding="utf-8"))
    if run_manifest.get("contract") != MEASUREMENT_UNCERTAINTY_CONTRACT:
        raise MeasurementUncertaintyError(
            "Source uncertainty run must use the v0.2 covariance contract."
        )
    manifest = pd.read_csv(root / "detector_profile_fit_manifest.csv")
    if (
        not {"profile_key", "npz_key"}.issubset(manifest.columns)
        or manifest["profile_key"].duplicated().any()
        or manifest["npz_key"].duplicated().any()
    ):
        raise MeasurementUncertaintyError(
            "Source detector profile manifest is invalid or duplicated."
        )
    profile_draws: dict[str, np.ndarray] = {}
    with np.load(root / "detector_profile_fit_draws.npz") as arrays:
        for row in manifest.itertuples(index=False):
            if str(row.npz_key) not in arrays.files:
                raise MeasurementUncertaintyError(
                    f"Source detector draw array is missing: {row.npz_key!r}."
                )
            values = np.asarray(arrays[str(row.npz_key)], dtype=float)
            if values.ndim != 2 or values.shape[0] < 3 or not np.isfinite(values).all():
                raise MeasurementUncertaintyError(
                    f"Source detector draws are invalid for {row.profile_key!r}."
                )
            profile_draws[str(row.profile_key)] = values
    summaries = pd.read_csv(root / "detector_measurement_uncertainty_summary.csv")
    required_summary = {
        "target_case_id",
        "deterministic_p_cancer",
        "p_cancer_low",
        "p_cancer_high",
        "threshold_crossing",
    }
    if not required_summary.issubset(summaries.columns) or summaries[
        "target_case_id"
    ].duplicated().any():
        raise MeasurementUncertaintyError("Source detector summaries are invalid.")
    checksums = {name: file_sha256(root / name) for name in sorted(required)}
    return SourceDetectorArtifacts(
        profile_draws=profile_draws,
        measurement_manifest=manifest.drop(columns=["npz_key"]),
        detector_summaries=summaries,
        lineage=lineage,
        run_manifest=run_manifest,
        checksums=checksums,
    )


def summarize_variant_metrics(
    *,
    name: str,
    model: LowRankCovarianceModel,
    summaries: pd.DataFrame,
    comparison: pd.DataFrame,
    convergence: pd.DataFrame,
    gates: dict[str, float],
) -> dict[str, Any]:
    """Reduce one full variant into auditable research gates."""
    final = convergence[convergence["draws"] == convergence["draws"].max()]
    max_endpoint_change = float(
        final[["abs_delta_low", "abs_delta_high"]].max(axis=1).max()
    )
    agreement = float(comparison["threshold_crossing_agreement"].mean())
    width_ratio = float(
        comparison["covariance_to_detector_width_ratio"].median()
    )
    crossing_pass = agreement >= gates["threshold_crossing_agreement_min"]
    width_pass = (
        gates["interval_width_ratio_min"]
        <= width_ratio
        <= gates["interval_width_ratio_max"]
    )
    convergence_pass = max_endpoint_change <= gates[
        "interval_endpoint_convergence_max"
    ]
    return {
        "variant": name,
        **model.diagnostics,
        "patients_scored": int(len(summaries)),
        "mean_interval_width": float(
            (summaries["p_cancer_high"] - summaries["p_cancer_low"]).mean()
        ),
        "threshold_crossing_cases": int(summaries["threshold_crossing"].sum()),
        "detector_reference_cases": int(len(comparison)),
        "threshold_crossing_agreement": agreement,
        "median_interval_width_ratio": width_ratio,
        "median_abs_probability_above_threshold_difference": float(
            comparison["abs_probability_above_threshold_difference"].median()
        ),
        "max_interval_endpoint_change": max_endpoint_change,
        "threshold_crossing_agreement_pass": bool(crossing_pass),
        "interval_width_ratio_pass": bool(width_pass),
        "interval_endpoint_convergence_pass": bool(convergence_pass),
        "all_provisional_gates_pass": bool(
            crossing_pass and width_pass and convergence_pass
        ),
    }


def load_rank_scan_config(path: str | Path) -> dict[str, Any]:
    """Load the intentional v0.1 rank-scan YAML contract."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Rank-scan config is unavailable: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_rank_scan_config(config)
    return config


def _fit_variant(
    variant: dict[str, Any],
    source: SourceDetectorArtifacts,
    *,
    minimum_diagonal_variance: float,
) -> LowRankCovarianceModel:
    estimator = variant["estimator"]
    if estimator == EMPIRICAL_ESTIMATOR:
        rank = int(variant["rank"])
        feature_count = int(next(iter(source.profile_draws.values())).shape[1])
        full_positive_spectrum = rank == feature_count
        model = fit_low_rank_covariance(
            source.profile_draws,
            source.measurement_manifest,
            explained_variance=1.0 - 1e-12,
            max_rank=rank,
            minimum_diagonal_variance=minimum_diagonal_variance,
            fixed_rank=None if full_positive_spectrum else rank,
        )
        selected_rank = int(model.diagnostics["selected_rank"])
        if not full_positive_spectrum and selected_rank != rank:
            raise MeasurementUncertaintyError(
                f"Variant {variant['name']!r} could not retain requested rank {rank}."
            )
        return replace(
            model,
            diagnostics={
                **model.diagnostics,
                "estimator": EMPIRICAL_ESTIMATOR,
                "requested_rank_budget": rank,
                "rank_policy": (
                    "full_positive_spectrum"
                    if full_positive_spectrum
                    else "fixed_rank"
                ),
            },
        )
    if estimator == SHRINKAGE_ESTIMATOR:
        return fit_full_shrinkage_covariance(
            source.profile_draws,
            source.measurement_manifest,
        )
    raise MeasurementUncertaintyError(f"Unsupported covariance estimator: {estimator}")


def _validate_source_lineage(
    source: SourceDetectorArtifacts,
    current: dict[str, Any],
) -> None:
    source_data = source.lineage.get("data_version", {})
    current_data = current["data_version"]
    source_model = source.lineage.get("model", {})
    current_model = current["model"]
    mismatches = {}
    for key in ("input_h5_sha256", "hash", "size_bytes"):
        if source_data.get(key) != current_data.get(key):
            mismatches[f"data.{key}"] = {
                "source": source_data.get(key),
                "current": current_data.get(key),
            }
    if source_model.get("sha256") != current_model.get("sha256"):
        mismatches["model.sha256"] = {
            "source": source_model.get("sha256"),
            "current": current_model.get("sha256"),
        }
    if mismatches:
        raise MeasurementUncertaintyError(
            f"Source uncertainty run lineage does not match current input: {mismatches}."
        )


def _select_source_reference_cases(
    source: SourceDetectorArtifacts,
    targets: list[Any],
) -> SourceDetectorArtifacts:
    available = {target.target_case_id.lower() for target in targets}
    selected = source.detector_summaries[
        source.detector_summaries["target_case_id"]
        .astype(str)
        .str.lower()
        .isin(available)
    ].copy()
    if selected.empty:
        raise MeasurementUncertaintyError(
            "No source detector-reference case is present in the requested targets."
        )
    missing = sorted(
        set(selected["target_case_id"].astype(str).str.lower()) - available
    )
    if missing:
        raise MeasurementUncertaintyError(
            f"Source detector cases are absent from current preprocessing: {missing}."
        )
    return replace(source, detector_summaries=selected.reset_index(drop=True))


def _write_rank_scan_artifacts(
    *,
    run_folder: Path,
    config: dict[str, Any],
    config_path: Path,
    effective_preprocessing: dict[str, Any],
    data_version: dict[str, Any],
    lineage: dict[str, Any],
    source: SourceDetectorArtifacts,
    metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    convergence: pd.DataFrame,
    diagnostics: pd.DataFrame,
    spectrum: pd.DataFrame,
    required_artifacts: tuple[str, ...],
) -> None:
    (run_folder / "effective_experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_folder / "effective_training_preprocessing.yaml").write_text(
        yaml.safe_dump(effective_preprocessing, sort_keys=False), encoding="utf-8"
    )
    pointer = resolve_config_path(data_version["pointer_path"], config_path)
    (run_folder / "dvc_data_pointer.dvc").write_bytes(pointer.read_bytes())
    _write_json(run_folder / "lineage.json", lineage)
    _write_json(run_folder / "source_artifact_checksums.json", source.checksums)
    source.detector_summaries.to_csv(
        run_folder / "source_detector_measurement_uncertainty_summary.csv",
        index=False,
    )
    metrics.to_csv(run_folder / "rank_scan_metrics.csv", index=False)
    summaries.to_csv(
        run_folder / "rank_scan_measurement_uncertainty_summary.csv", index=False
    )
    comparisons.to_csv(run_folder / "rank_scan_case_comparison.csv", index=False)
    convergence.to_csv(
        run_folder / "rank_scan_monte_carlo_convergence.csv", index=False
    )
    diagnostics.to_csv(
        run_folder / "rank_scan_covariance_diagnostics.csv", index=False
    )
    spectrum.to_csv(run_folder / "rank_scan_eigen_spectrum.csv", index=False)
    manifest = {
        "contract": RANK_SCAN_CONTRACT,
        "clinical_stage": "research_only",
        "product_artifact_modified": False,
        "uncertainty_scope": "photon_statistical_detector_component_only",
        "source_detector_reference_cases": int(len(source.detector_summaries)),
        "source_detector_fit_measurements": int(len(source.measurement_manifest)),
        "patients_scored": int(summaries["target_case_id"].nunique()),
        "variants": metrics.to_dict(orient="records"),
        "source_mlflow_run_id": config["source_run"]["mlflow_run_id"],
        "source_artifact_checksums": source.checksums,
        "limitations": [
            "photon/statistical uncertainty only",
            "single retrospective training archive",
            "shared covariance transfer assumption",
            "no calibration, thickness, positioning, biological, or model uncertainty",
            "not an independent clinical validation",
        ],
        "required_artifacts": list(required_artifacts),
    }
    _write_json(run_folder / "run_manifest.json", manifest)
    missing = [name for name in required_artifacts if not (run_folder / name).is_file()]
    if missing:
        raise MeasurementUncertaintyError(
            f"Rank-scan artifacts are incomplete: {missing}."
        )


def _log_rank_scan_mlflow(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_folder: Path,
    lineage: dict[str, Any],
    metrics: pd.DataFrame,
    required_artifacts: tuple[str, ...],
) -> dict[str, Any]:
    mlflow_config = config["mlflow"]
    tracking_uri = _tracking_uri(str(mlflow_config["tracking_uri"]), config_path)
    metric_values: dict[str, float] = {}
    for row in metrics.to_dict(orient="records"):
        prefix = f"variant.{row['variant']}"
        for key, value in row.items():
            if key == "variant" or isinstance(value, bool):
                continue
            if isinstance(value, int | float) and np.isfinite(float(value)):
                metric_values[f"{prefix}.{key}"] = float(value)
        metric_values[f"{prefix}.all_provisional_gates_pass"] = float(
            bool(row["all_provisional_gates_pass"])
        )
    params = rank_scan_mlflow_params(config)
    tags = {
        "product": "aramina",
        "clinical_stage": "research_draft",
        "uncertainty_scope": "photon_statistical_detector_component_only",
        "input_h5_checksum": lineage["data_version"]["input_h5_sha256"],
        "model_artifact_sha256": lineage["model"]["sha256"],
        "source_mlflow_run_id": config["source_run"]["mlflow_run_id"],
        "source_code": lineage["source_code"],
    }
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
            required_files=required_artifacts,
            artifact_path="measurement_uncertainty_rank_scan",
        )
        run_id = run.run_id
    return {
        "enabled": True,
        "run_id": run_id,
        "status": run.status,
        "tracking_uri": tracking_uri,
    }


def rank_scan_mlflow_params(config: dict[str, Any]) -> dict[str, Any]:
    """Return MLflow-safe scalar parameters for the rank-scan contract."""
    variants = config["rank_scan"]["variants"]
    monte_carlo = config["monte_carlo"]
    return {
        "experiment": config["experiment"],
        "rank_scan": {
            "minimum_diagonal_variance": config["rank_scan"][
                "minimum_diagonal_variance"
            ],
            "variants": ";".join(
                f"{item['name']}:{item['estimator']}:rank={item['rank']}"
                for item in variants
            ),
        },
        "monte_carlo": {
            "draws": monte_carlo["draws"],
            "seed": monte_carlo["seed"],
            "interval_quantiles": ",".join(
                str(value) for value in monte_carlo["interval_quantiles"]
            ),
            "convergence_draws": ",".join(
                str(value) for value in monte_carlo["convergence_draws"]
            ),
        },
        "source_run": config["source_run"],
    }


def _validate_rank_scan_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise MeasurementUncertaintyError("Rank-scan config must be a mapping.")
    _exact_keys(
        config,
        {
            "contract",
            "experiment",
            "input",
            "data_version",
            "targets",
            "source_run",
            "rank_scan",
            "monte_carlo",
            "provisional_gates",
            "mlflow",
            "output",
        },
        "rank-scan config",
    )
    if config["contract"] != RANK_SCAN_CONTRACT:
        raise MeasurementUncertaintyError(
            f"Unsupported rank-scan contract: {config['contract']!r}."
        )
    _exact_keys(config["experiment"], {"name", "model_name", "model_version"}, "experiment")
    if (
        config["experiment"]["model_name"] != FROZEN_MODEL_NAME
        or config["experiment"]["model_version"] != FROZEN_MODEL_VERSION
    ):
        raise MeasurementUncertaintyError("Rank scan must pin frozen Aramina 0.2.14-beta.")
    _exact_keys(config["input"], {"input_h5_path", "model_joblib_path"}, "input")
    _exact_keys(
        config["data_version"],
        {"contract", "system", "dataset_id", "dvc_version", "pointer_path"},
        "data_version",
    )
    if (
        config["data_version"]["contract"] != DVC_DATA_CONTRACT
        or config["data_version"]["system"] != "dvc"
    ):
        raise MeasurementUncertaintyError("Rank scan requires the DVC input contract.")
    _exact_keys(config["targets"], {"mode", "selected"}, "targets")
    target_mode = config["targets"]["mode"]
    selected_targets = config["targets"]["selected"]
    if target_mode == "all_training_target_cases":
        if selected_targets:
            raise MeasurementUncertaintyError(
                "all_training_target_cases requires an empty selected list."
            )
    elif target_mode == "selected":
        if not isinstance(selected_targets, list) or not selected_targets:
            raise MeasurementUncertaintyError(
                "selected target mode requires at least one target mapping."
            )
        for index, target in enumerate(selected_targets):
            _exact_keys(target, {"patient_id", "target_side"}, f"targets.selected[{index}]")
            if target["target_side"] not in {"left", "right"}:
                raise MeasurementUncertaintyError(
                    "Selected target_side must be left or right."
                )
    else:
        raise MeasurementUncertaintyError(f"Unsupported target mode: {target_mode!r}.")
    _exact_keys(config["source_run"], {"folder", "mlflow_run_id"}, "source_run")
    scan = config["rank_scan"]
    _exact_keys(scan, {"variants", "minimum_diagonal_variance"}, "rank_scan")
    variants = scan["variants"]
    if not isinstance(variants, list) or len(variants) < 2:
        raise MeasurementUncertaintyError("rank_scan.variants requires at least two variants.")
    names: set[str] = set()
    empirical_ranks: list[int] = []
    shrinkage_count = 0
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise MeasurementUncertaintyError(f"rank_scan.variants[{index}] must be a mapping.")
        _exact_keys(variant, {"name", "estimator", "rank"}, f"rank_scan.variants[{index}]")
        name = str(variant["name"])
        if not name or name in names:
            raise MeasurementUncertaintyError("Rank-scan variant names must be unique and non-empty.")
        names.add(name)
        estimator = variant["estimator"]
        rank = variant["rank"]
        if estimator == EMPIRICAL_ESTIMATOR:
            if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 100:
                raise MeasurementUncertaintyError("Empirical rank must be an integer in 1-100.")
            empirical_ranks.append(rank)
        elif estimator == SHRINKAGE_ESTIMATOR:
            if rank != 100:
                raise MeasurementUncertaintyError("Full shrinkage variant must declare rank 100.")
            shrinkage_count += 1
        else:
            raise MeasurementUncertaintyError(f"Unsupported estimator: {estimator!r}.")
    if empirical_ranks != [30, 50, 75, 100] or shrinkage_count != 1:
        raise MeasurementUncertaintyError(
            "Rank scan must contain empirical ranks 30/50/75/100 and one full shrinkage variant."
        )
    minimum = scan["minimum_diagonal_variance"]
    if isinstance(minimum, bool) or not isinstance(minimum, int | float) or minimum < 0:
        raise MeasurementUncertaintyError("minimum_diagonal_variance must be nonnegative.")
    monte_carlo = config["monte_carlo"]
    _exact_keys(
        monte_carlo,
        {"draws", "seed", "interval_quantiles", "convergence_draws"},
        "monte_carlo",
    )
    draws = monte_carlo["draws"]
    checkpoints = monte_carlo["convergence_draws"]
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1000:
        raise MeasurementUncertaintyError("Rank scan requires at least 1000 Monte Carlo draws.")
    if (
        not isinstance(checkpoints, list)
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[-1] != draws
    ):
        raise MeasurementUncertaintyError("convergence_draws must be increasing and end at draws.")
    _validate_quantiles(tuple(monte_carlo["interval_quantiles"]))
    gates = config["provisional_gates"]
    _exact_keys(
        gates,
        {
            "threshold_crossing_agreement_min",
            "interval_width_ratio_min",
            "interval_width_ratio_max",
            "interval_endpoint_convergence_max",
        },
        "provisional_gates",
    )
    _exact_keys(config["mlflow"], {"enabled", "tracking_uri", "experiment_name"}, "mlflow")
    if config["mlflow"]["enabled"] is not True:
        raise MeasurementUncertaintyError("Rank-scan MLflow logging is required.")
    _exact_keys(config["output"], {"folder"}, "output")
    for mapping, keys in (
        (config["experiment"], ("name",)),
        (config["input"], ("input_h5_path", "model_joblib_path")),
        (config["source_run"], ("folder", "mlflow_run_id")),
        (config["mlflow"], ("tracking_uri", "experiment_name")),
        (config["output"], ("folder",)),
    ):
        for key in keys:
            if not isinstance(mapping[key], str) or not mapping[key].strip():
                raise MeasurementUncertaintyError(f"{key} must be non-empty text.")


def _exact_keys(value: Any, expected: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise MeasurementUncertaintyError(f"{where} must be a mapping.")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise MeasurementUncertaintyError(
            f"{where} fields invalid; missing={missing}, unknown={unknown}."
        )


def _create_run_folder(config: dict[str, Any], config_path: Path) -> Path:
    root = _resolve_path(config["output"]["folder"], config_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"measurement_uncertainty_rank_scan_{timestamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _resolve_path(value: str, config_path: Path) -> Path:
    return resolve_config_path(value, config_path)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")
