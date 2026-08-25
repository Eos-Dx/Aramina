"""Patient-safe polar-basis compression experiment for Aramina.

The module is research-only. It reads the immutable 0.2.14 product artifact for
lineage and architecture settings but never modifies its score, threshold,
reports, or serialized model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.special import jn_zeros, jv
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml
from xrd_preprocessing import perform_polar_cake_integration

from ..config_paths import resolve_config_path
from ..data_versioning import DVC_DATA_CONTRACT, verify_dvc_input
from ..mlflow_tracking import MlflowRun
from ..model_metrics import binary_metric_values
from ..model_utils import compute_binary_thresholds, profile_matrix
from ..patient_features import (
    TARGET_CASE_ID,
    empty_lr1_scores,
    lr1_training_rows,
    normalize_side,
    patient_feature_table,
    row_labels,
    score_lr1_rows,
)
from ..pipelines import run_preprocessing_pipeline
from ..runtime_identity import file_sha256
from ..target_breast_model import GatedSymmetryLogistic
from ..training_evaluation import _patient_split_pairs
from ..training_config import PRODUCT_MODEL_NAME
from .detector_uncertainty import (
    CALIBRATION_SESSION_COLUMN,
    MASK_COLUMN,
    RAW_FRAME_COLUMN,
)
from .measurement_uncertainty import (
    FROZEN_MODEL_NAME,
    FROZEN_MODEL_VERSION,
    _experimental_preprocessing_config,
    _lineage,
    _load_frozen_model,
    _tracking_uri,
    _verify_model_data_lineage,
)


CONTRACT = "aramina_polar_basis_compression_v0_1"
CANDIDATE_MODES = (0, 2, 4)
QC_MODES = (1, 3)
COEFFICIENT_BUDGETS = (15, 30, 50)
REPRESENTATIONS = ("fourier_bspline", "fourier_bessel", "fourier_fpca")
PROFILE_SCORE_COLUMNS = (
    "profile_p_cancer_probability_mean",
    "profile_p_cancer_logit_average",
    "profile_p_cancer_n_measurements",
)
REQUIRED_ARTIFACTS = (
    "basis.joblib",
    "basis_metadata.json",
    "coefficient_table.parquet",
    "cohort_manifest.csv",
    "confounder_analysis.csv",
    "confounder_availability.json",
    "dvc_data_pointer.dvc",
    "effective_experiment_config.yaml",
    "effective_training_preprocessing.yaml",
    "fold_manifest.csv",
    "fold_metrics.csv",
    "lineage.json",
    "metrics.csv",
    "polar_to_raw100_comparison.csv",
    "predictions.csv",
    "polar_cake_manifest.csv",
    "q_chi_axes.npz",
    "raw100_fold_metrics.csv",
    "raw100_metrics.csv",
    "raw100_predictions.csv",
    "reconstruction_examples.npz",
    "reconstruction_metrics.csv",
    "run_manifest.json",
)
CONTINUOUS_CONFOUNDERS = {
    "age": "age",
    "thickness": "target_thickness_mm",
    "date": "target_date_ordinal",
}


class PolarBasisExperimentError(ValueError):
    """Raised when the research experiment cannot be reproduced safely."""


@dataclass(frozen=True)
class RepresentationSpec:
    """One controlled candidate representation."""

    family: Literal["fourier_bspline", "fourier_bessel", "fourier_fpca"]
    budget: int

    @property
    def name(self) -> str:
        return f"{self.family}_{self.budget}"


@dataclass(frozen=True)
class PolarAxes:
    """Shared fixed polar grid."""

    q: np.ndarray
    chi: np.ndarray
    harmonic_q_mask: np.ndarray | None = None

    @property
    def harmonic_q(self) -> np.ndarray:
        if self.harmonic_q_mask is None:
            return self.q
        return self.q[np.asarray(self.harmonic_q_mask, dtype=bool)]


class PolarBasisEncoder:
    """Fold-local compressor for candidate angular harmonics."""

    def __init__(self, *, spec: RepresentationSpec, q: np.ndarray, seed: int) -> None:
        self.spec = spec
        self.q = np.asarray(q, dtype=float)
        self.seed = int(seed)

    def fit(self, values: np.ndarray) -> "PolarBasisEncoder":
        matrix = _candidate_tensor(values)
        if self.spec.family == "fourier_fpca":
            flat = matrix.reshape(len(matrix), -1)
            if self.spec.budget >= min(flat.shape):
                raise PolarBasisExperimentError(
                    f"FPCA budget {self.spec.budget} must be smaller than "
                    f"training matrix limit {min(flat.shape)}."
                )
            self.transformer_ = PCA(
                n_components=self.spec.budget,
                svd_solver="full",
                random_state=self.seed,
            ).fit(flat)
            self.radial_basis_ = None
        else:
            radial_terms = _budget_allocation(self.spec.budget)
            self.radial_basis_ = _radial_basis(
                self.spec.family,
                q=self.q,
                terms_by_channel=radial_terms,
            )
            weights = (
                self.q / float(np.mean(self.q))
                if self.spec.family == "fourier_bessel"
                else np.ones_like(self.q)
            )
            sqrt_weights = np.sqrt(weights)
            self.radial_projector_ = {
                name: np.linalg.pinv(basis * sqrt_weights[:, None])
                * sqrt_weights[None, :]
                for name, basis in self.radial_basis_.items()
            }
            self.transformer_ = None
        self.training_rows_ = int(len(matrix))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = _candidate_tensor(values)
        if self.spec.family == "fourier_fpca":
            return self.transformer_.transform(matrix.reshape(len(matrix), -1))
        coefficients = []
        for channel_index, channel_name in enumerate(_candidate_channel_names()):
            projector = self.radial_projector_[channel_name]
            coefficients.append(matrix[:, channel_index, :] @ projector.T)
        return np.hstack(coefficients)

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        values = np.asarray(coefficients, dtype=float)
        if self.spec.family == "fourier_fpca":
            reconstructed = self.transformer_.inverse_transform(values)
            return reconstructed.reshape(
                len(values), len(_candidate_channel_names()), -1
            )
        radial_terms = _budget_allocation(self.spec.budget)
        channels = []
        offset = 0
        for channel_name in _candidate_channel_names():
            basis = self.radial_basis_[channel_name]
            channel_terms = radial_terms[channel_name]
            channels.append(values[:, offset : offset + channel_terms] @ basis.T)
            offset += channel_terms
        return np.stack(channels, axis=1)

    def metadata(self) -> dict[str, Any]:
        """Return auditable basis metadata without duplicating binary artifacts."""
        out: dict[str, Any] = {
            "family": self.spec.family,
            "budget": self.spec.budget,
            "candidate_modes": list(CANDIDATE_MODES),
            "candidate_channels": list(_candidate_channel_names()),
            "training_basis_rows": self.training_rows_,
            "training_basis_balance": "one_mean_harmonic_tensor_per_target_case",
            "q_bins": int(len(self.q)),
        }
        if self.transformer_ is not None:
            out.update(
                {
                    "fit_scope": "training_patients_only",
                    "explained_variance_ratio": self.transformer_.explained_variance_ratio_.tolist(),
                    "basis_fingerprint": _array_fingerprint(
                        self.transformer_.components_, self.transformer_.mean_
                    ),
                }
            )
        else:
            out.update(
                {
                    "fit_scope": "fixed_q_grid_recreated_inside_training_fold",
                    "radial_terms_per_candidate_channel": _budget_allocation(
                        self.spec.budget
                    ),
                    "basis_fingerprint": _array_fingerprint(
                        *(
                            self.radial_basis_[name]
                            for name in _candidate_channel_names()
                        ),
                        *(
                            self.radial_projector_[name]
                            for name in _candidate_channel_names()
                        ),
                    ),
                    "radial_fit_weight": (
                        "q_dq_discrete_weight"
                        if self.spec.family == "fourier_bessel"
                        else "uniform_q_grid"
                    ),
                }
            )
        return out


def run_polar_basis_compression_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run all polar representations using one patient-safe fold manifest."""
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    input_h5 = _resolve_path(config["input"]["input_h5_path"], path)
    model_path = _resolve_path(config["input"]["model_joblib_path"], path)
    data_version = verify_dvc_input(
        {"data_version": config["data_version"]},
        config_path=path,
        input_h5_path=input_h5,
    )
    if data_version is None:
        raise PolarBasisExperimentError("Polar experiment requires DVC data lineage.")
    model_artifact = _load_frozen_model(model_path)
    _verify_model_data_lineage(model_artifact, data_version)
    model_definition = _model_definition(model_artifact)
    model_info = model_artifact["models"][PRODUCT_MODEL_NAME]

    run_folder = _create_run_folder(config, path)
    scratch_path = run_folder / "_scratch_preprocessed.joblib"
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
    dataframe = _select_pilot_cohort(dataframe, config, model_definition)
    context = _build_context(dataframe, model_definition)
    target_rows = _target_measurement_rows(dataframe, model_definition)
    cake_manifest, axes = build_or_reuse_polar_cakes(
        target_rows,
        cache_folder=_resolve_path(config["polar_cakes"]["cache_folder"], path),
        dataset_sha256=data_version["input_h5_sha256"],
        n_q=int(config["polar_cakes"]["n_q"]),
        n_chi=int(config["polar_cakes"]["n_chi"]),
        radial_q_range=tuple(config["polar_cakes"]["radial_q_range"]),
        azimuthal_range=tuple(config["polar_cakes"]["azimuthal_range"]),
        force_rebuild=bool(config["polar_cakes"]["force_rebuild"]),
        verbose=verbose,
    )
    harmonic_q_range = tuple(config["polar_cakes"]["harmonic_q_range"])
    harmonic_q_mask = (axes.q >= harmonic_q_range[0]) & (axes.q <= harmonic_q_range[1])
    if int(np.sum(harmonic_q_mask)) < max(COEFFICIENT_BUDGETS):
        raise PolarBasisExperimentError(
            "Configured harmonic q range has too few q bins for the experiment."
        )
    axes = PolarAxes(
        q=axes.q,
        chi=axes.chi,
        harmonic_q_mask=harmonic_q_mask,
    )
    polar_rows = load_polar_harmonics(
        target_rows,
        cake_manifest,
        cache_folder=_resolve_path(config["polar_cakes"]["cache_folder"], path),
        normalization_q_range=tuple(config["polar_cakes"]["normalization_q_range"]),
        axes=axes,
    )
    cake_manifest = cake_manifest.merge(
        polar_rows[
            [
                "measurement_key",
                "normalization_scale",
                "qc_m1_energy",
                "qc_m3_energy",
            ]
        ],
        on="measurement_key",
        how="left",
        validate="one_to_one",
    )
    split_pairs, fold_manifest = _shared_patient_folds(
        context,
        folds=int(config["evaluation"]["folds"]),
        repeats=int(config["evaluation"]["repeats"]),
        seed=int(config["evaluation"]["seed"]),
    )
    result = evaluate_representations(
        dataframe=dataframe,
        context=context,
        polar_rows=polar_rows,
        axes=axes,
        split_pairs=split_pairs,
        fold_manifest=fold_manifest,
        model_definition=model_definition,
        model_info=model_info,
        config=config,
    )
    lineage = _lineage(
        model_artifact=model_artifact,
        model_path=model_path,
        data_version=data_version,
    )
    _write_artifacts(
        run_folder=run_folder,
        config=config,
        config_path=path,
        effective_preprocessing=effective_preprocessing,
        data_version=data_version,
        lineage=lineage,
        axes=axes,
        cake_manifest=cake_manifest,
        result=result,
        product_threshold=float(model_info["thresholds"]["threshold_target"]),
    )
    mlflow = _log_mlflow(
        config=config,
        config_path=path,
        run_folder=run_folder,
        lineage=lineage,
        result=result,
    )
    return {
        "contract": CONTRACT,
        "run_folder": run_folder,
        "variants": int(len(result["summary"])),
        "target_cases": int(len(context)),
        "mlflow": mlflow,
    }


def build_or_reuse_polar_cakes(
    target_rows: pd.DataFrame,
    *,
    cache_folder: Path,
    dataset_sha256: str,
    n_q: int,
    n_chi: int,
    radial_q_range: tuple[float, float] = (2.0, 23.0),
    azimuthal_range: tuple[float, float] = (-180.0, 180.0),
    force_rebuild: bool,
    verbose: bool = False,
) -> tuple[pd.DataFrame, PolarAxes]:
    """Create only missing fixed cakes, one detector frame at a time."""
    cache_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_folder / "polar_cake_cache_manifest.csv"
    axis_contract = _axis_contract_fingerprint(
        n_q=n_q,
        n_chi=n_chi,
        radial_q_range=radial_q_range,
        azimuthal_range=azimuthal_range,
    )
    canonical_axes = _canonical_axes(
        n_q=n_q,
        n_chi=n_chi,
        radial_q_range=radial_q_range,
        azimuthal_range=azimuthal_range,
    )
    existing = (
        pd.read_csv(manifest_path, dtype={"measurement_key": str})
        if manifest_path.is_file() and not force_rebuild
        else pd.DataFrame()
    )
    records: dict[str, dict[str, Any]] = {}
    for row in existing.itertuples(index=False):
        artifact_path = cache_folder / str(row.artifact)
        if (
            row.dataset_sha256 != dataset_sha256
            or int(row.n_q) != n_q
            or int(row.n_chi) != n_chi
            or not artifact_path.is_file()
        ):
            continue
        record = row._asdict()
        cached_contract = str(record.get("axis_contract_sha256", ""))
        if cached_contract and cached_contract != axis_contract:
            continue
        with np.load(artifact_path) as cached:
            q = np.asarray(cached["q"], dtype=float)
            chi = np.asarray(cached["chi"], dtype=float)
        if not _axes_match_contract(
            q,
            chi,
            n_q=n_q,
            n_chi=n_chi,
            radial_q_range=radial_q_range,
            azimuthal_range=azimuthal_range,
        ):
            continue
        record.update(
            {
                "axis_contract_sha256": axis_contract,
                "radial_q_min": float(radial_q_range[0]),
                "radial_q_max": float(radial_q_range[1]),
                "azimuthal_min": float(azimuthal_range[0]),
                "azimuthal_max": float(azimuthal_range[1]),
            }
        )
        records[str(row.measurement_key)] = record
    shared_q: np.ndarray | None = canonical_axes.q
    shared_chi: np.ndarray | None = canonical_axes.chi
    for position, row in target_rows.reset_index(drop=True).iterrows():
        key = str(row["measurement_key"])
        recovered_artifact = cache_folder / f"cakes/{key}.npz"
        if key not in records and not force_rebuild and recovered_artifact.is_file():
            with np.load(recovered_artifact) as cached:
                q = np.asarray(cached["q"], dtype=float)
                chi = np.asarray(cached["chi"], dtype=float)
            if _axes_match_contract(
                q,
                chi,
                n_q=n_q,
                n_chi=n_chi,
                radial_q_range=radial_q_range,
                azimuthal_range=azimuthal_range,
            ):
                records[key] = _cake_manifest_record(
                    row,
                    key=key,
                    artifact=f"cakes/{key}.npz",
                    dataset_sha256=dataset_sha256,
                    n_q=n_q,
                    n_chi=n_chi,
                    axis_contract=axis_contract,
                    radial_q_range=radial_q_range,
                    azimuthal_range=azimuthal_range,
                )
        if key in records:
            artifact_path = cache_folder / str(records[key]["artifact"])
            with np.load(artifact_path) as cached:
                q = np.asarray(cached["q"], dtype=float)
                chi = np.asarray(cached["chi"], dtype=float)
            shared_q, shared_chi = _validate_shared_axes(
                shared_q, shared_chi, q, chi
            )
            if verbose:
                print(f"polar_cake_cached={position + 1}/{len(target_rows)}")
            continue
        integration_row = row.copy()
        integration_row["interpolation_q_range"] = tuple(radial_q_range)
        integration_row["azimuthal_range"] = tuple(azimuthal_range)
        cake = perform_polar_cake_integration(
            integration_row,
            column=RAW_FRAME_COLUMN,
            npt=n_q,
            npt_azimuthal=n_chi,
            mask_column=MASK_COLUMN,
            calibration_mode="poni",
            error_model="poisson",
            thickness_adjustment=True,
            require_thickness_adjustment=True,
            thickness_reference_column="calibrant_thickness_mm",
            sample_thickness_column="sample_thickness_mm",
        )
        q = np.asarray(cake.q, dtype=float)
        chi = np.asarray(cake.azimuth, dtype=float)
        if not _axes_match_contract(
            q,
            chi,
            n_q=n_q,
            n_chi=n_chi,
            radial_q_range=radial_q_range,
            azimuthal_range=azimuthal_range,
        ):
            raise PolarBasisExperimentError(
                f"Polar cake {key} violates the configured q/chi axis contract."
            )
        shared_q, shared_chi = _validate_shared_axes(shared_q, shared_chi, q, chi)
        q = canonical_axes.q
        chi = canonical_axes.chi
        artifact = f"cakes/{key}.npz"
        artifact_path = cache_folder / artifact
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            intensity=np.asarray(cake.intensity, dtype=np.float32),
            count=np.asarray(cake.count, dtype=np.float32),
            sigma=np.asarray(cake.sigma, dtype=np.float32),
            sum_variance=np.asarray(
                getattr(cake, "sum_variance", np.square(cake.sigma)),
                dtype=np.float32,
            ),
            q=q,
            chi=chi,
        )
        records[key] = _cake_manifest_record(
            row,
            key=key,
            artifact=artifact,
            dataset_sha256=dataset_sha256,
            n_q=n_q,
            n_chi=n_chi,
            axis_contract=axis_contract,
            radial_q_range=radial_q_range,
            azimuthal_range=azimuthal_range,
        )
        if verbose:
            print(f"polar_cake_generated={position + 1}/{len(target_rows)}")
    manifest = pd.DataFrame(list(records.values())).sort_values("measurement_key")
    manifest.to_csv(manifest_path, index=False)
    selected = manifest[
        manifest["measurement_key"].isin(target_rows["measurement_key"].astype(str))
    ].copy()
    if len(selected) != len(target_rows):
        raise PolarBasisExperimentError("Polar cake cache is incomplete.")
    return selected.reset_index(drop=True), canonical_axes


def load_polar_harmonics(
    target_rows: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    cache_folder: Path,
    normalization_q_range: tuple[float, float],
    axes: PolarAxes,
) -> pd.DataFrame:
    """Load cached cakes and derive normalized m=0..4 harmonic channels."""
    artifact_by_key = manifest.set_index("measurement_key")["artifact"].to_dict()
    records: list[dict[str, Any]] = []
    for _, row in target_rows.iterrows():
        key = str(row["measurement_key"])
        with np.load(cache_folder / str(artifact_by_key[key])) as cake:
            intensity = np.asarray(cake["intensity"], dtype=float)
            count = np.asarray(cake["count"], dtype=float)
            q = np.asarray(cake["q"], dtype=float)
            chi = np.asarray(cake["chi"], dtype=float)
        _validate_axes(axes, q, chi)
        q = axes.q
        chi = axes.chi
        normalized = _normalize_cake(
            intensity,
            count,
            q,
            normalization_q_range=normalization_q_range,
        )
        q_mask = (
            np.ones(len(q), dtype=bool)
            if axes.harmonic_q_mask is None
            else np.asarray(axes.harmonic_q_mask, dtype=bool)
        )
        harmonics = angular_harmonic_channels(
            normalized[:, q_mask],
            count[:, q_mask],
            chi,
            max_mode=4,
        )
        metadata = {
            column: row[column]
            for column in (
                "patientId",
                "specimenId",
                "side",
                "product_status_group",
                "biopsy",
                TARGET_CASE_ID,
                "_label",
                "measurement_key",
            )
            if column in target_rows.columns
        }
        records.append(
            {
                **metadata,
                "harmonic_matrix": harmonics.astype(np.float32),
                "qc_m1_energy": _mode_energy(harmonics, 1),
                "qc_m3_energy": _mode_energy(harmonics, 3),
                "normalization_scale": _cake_normalization_scale(
                    intensity,
                    count,
                    q,
                    normalization_q_range=normalization_q_range,
                ),
            }
        )
    return pd.DataFrame(records)


def angular_harmonic_channels(
    intensity: np.ndarray,
    count: np.ndarray,
    chi_degrees: np.ndarray,
    *,
    max_mode: int,
) -> np.ndarray:
    """Fit weighted Fourier channels independently at every q bin."""
    values = np.asarray(intensity, dtype=float)
    weights = np.asarray(count, dtype=float)
    chi = np.deg2rad(np.asarray(chi_degrees, dtype=float).ravel())
    if values.ndim != 2 or values.shape != weights.shape or values.shape[0] != len(chi):
        raise PolarBasisExperimentError(
            "Polar cake intensity/count shapes are invalid."
        )
    design_columns = [np.ones_like(chi)]
    for mode in range(1, max_mode + 1):
        design_columns.extend([np.cos(mode * chi), np.sin(mode * chi)])
    design = np.column_stack(design_columns)
    channels = np.empty((design.shape[1], values.shape[1]), dtype=float)
    for q_index in range(values.shape[1]):
        valid = (
            np.isfinite(values[:, q_index])
            & np.isfinite(weights[:, q_index])
            & (weights[:, q_index] > 0.0)
        )
        if int(np.sum(valid)) < design.shape[1]:
            raise PolarBasisExperimentError(
                f"Polar q bin {q_index} has insufficient angular support."
            )
        weighted_design = design[valid] * np.sqrt(weights[valid, q_index])[:, None]
        weighted_values = values[valid, q_index] * np.sqrt(weights[valid, q_index])
        channels[:, q_index] = np.linalg.lstsq(
            weighted_design,
            weighted_values,
            rcond=None,
        )[0]
    if not np.isfinite(channels).all():
        raise PolarBasisExperimentError("Polar harmonic coefficients are non-finite.")
    return channels


def evaluate_representations(
    *,
    dataframe: pd.DataFrame,
    context: pd.DataFrame,
    polar_rows: pd.DataFrame,
    axes: PolarAxes,
    split_pairs: list[tuple[np.ndarray, np.ndarray]],
    fold_manifest: pd.DataFrame,
    model_definition: dict[str, Any],
    model_info: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every representation on the exact same held-out patients."""
    specifications = [
        RepresentationSpec(family=family, budget=budget)
        for family in REPRESENTATIONS
        for budget in COEFFICIENT_BUDGETS
    ]
    fold_metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    reconstruction: list[pd.DataFrame] = []
    confounders: list[pd.DataFrame] = []
    fitted_bases: dict[str, PolarBasisEncoder] = {}
    basis_metadata: dict[str, Any] = {}
    examples: dict[str, np.ndarray] = {}
    threshold_policy = config["evaluation"]["threshold_policy"]
    target_sensitivity = float(config["evaluation"]["target_sensitivity"])
    product_threshold = float(model_info["thresholds"]["threshold_target"])
    seed = int(config["evaluation"]["seed"])
    max_examples = int(config["runtime"]["reconstruction_examples_per_variant"])
    raw100 = _evaluate_raw100_baseline(
        dataframe=dataframe,
        context=context,
        split_pairs=split_pairs,
        model_definition=model_definition,
        threshold_policy=threshold_policy,
        target_sensitivity=target_sensitivity,
        product_threshold=product_threshold,
        seed=seed,
        folds=int(config["evaluation"]["folds"]),
    )

    for specification in specifications:
        for split_id, (train_index, test_index) in enumerate(split_pairs):
            train_patients = set(context.iloc[train_index]["patientId"].astype(str))
            test_patients = set(context.iloc[test_index]["patientId"].astype(str))
            if train_patients.intersection(test_patients):
                raise RuntimeError("Patient leakage detected before polar encoding.")
            train_rows = polar_rows[
                polar_rows["patientId"].astype(str).isin(train_patients)
            ].copy()
            test_rows = polar_rows[
                polar_rows["patientId"].astype(str).isin(test_patients)
            ].copy()
            encoder = PolarBasisEncoder(
                spec=specification,
                q=axes.harmonic_q,
                seed=seed + split_id,
            ).fit(_case_balanced_harmonics(train_rows))
            train_coefficients = encoder.transform(_stack_harmonics(train_rows))
            test_coefficients = encoder.transform(_stack_harmonics(test_rows))
            fitted = _fit_product_fold(
                dataframe=dataframe,
                context=context,
                train_rows=train_rows,
                test_rows=test_rows,
                train_coefficients=train_coefficients,
                test_coefficients=test_coefficients,
                train_patients=train_patients,
                test_patients=test_patients,
                model_definition=model_definition,
                threshold_policy=threshold_policy,
                target_sensitivity=target_sensitivity,
                product_threshold=product_threshold,
                seed=seed + split_id,
            )
            metric = _classification_metrics(
                fitted["test_features"]["label"].to_numpy(dtype=int),
                fitted["test_scores"],
                threshold=fitted["threshold"],
            )
            fold_metrics.append(
                {
                    "representation": specification.family,
                    "budget": specification.budget,
                    "split_id": split_id,
                    "repeat_id": split_id // int(config["evaluation"]["folds"]),
                    "fold_id": split_id % int(config["evaluation"]["folds"]),
                    "threshold_policy": threshold_policy,
                    **metric,
                }
            )
            prediction = fitted["test_features"][
                [TARGET_CASE_ID, "patientId", "target_side", "label", "label_name"]
            ].copy()
            prediction.insert(0, "representation", specification.family)
            prediction.insert(1, "budget", specification.budget)
            prediction.insert(2, "split_id", split_id)
            prediction["p_cancer"] = fitted["test_scores"]
            prediction["threshold"] = fitted["threshold"]
            prediction["suggested_class"] = np.where(
                prediction["p_cancer"] >= prediction["threshold"],
                "CANCER",
                "BENIGN",
            )
            predictions.append(prediction)
            train_case_coefficients = _aggregate_case_coefficients(
                train_rows, train_coefficients
            )
            test_case_coefficients = _aggregate_case_coefficients(
                test_rows, test_coefficients
            )
            coefficients.append(
                _coefficient_long_table(
                    test_case_coefficients,
                    representation=specification.family,
                    budget=specification.budget,
                    split_id=split_id,
                )
            )
            reconstruction.append(
                _reconstruction_metrics(
                    encoder,
                    test_rows,
                    test_coefficients,
                    representation=specification.family,
                    budget=specification.budget,
                    split_id=split_id,
                )
            )
            confounders.append(
                _confounder_analysis(
                    train_case_coefficients,
                    test_case_coefficients,
                    context=context,
                    representation=specification.family,
                    budget=specification.budget,
                    split_id=split_id,
                )
            )
            basis_key = f"{specification.name}/split_{split_id:03d}"
            fitted_bases[basis_key] = encoder
            basis_metadata[basis_key] = {
                **encoder.metadata(),
                "training_patient_count": len(train_patients),
                "training_patient_fingerprint": _text_fingerprint(train_patients),
            }
            if split_id == 0:
                _record_reconstruction_examples(
                    examples,
                    encoder,
                    test_rows,
                    test_coefficients,
                    prefix=specification.name,
                    limit=max_examples,
                )
    metrics_frame = pd.DataFrame(fold_metrics)
    reconstruction_frame = pd.concat(reconstruction, ignore_index=True)
    summary = _summarize_metrics(metrics_frame, reconstruction_frame)
    return {
        "fold_manifest": fold_manifest,
        "fold_metrics": metrics_frame,
        "summary": summary,
        "raw100_fold_metrics": raw100["fold_metrics"],
        "raw100_summary": raw100["summary"],
        "raw100_predictions": raw100["predictions"],
        "polar_to_raw100": _compare_to_raw100(summary, raw100["summary"]),
        "predictions": pd.concat(predictions, ignore_index=True),
        "coefficients": pd.concat(coefficients, ignore_index=True),
        "reconstruction": reconstruction_frame,
        "reconstruction_examples": examples,
        "confounders": pd.concat(confounders, ignore_index=True),
        "confounder_availability": _confounder_availability(context),
        "bases": fitted_bases,
        "basis_metadata": basis_metadata,
    }


def _evaluate_raw100_baseline(
    *,
    dataframe: pd.DataFrame,
    context: pd.DataFrame,
    split_pairs: list[tuple[np.ndarray, np.ndarray]],
    model_definition: dict[str, Any],
    threshold_policy: str,
    target_sensitivity: float,
    product_threshold: float,
    seed: int,
    folds: int,
) -> dict[str, pd.DataFrame]:
    """Evaluate the uncompressed 100-bin product profile on matched folds."""
    target_rows = _target_measurement_rows(dataframe, model_definition)
    profile_column = str(model_definition["profile_column"])
    fold_metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for split_id, (train_index, test_index) in enumerate(split_pairs):
        train_patients = set(context.iloc[train_index]["patientId"].astype(str))
        test_patients = set(context.iloc[test_index]["patientId"].astype(str))
        if train_patients.intersection(test_patients):
            raise RuntimeError("Patient leakage detected in raw100 baseline.")
        train_rows = target_rows[
            target_rows["patientId"].astype(str).isin(train_patients)
        ].copy()
        test_rows = target_rows[
            target_rows["patientId"].astype(str).isin(test_patients)
        ].copy()
        train_profiles = profile_matrix(train_rows, profile_column)
        test_profiles = profile_matrix(test_rows, profile_column)
        if train_profiles.shape[1] != 100 or test_profiles.shape[1] != 100:
            raise PolarBasisExperimentError(
                "raw100 baseline requires the frozen 100-bin product profile."
            )
        fitted = _fit_product_fold(
            dataframe=dataframe,
            context=context,
            train_rows=train_rows,
            test_rows=test_rows,
            train_coefficients=train_profiles,
            test_coefficients=test_profiles,
            train_patients=train_patients,
            test_patients=test_patients,
            model_definition=model_definition,
            threshold_policy=threshold_policy,
            target_sensitivity=target_sensitivity,
            product_threshold=product_threshold,
            seed=seed + split_id,
        )
        metric = _classification_metrics(
            fitted["test_features"]["label"].to_numpy(dtype=int),
            fitted["test_scores"],
            threshold=fitted["threshold"],
        )
        fold_metrics.append(
            {
                "representation": "raw100",
                "budget": 100,
                "split_id": split_id,
                "repeat_id": split_id // folds,
                "fold_id": split_id % folds,
                "threshold_policy": threshold_policy,
                **metric,
            }
        )
        prediction = fitted["test_features"][
            [TARGET_CASE_ID, "patientId", "target_side", "label", "label_name"]
        ].copy()
        prediction.insert(0, "representation", "raw100")
        prediction.insert(1, "budget", 100)
        prediction.insert(2, "split_id", split_id)
        prediction["p_cancer"] = fitted["test_scores"]
        prediction["threshold"] = fitted["threshold"]
        prediction["suggested_class"] = np.where(
            prediction["p_cancer"] >= prediction["threshold"],
            "CANCER",
            "BENIGN",
        )
        predictions.append(prediction)
    metrics = pd.DataFrame(fold_metrics)
    return {
        "fold_metrics": metrics,
        "summary": _summarize_classification_metrics(metrics),
        "predictions": pd.concat(predictions, ignore_index=True),
    }


def _fit_product_fold(
    *,
    dataframe: pd.DataFrame,
    context: pd.DataFrame,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    train_coefficients: np.ndarray,
    test_coefficients: np.ndarray,
    train_patients: set[str],
    test_patients: set[str],
    model_definition: dict[str, Any],
    threshold_policy: str,
    target_sensitivity: float,
    product_threshold: float,
    seed: int,
) -> dict[str, Any]:
    train = train_rows.copy()
    test = test_rows.copy()
    train["polar_coefficients"] = list(train_coefficients)
    test["polar_coefficients"] = list(test_coefficients)
    lr1 = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(model_definition["lr1_logreg_c"]),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    ).fit(
        profile_matrix(train, "polar_coefficients"),
        row_labels(train, model_definition["label_column"]),
    )
    train_partition = dataframe[
        dataframe[model_definition["group_column"]].astype(str).isin(train_patients)
    ]
    test_partition = dataframe[
        dataframe[model_definition["group_column"]].astype(str).isin(test_patients)
    ]
    train_scores = score_lr1_rows(
        lr1,
        train,
        full_df=train_partition,
        profile_column="polar_coefficients",
        group_column=model_definition["group_column"],
        side_column=model_definition["side_column"],
        label_column=model_definition["label_column"],
        biopsy_column=model_definition["biopsy_column"],
    )
    test_scores = score_lr1_rows(
        lr1,
        test,
        full_df=test_partition,
        profile_column="polar_coefficients",
        group_column=model_definition["group_column"],
        side_column=model_definition["side_column"],
        label_column=model_definition["label_column"],
        biopsy_column=model_definition["biopsy_column"],
    )
    train_features = _attach_profile_scores(context, train_scores, train_patients)
    test_features = _attach_profile_scores(context, test_scores, test_patients)
    final_model = GatedSymmetryLogistic(
        logreg_c=float(model_definition["lr2_logreg_c"]),
        random_state=seed,
    ).fit(train_features, train_features["label"].to_numpy(dtype=int))
    train_final_scores = final_model.predict_proba(train_features)[:, 1]
    test_final_scores = final_model.predict_proba(test_features)[:, 1]
    if threshold_policy == "training_fold_target_sensitivity":
        threshold = float(
            compute_binary_thresholds(
                train_features["label"].to_numpy(dtype=int),
                train_final_scores,
                target_sensitivity=target_sensitivity,
            )["threshold_target"]
        )
    elif threshold_policy == "frozen_product_threshold":
        threshold = product_threshold
    else:
        raise PolarBasisExperimentError(
            f"Unsupported threshold policy: {threshold_policy!r}."
        )
    return {
        "lr1": lr1,
        "final_model": final_model,
        "train_features": train_features,
        "test_features": test_features,
        "test_scores": test_final_scores,
        "threshold": threshold,
    }


def _classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    thresholds = np.full(len(labels), threshold, dtype=float)
    values = binary_metric_values(labels, scores, thresholds)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        **values,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "test_target_cases": int(len(labels)),
    }


def _summarize_metrics(
    metrics: pd.DataFrame,
    reconstruction: pd.DataFrame,
) -> pd.DataFrame:
    summary = _summarize_classification_metrics(metrics)
    reconstruction_summary = reconstruction.groupby(
        ["representation", "budget"], as_index=False
    ).agg(
        reconstruction_relative_rmse_mean=(
            "reconstruction_relative_rmse",
            "mean",
        ),
        reconstruction_relative_rmse_std=(
            "reconstruction_relative_rmse",
            "std",
        ),
        radial_profile_relative_rmse_mean=(
            "radial_profile_relative_rmse",
            "mean",
        ),
        radial_profile_relative_rmse_std=(
            "radial_profile_relative_rmse",
            "std",
        ),
    )
    return summary.merge(
        reconstruction_summary,
        on=["representation", "budget"],
        how="left",
        validate="one_to_one",
    )


def _summarize_classification_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "sensitivity",
        "specificity",
        "roc_auc",
        "balanced_accuracy",
        "ppv",
        "npv",
        "threshold",
    )
    rows = []
    for (representation, budget), group in metrics.groupby(
        ["representation", "budget"], sort=False
    ):
        row: dict[str, Any] = {
            "representation": representation,
            "budget": int(budget),
            "splits": int(len(group)),
        }
        for column in columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _compare_to_raw100(
    polar_summary: pd.DataFrame,
    raw100_summary: pd.DataFrame,
) -> pd.DataFrame:
    if len(raw100_summary) != 1:
        raise PolarBasisExperimentError("raw100 summary must contain one row.")
    out = polar_summary.copy()
    baseline = raw100_summary.iloc[0]
    for metric in (
        "sensitivity_mean",
        "specificity_mean",
        "roc_auc_mean",
        "balanced_accuracy_mean",
        "ppv_mean",
        "npv_mean",
    ):
        out[f"raw100_{metric}"] = float(baseline[metric])
        out[f"delta_{metric}"] = out[metric] - float(baseline[metric])
    return out


def _reconstruction_metrics(
    encoder: PolarBasisEncoder,
    rows: pd.DataFrame,
    coefficients: np.ndarray,
    *,
    representation: str,
    budget: int,
    split_id: int,
) -> pd.DataFrame:
    original = _candidate_tensor(_stack_harmonics(rows))
    reconstructed = encoder.inverse_transform(coefficients)
    records = []
    for index, row in enumerate(rows.itertuples(index=False)):
        records.append(
            {
                "representation": representation,
                "budget": budget,
                "split_id": split_id,
                "measurement_key": row.measurement_key,
                TARGET_CASE_ID: getattr(row, TARGET_CASE_ID),
                "reconstruction_relative_rmse": _relative_rmse(
                    reconstructed[index], original[index]
                ),
                "radial_profile_relative_rmse": _relative_rmse(
                    reconstructed[index, 0], original[index, 0]
                ),
            }
        )
    return pd.DataFrame(records)


def _aggregate_case_coefficients(
    rows: pd.DataFrame,
    coefficients: np.ndarray,
) -> pd.DataFrame:
    frame = rows[[TARGET_CASE_ID, "patientId", "qc_m1_energy", "qc_m3_energy"]].copy()
    frame["coefficients"] = list(np.asarray(coefficients, dtype=float))
    records = []
    for target_case_id, group in frame.groupby(TARGET_CASE_ID, sort=True):
        records.append(
            {
                TARGET_CASE_ID: target_case_id,
                "patientId": str(group["patientId"].iloc[0]),
                "coefficients": np.mean(np.vstack(group["coefficients"]), axis=0),
                "qc_m1_energy": float(group["qc_m1_energy"].mean()),
                "qc_m3_energy": float(group["qc_m3_energy"].mean()),
            }
        )
    return pd.DataFrame(records)


def _coefficient_long_table(
    cases: pd.DataFrame,
    *,
    representation: str,
    budget: int,
    split_id: int,
) -> pd.DataFrame:
    rows = []
    for case in cases.itertuples(index=False):
        for index, value in enumerate(case.coefficients):
            rows.append(
                {
                    "representation": representation,
                    "budget": budget,
                    "split_id": split_id,
                    TARGET_CASE_ID: getattr(case, TARGET_CASE_ID),
                    "patientId": case.patientId,
                    "coefficient": f"c{index:03d}",
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def _confounder_analysis(
    train_cases: pd.DataFrame,
    test_cases: pd.DataFrame,
    *,
    context: pd.DataFrame,
    representation: str,
    budget: int,
    split_id: int,
) -> pd.DataFrame:
    metadata = context[
        [
            TARGET_CASE_ID,
            *CONTINUOUS_CONFOUNDERS.values(),
            "target_session",
        ]
    ]
    train = train_cases.merge(metadata, on=TARGET_CASE_ID, how="left")
    test = test_cases.merge(metadata, on=TARGET_CASE_ID, how="left")
    rows: list[dict[str, Any]] = []
    for feature_set in ("candidate", "candidate_plus_qc"):
        train_x = _confounder_matrix(train, feature_set)
        test_x = _confounder_matrix(test, feature_set)
        for name, column in CONTINUOUS_CONFOUNDERS.items():
            train_y = pd.to_numeric(train[column], errors="coerce").to_numpy(float)
            test_y = pd.to_numeric(test[column], errors="coerce").to_numpy(float)
            valid_train = np.isfinite(train_y)
            valid_test = np.isfinite(test_y)
            if int(valid_train.sum()) < 5 or int(valid_test.sum()) < 2:
                rows.append(
                    _unavailable_confounder_row(
                        representation,
                        budget,
                        split_id,
                        name,
                        feature_set,
                        "insufficient_finite_values",
                    )
                )
                continue
            model = Pipeline(
                [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
            ).fit(train_x[valid_train], train_y[valid_train])
            predicted = model.predict(test_x[valid_test])
            rows.append(
                {
                    "representation": representation,
                    "budget": budget,
                    "split_id": split_id,
                    "confounder": name,
                    "feature_set": feature_set,
                    "status": "available",
                    "reason": "",
                    "test_rows": int(valid_test.sum()),
                    "mae": float(mean_absolute_error(test_y[valid_test], predicted)),
                    "r2": float(r2_score(test_y[valid_test], predicted))
                    if int(valid_test.sum()) >= 2
                    else float("nan"),
                    "accuracy": float("nan"),
                    "balanced_accuracy": float("nan"),
                    "unseen_test_classes": 0,
                }
            )
        rows.append(
            _session_confounder_row(
                train,
                test,
                train_x,
                test_x,
                representation=representation,
                budget=budget,
                split_id=split_id,
                feature_set=feature_set,
            )
        )
    return pd.DataFrame(rows)


def _session_confounder_row(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    representation: str,
    budget: int,
    split_id: int,
    feature_set: str,
) -> dict[str, Any]:
    train_y = train["target_session"].fillna("").astype(str).to_numpy()
    test_y = test["target_session"].fillna("").astype(str).to_numpy()
    valid_train = train_y != ""
    valid_test = test_y != ""
    train_classes = set(train_y[valid_train])
    train_counts = Counter(train_y[valid_train])
    sparse_or_high_cardinality = (
        len(train_classes) > min(10, max(2, int(valid_train.sum()) // 5))
        or min(train_counts.values(), default=0) < 2
    )
    if (
        len(train_classes) < 2
        or int(valid_test.sum()) < 2
        or sparse_or_high_cardinality
    ):
        return _unavailable_confounder_row(
            representation,
            budget,
            split_id,
            "session",
            feature_set,
            "sparse_or_high_cardinality_session_labels",
        )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=split_id,
                    solver="lbfgs",
                ),
            ),
        ]
    ).fit(train_x[valid_train], train_y[valid_train])
    predicted = model.predict(test_x[valid_test])
    observed = test_y[valid_test]
    unseen = int(sum(value not in train_classes for value in observed))
    return {
        "representation": representation,
        "budget": budget,
        "split_id": split_id,
        "confounder": "session",
        "feature_set": feature_set,
        "status": "available",
        "reason": "",
        "test_rows": int(valid_test.sum()),
        "mae": float("nan"),
        "r2": float("nan"),
        "accuracy": float(accuracy_score(observed, predicted)),
        "balanced_accuracy": _observed_class_balanced_accuracy(observed, predicted),
        "unseen_test_classes": unseen,
    }


def _unavailable_confounder_row(
    representation: str,
    budget: int,
    split_id: int,
    confounder: str,
    feature_set: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "representation": representation,
        "budget": budget,
        "split_id": split_id,
        "confounder": confounder,
        "feature_set": feature_set,
        "status": "unavailable",
        "reason": reason,
        "test_rows": 0,
        "mae": float("nan"),
        "r2": float("nan"),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "unseen_test_classes": 0,
    }


def _observed_class_balanced_accuracy(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    classes = np.unique(observed)
    return float(
        np.mean([np.mean(predicted[observed == label] == label) for label in classes])
    )


def _confounder_matrix(cases: pd.DataFrame, feature_set: str) -> np.ndarray:
    candidate = np.vstack(cases["coefficients"])
    if feature_set == "candidate":
        return candidate
    qc = cases[["qc_m1_energy", "qc_m3_energy"]].to_numpy(dtype=float)
    return np.hstack([candidate, qc])


def _confounder_availability(context: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, column in CONTINUOUS_CONFOUNDERS.items():
        values = pd.to_numeric(context[column], errors="coerce")
        count = int(values.notna().sum())
        output[name] = {
            "source_column": column,
            "available_target_cases": count,
            "unique_values": int(values.dropna().nunique()),
            "status": "available" if count >= 2 else "unavailable",
            "unavailable_reason": "" if count >= 2 else "insufficient_finite_values",
        }
    sessions = context["target_session"].replace("", np.nan).dropna().astype(str)
    output["session"] = {
        "source_column": CALIBRATION_SESSION_COLUMN,
        "available_target_cases": int(len(sessions)),
        "unique_values": int(sessions.nunique()),
        "status": "available" if sessions.nunique() >= 2 else "unavailable",
        "unavailable_reason": ""
        if sessions.nunique() >= 2
        else "fewer_than_two_session_labels",
    }
    output["unavailable_confounders_are_not_imputed"] = True
    return output


def _record_reconstruction_examples(
    destination: dict[str, np.ndarray],
    encoder: PolarBasisEncoder,
    rows: pd.DataFrame,
    coefficients: np.ndarray,
    *,
    prefix: str,
    limit: int,
) -> None:
    original = _candidate_tensor(_stack_harmonics(rows))[:limit]
    reconstructed = encoder.inverse_transform(coefficients[:limit])
    destination[f"{prefix}_measurement_keys"] = (
        rows["measurement_key"].astype(str).to_numpy()[:limit]
    )
    destination[f"{prefix}_original_candidate_harmonics"] = original.astype(np.float32)
    destination[f"{prefix}_reconstructed_candidate_harmonics"] = reconstructed.astype(
        np.float32
    )


def _write_artifacts(
    *,
    run_folder: Path,
    config: dict[str, Any],
    config_path: Path,
    effective_preprocessing: dict[str, Any],
    data_version: dict[str, Any],
    lineage: dict[str, Any],
    axes: PolarAxes,
    cake_manifest: pd.DataFrame,
    result: dict[str, Any],
    product_threshold: float,
) -> None:
    (run_folder / "effective_experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_folder / "effective_training_preprocessing.yaml").write_text(
        yaml.safe_dump(effective_preprocessing, sort_keys=False), encoding="utf-8"
    )
    pointer = resolve_config_path(data_version["pointer_path"], config_path)
    (run_folder / "dvc_data_pointer.dvc").write_bytes(pointer.read_bytes())
    cohort_manifest = (
        result["fold_manifest"][[TARGET_CASE_ID, "patientId", "label"]]
        .drop_duplicates()
        .merge(
            cake_manifest.groupby(TARGET_CASE_ID, as_index=False).agg(
                target_measurements=("measurement_key", "nunique")
            ),
            on=TARGET_CASE_ID,
            how="left",
            validate="one_to_one",
        )
        .sort_values(["patientId", TARGET_CASE_ID])
    )
    cohort_manifest.to_csv(run_folder / "cohort_manifest.csv", index=False)
    joblib.dump(result["bases"], run_folder / "basis.joblib", compress=3)
    basis_metadata = deepcopy(result["basis_metadata"])
    basis_metadata["basis_joblib_sha256"] = file_sha256(run_folder / "basis.joblib")
    _write_json(run_folder / "basis_metadata.json", basis_metadata)
    np.savez_compressed(
        run_folder / "q_chi_axes.npz",
        q=axes.q,
        chi=axes.chi,
        harmonic_q_mask=np.asarray(axes.harmonic_q_mask, dtype=bool),
        harmonic_q=axes.harmonic_q,
    )
    result["coefficients"].to_parquet(
        run_folder / "coefficient_table.parquet", index=False
    )
    result["fold_manifest"].to_csv(run_folder / "fold_manifest.csv", index=False)
    result["fold_metrics"].to_csv(run_folder / "fold_metrics.csv", index=False)
    result["summary"].to_csv(run_folder / "metrics.csv", index=False)
    result["predictions"].to_csv(run_folder / "predictions.csv", index=False)
    result["raw100_fold_metrics"].to_csv(
        run_folder / "raw100_fold_metrics.csv", index=False
    )
    result["raw100_summary"].to_csv(run_folder / "raw100_metrics.csv", index=False)
    result["raw100_predictions"].to_csv(
        run_folder / "raw100_predictions.csv", index=False
    )
    result["polar_to_raw100"].to_csv(
        run_folder / "polar_to_raw100_comparison.csv", index=False
    )
    cake_manifest.to_csv(run_folder / "polar_cake_manifest.csv", index=False)
    experiment_lineage = {
        "effective_config_sha256": file_sha256(
            run_folder / "effective_experiment_config.yaml"
        ),
        "fold_manifest_sha256": file_sha256(run_folder / "fold_manifest.csv"),
        "cohort_manifest_sha256": file_sha256(run_folder / "cohort_manifest.csv"),
        "polar_cake_manifest_sha256": file_sha256(
            run_folder / "polar_cake_manifest.csv"
        ),
        "dataset_fingerprint": _text_fingerprint(
            [
                data_version["input_h5_sha256"],
                *cohort_manifest[TARGET_CASE_ID].astype(str),
                *cake_manifest["measurement_key"].astype(str),
            ]
        ),
    }
    lineage["experiment"] = experiment_lineage
    _write_json(run_folder / "lineage.json", lineage)
    basis_metadata["fold_manifest_sha256"] = experiment_lineage["fold_manifest_sha256"]
    basis_metadata["effective_config_sha256"] = experiment_lineage[
        "effective_config_sha256"
    ]
    _write_json(run_folder / "basis_metadata.json", basis_metadata)
    result["reconstruction"].to_csv(
        run_folder / "reconstruction_metrics.csv", index=False
    )
    np.savez_compressed(
        run_folder / "reconstruction_examples.npz", **result["reconstruction_examples"]
    )
    result["confounders"].to_csv(run_folder / "confounder_analysis.csv", index=False)
    _write_json(
        run_folder / "confounder_availability.json", result["confounder_availability"]
    )
    _write_json(
        run_folder / "run_manifest.json",
        {
            "contract": CONTRACT,
            "clinical_stage": "research_only",
            "endpoint": "target_breast_BENIGN_vs_CANCER_decision_support",
            "representations": list(REPRESENTATIONS),
            "matched_baseline": "raw100_product_architecture",
            "coefficient_budgets": list(COEFFICIENT_BUDGETS),
            "candidate_modes": list(CANDIDATE_MODES),
            "qc_modes": list(QC_MODES),
            "qc_modes_used_for_cancer_prediction": False,
            "product_architecture": "LR1_measurement_then_logit_average_then_age_and_gated_SK_Core4_LR2",
            "threshold_policy": config["evaluation"]["threshold_policy"],
            "immutable_product_threshold": product_threshold,
            "product_artifact_modified": False,
            "full_cohort_polar_cakes_may_be_generated_from_existing_h5": True,
            "new_measurements_allowed": False,
            "cached_polar_cakes": int(len(cake_manifest)),
            "full_polar_q_bins": int(len(axes.q)),
            "harmonic_model_q_bins": int(len(axes.harmonic_q)),
            "harmonic_model_q_range": [
                float(axes.harmonic_q[0]),
                float(axes.harmonic_q[-1]),
            ],
            "post_preprocessing_accepted_target_measurements": int(
                cake_manifest["measurement_key"].nunique()
            ),
            "preprocessing_drop_accounting_source": (
                "effective_training_preprocessing_and_canonical_pipeline"
            ),
            "dataset_fingerprint": experiment_lineage["dataset_fingerprint"],
            "fold_manifest_sha256": experiment_lineage["fold_manifest_sha256"],
            "target_cases": int(result["fold_manifest"][TARGET_CASE_ID].nunique()),
            "required_artifacts": list(REQUIRED_ARTIFACTS),
            "limitations": [
                "single retrospective training archive",
                "no independent blind validation cohort",
                "basis comparison is exploratory and not a product release",
                "polar compression and the common harmonic q-range restriction are not separated",
                "session/date/thickness analyses depend on recorded metadata availability",
                "m1 and m3 are QC/confounder channels and are excluded from cancer prediction",
            ],
        },
    )
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_folder / name).is_file()]
    if missing:
        raise PolarBasisExperimentError(
            f"Required experiment artifacts missing: {missing}"
        )


def _log_mlflow(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_folder: Path,
    lineage: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    tracking = config["mlflow"]
    uri = _tracking_uri(str(tracking["tracking_uri"]), config_path)
    params = {
        "contract": CONTRACT,
        "polar.n_q": config["polar_cakes"]["n_q"],
        "polar.n_chi": config["polar_cakes"]["n_chi"],
        "polar.normalization_q_range": ",".join(
            str(value) for value in config["polar_cakes"]["normalization_q_range"]
        ),
        "polar.harmonic_q_range": ",".join(
            str(value) for value in config["polar_cakes"]["harmonic_q_range"]
        ),
        "representations.families": ",".join(config["representations"]["families"]),
        "representations.candidate_modes": ",".join(
            str(value) for value in config["representations"]["candidate_modes"]
        ),
        "representations.qc_modes": ",".join(
            str(value) for value in config["representations"]["qc_modes"]
        ),
        "representations.coefficient_budgets": ",".join(
            str(value) for value in config["representations"]["coefficient_budgets"]
        ),
        "evaluation.method": config["evaluation"]["method"],
        "evaluation.folds": config["evaluation"]["folds"],
        "evaluation.repeats": config["evaluation"]["repeats"],
        "evaluation.seed": config["evaluation"]["seed"],
        "evaluation.target_sensitivity": config["evaluation"]["target_sensitivity"],
        "evaluation.threshold_policy": config["evaluation"]["threshold_policy"],
        "runtime.reconstruction_examples_per_variant": config["runtime"][
            "reconstruction_examples_per_variant"
        ],
    }
    tags = {
        "product": "aramina",
        "clinical_stage": "research_only",
        "endpoint": "target_breast_BENIGN_vs_CANCER",
        "input_h5_checksum": lineage["data_version"]["input_h5_sha256"],
        "dataset_fingerprint": lineage["experiment"]["dataset_fingerprint"],
        "fold_manifest_sha256": lineage["experiment"]["fold_manifest_sha256"],
        "effective_config_sha256": lineage["experiment"]["effective_config_sha256"],
        "dvc": lineage["data_version"],
        "source_code": lineage["source_code"],
        "model": lineage["model"],
    }
    metrics = {}
    raw100 = result["raw100_summary"].iloc[0]
    for name in (
        "sensitivity_mean",
        "specificity_mean",
        "roc_auc_mean",
        "balanced_accuracy_mean",
        "ppv_mean",
        "npv_mean",
    ):
        metrics[f"raw100.{name}"] = float(raw100[name])
    for row in result["summary"].itertuples(index=False):
        prefix = f"{row.representation}.{row.budget}"
        for name in (
            "sensitivity_mean",
            "specificity_mean",
            "roc_auc_mean",
            "balanced_accuracy_mean",
            "ppv_mean",
            "npv_mean",
        ):
            metrics[f"{prefix}.{name}"] = float(getattr(row, name))
            metrics[f"{prefix}.delta_{name}"] = float(
                getattr(row, name) - raw100[name]
            )
    run_name = f"{config['experiment']['name']}_{run_folder.name.rsplit('_', 1)[-1]}"
    with MlflowRun(
        enabled=True,
        tracking_uri=uri,
        experiment_name=str(tracking["experiment_name"]),
        run_name=run_name,
        params=params,
        tags=tags,
    ) as run:
        run.log_metrics(metrics)
        run.log_artifact_directory(
            run_folder,
            required_files=REQUIRED_ARTIFACTS,
            artifact_path="polar_basis_compression",
        )
        run_id = run.run_id
    return {
        "enabled": True,
        "run_id": run_id,
        "status": run.status,
        "tracking_uri": uri,
    }


def _shared_patient_folds(
    context: pd.DataFrame,
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    split_pairs = _patient_split_pairs(
        mode="stratified_kfold",
        base_features=context,
        y_patients=context["label"].to_numpy(dtype=int),
        n_splits=folds,
        n_repeats=repeats,
        random_state=seed,
    )
    rows = []
    for split_id, (train_index, test_index) in enumerate(split_pairs):
        train_cases = context.iloc[train_index]
        test_cases = context.iloc[test_index]
        train_patients = set(train_cases["patientId"].astype(str))
        test_patients = set(test_cases["patientId"].astype(str))
        if train_patients.intersection(test_patients):
            raise RuntimeError("Patient leakage detected in shared fold manifest.")
        for partition, cases in (("train", train_cases), ("test", test_cases)):
            for case in cases.itertuples(index=False):
                rows.append(
                    {
                        "split_id": split_id,
                        "repeat_id": split_id // folds,
                        "fold_id": split_id % folds,
                        "partition": partition,
                        "patientId": str(case.patientId),
                        TARGET_CASE_ID: getattr(case, TARGET_CASE_ID),
                        "label": int(case.label),
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest.duplicated(["split_id", TARGET_CASE_ID]).any():
        raise RuntimeError("A target case appears twice in one shared split manifest.")
    return split_pairs, manifest


def _build_context(dataframe: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    neutral = empty_lr1_scores(
        dataframe,
        group_column=model["group_column"],
        side_column=model["side_column"],
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
    )
    context = patient_feature_table(
        dataframe,
        neutral,
        profile_column=model["profile_column"],
        label_column=model["label_column"],
        group_column=model["group_column"],
        specimen_column=model["specimen_column"],
        side_column=model["side_column"],
        q_column=model["q_column"],
        age_column=model["age_column"],
        biopsy_column=model["biopsy_column"],
    )
    metadata = _target_case_metadata(dataframe, model)
    return context.drop(columns=list(PROFILE_SCORE_COLUMNS)).merge(
        metadata, on=TARGET_CASE_ID, how="left", validate="one_to_one"
    )


def _target_case_metadata(
    dataframe: pd.DataFrame, model: dict[str, Any]
) -> pd.DataFrame:
    rows = _target_measurement_rows(dataframe, model)
    records = []
    for target_case_id, group in rows.groupby(TARGET_CASE_ID, sort=True):
        started_source = (
            group["started_at"]
            if "started_at" in group
            else pd.Series(pd.NaT, index=group.index)
        )
        started = pd.to_datetime(started_source, errors="coerce", utc=True)
        date_value = (
            float(started.dropna().astype("int64").median() / 86_400e9)
            if started.notna().any()
            else float("nan")
        )
        sessions = (
            group.get(CALIBRATION_SESSION_COLUMN, pd.Series(dtype=object))
            .dropna()
            .astype(str)
        )
        session = Counter(sessions).most_common(1)[0][0] if len(sessions) else ""
        thickness_source = (
            group["sample_thickness_mm"]
            if "sample_thickness_mm" in group
            else pd.Series(float("nan"), index=group.index)
        )
        thickness = pd.to_numeric(thickness_source, errors="coerce")
        records.append(
            {
                TARGET_CASE_ID: target_case_id,
                "target_thickness_mm": float(thickness.median())
                if thickness.notna().any()
                else float("nan"),
                "target_date_ordinal": date_value,
                "target_session": session,
            }
        )
    return pd.DataFrame(records)


def _target_measurement_rows(
    dataframe: pd.DataFrame, model: dict[str, Any]
) -> pd.DataFrame:
    rows = lr1_training_rows(
        dataframe,
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
        lr1_row_policy=model["lr1_row_policy"],
    ).copy()
    rows["_side_norm"] = rows[model["side_column"]].map(normalize_side)
    if rows["_side_norm"].isna().any():
        raise PolarBasisExperimentError(
            "Target measurement has unsupported breast side."
        )
    rows[TARGET_CASE_ID] = (
        rows[model["group_column"]].astype(str) + "::" + rows["_side_norm"].astype(str)
    )
    rows["_label"] = row_labels(rows, model["label_column"])
    rows["measurement_key"] = [
        _measurement_key(row, index) for index, (_, row) in enumerate(rows.iterrows())
    ]
    if rows["measurement_key"].duplicated().any():
        raise PolarBasisExperimentError("Polar measurement keys are not unique.")
    return rows.reset_index(drop=True)


def _select_pilot_cohort(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    model: dict[str, Any],
) -> pd.DataFrame:
    maximum = config["cohort"]["max_patients_per_class"]
    if maximum is None:
        return dataframe.reset_index(drop=True)
    context = _build_context(dataframe, model)
    patient_labels = context.groupby("patientId", as_index=False)["label"].max()
    selected = []
    for label in (0, 1):
        selected.extend(
            patient_labels.loc[patient_labels["label"] == label, "patientId"]
            .astype(str)
            .sort_values()
            .head(int(maximum))
            .tolist()
        )
    out = dataframe[dataframe["patientId"].astype(str).isin(selected)].copy()
    if out.empty:
        raise PolarBasisExperimentError("Pilot cohort selection produced no rows.")
    return out.reset_index(drop=True)


def _attach_profile_scores(
    context: pd.DataFrame,
    scores: pd.DataFrame,
    patients: set[str],
) -> pd.DataFrame:
    subset = context[context["patientId"].astype(str).isin(patients)].copy()
    out = subset.merge(scores, on=TARGET_CASE_ID, how="inner", validate="one_to_one")
    if len(out) != len(subset):
        raise RuntimeError("LR1 scores do not cover every target case in the fold.")
    return out.reset_index(drop=True)


def _model_definition(model_artifact: dict[str, Any]) -> dict[str, Any]:
    raw = yaml.safe_load(model_artifact["model_definition_yaml"])
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise PolarBasisExperimentError("Frozen artifact has no model definition.")
    model = raw["model"]
    required = {
        "profile_column",
        "label_column",
        "group_column",
        "specimen_column",
        "side_column",
        "q_column",
        "age_column",
        "biopsy_column",
        "lr1_row_policy",
        "lr1_logreg_c",
        "lr2_logreg_c",
    }
    missing = sorted(required.difference(model))
    if missing:
        raise PolarBasisExperimentError(f"Frozen model definition missing: {missing}")
    return model


def _radial_basis(
    family: str,
    *,
    q: np.ndarray,
    terms_by_channel: dict[str, int],
) -> dict[str, np.ndarray]:
    q_values = np.asarray(q, dtype=float)
    scaled = (q_values - q_values.min()) / (q_values.max() - q_values.min())
    bases = {}
    for channel in _candidate_channel_names():
        mode = int(channel.removeprefix("m").removeprefix("A"))
        terms = terms_by_channel[channel]
        if family == "fourier_bspline":
            basis = _bspline_design(scaled, terms)
        elif family == "fourier_bessel":
            zeros = jn_zeros(mode, terms)
            basis = np.column_stack([jv(mode, zero * scaled) for zero in zeros])
        else:
            raise PolarBasisExperimentError(f"Unknown fixed radial basis: {family}")
        bases[channel] = basis
    return bases


def _bspline_design(x: np.ndarray, n_basis: int) -> np.ndarray:
    degree = min(3, n_basis - 1)
    internal_count = n_basis - degree - 1
    internal = (
        np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
        if internal_count > 0
        else np.array([], dtype=float)
    )
    knots = np.concatenate(
        [np.repeat(0.0, degree + 1), internal, np.repeat(1.0, degree + 1)]
    )
    return BSpline.design_matrix(x, knots, degree, extrapolate=False).toarray()


def _candidate_channel_names() -> tuple[str, ...]:
    return ("m0", "A2", "A4")


def _budget_allocation(budget: int) -> dict[str, int]:
    channels = _candidate_channel_names()
    base, remainder = divmod(int(budget), len(channels))
    return {
        channel: base + int(index < remainder) for index, channel in enumerate(channels)
    }


def _candidate_tensor(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 3 or matrix.shape[1] < 9:
        raise PolarBasisExperimentError("Expected m=0..4 harmonic tensor.")
    return np.stack(
        [
            matrix[:, 0, :],
            np.hypot(matrix[:, 3, :], matrix[:, 4, :]),
            np.hypot(matrix[:, 7, :], matrix[:, 8, :]),
        ],
        axis=1,
    )


def _stack_harmonics(rows: pd.DataFrame) -> np.ndarray:
    return np.stack(rows["harmonic_matrix"].to_numpy())


def _case_balanced_harmonics(rows: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            np.mean(np.stack(group["harmonic_matrix"].to_numpy()), axis=0)
            for _, group in rows.groupby(TARGET_CASE_ID, sort=True)
        ]
    )


def _mode_energy(harmonics: np.ndarray, mode: int) -> float:
    offset = 1 + 2 * (mode - 1)
    return float(np.sqrt(np.mean(harmonics[offset : offset + 2] ** 2)))


def _normalize_cake(
    intensity: np.ndarray,
    count: np.ndarray,
    q: np.ndarray,
    *,
    normalization_q_range: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(intensity, dtype=float)
    weights = np.asarray(count, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    clean = np.where(valid, values, 0.0)
    scale = _cake_normalization_scale(
        values,
        weights,
        q,
        normalization_q_range=normalization_q_range,
    )
    if not np.isfinite(scale) or scale <= 1e-12:
        raise PolarBasisExperimentError("Polar cake normalization scale is invalid.")
    return clean / scale


def _cake_normalization_scale(
    intensity: np.ndarray,
    count: np.ndarray,
    q: np.ndarray,
    *,
    normalization_q_range: tuple[float, float],
) -> float:
    values = np.asarray(intensity, dtype=float)
    weights = np.asarray(count, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    numerator = np.sum(np.where(valid, values * weights, 0.0), axis=0)
    denominator = np.sum(np.where(valid, weights, 0.0), axis=0)
    radial = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
    band = (q >= normalization_q_range[0]) & (q <= normalization_q_range[1])
    return float(np.nanmedian(radial[band]))


def _relative_rmse(observed: np.ndarray, reference: np.ndarray) -> float:
    observed_values = np.asarray(observed, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    scale = float(np.sqrt(np.mean(reference_values**2)))
    if scale <= 1e-12:
        return float("nan")
    return float(np.sqrt(np.mean((observed_values - reference_values) ** 2)) / scale)


def _validate_shared_axes(
    shared_q: np.ndarray | None,
    shared_chi: np.ndarray | None,
    q: np.ndarray,
    chi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if shared_q is None or shared_chi is None:
        return q, chi
    _validate_axes(PolarAxes(shared_q, shared_chi), q, chi)
    return shared_q, shared_chi


def _axis_contract_fingerprint(
    *,
    n_q: int,
    n_chi: int,
    radial_q_range: tuple[float, float],
    azimuthal_range: tuple[float, float],
) -> str:
    payload = {
        "n_q": int(n_q),
        "n_chi": int(n_chi),
        "radial_q_range": [float(value) for value in radial_q_range],
        "azimuthal_range": [float(value) for value in azimuthal_range],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _cake_manifest_record(
    row: pd.Series,
    *,
    key: str,
    artifact: str,
    dataset_sha256: str,
    n_q: int,
    n_chi: int,
    axis_contract: str,
    radial_q_range: tuple[float, float],
    azimuthal_range: tuple[float, float],
) -> dict[str, Any]:
    return {
        "measurement_key": key,
        "dataset_sha256": dataset_sha256,
        "patient_id": str(row["patientId"]),
        "target_case_id": str(row[TARGET_CASE_ID]),
        "label": int(row["_label"]),
        "n_q": n_q,
        "n_chi": n_chi,
        "axis_contract_sha256": axis_contract,
        "radial_q_min": float(radial_q_range[0]),
        "radial_q_max": float(radial_q_range[1]),
        "azimuthal_min": float(azimuthal_range[0]),
        "azimuthal_max": float(azimuthal_range[1]),
        "artifact": artifact,
        "error_model": "poisson",
        "integration_method": "pyfai_integrate2d_default",
        "unit": "q_nm^-1",
        "calibration_session_uid": str(row.get(CALIBRATION_SESSION_COLUMN, "")),
        "poni_sha256": _poni_fingerprint(row.get("ponifile")),
        "sample_thickness_mm": _finite_or_nan(row.get("sample_thickness_mm")),
        "calibrant_thickness_mm": _finite_or_nan(
            row.get("calibrant_thickness_mm")
        ),
        "mask_fraction": _mask_fraction(
            row.get(MASK_COLUMN), row.get(RAW_FRAME_COLUMN)
        ),
    }


def _axes_match_contract(
    q: np.ndarray,
    chi: np.ndarray,
    *,
    n_q: int,
    n_chi: int,
    radial_q_range: tuple[float, float],
    azimuthal_range: tuple[float, float],
) -> bool:
    expected = _canonical_axes(
        n_q=n_q,
        n_chi=n_chi,
        radial_q_range=radial_q_range,
        azimuthal_range=azimuthal_range,
    )
    return bool(
        np.asarray(q).shape == expected.q.shape
        and np.asarray(chi).shape == expected.chi.shape
        and np.allclose(q, expected.q, rtol=1e-7, atol=1e-4)
        and np.allclose(chi, expected.chi, rtol=1e-7, atol=1e-4)
    )


def _canonical_axes(
    *,
    n_q: int,
    n_chi: int,
    radial_q_range: tuple[float, float],
    azimuthal_range: tuple[float, float],
) -> PolarAxes:
    q_step = (radial_q_range[1] - radial_q_range[0]) / n_q
    chi_step = (azimuthal_range[1] - azimuthal_range[0]) / n_chi
    return PolarAxes(
        q=np.linspace(
            radial_q_range[0] + 0.5 * q_step,
            radial_q_range[1] - 0.5 * q_step,
            n_q,
        ),
        chi=np.linspace(
            azimuthal_range[0] + 0.5 * chi_step,
            azimuthal_range[1] - 0.5 * chi_step,
            n_chi,
        ),
    )


def _validate_axes(axes: PolarAxes, q: np.ndarray, chi: np.ndarray) -> None:
    if not np.allclose(axes.q, q, rtol=1e-7, atol=1e-4) or not np.allclose(
        axes.chi, chi, rtol=1e-7, atol=1e-4
    ):
        raise PolarBasisExperimentError("Polar cakes do not share fixed q/chi axes.")


def _measurement_key(row: pd.Series, index: int) -> str:
    raw = "|".join(
        str(row.get(column, ""))
        for column in ("patientId", "specimenId", "side", "position", "started_at")
    )
    digest = hashlib.sha256(raw.encode("utf-8"))
    frame = np.asarray(row.get(RAW_FRAME_COLUMN))
    if frame.ndim == 2:
        digest.update(np.ascontiguousarray(frame).tobytes())
    else:
        digest.update(f"row={index}".encode("ascii"))
    return digest.hexdigest()[:24]


def _poni_fingerprint(value: Any) -> str:
    text = str(value)
    if "\n" not in text and len(text) < 1024:
        path = Path(text).expanduser()
        try:
            if path.is_file():
                return file_sha256(path)
        except OSError:
            pass
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _mask_fraction(mask_value: Any, frame_value: Any) -> float:
    frame = np.asarray(frame_value)
    mask = np.asarray(mask_value)
    if frame.ndim != 2 or frame.size == 0:
        return float("nan")
    if mask.shape == frame.shape:
        return float(np.mean(mask.astype(bool)))
    if mask.ndim == 2 and mask.shape[1] == 2:
        return float(len(mask) / frame.size)
    if mask.size == 0:
        return 0.0
    return float("nan")


def _array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _text_fingerprint(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Polar basis config is unavailable: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate_config(config)
    return config


def _validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise PolarBasisExperimentError("Polar basis config must be a mapping.")
    required = {
        "contract",
        "experiment",
        "input",
        "data_version",
        "cohort",
        "polar_cakes",
        "representations",
        "evaluation",
        "confounders",
        "runtime",
        "mlflow",
        "output",
    }
    if set(config) != required:
        raise PolarBasisExperimentError(
            f"Polar basis config fields invalid; missing={sorted(required - set(config))}, "
            f"unknown={sorted(set(config) - required)}."
        )
    if config["contract"] != CONTRACT:
        raise PolarBasisExperimentError(f"Unsupported contract: {config['contract']!r}")
    _exact_keys(
        config["experiment"],
        {"name", "model_name", "model_version"},
        "experiment",
    )
    if (
        config["experiment"].get("model_name") != FROZEN_MODEL_NAME
        or config["experiment"].get("model_version") != FROZEN_MODEL_VERSION
    ):
        raise PolarBasisExperimentError(
            "Experiment must pin frozen Aramina 0.2.14-beta."
        )
    _nonempty(config["experiment"].get("name"), "experiment.name")
    _exact_keys(config["input"], {"input_h5_path", "model_joblib_path"}, "input")
    for key in ("input_h5_path", "model_joblib_path"):
        _nonempty(config["input"].get(key), f"input.{key}")
    data = config["data_version"]
    _exact_keys(
        data,
        {"contract", "system", "dataset_id", "dvc_version", "pointer_path"},
        "data_version",
    )
    if data.get("contract") != DVC_DATA_CONTRACT or data.get("system") != "dvc":
        raise PolarBasisExperimentError(
            "Experiment requires Aramina DVC input contract."
        )
    for key in ("dataset_id", "dvc_version", "pointer_path"):
        _nonempty(data.get(key), f"data_version.{key}")
    _exact_keys(config["cohort"], {"max_patients_per_class"}, "cohort")
    maximum = config["cohort"]["max_patients_per_class"]
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 2
    ):
        raise PolarBasisExperimentError(
            "cohort.max_patients_per_class must be null or an integer >=2."
        )
    polar = config["polar_cakes"]
    _exact_keys(
        polar,
        {
            "n_q",
            "n_chi",
            "radial_q_range",
            "azimuthal_range",
            "normalization_q_range",
            "harmonic_q_range",
            "cache_folder",
            "force_rebuild",
        },
        "polar_cakes",
    )
    if int(polar.get("n_q", 0)) != 256 or int(polar.get("n_chi", 0)) != 36:
        raise PolarBasisExperimentError(
            "Polar grid must remain fixed at 256 q x 36 chi."
        )
    if polar.get("radial_q_range") != [2.0, 23.0]:
        raise PolarBasisExperimentError(
            "Polar radial q range must remain fixed at 2.0-23.0 nm^-1."
        )
    if polar.get("azimuthal_range") != [-180.0, 180.0]:
        raise PolarBasisExperimentError(
            "Polar azimuthal range must remain fixed at -180 to 180 degrees."
        )
    if polar.get("normalization_q_range") != [6.7, 7.1]:
        raise PolarBasisExperimentError(
            "Polar normalization q range must remain 6.7-7.1."
        )
    harmonic_q_range = polar.get("harmonic_q_range")
    if (
        not isinstance(harmonic_q_range, list)
        or len(harmonic_q_range) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in harmonic_q_range
        )
        or not 2.0 <= float(harmonic_q_range[0]) < float(harmonic_q_range[1]) <= 23.0
    ):
        raise PolarBasisExperimentError(
            "polar_cakes.harmonic_q_range must be increasing inside 2.0-23.0."
        )
    if not isinstance(polar.get("force_rebuild"), bool):
        raise PolarBasisExperimentError("polar_cakes.force_rebuild must be boolean.")
    representations = config["representations"]
    _exact_keys(
        representations,
        {"families", "candidate_modes", "qc_modes", "coefficient_budgets"},
        "representations",
    )
    if tuple(representations.get("families", [])) != REPRESENTATIONS:
        raise PolarBasisExperimentError(
            f"representations.families must be {REPRESENTATIONS}."
        )
    if (
        tuple(representations.get("candidate_modes", [])) != CANDIDATE_MODES
        or tuple(representations.get("qc_modes", [])) != QC_MODES
    ):
        raise PolarBasisExperimentError(
            "Angular candidate/QC modes are fixed by contract."
        )
    if tuple(representations.get("coefficient_budgets", [])) != COEFFICIENT_BUDGETS:
        raise PolarBasisExperimentError(
            f"Coefficient budgets must be {COEFFICIENT_BUDGETS}."
        )
    evaluation = config["evaluation"]
    _exact_keys(
        evaluation,
        {
            "method",
            "folds",
            "repeats",
            "seed",
            "target_sensitivity",
            "threshold_policy",
        },
        "evaluation",
    )
    if evaluation.get("method") != "repeated_stratified_patient_kfold":
        raise PolarBasisExperimentError(
            "Evaluation must be patient-safe repeated stratified k-fold."
        )
    if evaluation.get("threshold_policy") not in {
        "training_fold_target_sensitivity",
        "frozen_product_threshold",
    }:
        raise PolarBasisExperimentError("Unsupported evaluation.threshold_policy.")
    if int(evaluation.get("folds", 0)) < 2 or int(evaluation.get("repeats", 0)) < 1:
        raise PolarBasisExperimentError("Evaluation folds/repeats are invalid.")
    seed = evaluation.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PolarBasisExperimentError("evaluation.seed must be non-negative integer.")
    sensitivity = evaluation.get("target_sensitivity")
    if (
        isinstance(sensitivity, bool)
        or not isinstance(sensitivity, int | float)
        or not 0.0 < float(sensitivity) <= 1.0
    ):
        raise PolarBasisExperimentError(
            "evaluation.target_sensitivity must be inside (0, 1]."
        )
    _exact_keys(config["confounders"], {"fields"}, "confounders")
    if config["confounders"].get("fields") != ["age", "thickness", "session", "date"]:
        raise PolarBasisExperimentError(
            "Confounder fields must be age, thickness, session, date."
        )
    _exact_keys(
        config["runtime"],
        {"reconstruction_examples_per_variant"},
        "runtime",
    )
    examples = config["runtime"]["reconstruction_examples_per_variant"]
    if isinstance(examples, bool) or not isinstance(examples, int) or examples < 1:
        raise PolarBasisExperimentError(
            "runtime.reconstruction_examples_per_variant must be positive integer."
        )
    _exact_keys(
        config["mlflow"],
        {"enabled", "tracking_uri", "experiment_name"},
        "mlflow",
    )
    if config["mlflow"].get("enabled") is not True:
        raise PolarBasisExperimentError("Polar experiment requires MLflow.")
    _exact_keys(config["output"], {"folder"}, "output")
    for section, key in (
        ("polar_cakes", "cache_folder"),
        ("mlflow", "tracking_uri"),
        ("mlflow", "experiment_name"),
        ("output", "folder"),
    ):
        _nonempty(config[section].get(key), f"{section}.{key}")


def _exact_keys(mapping: Any, expected: set[str], where: str) -> None:
    if not isinstance(mapping, dict):
        raise PolarBasisExperimentError(f"{where} must be a mapping.")
    missing = sorted(expected.difference(mapping))
    unknown = sorted(set(mapping).difference(expected))
    if missing or unknown:
        raise PolarBasisExperimentError(
            f"{where} fields invalid; missing={missing}, unknown={unknown}."
        )


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolarBasisExperimentError(f"{where} must be a non-empty string.")
    return value.strip()


def _resolve_path(value: str, config_path: Path) -> Path:
    return resolve_config_path(_nonempty(value, "path"), config_path)


def _create_run_folder(config: dict[str, Any], config_path: Path) -> Path:
    root = _resolve_path(config["output"]["folder"], config_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"polar_basis_compression_{timestamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
