"""Direct detector Monte Carlo with bounded geometry sensitivity scenarios."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import importlib.util
import json
import os
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
RESULT_CONTRACT = "aramina_joint_measurement_uncertainty_results_v0_3"
COMPONENTS = ("photon", "thickness", "beam_center", "detector_distance")
RESUME_CONTRACT = "aramina_joint_measurement_uncertainty_resume_v0_2"
RUN_STATE_FILENAME = "run_state.json"
PROGRESS_FILENAME = "progress.json"
PROBABILITY_FILENAME = "p_cancer_probability_cube.npy"
SELECTED_FRAME_CACHE = "selected_detector_frame.joblib"
SELECTED_CASES_CACHE = "selected_cases_checkpoint.joblib"
PYFAI_PARITY_CACHE = "pyfai_parity_checkpoint.joblib"
UNIT_CHECKPOINT_FOLDER = "unit_checkpoints"
CONVERGENCE_FOLDER = "convergence"
STOP_REQUEST_FILENAME = "STOP_REQUESTED"
NUISANCE_SCOPE_FILENAME = "nuisance_scope_manifest.csv"


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


@dataclass(frozen=True)
class CohortNuisanceDraws:
    """One cohort-level outer random field indexed by stable measurements."""

    measurement_keys: tuple[str, ...]
    thickness_delta_mm: np.ndarray
    beam_center_row_delta_px: np.ndarray
    beam_center_col_delta_px: np.ndarray
    detector_distance_delta_mm: np.ndarray
    photon_measurement_seeds: np.ndarray

    def for_frame(self, frame: pd.DataFrame) -> NuisanceDraws:
        """Select one patient without changing cohort draw alignment."""
        index_by_key = {key: index for index, key in enumerate(self.measurement_keys)}
        keys = tuple(_measurement_key(row) for _, row in frame.iterrows())
        if len(set(keys)) != len(keys):
            raise ValueError("Patient frame contains duplicate measurement keys.")
        try:
            indices = np.asarray([index_by_key[key] for key in keys], dtype=int)
        except KeyError as error:
            raise ValueError(
                "Patient frame contains a measurement outside the cohort field."
            ) from error
        return NuisanceDraws(
            thickness_delta_mm=self.thickness_delta_mm[:, indices],
            beam_center_row_delta_px=self.beam_center_row_delta_px[:, indices],
            beam_center_col_delta_px=self.beam_center_col_delta_px[:, indices],
            detector_distance_delta_mm=self.detector_distance_delta_mm[:, indices],
            photon_measurement_seeds=self.photon_measurement_seeds[indices],
        )


@dataclass(frozen=True)
class MonteCarloDesign:
    """Explicit independent-geometry and conditional-photon sample counts."""

    mode: str
    geometry_draws: int
    photon_replicates: int
    geometry_stage_draws: int

    @property
    def output_draws(self) -> int:
        return self.geometry_draws * self.photon_replicates

    @property
    def output_stage_draws(self) -> int:
        return self.geometry_stage_draws * self.photon_replicates


@dataclass
class PatientMetalContext:
    """Patient detector frames and prepared-geometry Metal executor."""

    session: Any
    q_grid: np.ndarray
    images: np.ndarray
    geometry_plan_cache: GeometryPlanCache | None = None
    backend_kind: str = "pyfai_prepared_csr_metal_photon_mc"
    nominal_effective_distance_m: np.ndarray | None = None
    nominal_poni1_m: np.ndarray | None = None
    nominal_poni2_m: np.ndarray | None = None
    pixel1_m: np.ndarray | None = None
    pixel2_m: np.ndarray | None = None

    def __enter__(self) -> PatientMetalContext:
        return self

    def __exit__(self, *_: object) -> None:
        close = getattr(self.session, "close", None)
        if close is not None:
            close()


@dataclass(frozen=True)
class PreparedGeometryDraw:
    """Exact pyFAI result and Metal plans for one immutable geometry state."""

    expected: np.ndarray
    plans: tuple[Any, ...]
    geometry_rows: tuple[dict[str, Any], ...]


class GeometryPlanCache:
    """Bounded runner cache for exact pyFAI geometry-plus-mask plans.

    A cache entry is valid only when every geometry input, frame-local mask,
    q-grid, and normalization denominator is identical.  It deliberately does
    not merge masks across frames or approximate nearby geometries.
    """

    def __init__(self, max_entries: int = 2) -> None:
        if max_entries < 1:
            raise ValueError("Geometry plan cache must retain at least one entry.")
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[str, PreparedGeometryDraw] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> PreparedGeometryDraw | None:
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: PreparedGeometryDraw) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


class RunnerPreparedMetalSession:
    """Aramina adapter for the existing xrd-analysis prepared-plan interface.

    xrd-analysis owns the GPU implementation.  This runner keeps one grouped
    Metal session alive while an exact plan set is reused, which avoids repeated
    image uploads and session creation for ``photon_only`` geometry states.
    """

    def __init__(
        self,
        images: list[np.ndarray],
        *,
        measurement_seeds: np.ndarray,
        profile_batch_size: int,
        max_entries: int = 2,
    ) -> None:
        if max_entries < 1:
            raise ValueError("Metal session cache must retain at least one entry.")
        self._images = tuple(np.asarray(image, dtype=np.float64) for image in images)
        self._measurement_seeds = np.asarray(measurement_seeds, dtype=np.uint64)
        self._profile_batch_size = int(profile_batch_size)
        self._max_entries = int(max_entries)
        self._sessions: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __enter__(self) -> RunnerPreparedMetalSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()

    @staticmethod
    def _plan_set_key(plans: list[Any] | tuple[Any, ...]) -> str:
        """Fingerprint exact ordered plans, including masks and denominators."""
        try:
            from xrdanalysis.direct_monte_carlo_metal_session import (
                metal_plan_fingerprint,
            )
        except ImportError as error:
            raise RuntimeError(
                "Joint uncertainty requires grouped prepared Metal support."
            ) from error
        digest = hashlib.sha256()
        for plan in plans:
            digest.update(metal_plan_fingerprint(plan).encode("ascii"))
        return digest.hexdigest()

    def _session_for_plans(self, plans: list[Any] | tuple[Any, ...]) -> Any:
        key = self._plan_set_key(plans)
        session = self._sessions.get(key)
        if session is not None:
            self._sessions.move_to_end(key)
            self.hits += 1
            return session
        try:
            from xrdanalysis.direct_monte_carlo_metal_session import (
                GroupedPersistentMetalMonteCarlo,
            )
        except ImportError as error:
            raise RuntimeError(
                "Joint uncertainty requires grouped prepared Metal support."
            ) from error
        self.misses += 1
        session = GroupedPersistentMetalMonteCarlo(
            plans,
            self._images,
            measurement_seeds=self._measurement_seeds,
            profile_batch_size=self._profile_batch_size,
        )
        self._sessions[key] = session
        while len(self._sessions) > self._max_entries:
            _, old_session = self._sessions.popitem(last=False)
            old_session.close()
        return session

    def run_geometry(
        self,
        plans: list[Any] | tuple[Any, ...],
        photon_replicates: int,
        *,
        seed: int,
        include_deterministic: bool,
    ) -> Any:
        if photon_replicates < 1:
            raise ValueError("photon_replicates must be positive.")
        session = self._session_for_plans(plans)
        profiles = session.run((1.0,), photon_replicates, seed=seed)[0]
        deterministic = session.integrate() if include_deterministic else None
        return type(
            "PreparedGeometryMetalResult",
            (),
            {
                "profiles": profiles,
                "deterministic_profiles": deterministic,
                "unique_plan_count": session.group_count,
            },
        )()


@dataclass
class RunCheckpoint:
    """Atomic patient/scenario completion state over one probability memmap."""

    run_folder: Path
    state: dict[str, Any]
    progress: dict[str, Any]
    probabilities: np.memmap

    @property
    def run_fingerprint(self) -> str:
        return str(self.state["run_fingerprint"])

    @staticmethod
    def unit_id(
        patient_id: str,
        scenario_name: str,
        draw_start: int = 0,
        draw_stop: int | None = None,
    ) -> str:
        if draw_stop is None:
            draw_stop = -1
        return json.dumps(
            [str(patient_id), str(scenario_name), int(draw_start), int(draw_stop)],
            separators=(",", ":"),
        )

    def completed_unit(
        self,
        *,
        patient_id: str,
        scenario_name: str,
        case_ids: list[str],
        case_indices: list[int],
        scenario_index: int,
        draw_start: int = 0,
        draw_stop: int | None = None,
    ) -> dict[str, Any] | None:
        if draw_stop is None:
            draw_stop = self.probabilities.shape[2]
        unit_id = self.unit_id(patient_id, scenario_name, draw_start, draw_stop)
        record = self.progress["completed_units"].get(unit_id)
        if record is None:
            return None
        checkpoint_path = self.run_folder / str(record["checkpoint_file"])
        if not checkpoint_path.is_file():
            raise MeasurementUncertaintyError(
                f"Completed unit checkpoint is missing: {unit_id}."
            )
        if file_sha256(checkpoint_path) != str(record["sha256"]):
            raise MeasurementUncertaintyError(
                f"Completed unit checkpoint fingerprint mismatch: {unit_id}."
            )
        payload = joblib.load(checkpoint_path)
        expected = {
            "contract": RESUME_CONTRACT,
            "run_fingerprint": self.run_fingerprint,
            "unit_id": unit_id,
            "patient_id": str(patient_id),
            "scenario": str(scenario_name),
            "case_ids": [str(value) for value in case_ids],
            "draw_start": int(draw_start),
            "draw_stop": int(draw_stop),
        }
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in expected.items()
        ):
            raise MeasurementUncertaintyError(
                f"Completed unit payload mismatch: {unit_id}."
            )
        values = self.probabilities[
            case_indices, scenario_index, draw_start:draw_stop
        ]
        if values.shape[0] != len(case_ids) or not np.isfinite(values).all():
            raise MeasurementUncertaintyError(
                f"Completed unit probability slice is partial: {unit_id}."
            )
        probability_sha256 = hashlib.sha256(
            np.ascontiguousarray(values, dtype=np.float32).tobytes()
        ).hexdigest()
        if payload.get("probability_sha256") != probability_sha256:
            raise MeasurementUncertaintyError(
                f"Completed unit probability fingerprint mismatch: {unit_id}."
            )
        return payload

    def complete_unit(
        self,
        *,
        patient_id: str,
        scenario_name: str,
        scenario_index: int,
        case_values: dict[str, np.ndarray],
        case_index: dict[str, int],
        parity_rows: list[dict[str, Any]],
        geometry_rows: list[dict[str, Any]],
        draw_start: int = 0,
        draw_stop: int | None = None,
    ) -> None:
        case_ids = [str(value) for value in case_values]
        if not case_ids or any(value not in case_index for value in case_ids):
            raise MeasurementUncertaintyError(
                "Checkpoint unit contains unknown or empty target cases."
            )
        if draw_stop is None:
            draw_stop = self.probabilities.shape[2]
        if not 0 <= draw_start < draw_stop <= self.probabilities.shape[2]:
            raise MeasurementUncertaintyError("Invalid checkpoint draw range.")
        expected_draws = draw_stop - draw_start
        for case_id in case_ids:
            values = np.asarray(case_values[case_id], dtype=np.float32)
            if values.shape != (expected_draws,) or not np.isfinite(values).all():
                raise MeasurementUncertaintyError(
                    "Checkpoint unit probabilities must be complete and finite."
                )
            self.probabilities[
                case_index[case_id], scenario_index, draw_start:draw_stop
            ] = values
        self.probabilities.flush()

        unit_id = self.unit_id(patient_id, scenario_name, draw_start, draw_stop)
        case_indices = [case_index[value] for value in case_ids]
        probability_sha256 = hashlib.sha256(
            np.ascontiguousarray(
                self.probabilities[
                    case_indices, scenario_index, draw_start:draw_stop
                ],
                dtype=np.float32,
            ).tobytes()
        ).hexdigest()
        checkpoint_folder = self.run_folder / UNIT_CHECKPOINT_FOLDER
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"{hashlib.sha256(unit_id.encode()).hexdigest()[:24]}.joblib"
        checkpoint_path = checkpoint_folder / checkpoint_name
        payload = {
            "contract": RESUME_CONTRACT,
            "run_fingerprint": self.run_fingerprint,
            "unit_id": unit_id,
            "patient_id": str(patient_id),
            "scenario": str(scenario_name),
            "case_ids": case_ids,
            "draw_start": int(draw_start),
            "draw_stop": int(draw_stop),
            "probability_sha256": probability_sha256,
            "parity_rows": parity_rows,
            "geometry_rows": geometry_rows,
        }
        _atomic_joblib_dump(payload, checkpoint_path, compress=3)
        completed = dict(self.progress["completed_units"])
        completed[unit_id] = {
            "checkpoint_file": str(checkpoint_path.relative_to(self.run_folder)),
            "sha256": file_sha256(checkpoint_path),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        self.progress = {
            **self.progress,
            "status": "running",
            "completed_units": completed,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(self.run_folder / PROGRESS_FILENAME, self.progress)

    def mark_complete(self) -> None:
        self.progress = {
            **self.progress,
            "status": "complete",
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(self.run_folder / PROGRESS_FILENAME, self.progress)

    def mark_paused(self, *, reason: str, completed_draws: int) -> None:
        self.progress = {
            **self.progress,
            "status": "paused",
            "pause_reason": reason,
            "completed_draws": int(completed_draws),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(self.run_folder / PROGRESS_FILENAME, self.progress)

    def mark_failed(self, *, reason: str, completed_draws: int) -> None:
        self.progress = {
            **self.progress,
            "status": "failed",
            "failure_reason": reason,
            "completed_draws": int(completed_draws),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(self.run_folder / PROGRESS_FILENAME, self.progress)

    def mark_stage_complete(
        self,
        *,
        completed_draws: int,
        convergence: dict[str, Any],
    ) -> None:
        stages = dict(self.progress.get("completed_stages", {}))
        stages[str(completed_draws)] = convergence
        self.progress = {
            **self.progress,
            "status": "running",
            "completed_draws": int(completed_draws),
            "completed_stages": stages,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(self.run_folder / PROGRESS_FILENAME, self.progress)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_joblib_dump(value: Any, path: Path, *, compress: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(value, temporary, compress=compress)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeasurementUncertaintyError(
            f"Resume JSON is missing or invalid: {path}."
        ) from error
    if not isinstance(value, dict):
        raise MeasurementUncertaintyError(f"Resume JSON must be a mapping: {path}.")
    return value


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _case_fingerprint(selected_cases: pd.DataFrame) -> str:
    return hashlib.sha256(
        selected_cases.to_json(
            orient="split",
            date_format="iso",
            double_precision=15,
        ).encode("utf-8")
    ).hexdigest()


def _scenario_fingerprint(scenarios: tuple[Scenario, ...]) -> str:
    return _json_fingerprint(
        [
            {
                "name": value.name,
                "photon": value.photon,
                "thickness": value.thickness,
                "beam_center": value.beam_center,
                "detector_distance": value.detector_distance,
                "beam_center_scale": value.beam_center_scale,
                "detector_distance_scale": value.detector_distance_scale,
            }
            for value in scenarios
        ]
    )


def _computational_config_fingerprint(config: dict[str, Any]) -> str:
    value = copy.deepcopy(config)
    value.setdefault("output", {}).pop("resume_run_folder", None)
    return _json_fingerprint(value)


def _base_resume_identity(
    config: dict[str, Any],
    *,
    model_path: Path,
    data_version: dict[str, Any],
    scenarios: tuple[Scenario, ...],
) -> dict[str, str]:
    native_spec = importlib.util.find_spec("xrdanalysis._native")
    native_library = (
        Path(next(iter(native_spec.submodule_search_locations)))
        / "libxrdanalysis_direct_monte_carlo_metal.dylib"
        if native_spec is not None and native_spec.submodule_search_locations
        else None
    )
    return {
        "config_fingerprint": _computational_config_fingerprint(config),
        "model_fingerprint": file_sha256(model_path),
        "data_fingerprint": _json_fingerprint(data_version),
        "scenario_fingerprint": _scenario_fingerprint(scenarios),
        "runner_fingerprint": file_sha256(Path(__file__)),
        "metal_library_fingerprint": (
            file_sha256(native_library)
            if native_library is not None and native_library.is_file()
            else "missing"
        ),
    }


def _cache_record(run_folder: Path, filename: str) -> dict[str, Any]:
    path = run_folder / filename
    return {
        "filename": filename,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _initialize_run_checkpoint(
    run_folder: Path,
    *,
    base_identity: dict[str, str],
    selected_frame: pd.DataFrame,
    selected_cases: pd.DataFrame,
    parity: pd.DataFrame,
    probability_shape: tuple[int, int, int],
) -> RunCheckpoint:
    state_path = run_folder / RUN_STATE_FILENAME
    if state_path.exists():
        raise MeasurementUncertaintyError(
            "New run folder already contains resumable state."
        )
    _atomic_joblib_dump(
        selected_frame, run_folder / SELECTED_FRAME_CACHE, compress=0
    )
    _atomic_joblib_dump(
        selected_cases, run_folder / SELECTED_CASES_CACHE, compress=3
    )
    _atomic_joblib_dump(parity, run_folder / PYFAI_PARITY_CACHE, compress=3)
    caches = {
        "selected_frame": _cache_record(run_folder, SELECTED_FRAME_CACHE),
        "selected_cases": _cache_record(run_folder, SELECTED_CASES_CACHE),
        "pyfai_parity": _cache_record(run_folder, PYFAI_PARITY_CACHE),
    }
    identity = {
        **base_identity,
        "case_fingerprint": _case_fingerprint(selected_cases),
        "selected_frame_fingerprint": caches["selected_frame"]["sha256"],
    }
    run_fingerprint = _json_fingerprint(
        {
            **identity,
            "probability_shape": list(probability_shape),
            "probability_dtype": "float32",
        }
    )
    state = {
        "contract": RESUME_CONTRACT,
        "run_fingerprint": run_fingerprint,
        "identity": identity,
        "probability_shape": list(probability_shape),
        "probability_dtype": "float32",
        "caches": caches,
        "created_at": datetime.now(UTC).isoformat(),
    }
    probabilities = np.lib.format.open_memmap(
        run_folder / PROBABILITY_FILENAME,
        mode="w+",
        dtype=np.float32,
        shape=probability_shape,
    )
    probabilities[:] = np.nan
    probabilities.flush()
    progress = {
        "contract": RESUME_CONTRACT,
        "run_fingerprint": run_fingerprint,
        "status": "running",
        "completed_units": {},
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(run_folder / PROGRESS_FILENAME, progress)
    _atomic_write_json(state_path, state)
    return RunCheckpoint(run_folder, state, progress, probabilities)


def _validate_resume_identity(
    state: dict[str, Any], expected_base_identity: dict[str, str]
) -> None:
    if state.get("contract") != RESUME_CONTRACT:
        raise MeasurementUncertaintyError("Resume state contract mismatch.")
    identity = state.get("identity")
    if not isinstance(identity, dict):
        raise MeasurementUncertaintyError("Resume state identity is missing.")
    mismatches = [
        key
        for key, expected in expected_base_identity.items()
        if identity.get(key) != expected
    ]
    if mismatches:
        raise MeasurementUncertaintyError(
            "Resume fingerprint mismatch: " + ", ".join(sorted(mismatches)) + "."
        )


def _load_resume_cache(
    run_folder: Path,
    state: dict[str, Any],
    cache_name: str,
) -> Any:
    record = state.get("caches", {}).get(cache_name)
    if not isinstance(record, dict):
        raise MeasurementUncertaintyError(f"Resume cache record missing: {cache_name}.")
    path = run_folder / str(record.get("filename", ""))
    if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
        raise MeasurementUncertaintyError(f"Resume cache missing: {cache_name}.")
    if file_sha256(path) != record.get("sha256"):
        raise MeasurementUncertaintyError(
            f"Resume cache fingerprint mismatch: {cache_name}."
        )
    return joblib.load(path)


def _open_run_checkpoint(
    run_folder: Path,
    *,
    expected_base_identity: dict[str, str],
    probability_shape: tuple[int, int, int] | None = None,
) -> tuple[RunCheckpoint, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state = _read_json(run_folder / RUN_STATE_FILENAME)
    _validate_resume_identity(state, expected_base_identity)
    selected_frame = _load_resume_cache(run_folder, state, "selected_frame")
    selected_cases = _load_resume_cache(run_folder, state, "selected_cases")
    parity = _load_resume_cache(run_folder, state, "pyfai_parity")
    if not all(isinstance(value, pd.DataFrame) for value in (selected_frame, selected_cases, parity)):
        raise MeasurementUncertaintyError("Resume caches must contain DataFrames.")
    identity = state["identity"]
    if _case_fingerprint(selected_cases) != identity.get("case_fingerprint"):
        raise MeasurementUncertaintyError("Resume case fingerprint mismatch.")
    if state["caches"]["selected_frame"]["sha256"] != identity.get(
        "selected_frame_fingerprint"
    ):
        raise MeasurementUncertaintyError("Resume selected-frame fingerprint mismatch.")
    state_shape = tuple(int(value) for value in state.get("probability_shape", ()))
    if probability_shape is not None and state_shape != probability_shape:
        raise MeasurementUncertaintyError("Resume probability shape mismatch.")
    expected_run_fingerprint = _json_fingerprint(
        {
            **identity,
            "probability_shape": list(state_shape),
            "probability_dtype": state.get("probability_dtype"),
        }
    )
    if state.get("run_fingerprint") != expected_run_fingerprint:
        raise MeasurementUncertaintyError("Resume run-state fingerprint mismatch.")
    probabilities = np.lib.format.open_memmap(
        run_folder / PROBABILITY_FILENAME,
        mode="r+",
    )
    if probabilities.shape != state_shape or probabilities.dtype != np.dtype(np.float32):
        raise MeasurementUncertaintyError("Resume probability memmap contract mismatch.")
    progress = _read_json(run_folder / PROGRESS_FILENAME)
    if (
        progress.get("contract") != RESUME_CONTRACT
        or progress.get("run_fingerprint") != state.get("run_fingerprint")
        or not isinstance(progress.get("completed_units"), dict)
    ):
        raise MeasurementUncertaintyError("Resume progress fingerprint mismatch.")
    return (
        RunCheckpoint(run_folder, state, progress, probabilities),
        selected_frame,
        selected_cases,
        parity,
    )


def run_joint_measurement_uncertainty_from_config(
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run bounded component and joint sensitivity scenarios on a frozen model."""
    started = perf_counter()
    path = Path(config_path).expanduser().resolve()
    config = _load_config(path)
    profile_max_tolerance, profile_p99_tolerance = _profile_parity_tolerances(
        config["validation"]
    )
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
    scenarios = tuple(_scenario(value) for value in config["scenarios"])
    design = _monte_carlo_design(config)
    draws = design.output_draws
    base_resume_identity = _base_resume_identity(
        config,
        model_path=model_path,
        data_version=data_version,
        scenarios=scenarios,
    )
    resume_value = config.get("output", {}).get("resume_run_folder")
    resuming = resume_value not in (None, "")
    if resuming:
        run_folder = _resolve_path(str(resume_value), path)
        if not run_folder.is_dir():
            raise MeasurementUncertaintyError(
                f"Resume run folder does not exist: {run_folder}."
            )
    else:
        run_folder = _create_run_folder(config, path)
    lock_stream = (run_folder / "run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise MeasurementUncertaintyError(
            f"Run folder is already active: {run_folder}."
        ) from error
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )
    )
    lock_stream.flush()
    effective_preprocessing = _experimental_preprocessing_config(
        model_artifact,
        input_h5_path=input_h5_path,
        output_joblib_path=run_folder / "preprocessed_joint_uncertainty.joblib",
        data_version=data_version,
    )
    if resuming:
        checkpoint, selected_frame, selected_cases, parity = _open_run_checkpoint(
            run_folder,
            expected_base_identity=base_resume_identity,
        )
        expected_shape = (len(selected_cases), len(scenarios), draws)
        if checkpoint.probabilities.shape != expected_shape:
            raise MeasurementUncertaintyError("Resume probability shape mismatch.")
    else:
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
        patient_ids_for_cache = (
            selected_cases["patient_id"].astype(str).unique().tolist()
        )
        selected_frame = dataframe[
            dataframe["patientId"].astype(str).isin(patient_ids_for_cache)
        ].reset_index(drop=True)
        parity = detector_integration_parity_check(
            selected_frame,
            measurement_count=int(config["validation"]["parity_measurements"]),
            tolerance=float(config["validation"]["pyfai_parity_tolerance"]),
        )
        if not bool(parity["parity_pass"].all()):
            raise MeasurementUncertaintyError(
                "Prepared pyFAI integration does not reproduce product profiles."
            )
        checkpoint = _initialize_run_checkpoint(
            run_folder,
            base_identity=base_resume_identity,
            selected_frame=selected_frame,
            selected_cases=selected_cases,
            parity=parity,
            probability_shape=(len(selected_cases), len(scenarios), draws),
        )

    nuisance_scope = _nuisance_scope_manifest(
        selected_frame,
        thickness_correlation=str(
            config["nuisance"]["sample_thickness"]["correlation"]
        ),
    )
    nuisance_scope_path = run_folder / NUISANCE_SCOPE_FILENAME
    if nuisance_scope_path.exists():
        existing_scope = pd.read_csv(nuisance_scope_path, dtype=str)
        if not existing_scope.equals(nuisance_scope.astype(str)):
            raise MeasurementUncertaintyError(
                "Resume nuisance scope differs from the stored cohort contract."
            )
    else:
        nuisance_scope.to_csv(nuisance_scope_path, index=False)

    patient_ids = selected_cases["patient_id"].astype(str).unique().tolist()
    quantiles = tuple(float(value) for value in config["monte_carlo"]["quantiles"])
    probability_path = run_folder / PROBABILITY_FILENAME
    probabilities = checkpoint.probabilities
    case_index = {
        str(case_id): index
        for index, case_id in enumerate(selected_cases["target_case_id"])
    }
    deterministic = np.full(len(selected_cases), np.nan, dtype=float)
    thresholds = np.full(len(selected_cases), np.nan, dtype=float)
    metal_parity_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    patient_inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for patient_id in patient_ids:
        patient_frame = selected_frame[
            selected_frame["patientId"].astype(str).eq(patient_id)
        ].reset_index(drop=True)
        patient_cases = selected_cases[
            selected_cases["patient_id"].astype(str).eq(patient_id)
        ].reset_index(drop=True)
        baseline_scores = _score_cases(
            patient_frame,
            patient_cases,
            model_info=model_info,
        )
        for case_id, score in baseline_scores.items():
            index = case_index[case_id]
            deterministic[index] = score["p_cancer"]
            thresholds[index] = score["threshold"]
        patient_inputs[patient_id] = (patient_frame, patient_cases)

    case_table = selected_cases.copy()
    case_table["deterministic_p_cancer"] = deterministic
    case_table["decision_threshold"] = thresholds
    draw_chunk_size = int(config["execution"]["draw_chunk_size"])
    profile_batch_size = int(config["execution"]["profile_batch_size"])
    backend_kind = str(config["backend"]["kind"])
    geometry_audit_draws = int(config["execution"]["geometry_audit_draws"])
    stage_geometry_draws = design.geometry_stage_draws
    normalization_q_range = tuple(
        float(value) for value in config["integration"]["normalization_q_range"]
    )
    stop_request_path = run_folder / STOP_REQUEST_FILENAME
    if resuming and stop_request_path.exists():
        stop_request_path.unlink()
    completed_stages = checkpoint.progress.get("completed_stages", {})
    latest_stage = (
        completed_stages.get(str(max(map(int, completed_stages))))
        if completed_stages
        else None
    )
    stable_checkpoint_count = int(
        latest_stage.get("consecutive_stable_checkpoints", 0)
        if isinstance(latest_stage, dict)
        else 0
    )
    completed_global_draws = int(checkpoint.progress.get("completed_draws", 0))
    cohort_nuisance = sample_cohort_nuisance_draws(
        selected_frame,
        draws=design.geometry_draws,
        seed=int(config["monte_carlo"]["seed"]),
        thickness_config=config["nuisance"]["sample_thickness"],
        beam_center_config=config["nuisance"]["beam_center"],
        detector_distance_config=config["nuisance"]["detector_distance"],
    )

    try:
        for geometry_start, geometry_stop in _stage_ranges(
            design.geometry_draws,
            stage_geometry_draws,
        ):
            draw_start = geometry_start * design.photon_replicates
            draw_stop = geometry_stop * design.photon_replicates
            if draw_stop <= completed_global_draws:
                continue
            for patient_id in patient_ids:
                patient_frame, patient_cases = patient_inputs[patient_id]
                patient_case_ids = [
                    str(value) for value in patient_cases["target_case_id"].tolist()
                ]
                patient_case_indices = [
                    case_index[value] for value in patient_case_ids
                ]
                incomplete_scenarios: list[tuple[int, Scenario]] = []
                for scenario_index, scenario in enumerate(scenarios):
                    completed = checkpoint.completed_unit(
                        patient_id=patient_id,
                        scenario_name=scenario.name,
                        case_ids=patient_case_ids,
                        case_indices=patient_case_indices,
                        scenario_index=scenario_index,
                        draw_start=draw_start,
                        draw_stop=draw_stop,
                    )
                    if completed is None:
                        incomplete_scenarios.append((scenario_index, scenario))
                        continue
                    parity_rows = completed.get("parity_rows")
                    completed_geometry = completed.get("geometry_rows")
                    if not isinstance(parity_rows, list) or not isinstance(
                        completed_geometry, list
                    ):
                        raise MeasurementUncertaintyError(
                            "Completed stage-unit audit payload is invalid."
                        )
                    metal_parity_rows.extend(parity_rows)
                    geometry_rows.extend(completed_geometry)
                if not incomplete_scenarios:
                    continue

                nuisance = cohort_nuisance.for_frame(patient_frame)
                with _prepare_patient_metal_context(
                    patient_frame,
                    nuisance=nuisance,
                    draw_chunk_size=draw_chunk_size,
                    profile_batch_size=profile_batch_size,
                    normalization_q_range=normalization_q_range,
                    backend_kind=backend_kind,
                ) as metal_context:
                    for scenario_index, scenario in incomplete_scenarios:
                        patient_probabilities, patient_parity, patient_geometry = (
                            _run_patient_scenario(
                                patient_frame,
                                patient_cases,
                                model_artifact=model_artifact,
                                model_info=model_info,
                                metal_context=metal_context,
                                scenario=scenario,
                                nuisance=nuisance,
                                draws=draws,
                                draw_start=draw_start,
                                draw_stop=draw_stop,
                                geometry_draw_start=geometry_start,
                                geometry_draw_stop=geometry_stop,
                                photon_replicates=design.photon_replicates,
                                draw_chunk_size=draw_chunk_size,
                                geometry_audit_draws=geometry_audit_draws,
                                normalization_q_range=normalization_q_range,
                                metal_profile_max_tolerance=profile_max_tolerance,
                                metal_profile_p99_tolerance=profile_p99_tolerance,
                                metal_p_cancer_parity_tolerance=float(
                                    config["validation"][
                                        "metal_p_cancer_parity_tolerance"
                                    ]
                                ),
                                random_seed=int(config["monte_carlo"]["seed"]),
                            )
                        )
                        checkpoint.complete_unit(
                            patient_id=patient_id,
                            scenario_name=scenario.name,
                            scenario_index=scenario_index,
                            case_values=patient_probabilities,
                            case_index=case_index,
                            parity_rows=patient_parity,
                            geometry_rows=patient_geometry,
                            draw_start=draw_start,
                            draw_stop=draw_stop,
                        )
                        metal_parity_rows.extend(patient_parity)
                        geometry_rows.extend(patient_geometry)
                        if stop_request_path.exists():
                            checkpoint.mark_paused(
                                reason="manual_stop_request",
                                completed_draws=completed_global_draws,
                            )
                            return _paused_result(
                                run_folder,
                                patient_ids=patient_ids,
                                selected_cases=selected_cases,
                                completed_draws=completed_global_draws,
                                reason="manual_stop_request",
                                resuming=resuming,
                            )

            stage_case_summary = _stage_case_summary(
                probabilities,
                case_table,
                scenarios=scenarios,
                quantiles=quantiles,
                completed_draws=draw_stop,
            )
            stage_cohort_summary = summarize_cohort_convergence(stage_case_summary)
            nested_axis_summary = None
            nested_axis_changes = None
            if design.mode == "nested_geometry_photon":
                nested_axis_summary = summarize_nested_axis_convergence(
                    probabilities[:, :, :draw_stop],
                    case_table,
                    scenarios=scenarios,
                    quantiles=quantiles,
                    geometry_draws=geometry_stop,
                    photon_replicates=design.photon_replicates,
                    geometry_prefixes=tuple(
                        int(value)
                        for value in config["convergence"]["geometry_prefixes"]
                    ),
                    photon_prefixes=tuple(
                        int(value)
                        for value in config["convergence"]["photon_prefixes"]
                    ),
                )
                nested_axis_changes = summarize_nested_axis_changes(
                    nested_axis_summary
                )
            previous_summary = None
            if completed_global_draws:
                previous_path = (
                    run_folder
                    / CONVERGENCE_FOLDER
                    / f"draws_{completed_global_draws:05d}"
                    / "case_summary.csv"
                )
                if previous_path.is_file():
                    previous_summary = pd.read_csv(previous_path)
            plateau_metrics, plateau_status = _stage_plateau_metrics(
                stage_case_summary,
                previous_summary,
                config=config,
                completed_draws=draw_stop,
                stable_checkpoint_count=stable_checkpoint_count,
            )
            geometry_minimum_reached = geometry_stop >= int(
                config["convergence"]["minimum_geometry_draws"]
            )
            plateau_status["independent_geometry_draws"] = geometry_stop
            plateau_status["geometry_minimum_reached"] = geometry_minimum_reached
            plateau_status["photon_replicates_per_geometry"] = (
                design.photon_replicates
            )
            plateau_status["plateau"] = bool(
                plateau_status["plateau"] and geometry_minimum_reached
            )
            stable_checkpoint_count = int(
                plateau_status["consecutive_stable_checkpoints"]
            )
            stage_status = _write_stage_convergence(
                run_folder,
                case_summary=stage_case_summary,
                cohort_summary=stage_cohort_summary,
                plateau_metrics=plateau_metrics,
                plateau_status=plateau_status,
                completed_draws=draw_stop,
                nested_axis_summary=nested_axis_summary,
                nested_axis_changes=nested_axis_changes,
            )
            completed_global_draws = draw_stop
            checkpoint.mark_stage_complete(
                completed_draws=draw_stop,
                convergence=stage_status,
            )
            if bool(config["convergence"]["auto_stop"]) and bool(
                plateau_status["plateau"]
            ):
                checkpoint.mark_paused(
                    reason="convergence_plateau",
                    completed_draws=draw_stop,
                )
                return _paused_result(
                    run_folder,
                    patient_ids=patient_ids,
                    selected_cases=selected_cases,
                    completed_draws=draw_stop,
                    reason="convergence_plateau",
                    resuming=resuming,
                )
    except KeyboardInterrupt:
        checkpoint.mark_paused(
            reason="keyboard_interrupt",
            completed_draws=completed_global_draws,
        )
        return _paused_result(
            run_folder,
            patient_ids=patient_ids,
            selected_cases=selected_cases,
            completed_draws=completed_global_draws,
            reason="keyboard_interrupt",
            resuming=resuming,
        )
    except Exception as error:
        checkpoint.mark_failed(
            reason=f"{type(error).__name__}: {error}",
            completed_draws=completed_global_draws,
        )
        raise

    expected_unit_ids = {
        checkpoint.unit_id(
            patient_id,
            scenario.name,
            geometry_start * design.photon_replicates,
            geometry_stop * design.photon_replicates,
        )
        for patient_id in patient_ids
        for scenario in scenarios
        for geometry_start, geometry_stop in _stage_ranges(
            design.geometry_draws,
            design.geometry_stage_draws,
        )
    }
    if set(checkpoint.progress["completed_units"]) != expected_unit_ids:
        raise MeasurementUncertaintyError(
            "Run cannot finalize with incomplete patient/scenario units."
        )
    if not np.isfinite(probabilities).all():
        raise MeasurementUncertaintyError(
            "Run cannot finalize with partial probability values."
        )

    summaries = summarize_case_uncertainty(
        probabilities,
        case_table,
        scenarios=scenarios,
        quantiles=quantiles,
    )
    nested_axis_summary = None
    nested_axis_changes = None
    if design.mode == "nested_geometry_photon":
        nested_axis_summary = summarize_nested_axis_convergence(
            probabilities,
            case_table,
            scenarios=scenarios,
            quantiles=quantiles,
            geometry_draws=design.geometry_draws,
            photon_replicates=design.photon_replicates,
            geometry_prefixes=tuple(
                int(value) for value in config["convergence"]["geometry_prefixes"]
            ),
            photon_prefixes=tuple(
                int(value) for value in config["convergence"]["photon_prefixes"]
            ),
        )
        nested_axis_changes = summarize_nested_axis_changes(nested_axis_summary)
        case_convergence = nested_axis_summary[
            nested_axis_summary["convergence_axis"].eq("geometry")
        ].copy()
        case_convergence["draw_prefix"] = case_convergence["geometry_draws"]
    else:
        case_convergence = summarize_case_convergence(
            probabilities,
            case_table,
            scenarios=scenarios,
            quantiles=quantiles,
        )
    cohort_convergence = summarize_cohort_convergence(case_convergence)
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
        case_convergence=case_convergence,
        cohort_convergence=cohort_convergence,
        nested_axis_summary=nested_axis_summary,
        nested_axis_changes=nested_axis_changes,
        parity=parity,
        metal_parity=pd.DataFrame(metal_parity_rows),
        geometry_draws=pd.DataFrame(geometry_rows),
        metadata_qc=metadata_qc,
        scenarios=scenarios,
        elapsed_seconds=perf_counter() - started,
    )
    checkpoint.mark_complete()
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
        "resumed": resuming,
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
    """Compatibility wrapper for one patient slice of the cohort field."""
    patient_ids = patient_frame["patientId"].astype(str).unique()
    if len(patient_ids) != 1:
        raise ValueError("Nuisance sampling requires exactly one patient.")
    return sample_cohort_nuisance_draws(
        patient_frame,
        draws=draws,
        seed=seed,
        thickness_config=thickness_config,
        beam_center_config=beam_center_config,
        detector_distance_config=detector_distance_config,
    ).for_frame(patient_frame)


def _nuisance_scope_manifest(
    cohort_frame: pd.DataFrame,
    *,
    thickness_correlation: str,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in cohort_frame.iterrows():
        measurement_key = _measurement_key(row)
        if thickness_correlation == "visit_shared":
            thickness_group = f"patient:{row['patientId']}"
        elif thickness_correlation == "measurement_independent":
            thickness_group = f"measurement:{measurement_key}"
        else:
            raise ValueError("Unsupported thickness correlation scope.")
        rows.append(
            {
                "measurement_key_sha256": hashlib.sha256(
                    measurement_key.encode("utf-8")
                ).hexdigest(),
                "patient_id": str(row["patientId"]),
                "specimen_id": str(row["specimenId"]),
                "side": str(row["side"]),
                "poni_geometry_sha256": _poni_geometry_key(row),
                "thickness_group": thickness_group,
                "photon_group": f"measurement:{measurement_key}",
            }
        )
    return pd.DataFrame(rows)


def sample_cohort_nuisance_draws(
    cohort_frame: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    thickness_config: dict[str, Any],
    beam_center_config: dict[str, Any],
    detector_distance_config: dict[str, Any],
) -> CohortNuisanceDraws:
    """Sample one aligned geometry field for the complete cohort."""
    measurements = len(cohort_frame)
    if draws < 1 or measurements < 1:
        raise ValueError("draws and measurements must be positive.")
    measurement_keys = tuple(
        _measurement_key(row) for _, row in cohort_frame.iterrows()
    )
    if len(set(measurement_keys)) != measurements:
        raise ValueError("Cohort frame contains duplicate measurement keys.")
    patient_ids = cohort_frame["patientId"].astype(str).to_numpy()
    thickness = pd.to_numeric(
        cohort_frame["sample_thickness_mm"], errors="coerce"
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
    if correlation == "visit_shared":
        patients, patient_index = np.unique(patient_ids, return_inverse=True)
        thickness_by_patient = np.column_stack(
            [
                _keyed_rng(seed, "thickness", patient_id).uniform(
                    -1.0, 1.0, size=draws
                )
                for patient_id in patients
            ]
        )
        thickness_unit = thickness_by_patient[:, patient_index]
    elif correlation == "measurement_independent":
        thickness_unit = np.column_stack(
            [
                _keyed_rng(seed, "thickness", measurement_key).uniform(
                    -1.0,
                    1.0,
                    size=draws,
                )
                for measurement_key in measurement_keys
            ]
        )
    else:
        raise ValueError("Thickness correlation must be visit_shared or independent.")
    thickness_delta = np.broadcast_to(
        thickness_unit, (draws, measurements)
    ).copy() * bounds

    if str(beam_center_config.get("correlation")) != "poni_file_shared":
        raise ValueError("Beam-center correlation must be poni_file_shared.")
    if str(detector_distance_config.get("correlation")) != "poni_file_shared":
        raise ValueError("Detector-distance correlation must be poni_file_shared.")
    geometry_keys = np.asarray(
        [_poni_geometry_key(row) for _, row in cohort_frame.iterrows()]
    )
    geometries, geometry_index = np.unique(geometry_keys, return_inverse=True)
    geometry_count = len(geometries)
    radius = float(beam_center_config["radius_px"])
    row_by_geometry = np.empty((draws, geometry_count), dtype=float)
    col_by_geometry = np.empty((draws, geometry_count), dtype=float)
    distance_by_geometry = np.empty((draws, geometry_count), dtype=float)
    for index, geometry_key in enumerate(geometries):
        geometry_rng = _keyed_rng(seed, "geometry", str(geometry_key))
        radial = radius * np.sqrt(geometry_rng.uniform(size=draws))
        angle = geometry_rng.uniform(0.0, 2.0 * np.pi, size=draws)
        row_by_geometry[:, index] = radial * np.sin(angle)
        col_by_geometry[:, index] = radial * np.cos(angle)
        distance_by_geometry[:, index] = geometry_rng.uniform(
            -float(detector_distance_config["half_width_mm"]),
            float(detector_distance_config["half_width_mm"]),
            size=draws,
        )
    photon_seeds = np.asarray(
        [
            _keyed_rng(seed, "photon", _measurement_key(row)).integers(
                0,
                np.iinfo(np.uint64).max,
                dtype=np.uint64,
            )
            for _, row in cohort_frame.iterrows()
        ],
        dtype=np.uint64,
    )
    return CohortNuisanceDraws(
        measurement_keys=measurement_keys,
        thickness_delta_mm=thickness_delta,
        beam_center_row_delta_px=row_by_geometry[:, geometry_index],
        beam_center_col_delta_px=col_by_geometry[:, geometry_index],
        detector_distance_delta_mm=distance_by_geometry[:, geometry_index],
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


def _profile_parity_tolerances(validation: dict[str, Any]) -> tuple[float, float]:
    required = (
        "metal_profile_max_abs_tolerance",
        "metal_profile_p99_abs_tolerance",
    )
    missing = [key for key in required if key not in validation]
    if missing:
        raise MeasurementUncertaintyError(
            "Profile parity requires explicit max and p99 tolerances: "
            + ", ".join(missing)
            + "."
        )
    maximum, p99 = (float(validation[key]) for key in required)
    if not np.isfinite((maximum, p99)).all() or maximum <= 0.0 or p99 <= 0.0:
        raise MeasurementUncertaintyError(
            "Profile parity tolerances must be finite and positive."
        )
    return maximum, p99


def _profile_parity_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    q_grid: np.ndarray,
    *,
    draw_start: int,
    maximum_tolerance: float,
    p99_tolerance: float,
) -> dict[str, Any]:
    absolute_errors = np.abs(np.asarray(actual) - np.asarray(expected))
    if absolute_errors.ndim != 3 or absolute_errors.shape[2] != q_grid.size:
        raise ValueError("Profile parity arrays must have shape (draw, measurement, q).")
    maximum_error = float(np.max(absolute_errors))
    p99_error = float(np.quantile(absolute_errors, 0.99))
    maximum_location = np.unravel_index(
        int(np.argmax(absolute_errors)), absolute_errors.shape
    )
    return {
        "maximum_absolute_error": maximum_error,
        "p99_absolute_error": p99_error,
        "maximum_error_draw_index": draw_start + int(maximum_location[0]),
        "maximum_error_measurement_index": int(maximum_location[1]),
        "maximum_error_q_nm_inv": float(q_grid[int(maximum_location[2])]),
        "profile_max_abs_tolerance": maximum_tolerance,
        "profile_p99_abs_tolerance": p99_tolerance,
        "parity_pass": bool(
            maximum_error <= maximum_tolerance and p99_error <= p99_tolerance
        ),
    }


def _prepare_patient_metal_context(
    patient_frame: pd.DataFrame,
    *,
    nuisance: NuisanceDraws,
    draw_chunk_size: int,
    profile_batch_size: int,
    normalization_q_range: tuple[float, float],
    backend_kind: str = "pyfai_prepared_csr_metal_photon_mc",
) -> PatientMetalContext:
    measurements = len(patient_frame)
    if nuisance.photon_measurement_seeds.shape != (measurements,):
        raise MeasurementUncertaintyError(
            "Photon seeds must contain one stable seed per measurement."
        )
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    q_rows: list[np.ndarray] = []
    nominal_distance: list[float] = []
    nominal_poni1: list[float] = []
    nominal_poni2: list[float] = []
    pixel1: list[float] = []
    pixel2: list[float] = []
    geometries: list[Any] = []
    for _, row in patient_frame.iterrows():
        image = _centered_poisson_observation(
            row[RAW_FRAME_COLUMN], row[MASK_COLUMN]
        )
        context, _ = _perturbed_context(
            row,
            thickness_delta_mm=0.0,
            center_row_delta_px=0.0,
            center_col_delta_px=0.0,
            distance_delta_mm=0.0,
        )
        q = np.asarray(row["q_range"], dtype=float).ravel()
        if q.size != context.npt or not np.isfinite(q).all():
            raise MeasurementUncertaintyError(
                "Product q grid is missing or incompatible with Metal integration."
            )
        images.append(image)
        q_rows.append(q)
        masks.append(np.asarray(context.mask, dtype=np.uint8))
        nominal_distance.append(float(context.integrator.dist))
        nominal_poni1.append(float(context.integrator.poni1))
        nominal_poni2.append(float(context.integrator.poni2))
        pixel1.append(float(context.integrator.detector.pixel1))
        pixel2.append(float(context.integrator.detector.pixel2))
        if backend_kind == "metal_geometry_aware_nested":
            try:
                from xrdanalysis.direct_monte_carlo_geometry_metal import (
                    MetalDetectorGeometry,
                )
            except ImportError as error:
                raise RuntimeError(
                    "Joint uncertainty requires geometry-aware Metal support."
                ) from error
            geometries.append(MetalDetectorGeometry.from_pyfai(context.integrator))

    q_grid = q_rows[0]
    if any(
        not np.allclose(q, q_grid, rtol=0.0, atol=1e-12) for q in q_rows[1:]
    ):
        raise MeasurementUncertaintyError(
            "Patient measurements do not share the fixed product q grid."
        )
    if backend_kind == "pyfai_prepared_csr_metal_photon_mc":
        session: Any = RunnerPreparedMetalSession(
            images,
            measurement_seeds=nuisance.photon_measurement_seeds,
            profile_batch_size=profile_batch_size,
        )
        geometry_plan_cache: GeometryPlanCache | None = GeometryPlanCache()
    elif backend_kind == "metal_geometry_aware_nested":
        try:
            from xrdanalysis.direct_monte_carlo_geometry_metal import (
                GeometryAwareMetalMonteCarlo,
            )
        except ImportError as error:
            raise RuntimeError(
                "Joint uncertainty requires geometry-aware Metal support."
            ) from error
        session = GeometryAwareMetalMonteCarlo(
            np.stack(images),
            np.stack(masks),
            geometries,
            q_grid,
            normalization_q_range,
            measurement_seeds=nuisance.photon_measurement_seeds,
            draw_capacity=max(1, int(draw_chunk_size)),
            profile_batch_size=max(int(profile_batch_size), 50),
        )
        geometry_plan_cache = None
    else:
        raise MeasurementUncertaintyError(
            f"Unsupported joint uncertainty backend: {backend_kind}."
        )
    return PatientMetalContext(
        session=session,
        q_grid=q_grid.copy(),
        images=np.stack(images),
        geometry_plan_cache=geometry_plan_cache,
        backend_kind=backend_kind,
        nominal_effective_distance_m=np.asarray(nominal_distance, dtype=float),
        nominal_poni1_m=np.asarray(nominal_poni1, dtype=float),
        nominal_poni2_m=np.asarray(nominal_poni2, dtype=float),
        pixel1_m=np.asarray(pixel1, dtype=float),
        pixel2_m=np.asarray(pixel2, dtype=float),
    )


def _scenario_component_deltas(
    scenario: Scenario,
    nuisance: NuisanceDraws,
    *,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (stop - start, nuisance.thickness_delta_mm.shape[1])
    zeros = np.zeros(shape, dtype=float)
    thickness = nuisance.thickness_delta_mm[start:stop] if scenario.thickness else zeros
    row = (
        nuisance.beam_center_row_delta_px[start:stop]
        * scenario.beam_center_scale
        if scenario.beam_center
        else zeros
    )
    column = (
        nuisance.beam_center_col_delta_px[start:stop]
        * scenario.beam_center_scale
        if scenario.beam_center
        else zeros
    )
    distance = (
        nuisance.detector_distance_delta_mm[start:stop]
        * scenario.detector_distance_scale
        if scenario.detector_distance
        else zeros
    )
    return thickness, row, column, distance


def _prepared_geometry_seed(base_seed: int, draw_index: int) -> int:
    """Derive a resumable independent photon stream for one geometry draw."""
    payload = f"{base_seed}:{draw_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _cache_array_digest(digest: Any, values: Any, *, dtype: np.dtype) -> None:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest.update(array.shape.__repr__().encode("ascii"))
    digest.update(array.tobytes())


def _geometry_plan_cache_key(
    patient_frame: pd.DataFrame,
    *,
    thickness: np.ndarray,
    row_delta: np.ndarray,
    column_delta: np.ndarray,
    distance_delta: np.ndarray,
    q_grid: np.ndarray,
    normalization_q_range: tuple[float, float],
) -> str:
    """Fingerprint every input which can alter a prepared pyFAI plan.

    The key includes each frame-local mask and the fixed normalization context.
    Thus a hit means exactly the same integration plan, rather than merely a
    matching PONI file or a nearby perturbation.
    """
    digest = hashlib.sha256()
    digest.update(b"aramina-prepared-geometry-cache-v1")
    _cache_array_digest(digest, q_grid, dtype=np.dtype(np.float64))
    _cache_array_digest(digest, normalization_q_range, dtype=np.dtype(np.float64))
    for values in (thickness, row_delta, column_delta, distance_delta):
        _cache_array_digest(digest, values, dtype=np.dtype(np.float64))
    for _, measurement in patient_frame.iterrows():
        image_shape = np.asarray(measurement[RAW_FRAME_COLUMN]).shape
        digest.update(repr(tuple(int(value) for value in image_shape)).encode("ascii"))
        digest.update(str(measurement["ponifile"]).encode("utf-8"))
        for column in (
            "sample_thickness_mm",
            "calibrant_thickness_mm",
            "interpolation_q_range",
            "azimuthal_range",
        ):
            digest.update(repr(measurement.get(column)).encode("utf-8"))
        _cache_array_digest(
            digest,
            measurement[MASK_COLUMN],
            dtype=np.dtype(np.int64),
        )
    return digest.hexdigest()


def _retag_geometry_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    scenario: Scenario,
    draw_index: int,
) -> list[dict[str, Any]]:
    return [
        {**row, "scenario": scenario.name, "draw_index": int(draw_index)}
        for row in rows
    ]


def _prepare_pyfai_geometry_draw(
    patient_frame: pd.DataFrame,
    *,
    scenario: Scenario,
    nuisance: NuisanceDraws,
    draw_index: int,
    q_grid: np.ndarray,
    normalization_q_range: tuple[float, float],
    prepare_metal_plans: bool = True,
    geometry_plan_cache: GeometryPlanCache | None = None,
) -> tuple[np.ndarray, list[Any], list[dict[str, Any]]]:
    if prepare_metal_plans:
        from xrdanalysis.direct_monte_carlo import prepare_native_plan
        from xrdanalysis.direct_monte_carlo_metal import prepare_metal_plan

    thickness, row_delta, column_delta, distance_delta = (
        _scenario_component_deltas(
            scenario,
            nuisance,
            start=draw_index,
            stop=draw_index + 1,
        )
    )
    cache_key = _geometry_plan_cache_key(
        patient_frame,
        thickness=thickness,
        row_delta=row_delta,
        column_delta=column_delta,
        distance_delta=distance_delta,
        q_grid=q_grid,
        normalization_q_range=normalization_q_range,
    )
    if geometry_plan_cache is not None:
        cached = geometry_plan_cache.get(cache_key)
        if cached is not None:
            if prepare_metal_plans and not cached.plans:
                raise MeasurementUncertaintyError(
                    "Cached geometry result has no required Metal plans."
                )
            return (
                cached.expected,
                list(cached.plans) if prepare_metal_plans else [],
                _retag_geometry_rows(
                    cached.geometry_rows,
                    scenario=scenario,
                    draw_index=draw_index,
                ),
            )
    expected = np.empty((len(patient_frame), q_grid.size), dtype=float)
    plans: list[Any] = []
    geometry_rows: list[dict[str, Any]] = []
    for measurement_index, (_, row) in enumerate(patient_frame.iterrows()):
        context, geometry = _perturbed_context(
            row,
            thickness_delta_mm=float(thickness[0, measurement_index]),
            center_row_delta_px=float(row_delta[0, measurement_index]),
            center_col_delta_px=float(column_delta[0, measurement_index]),
            distance_delta_mm=float(distance_delta[0, measurement_index]),
        )
        image = _centered_poisson_observation(
            row[RAW_FRAME_COLUMN], row[MASK_COLUMN]
        )
        result = context.integrator.integrate1d(
            image,
            context.npt,
            radial_range=context.radial_range,
            azimuth_range=context.azimuth_range,
            mask=context.mask,
            error_model="poisson",
        )
        q = np.asarray(result.radial, dtype=float)
        intensity = np.asarray(result.intensity, dtype=float)
        normalization = np.asarray(result.sum_normalization, dtype=float)
        supported = np.isfinite(normalization) & (normalization > 0.0)
        normalization_band = (
            (q >= normalization_q_range[0])
            & (q <= normalization_q_range[1])
        )
        if not np.all(supported):
            raise MeasurementUncertaintyError(
                "Perturbed pyFAI geometry has unsupported product q bins."
            )
        if not np.all(supported[normalization_band]):
            raise MeasurementUncertaintyError(
                "Perturbed pyFAI geometry has unsupported normalization bins."
            )
        if not np.allclose(q, q_grid, rtol=0.0, atol=1e-12):
            raise MeasurementUncertaintyError(
                "Perturbed pyFAI geometry changed the fixed product q grid."
            )
        expected[measurement_index] = normalize_profile(
            q, intensity, q_range=normalization_q_range
        )
        if prepare_metal_plans:
            plans.append(
                prepare_metal_plan(
                    prepare_native_plan(
                        context.integrator,
                        image.shape,
                        normalization_denominators=normalization,
                        q_grid=q,
                        q_normalization_band=normalization_q_range,
                    )
                )
            )
        geometry_rows.append(
            {
                "patient_id": str(row["patientId"]),
                "specimen_id": str(row["specimenId"]),
                "scenario": scenario.name,
                "draw_index": draw_index,
                "measurement_index": measurement_index,
                "supported_q_bin_fraction": float(np.mean(supported)),
                "normalization_band_supported": True,
                **geometry,
            }
        )
    if geometry_plan_cache is not None:
        expected.setflags(write=False)
        geometry_plan_cache.put(
            cache_key,
            PreparedGeometryDraw(
                expected=expected,
                plans=tuple(plans),
                geometry_rows=tuple(geometry_rows),
            ),
        )
    return expected, plans, geometry_rows


def _pyfai_oracle_profiles(
    patient_frame: pd.DataFrame,
    *,
    scenario: Scenario,
    nuisance: NuisanceDraws,
    start: int,
    stop: int,
    q_grid: np.ndarray,
    normalization_q_range: tuple[float, float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    expected: list[np.ndarray] = []
    geometry_rows: list[dict[str, Any]] = []
    for draw_index in range(start, stop):
        profiles, _, rows = _prepare_pyfai_geometry_draw(
            patient_frame,
            scenario=scenario,
            nuisance=nuisance,
            draw_index=draw_index,
            q_grid=q_grid,
            normalization_q_range=normalization_q_range,
            prepare_metal_plans=False,
        )
        expected.append(profiles)
        geometry_rows.extend(rows)
    return np.stack(expected), geometry_rows


def _geometry_aware_draw_arrays(
    metal_context: PatientMetalContext,
    *,
    scenario: Scenario,
    nuisance: NuisanceDraws,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact draw-specific geometry vectors for the persistent Metal session."""
    static = (
        metal_context.nominal_effective_distance_m,
        metal_context.nominal_poni1_m,
        metal_context.nominal_poni2_m,
        metal_context.pixel1_m,
        metal_context.pixel2_m,
    )
    if any(value is None for value in static):
        raise MeasurementUncertaintyError(
            "Geometry-aware Metal context is missing nominal geometry vectors."
        )
    thickness, row_delta, column_delta, distance_delta = _scenario_component_deltas(
        scenario,
        nuisance,
        start=start,
        stop=stop,
    )
    nominal_distance, nominal_poni1, nominal_poni2, pixel1, pixel2 = static
    return (
        np.ascontiguousarray(
            nominal_distance[np.newaxis, :]
            + 1e-3 * (distance_delta - 0.5 * thickness),
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            nominal_poni1[np.newaxis, :] + row_delta * pixel1[np.newaxis, :],
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            nominal_poni2[np.newaxis, :] + column_delta * pixel2[np.newaxis, :],
            dtype=np.float64,
        ),
    )


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
                    "scenario_draw_fraction_at_or_above_threshold": float(
                        np.mean(draws >= threshold)
                    ),
                    "scenario_class_flip_fraction": float(
                        np.mean((draws >= threshold) != baseline_class)
                    ),
                    "threshold_crossing": bool(lower < threshold <= upper),
                }
            )
    return pd.DataFrame(rows)


def convergence_draw_prefixes(
    draws: int,
    *,
    stage_draws: int = 250,
) -> tuple[int, ...]:
    """Return every global stage boundary plus the final draw count."""
    if draws < 1:
        raise ValueError("draws must be positive.")
    if stage_draws < 1:
        raise ValueError("stage_draws must be positive.")
    return tuple(range(stage_draws, draws, stage_draws)) + (draws,)


def _stage_ranges(draws: int, stage_draws: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(start + stage_draws, draws))
        for start in range(0, draws, stage_draws)
    )


def _stage_case_summary(
    probabilities: np.ndarray,
    case_table: pd.DataFrame,
    *,
    scenarios: tuple[Scenario, ...],
    quantiles: tuple[float, float, float],
    completed_draws: int,
) -> pd.DataFrame:
    summary = summarize_case_uncertainty(
        probabilities[:, :, :completed_draws],
        case_table,
        scenarios=scenarios,
        quantiles=quantiles,
    )
    summary.insert(5, "draw_prefix", completed_draws)
    return summary


def _stage_plateau_metrics(
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
    *,
    config: dict[str, Any],
    completed_draws: int,
    stable_checkpoint_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if previous is not None:
        keys = ["target_case_id", "scenario"]
        merged = current.merge(previous, on=keys, suffixes=("", "_previous"))
        for scenario, frame in merged.groupby("scenario", sort=False):
            endpoint_change = np.maximum.reduce(
                [
                    np.abs(
                        frame["p_cancer_p025"]
                        - frame["p_cancer_p025_previous"]
                    ),
                    np.abs(
                        frame["p_cancer_p50"]
                        - frame["p_cancer_p50_previous"]
                    ),
                    np.abs(
                        frame["p_cancer_p975"]
                        - frame["p_cancer_p975_previous"]
                    ),
                ]
            )
            crossing_change = abs(
                int(frame["threshold_crossing"].sum())
                - int(frame["threshold_crossing_previous"].sum())
            )
            threshold_status_change_count = int(
                np.sum(
                    frame["threshold_crossing"].astype(bool).to_numpy()
                    != frame["threshold_crossing_previous"].astype(bool).to_numpy()
                )
            )
            rows.append(
                {
                    "scenario": scenario,
                    "completed_draws": completed_draws,
                    "median_endpoint_change": float(np.median(endpoint_change)),
                    "p90_endpoint_change": float(np.quantile(endpoint_change, 0.9)),
                    "threshold_crossing_count": int(
                        frame["threshold_crossing"].sum()
                    ),
                    "threshold_crossing_count_change": crossing_change,
                    "threshold_status_change_count": threshold_status_change_count,
                }
            )
    metrics = pd.DataFrame(rows)
    convergence = config["convergence"]
    stable = bool(
        not metrics.empty
        and (metrics["median_endpoint_change"] <= float(
            convergence["median_endpoint_change_tolerance"]
        )).all()
        and (metrics["p90_endpoint_change"] <= float(
            convergence["p90_endpoint_change_tolerance"]
        )).all()
        and (metrics["threshold_crossing_count_change"] <= int(
            convergence["max_threshold_crossing_count_change"]
        )).all()
        and (metrics["threshold_status_change_count"] <= int(
            convergence["max_threshold_status_change_count"]
        )).all()
    )
    consecutive = stable_checkpoint_count + 1 if stable else 0
    plateau = bool(
        completed_draws >= int(convergence["minimum_draws"])
        and consecutive >= int(convergence["required_stable_checkpoints"])
    )
    return metrics, {
        "stable_checkpoint": stable,
        "consecutive_stable_checkpoints": consecutive,
        "plateau": plateau,
    }


def _write_stage_convergence(
    run_folder: Path,
    *,
    case_summary: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    plateau_metrics: pd.DataFrame,
    plateau_status: dict[str, Any],
    completed_draws: int,
    nested_axis_summary: pd.DataFrame | None = None,
    nested_axis_changes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    stage_folder = run_folder / CONVERGENCE_FOLDER / f"draws_{completed_draws:05d}"
    stage_folder.mkdir(parents=True, exist_ok=True)
    case_summary.to_csv(stage_folder / "case_summary.csv", index=False)
    cohort_summary.to_csv(stage_folder / "cohort_summary.csv", index=False)
    plateau_metrics.to_csv(stage_folder / "plateau_metrics.csv", index=False)
    nested_axis_summary_path = None
    nested_axis_changes_path = None
    if nested_axis_summary is not None:
        nested_axis_summary_path = stage_folder / "nested_axis_case_summary.csv"
        nested_axis_summary.to_csv(nested_axis_summary_path, index=False)
    if nested_axis_changes is not None:
        nested_axis_changes_path = stage_folder / "nested_axis_changes.csv"
        nested_axis_changes.to_csv(nested_axis_changes_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    scenario_order = cohort_summary["scenario"].astype(str).tolist()
    axes[0].barh(scenario_order, cohort_summary["median_interval_width"])
    axes[0].set_xlabel("Median 95% interval width")
    axes[0].set_title(f"Measurement uncertainty after {completed_draws} draws")
    axes[1].barh(scenario_order, cohort_summary["threshold_crossing_fraction"])
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Fraction of cases crossing threshold")
    axes[1].set_title("Decision instability")
    dashboard_path = stage_folder / "convergence_dashboard.png"
    figure.savefig(dashboard_path, dpi=160)
    plt.close(figure)

    history_rows: list[pd.DataFrame] = []
    for path in sorted((run_folder / CONVERGENCE_FOLDER).glob("draws_*/cohort_summary.csv")):
        history_rows.append(pd.read_csv(path))
    history = pd.concat(history_rows, ignore_index=True)
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for scenario, frame in history.groupby("scenario", sort=False):
        axis.plot(
            frame["draw_prefix"],
            frame["median_interval_width"],
            marker="o",
            linewidth=1.2,
            label=scenario,
        )
    axis.set_xlabel("Completed draws")
    axis.set_ylabel("Median 95% interval width")
    axis.set_title("Convergence by uncertainty scenario")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    history_path = run_folder / CONVERGENCE_FOLDER / "convergence_history.png"
    figure.savefig(history_path, dpi=160)
    plt.close(figure)

    status = {
        "completed_draws": int(completed_draws),
        **plateau_status,
        "case_summary": str(stage_folder / "case_summary.csv"),
        "cohort_summary": str(stage_folder / "cohort_summary.csv"),
        "dashboard": str(dashboard_path),
        "history_plot": str(history_path),
        "nested_axis_case_summary": (
            str(nested_axis_summary_path)
            if nested_axis_summary_path is not None
            else None
        ),
        "nested_axis_changes": (
            str(nested_axis_changes_path)
            if nested_axis_changes_path is not None
            else None
        ),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(stage_folder / "stage_summary.json", status)
    _atomic_write_json(run_folder / CONVERGENCE_FOLDER / "latest.json", status)
    shutil.copy2(dashboard_path, run_folder / CONVERGENCE_FOLDER / "latest.png")
    return status


def summarize_case_convergence(
    probabilities: np.ndarray,
    case_table: pd.DataFrame,
    *,
    scenarios: tuple[Scenario, ...],
    quantiles: tuple[float, float, float],
) -> pd.DataFrame:
    """Summarize case-level uncertainty at deterministic draw prefixes."""
    values = np.asarray(probabilities)
    prefixes = convergence_draw_prefixes(values.shape[2])
    rows: list[pd.DataFrame] = []
    for prefix in prefixes:
        summary = summarize_case_uncertainty(
            values[:, :, :prefix],
            case_table,
            scenarios=scenarios,
            quantiles=quantiles,
        )
        summary.insert(5, "draw_prefix", prefix)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def summarize_nested_axis_convergence(
    probabilities: np.ndarray,
    case_table: pd.DataFrame,
    *,
    scenarios: tuple[Scenario, ...],
    quantiles: tuple[float, float, float],
    geometry_draws: int,
    photon_replicates: int,
    geometry_prefixes: tuple[int, ...] = (),
    photon_prefixes: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Summarize geometry and photon convergence without mixing their counts."""
    values = np.asarray(probabilities)
    expected_draws = geometry_draws * photon_replicates
    if values.ndim != 3 or values.shape[2] != expected_draws:
        raise ValueError("Probability cube does not match the nested design.")
    nested = values.reshape(
        values.shape[0],
        values.shape[1],
        geometry_draws,
        photon_replicates,
    )
    geometry_prefixes = _axis_prefixes(geometry_prefixes, geometry_draws)
    photon_prefixes = _axis_prefixes(photon_prefixes, photon_replicates)
    rows: list[pd.DataFrame] = []
    for axis, prefixes in (
        ("geometry", geometry_prefixes),
        ("photon", photon_prefixes),
    ):
        for prefix in prefixes:
            if axis == "geometry":
                used_geometry = prefix
                used_photon = photon_replicates
                selected = nested[:, :, :prefix, :]
            else:
                used_geometry = geometry_draws
                used_photon = prefix
                selected = nested[:, :, :, :prefix]
            flattened = selected.reshape(
                values.shape[0],
                values.shape[1],
                used_geometry * used_photon,
            )
            summary = summarize_case_uncertainty(
                flattened,
                case_table,
                scenarios=scenarios,
                quantiles=quantiles,
            )
            summary.insert(5, "convergence_axis", axis)
            summary.insert(6, "axis_prefix", prefix)
            summary.insert(7, "geometry_draws", used_geometry)
            summary.insert(8, "photon_replicates_per_geometry", used_photon)
            summary.insert(9, "effective_draws", used_geometry * used_photon)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def summarize_nested_axis_changes(case_convergence: pd.DataFrame) -> pd.DataFrame:
    """Compare interval endpoints at consecutive geometry or photon prefixes."""
    required = {
        "target_case_id",
        "scenario",
        "convergence_axis",
        "axis_prefix",
        "p_cancer_p025",
        "p_cancer_p50",
        "p_cancer_p975",
        "threshold_crossing",
    }
    missing = required.difference(case_convergence.columns)
    if missing:
        raise ValueError(f"Nested convergence columns are missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    keys = ["target_case_id", "scenario"]
    for axis, axis_frame in case_convergence.groupby(
        "convergence_axis", sort=False
    ):
        prefixes = sorted(axis_frame["axis_prefix"].astype(int).unique())
        for previous_prefix, current_prefix in zip(prefixes, prefixes[1:]):
            previous = axis_frame[
                axis_frame["axis_prefix"].astype(int).eq(previous_prefix)
            ]
            current = axis_frame[
                axis_frame["axis_prefix"].astype(int).eq(current_prefix)
            ]
            merged = current.merge(previous, on=keys, suffixes=("", "_previous"))
            for scenario, frame in merged.groupby("scenario", sort=False):
                endpoint_change = np.maximum.reduce(
                    [
                        np.abs(
                            frame["p_cancer_p025"]
                            - frame["p_cancer_p025_previous"]
                        ),
                        np.abs(
                            frame["p_cancer_p50"]
                            - frame["p_cancer_p50_previous"]
                        ),
                        np.abs(
                            frame["p_cancer_p975"]
                            - frame["p_cancer_p975_previous"]
                        ),
                    ]
                )
                rows.append(
                    {
                        "convergence_axis": axis,
                        "scenario": scenario,
                        "previous_prefix": previous_prefix,
                        "current_prefix": current_prefix,
                        "median_endpoint_change": float(
                            np.median(endpoint_change)
                        ),
                        "p90_endpoint_change": float(
                            np.quantile(endpoint_change, 0.9)
                        ),
                        "threshold_crossing_count": int(
                            frame["threshold_crossing"].astype(bool).sum()
                        ),
                        "threshold_status_change_count": int(
                            np.sum(
                                frame["threshold_crossing"].astype(bool).to_numpy()
                                != frame["threshold_crossing_previous"]
                                .astype(bool)
                                .to_numpy()
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _axis_prefixes(prefixes: tuple[int, ...], maximum: int) -> tuple[int, ...]:
    if maximum < 1:
        raise ValueError("Nested convergence maximum must be positive.")
    selected = {int(value) for value in prefixes if 0 < int(value) <= maximum}
    selected.add(maximum)
    return tuple(sorted(selected))


def summarize_cohort_convergence(case_convergence: pd.DataFrame) -> pd.DataFrame:
    """Aggregate convergence by scenario without replacing case-level results."""
    return (
        case_convergence.groupby(["scenario", "draw_prefix"], sort=False)
        .agg(
            target_cases=("target_case_id", "size"),
            median_interval_width=("interval_width", "median"),
            mean_interval_width=("interval_width", "mean"),
            mean_scenario_draw_fraction_at_or_above_threshold=(
                "scenario_draw_fraction_at_or_above_threshold",
                "mean",
            ),
            median_scenario_class_flip_fraction=(
                "scenario_class_flip_fraction",
                "median",
            ),
            threshold_crossing_fraction=("threshold_crossing", "mean"),
        )
        .reset_index()
    )


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
    metal_context: PatientMetalContext,
    scenario: Scenario,
    nuisance: NuisanceDraws,
    draws: int,
    draw_start: int = 0,
    draw_stop: int | None = None,
    geometry_draw_start: int | None = None,
    geometry_draw_stop: int | None = None,
    photon_replicates: int = 1,
    draw_chunk_size: int,
    geometry_audit_draws: int,
    normalization_q_range: tuple[float, float],
    metal_profile_max_tolerance: float,
    metal_profile_p99_tolerance: float,
    metal_p_cancer_parity_tolerance: float,
    random_seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    if draw_stop is None:
        draw_stop = draws
    if not 0 <= draw_start < draw_stop <= draws:
        raise MeasurementUncertaintyError("Invalid patient/scenario draw range.")
    if geometry_draw_start is None:
        geometry_draw_start = draw_start
    if geometry_draw_stop is None:
        geometry_draw_stop = draw_stop
    if photon_replicates < 1 or (
        geometry_draw_stop - geometry_draw_start
    ) * photon_replicates != draw_stop - draw_start:
        raise MeasurementUncertaintyError(
            "Geometry draw range and photon replicates do not match output range."
        )
    if (
        draw_start != geometry_draw_start * photon_replicates
        or draw_stop != geometry_draw_stop * photon_replicates
    ):
        raise MeasurementUncertaintyError(
            "Nested output draw range must be geometry range times replicates."
        )
    stage_draws = draw_stop - draw_start
    case_values = {
        str(case_id): np.empty(stage_draws, dtype=np.float32)
        for case_id in patient_cases["target_case_id"]
    }
    metal_audit_chunks: list[np.ndarray] = []
    expected_audit_chunks: list[np.ndarray] = []
    geometry_rows: list[dict[str, Any]] = []
    for start in range(geometry_draw_start, geometry_draw_stop, draw_chunk_size):
        stop = min(geometry_draw_stop, start + draw_chunk_size)
        profiles, metal_nominal, expected, q_values, geometry = (
            _metal_profile_chunk(
                patient_frame,
                metal_context=metal_context,
                scenario=scenario,
                nuisance=nuisance,
                start=start,
                stop=stop,
                audit_draw_start=geometry_draw_start,
                geometry_audit_draws=geometry_audit_draws,
                normalization_q_range=normalization_q_range,
                random_seed=random_seed,
                photon_replicates=photon_replicates,
            )
        )
        geometry_rows.extend(geometry)
        if metal_nominal.shape[0]:
            metal_audit_chunks.append(metal_nominal)
            expected_audit_chunks.append(expected)
        score_kwargs = {
            "patient_manifest": patient_frame,
            "q_grid": q_values,
            "target_manifest": patient_cases,
            "model_artifact": model_artifact,
        }
        if not np.isfinite(profiles).all():
            raise MeasurementUncertaintyError(
                "Metal integration produced non-finite normalized profiles."
            )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scores = score_frozen_aramina_0_2_15_cube(profiles, **score_kwargs)
        if not np.isfinite(scores.p_cancer).all():
            raise MeasurementUncertaintyError(
                "Frozen model produced non-finite p_cancer values."
            )
        for target_index, case_id in enumerate(scores.target_case_ids):
            output_start = (start - geometry_draw_start) * photon_replicates
            output_stop = (stop - geometry_draw_start) * photon_replicates
            case_values[case_id][
                output_start:output_stop
            ] = scores.p_cancer[:, target_index]
    if not metal_audit_chunks:
        raise MeasurementUncertaintyError(
            "No geometry audit draws were produced for fail-closed parity."
        )
    metal_audit = np.concatenate(metal_audit_chunks, axis=0)
    expected_audit = np.concatenate(expected_audit_chunks, axis=0)
    profile_metrics = _profile_parity_metrics(
        metal_audit,
        expected_audit,
        metal_context.q_grid,
        draw_start=geometry_draw_start,
        maximum_tolerance=metal_profile_max_tolerance,
        p99_tolerance=metal_profile_p99_tolerance,
    )
    geometry_aware_backend = (
        getattr(metal_context, "backend_kind", None)
        == "metal_geometry_aware_nested"
    )
    parity = {
        "patient_id": str(patient_frame["patientId"].iloc[0]),
        "scenario": scenario.name,
        "draw_start": int(draw_start),
        "draw_stop": int(draw_stop),
        "geometry_draw_start": int(geometry_draw_start),
        "geometry_draw_stop": int(geometry_draw_stop),
        "photon_replicates_per_geometry": int(photon_replicates),
        "oracle_draws": int(metal_audit.shape[0]),
        **profile_metrics,
        "geometry_backend": (
            "metal_dynamic_bbox"
            if geometry_aware_backend
            else "pyfai_bbox_csr"
        ),
        "photon_backend": (
            "metal_geometry_aware_nested"
            if geometry_aware_backend and scenario.photon
            else "metal_prepared_csr"
            if scenario.photon
            else "none"
        ),
        "gpu_geometry_recalculation": geometry_aware_backend,
    }
    if not profile_metrics["parity_pass"]:
        raise MeasurementUncertaintyError(
            "Metal profile parity exceeds configured max/p99 tolerances: "
            f"max {profile_metrics['maximum_absolute_error']:.8g}, "
            f"limit {metal_profile_max_tolerance:.8g}; "
            f"p99 {profile_metrics['p99_absolute_error']:.8g}, "
            f"limit {metal_profile_p99_tolerance:.8g}."
        )
    audit_kwargs = {
        "patient_manifest": patient_frame,
        "q_grid": metal_context.q_grid,
        "target_manifest": patient_cases,
        "model_artifact": model_artifact,
    }
    if not np.isfinite(metal_audit).all() or not np.isfinite(expected_audit).all():
        raise MeasurementUncertaintyError(
            "Metal/pyFAI parity audit contains non-finite profiles."
        )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        metal_scores = score_frozen_aramina_0_2_15_cube(
            metal_audit, **audit_kwargs
        )
        expected_scores = score_frozen_aramina_0_2_15_cube(
            expected_audit, **audit_kwargs
        )
    if not np.isfinite(metal_scores.p_cancer).all() or not np.isfinite(
        expected_scores.p_cancer
    ).all():
        raise MeasurementUncertaintyError(
            "Metal/pyFAI parity audit produced non-finite p_cancer values."
        )
    maximum_score_error = float(
        np.max(np.abs(metal_scores.p_cancer - expected_scores.p_cancer))
    )
    decision_class_equal = bool(
        np.array_equal(
            metal_scores.p_cancer >= metal_scores.threshold,
            expected_scores.p_cancer >= expected_scores.threshold,
        )
    )
    parity["maximum_absolute_p_cancer_error"] = maximum_score_error
    parity["p_cancer_parity_tolerance"] = metal_p_cancer_parity_tolerance
    parity["p_cancer_parity_pass"] = bool(
        maximum_score_error <= metal_p_cancer_parity_tolerance
    )
    parity["decision_class_equal"] = decision_class_equal
    if not decision_class_equal:
        raise MeasurementUncertaintyError(
            "Metal and pyFAI parity changed the audited decision class."
        )
    if maximum_score_error > metal_p_cancer_parity_tolerance:
        raise MeasurementUncertaintyError(
            "Metal p_cancer parity exceeds configured tolerance: "
            f"{maximum_score_error:.8g} > "
            f"{metal_p_cancer_parity_tolerance:.8g}."
        )
    return case_values, [parity], geometry_rows


def _metal_profile_chunk(
    patient_frame: pd.DataFrame,
    *,
    metal_context: PatientMetalContext,
    scenario: Scenario,
    nuisance: NuisanceDraws,
    start: int,
    stop: int,
    audit_draw_start: int = 0,
    geometry_audit_draws: int,
    normalization_q_range: tuple[float, float],
    random_seed: int,
    photon_replicates: int = 1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    if getattr(metal_context, "backend_kind", None) == "metal_geometry_aware_nested":
        return _geometry_aware_metal_profile_chunk(
            patient_frame,
            metal_context=metal_context,
            scenario=scenario,
            nuisance=nuisance,
            start=start,
            stop=stop,
            audit_draw_start=audit_draw_start,
            geometry_audit_draws=geometry_audit_draws,
            normalization_q_range=normalization_q_range,
            random_seed=random_seed,
            photon_replicates=photon_replicates,
        )
    audit_stop = min(stop, audit_draw_start + geometry_audit_draws)
    if start < audit_draw_start < stop:
        raise MeasurementUncertaintyError(
            "Metal chunk starts before its audit stage boundary."
        )
    profile_chunks: list[np.ndarray] = []
    metal_audit_chunks: list[np.ndarray] = []
    expected_audit_chunks: list[np.ndarray] = []
    geometry_rows: list[dict[str, Any]] = []
    for draw_index in range(start, stop):
        expected, plans, rows = _prepare_pyfai_geometry_draw(
            patient_frame,
            scenario=scenario,
            nuisance=nuisance,
            draw_index=draw_index,
            q_grid=metal_context.q_grid,
            normalization_q_range=normalization_q_range,
            prepare_metal_plans=scenario.photon,
            geometry_plan_cache=getattr(metal_context, "geometry_plan_cache", None),
        )
        audited = audit_draw_start <= draw_index < audit_stop
        if scenario.photon:
            result = metal_context.session.run_geometry(
                plans,
                photon_replicates,
                seed=_prepared_geometry_seed(random_seed, draw_index),
                include_deterministic=audited,
            )
            profile_chunks.append(result.profiles)
            if audited:
                if result.deterministic_profiles is None:
                    raise MeasurementUncertaintyError(
                        "Prepared Metal audit omitted deterministic profiles."
                    )
                metal_audit_chunks.append(
                    result.deterministic_profiles[np.newaxis, ...]
                )
        else:
            profile_chunks.append(
                np.repeat(expected[np.newaxis, ...], photon_replicates, axis=0)
            )
            if audited:
                metal_audit_chunks.append(expected[np.newaxis, ...])
        if audited:
            expected_audit_chunks.append(expected[np.newaxis, ...])
            geometry_rows.extend(rows)

    profile_cube = np.concatenate(profile_chunks, axis=0)
    if metal_audit_chunks:
        metal_nominal_cube = np.concatenate(metal_audit_chunks, axis=0)
        expected_cube = np.concatenate(expected_audit_chunks, axis=0)
    else:
        empty_shape = (0, len(patient_frame), metal_context.q_grid.size)
        metal_nominal_cube = np.empty(empty_shape, dtype=float)
        expected_cube = np.empty(empty_shape, dtype=float)
    return (
        profile_cube,
        metal_nominal_cube,
        expected_cube,
        metal_context.q_grid,
        geometry_rows,
    )


def _geometry_aware_metal_profile_chunk(
    patient_frame: pd.DataFrame,
    *,
    metal_context: PatientMetalContext,
    scenario: Scenario,
    nuisance: NuisanceDraws,
    start: int,
    stop: int,
    audit_draw_start: int,
    geometry_audit_draws: int,
    normalization_q_range: tuple[float, float],
    random_seed: int,
    photon_replicates: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    """Run dynamic geometry in Metal and reserve pyFAI for bounded oracle audits."""
    if start < audit_draw_start < stop:
        raise MeasurementUncertaintyError(
            "Metal chunk starts before its audit stage boundary."
        )
    geometry_draws = stop - start
    distances, poni1, poni2 = _geometry_aware_draw_arrays(
        metal_context,
        scenario=scenario,
        nuisance=nuisance,
        start=start,
        stop=stop,
    )
    if scenario.photon:
        nested = metal_context.session.run_nested(
            (1.0,),
            geometry_draws,
            photon_replicates,
            effective_distance_m=distances,
            poni1_m=poni1,
            poni2_m=poni2,
            seed=random_seed,
            geometry_draw_offset=start,
            photon_draw_offset=start * photon_replicates,
            geometry_chunk_size=geometry_draws,
        )
        profile_cube = np.ascontiguousarray(
            nested[0].reshape(
                geometry_draws * photon_replicates,
                len(patient_frame),
                metal_context.q_grid.size,
            )
        )
    else:
        deterministic = metal_context.session.integrate(
            geometry_draws,
            effective_distance_m=distances,
            poni1_m=poni1,
            poni2_m=poni2,
            draw_offset=start,
            draw_chunk_size=geometry_draws,
        )
        profile_cube = np.repeat(deterministic, photon_replicates, axis=0)

    audit_stop = min(stop, audit_draw_start + geometry_audit_draws)
    if not audit_draw_start <= start < audit_stop:
        empty_shape = (0, len(patient_frame), metal_context.q_grid.size)
        return (
            profile_cube,
            np.empty(empty_shape, dtype=float),
            np.empty(empty_shape, dtype=float),
            metal_context.q_grid,
            [],
        )
    audited_count = audit_stop - start
    audit_distances = distances[:audited_count]
    audit_poni1 = poni1[:audited_count]
    audit_poni2 = poni2[:audited_count]
    metal_nominal = metal_context.session.integrate(
        audited_count,
        effective_distance_m=audit_distances,
        poni1_m=audit_poni1,
        poni2_m=audit_poni2,
        draw_offset=start,
        draw_chunk_size=audited_count,
    )
    expected, geometry_rows = _pyfai_oracle_profiles(
        patient_frame,
        scenario=scenario,
        nuisance=nuisance,
        start=start,
        stop=audit_stop,
        q_grid=metal_context.q_grid,
        normalization_q_range=normalization_q_range,
    )
    return (
        profile_cube,
        metal_nominal,
        expected,
        metal_context.q_grid,
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


def _poni_geometry_key(row: pd.Series) -> str:
    value = row.get("ponifile")
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Every measurement requires a non-empty PONI file.")
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _monte_carlo_design(config: dict[str, Any]) -> MonteCarloDesign:
    monte_carlo = config["monte_carlo"]
    execution = config["execution"]
    mode = str(monte_carlo.get("design", "direct_joint"))
    if mode == "direct_joint":
        draws = int(monte_carlo["draws"])
        stage = int(execution.get("global_stage_draws", 250))
        design = MonteCarloDesign(mode, draws, 1, stage)
    elif mode == "nested_geometry_photon":
        geometry_draws = int(monte_carlo["geometry_draws"])
        photon_replicates = int(monte_carlo["photon_replicates_per_geometry"])
        geometry_stage = int(execution["global_stage_geometry_draws"])
        design = MonteCarloDesign(
            mode,
            geometry_draws,
            photon_replicates,
            geometry_stage,
        )
    else:
        raise ValueError(
            "Monte Carlo design must be direct_joint or nested_geometry_photon."
        )
    if (
        design.geometry_draws < 1
        or design.photon_replicates < 1
        or design.geometry_stage_draws < 1
        or design.geometry_stage_draws > design.geometry_draws
    ):
        raise ValueError("Monte Carlo design counts are invalid.")
    return design


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
    backend_kind = config.get("backend", {}).get("kind")
    if backend_kind not in {
        "pyfai_prepared_csr_metal_photon_mc",
        "metal_geometry_aware_nested",
    }:
        raise ValueError(
            "Joint uncertainty requires a supported prepared-plan or "
            "geometry-aware Metal backend."
        )
    if int(config.get("integration", {}).get("npt", 0)) != 100:
        raise ValueError("Frozen Aramina 0.2.15 requires 100-bin integration.")
    nuisance = config.get("nuisance", {})
    if nuisance.get("beam_center", {}).get("correlation") != "poni_file_shared":
        raise ValueError("Beam-centre uncertainty must be shared by PONI file.")
    if nuisance.get("detector_distance", {}).get("correlation") != (
        "poni_file_shared"
    ):
        raise ValueError("Detector-distance uncertainty must be shared by PONI file.")
    quantiles = config["monte_carlo"]["quantiles"]
    if len(quantiles) != 3 or not 0 < quantiles[0] < quantiles[1] < quantiles[2] < 1:
        raise ValueError("Monte Carlo quantiles must be ordered inside (0, 1).")
    design = _monte_carlo_design(config)
    if design.mode == "direct_joint":
        config["execution"]["global_stage_draws"] = design.geometry_stage_draws
    convergence = config.setdefault("convergence", {})
    convergence.setdefault("auto_stop", False)
    convergence.setdefault("minimum_draws", min(2000, design.output_draws))
    convergence.setdefault(
        "minimum_geometry_draws",
        min(400, design.geometry_draws),
    )
    convergence.setdefault("required_stable_checkpoints", 3)
    convergence.setdefault("median_endpoint_change_tolerance", 0.0025)
    convergence.setdefault("p90_endpoint_change_tolerance", 0.01)
    convergence.setdefault("max_threshold_crossing_count_change", 1)
    convergence.setdefault("max_threshold_status_change_count", 1)
    convergence.setdefault("geometry_prefixes", [100, 250, 500, 1000, 2000])
    convergence.setdefault("photon_prefixes", [10, 20, 30, 40, 50])
    convergence.setdefault("minimum_photon_replicates_per_geometry", 1)
    minimum_photon_replicates = int(
        convergence["minimum_photon_replicates_per_geometry"]
    )
    if minimum_photon_replicates < 1:
        raise ValueError("Minimum photon replicates must be positive.")
    if (
        design.mode == "nested_geometry_photon"
        and design.photon_replicates < minimum_photon_replicates
    ):
        raise ValueError(
            "Nested design has fewer photon replicates than the declared minimum."
        )
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


def _paused_result(
    run_folder: Path,
    *,
    patient_ids: list[str],
    selected_cases: pd.DataFrame,
    completed_draws: int,
    reason: str,
    resuming: bool,
) -> dict[str, Any]:
    return {
        "status": "paused",
        "pause_reason": reason,
        "run_folder": str(run_folder),
        "summary_path": str(run_folder / CONVERGENCE_FOLDER / "latest.json"),
        "probability_path": str(run_folder / PROBABILITY_FILENAME),
        "patients": len(patient_ids),
        "target_cases": len(selected_cases),
        "completed_draws": int(completed_draws),
        "resumed": resuming,
        "mlflow": {"run_id": None, "status": "not_logged_until_complete"},
        "manifest": None,
    }


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
    case_convergence: pd.DataFrame,
    cohort_convergence: pd.DataFrame,
    nested_axis_summary: pd.DataFrame | None,
    nested_axis_changes: pd.DataFrame | None,
    parity: pd.DataFrame,
    metal_parity: pd.DataFrame,
    geometry_draws: pd.DataFrame,
    metadata_qc: pd.DataFrame,
    scenarios: tuple[Scenario, ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    design = _monte_carlo_design(config)
    shutil.copy2(config_path, run_folder / "effective_experiment_config.yaml")
    (run_folder / "effective_training_preprocessing.yaml").write_text(
        yaml.safe_dump(effective_preprocessing, sort_keys=False), encoding="utf-8"
    )
    pointer = _resolve_path(config["data_version"]["pointer_path"], config_path)
    shutil.copy2(pointer, run_folder / "dvc_data_pointer.dvc")
    selected_cases.to_csv(run_folder / "selected_cases.csv", index=False)
    summaries.to_csv(run_folder / "case_uncertainty_summary.csv", index=False)
    case_convergence.to_csv(
        run_folder / "case_uncertainty_convergence.csv", index=False
    )
    cohort_convergence.to_csv(
        run_folder / "cohort_uncertainty_convergence.csv", index=False
    )
    if nested_axis_summary is not None:
        nested_axis_summary.to_csv(
            run_folder / "nested_axis_case_convergence.csv", index=False
        )
    if nested_axis_changes is not None:
        nested_axis_changes.to_csv(
            run_folder / "nested_axis_changes.csv", index=False
        )
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
    nuisance_scope = pd.read_csv(run_folder / NUISANCE_SCOPE_FILENAME)
    manifest = {
        "contract": RESULT_CONTRACT,
        "status": "complete",
        "patients": int(selected_cases["patient_id"].nunique()),
        "target_cases": len(selected_cases),
        "monte_carlo_design": design.mode,
        "draws": design.output_draws,
        "independent_geometry_draws": design.geometry_draws,
        "photon_replicates_per_geometry": design.photon_replicates,
        "poni_geometry_groups": int(
            nuisance_scope["poni_geometry_sha256"].nunique()
        ),
        "thickness_groups": int(nuisance_scope["thickness_group"].nunique()),
        "outer_draw_scope": "cohort_aligned_by_poni_geometry_group",
        "convergence_draw_prefixes": list(
            convergence_draw_prefixes(
                design.output_draws,
                stage_draws=design.output_stage_draws,
            )
        ),
        "geometry_convergence_prefixes": list(
            config["convergence"]["geometry_prefixes"]
        ),
        "photon_convergence_prefixes": list(
            config["convergence"]["photon_prefixes"]
        ),
        "scenarios": [value.name for value in scenarios],
        "probability_values": int(
            len(selected_cases)
            * len(scenarios)
            * design.output_draws
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
        "case_uncertainty_convergence.csv",
        "cohort_uncertainty_convergence.csv",
        "pyfai_parity.csv",
        "metal_parity.csv",
        "geometry_draws.csv",
        "thickness_metadata_qc.csv",
        NUISANCE_SCOPE_FILENAME,
        "lineage.json",
        "run_manifest.json",
        PROBABILITY_FILENAME,
        RUN_STATE_FILENAME,
        PROGRESS_FILENAME,
    ]
    if manifest["monte_carlo_design"] == "nested_geometry_photon":
        required.extend(
            [
                "nested_axis_case_convergence.csv",
                "nested_axis_changes.csv",
            ]
        )
    with run:
        for step, scenario in enumerate(manifest["scenarios"]):
            subset = summaries[summaries["scenario"].eq(scenario)]
            run.log_metrics(
                {
                    "median_interval_width": float(subset["interval_width"].median()),
                    "threshold_crossing_fraction": float(
                        subset["threshold_crossing"].mean()
                    ),
                    "median_scenario_class_flip_fraction": float(
                        subset["scenario_class_flip_fraction"].median()
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
    "CohortNuisanceDraws",
    "NuisanceDraws",
    "Scenario",
    "convergence_draw_prefixes",
    "effective_detector_distance_m",
    "run_joint_measurement_uncertainty_from_config",
    "sample_cohort_nuisance_draws",
    "sample_nuisance_draws",
    "summarize_case_convergence",
    "summarize_case_uncertainty",
    "summarize_cohort_convergence",
    "summarize_nested_axis_changes",
    "summarize_nested_axis_convergence",
    "thickness_metadata_audit",
]
