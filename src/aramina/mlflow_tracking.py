"""Bounded MLflow tracking for one product training run."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
from math import isfinite
from numbers import Integral, Real
from os import environ
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, Self


class MlflowTrackingError(RuntimeError):
    """Raised when enabled MLflow tracking cannot complete."""


class MlflowRun:
    """Create and manage exactly one explicit MLflow training run."""

    def __init__(
        self,
        *,
        enabled: bool,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._params = params or {}
        self._tags = tags or {}
        self._client: Any | None = None
        self.run_id: str | None = None
        self.status = "NOT_STARTED"

    def __enter__(self) -> Self:
        if self.status != "NOT_STARTED":
            raise RuntimeError("An MlflowRun instance can be entered only once.")
        if not self.enabled:
            self.status = "DISABLED"
            return self
        if not self.tracking_uri:
            raise ValueError("Enabled MLflow tracking requires tracking_uri.")
        if not self.experiment_name:
            raise ValueError("Enabled MLflow tracking requires experiment_name.")

        params = _flatten_scalars(self._params, value_kind="parameter")
        tags = _flatten_scalars(self._tags, value_kind="tag")
        if self.run_name:
            tags["mlflow.runName"] = _scalar_text(self.run_name, "run name")

        try:
            tracking = import_module("mlflow.tracking")
            self._client = _new_mlflow_client(tracking, self.tracking_uri)
            experiment_id = self._get_or_create_experiment_id()
            run = self._client.create_run(experiment_id, tags=tags)
            self.run_id = run.info.run_id
            self.status = run.info.status
            self.log_params(params)
        except Exception as exc:
            self._fail_started_run()
            if isinstance(exc, MlflowTrackingError):
                raise
            raise MlflowTrackingError(
                f"MLflow run initialization failed for {self.tracking_uri!r}."
            ) from exc
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, traceback
        if not self.enabled:
            return False
        target_status = "FAILED" if exc_value is not None else "FINISHED"
        try:
            self._terminate(target_status)
        except Exception as exc:
            self.status = "TRACKING_FAILED"
            raise MlflowTrackingError(
                f"MLflow run {self.run_id!r} could not be terminated as "
                f"{target_status}."
            ) from exc_value or exc
        return False

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log flattened scalar parameters to the active run."""
        if not self.enabled:
            return
        client, run_id = self._active_client_and_run()
        flattened = _flatten_scalars(params, value_kind="parameter")
        try:
            for key, value in flattened.items():
                client.log_param(run_id, key, value)
        except Exception as exc:
            raise MlflowTrackingError(
                f"MLflow parameter logging failed for run {run_id!r}."
            ) from exc

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        """Set flattened scalar tags on the active run."""
        if not self.enabled:
            return
        client, run_id = self._active_client_and_run()
        flattened = _flatten_scalars(tags, value_kind="tag")
        try:
            for key, value in flattened.items():
                client.set_tag(run_id, key, value)
        except Exception as exc:
            raise MlflowTrackingError(
                f"MLflow tag logging failed for run {run_id!r}."
            ) from exc

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int = 0) -> None:
        """Log finite numeric metrics to the active run."""
        if not self.enabled:
            return
        client, run_id = self._active_client_and_run()
        flattened = _flatten_metrics(metrics)
        try:
            for key, value in flattened.items():
                client.log_metric(run_id, key, value, step=step)
        except Exception as exc:
            raise MlflowTrackingError(
                f"MLflow metric logging failed for run {run_id!r}."
            ) from exc

    def log_artifact_directory(
        self,
        directory: str | Path,
        *,
        required_files: Iterable[str | Path],
        artifact_path: str | None = None,
    ) -> None:
        """Validate and log one complete artifact directory."""
        if not self.enabled:
            return
        client, run_id = self._active_client_and_run()
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise MlflowTrackingError(f"Artifact directory does not exist: {root}")
        missing = [
            str(relative)
            for relative in required_files
            if not _required_artifact(root, relative).is_file()
        ]
        if missing:
            raise MlflowTrackingError(
                f"Artifact directory is incomplete; missing required files: {missing}"
            )
        destination = _artifact_destination(artifact_path)
        try:
            client.log_artifacts(run_id, str(root), artifact_path=destination)
        except Exception as exc:
            raise MlflowTrackingError(
                f"MLflow artifact logging failed for run {run_id!r}."
            ) from exc

    def _get_or_create_experiment_id(self) -> str:
        experiment = self._client.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self._client.create_experiment(self.experiment_name)
        except Exception:
            experiment = self._client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise
            return experiment.experiment_id

    def _active_client_and_run(self) -> tuple[Any, str]:
        if self.status != "RUNNING" or self._client is None or self.run_id is None:
            raise RuntimeError("MLflow logging requires an active RUNNING context.")
        return self._client, self.run_id

    def _terminate(self, status: str) -> None:
        if self._client is None or self.run_id is None:
            raise RuntimeError("Enabled MLflow run was not initialized.")
        self._client.set_terminated(self.run_id, status=status)
        stored_status = self._client.get_run(self.run_id).info.status
        if stored_status != status:
            raise RuntimeError(
                f"MLflow stored status {stored_status!r}, expected {status!r}."
            )
        self.status = stored_status

    def _fail_started_run(self) -> None:
        if self._client is None or self.run_id is None:
            self.status = "TRACKING_FAILED"
            return
        try:
            self._terminate("FAILED")
        except Exception:
            self.status = "TRACKING_FAILED"


def _flatten_scalars(
    values: Mapping[str, Any],
    *,
    value_kind: str,
) -> dict[str, str]:
    flattened: dict[str, str] = {}

    def visit(mapping: Mapping[str, Any], prefix: str = "") -> None:
        for raw_key, value in mapping.items():
            key = _joined_key(prefix, raw_key)
            if isinstance(value, Mapping):
                visit(value, key)
                continue
            if key in flattened:
                raise ValueError(f"Duplicate flattened {value_kind} key: {key!r}.")
            flattened[key] = _scalar_text(value, value_kind)

    visit(values)
    return flattened


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}

    def visit(mapping: Mapping[str, Any], prefix: str = "") -> None:
        for raw_key, value in mapping.items():
            key = _joined_key(prefix, raw_key)
            if isinstance(value, Mapping):
                visit(value, key)
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Metric {key!r} must be numeric, got {type(value).__name__}.")
            number = float(value)
            if not isfinite(number):
                raise ValueError(f"Metric {key!r} must be finite.")
            if key in flattened:
                raise ValueError(f"Duplicate flattened metric key: {key!r}.")
            flattened[key] = number

    visit(metrics)
    return flattened


def _joined_key(prefix: str, raw_key: Any) -> str:
    if not isinstance(raw_key, str) or not raw_key.strip():
        raise ValueError("MLflow keys must be non-empty strings.")
    if any(ord(char) < 32 for char in raw_key):
        raise ValueError(f"MLflow key contains a control character: {raw_key!r}.")
    return f"{prefix}.{raw_key}" if prefix else raw_key


def _scalar_text(value: Any, value_kind: str) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"MLflow {value_kind} values must be finite.")
        return repr(number)
    if isinstance(value, (str, Path)):
        text = str(value)
        if "\x00" in text:
            raise ValueError(f"MLflow {value_kind} values cannot contain NUL.")
        return text
    raise TypeError(
        f"MLflow {value_kind} values must be scalar, got {type(value).__name__}."
    )


def _required_artifact(root: Path, relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Required artifact path must stay inside its directory: {path}")
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Required artifact path escapes its directory: {path}")
    return candidate


def _artifact_destination(value: str | None) -> str | None:
    if value is None:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"Invalid MLflow artifact_path: {value!r}.")
    return str(path)


def _new_mlflow_client(tracking: Any, tracking_uri: str) -> Any:
    env_name = "MLFLOW_ALLOW_FILE_STORE"
    previous = environ.get(env_name)
    if tracking_uri.startswith("file:"):
        environ[env_name] = "true"
    try:
        return tracking.MlflowClient(tracking_uri=tracking_uri)
    finally:
        if previous is None:
            environ.pop(env_name, None)
        else:
            environ[env_name] = previous
