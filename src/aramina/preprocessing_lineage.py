"""Semantic identity and compatibility checks for product preprocessing."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml
from xrd_preprocessing import (
    PREPROCESSING_ARTIFACT_VERSION,
    pipeline_spec_sha256,
    resolve_pipeline_spec,
    validate_preprocessing_artifact,
)

from .preprocessing_contract import (
    ARAMINA_PREPROCESSING_CONTRACT,
    validate_aramina_preprocessing_config,
)


ARAMINA_PREPROCESSING_LINEAGE_CONTRACT = "aramina_preprocessing_lineage_v0_2"
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def build_preprocessing_lineage(config: dict[str, Any]) -> dict[str, Any]:
    """Build immutable product and XRD identities for one resolved config."""
    validate_aramina_preprocessing_config(config)
    spec = resolve_pipeline_spec(config)
    xrd_policy = _mapping(config, "xrd_preprocessing")
    release_tag = _nonempty_string(xrd_policy, "release_tag", "xrd_preprocessing")
    if release_tag == "local":
        raise ValueError("Aramina product preprocessing requires an immutable XRD release_tag.")
    runtime = xrd_runtime_identity()
    _require_exact_runtime_identity(runtime)
    declared_version = _nonempty_string(
        xrd_policy,
        "version",
        "xrd_preprocessing",
    )
    if runtime["version"] != declared_version:
        raise ValueError(
            "Aramina preprocessing XRD package version differs from the "
            f"declared product version: {runtime['version']!r} != "
            f"{declared_version!r}."
        )
    declared_commit = _nonempty_string(
        xrd_policy,
        "git_commit",
        "xrd_preprocessing",
    )
    if runtime["git_commit"] != declared_commit:
        raise ValueError(
            "Aramina preprocessing XRD git commit differs from the declared "
            f"product commit: {runtime['git_commit']!r} != {declared_commit!r}."
        )
    product = _mapping(config, "aramina_preprocessing")
    return {
        "contract": ARAMINA_PREPROCESSING_LINEAGE_CONTRACT,
        "artifact_version": PREPROCESSING_ARTIFACT_VERSION,
        "product": {
            "contract": ARAMINA_PREPROCESSING_CONTRACT,
            "name": product["name"],
            "route": product["route"],
            "version": product["version"],
        },
        "xrd_preprocessing": {
            "release_tag": release_tag,
            **runtime,
        },
        "resolved_pipeline_spec": spec,
        "pipeline_fingerprint": pipeline_spec_sha256(spec),
    }


def require_training_preprocessing_artifact(
    artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    """Require a v0.2 artifact with exact product and executable lineage."""
    if not isinstance(artifact, dict):
        raise ValueError("Training requires a preprocessing artifact with provenance.")
    validate_preprocessing_artifact(artifact)
    if artifact.get("version") != PREPROCESSING_ARTIFACT_VERSION:
        raise ValueError(
            "New Aramina training requires preprocessing artifact version "
            f"{PREPROCESSING_ARTIFACT_VERSION}; legacy v0.1 is read-only."
        )
    config = _artifact_config(artifact)
    expected = build_preprocessing_lineage(config)
    if expected["product"]["route"] != "training":
        raise ValueError("Training requires the Aramina training preprocessing route.")
    actual = artifact.get("metadata", {}).get("aramina_preprocessing_lineage")
    if actual != expected:
        raise ValueError(
            "Training preprocessing artifact lineage differs from the resolved "
            "product config or installed XRD-preprocessing revision."
        )
    if artifact.get("resolved_pipeline_spec") != expected["resolved_pipeline_spec"]:
        raise ValueError(
            "Training preprocessing artifact resolved_pipeline_spec differs from "
            "the product config."
        )
    if artifact.get("pipeline_fingerprint") != expected["pipeline_fingerprint"]:
        raise ValueError(
            "Training preprocessing artifact pipeline_fingerprint differs from "
            "the product config."
        )
    return expected


def validate_prediction_preprocessing_compatibility(
    model_artifact: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Validate new model lineage while retaining frozen legacy prediction."""
    expected = model_artifact.get("prediction_preprocessing_lineage")
    if expected is None:
        validate_aramina_preprocessing_config(config, allow_legacy=True)
        return
    validate_aramina_preprocessing_config(config)
    actual = build_preprocessing_lineage(config)
    if actual["product"]["route"] != "prediction":
        raise ValueError("Prediction requires the Aramina prediction preprocessing route.")
    if actual != expected:
        raise ValueError(
            "Prediction preprocessing differs from the model-held resolved "
            "pipeline identity."
        )


def xrd_runtime_identity() -> dict[str, str]:
    """Return installed XRD package version, requested revision, and full commit."""
    result = {"version": _installed_version("xrd-preprocessing")}
    requested = os.environ.get("XRD_PREPROCESSING_REQUESTED_REVISION")
    commit = os.environ.get("XRD_PREPROCESSING_GIT_COMMIT")
    direct_url: dict[str, Any] = {}
    try:
        payload = distribution("xrd-preprocessing").read_text("direct_url.json")
    except PackageNotFoundError:
        payload = None
    if payload:
        direct_url = json.loads(payload)
        source_url = direct_url.get("url")
        vcs_info = direct_url.get("vcs_info")
        if isinstance(vcs_info, dict):
            requested = requested or vcs_info.get("requested_revision")
            commit = commit or vcs_info.get("commit_id")
        if commit is None and source_url:
            commit = _git_commit_from_file_url(source_url)
            requested = requested or commit
    source_root = _xrd_source_root_from_package_source()
    dir_info = direct_url.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable") is True:
        source_root = _xrd_source_root_from_file_url(direct_url.get("url"))
    if source_root is not None:
        _require_clean_git_source(source_root)
        commit = _git_commit(source_root)
        result["version"] = _package_version_from_source(source_root)
        requested = os.environ.get(
            "XRD_PREPROCESSING_REQUESTED_REVISION",
            commit,
        )
    result["requested_revision"] = str(requested or "unavailable")
    result["git_commit"] = str(commit or "unavailable")
    return result


def _artifact_config(artifact: dict[str, Any]) -> dict[str, Any]:
    text = artifact.get("preprocessing_config_yaml")
    if not isinstance(text, str):
        raise ValueError("Training artifact is missing preprocessing_config_yaml.")
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("Training preprocessing_config_yaml must contain a mapping.")
    return config


def _require_exact_runtime_identity(identity: dict[str, str]) -> None:
    commit = identity.get("git_commit", "")
    requested = identity.get("requested_revision", "")
    if not _FULL_GIT_SHA.fullmatch(commit):
        raise ValueError(
            "Aramina requires the full XRD-preprocessing git commit in runtime "
            "provenance."
        )
    if not requested or requested == "unavailable":
        raise ValueError("Aramina requires the requested XRD-preprocessing revision.")


def _git_commit_from_file_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    root = _xrd_source_root_from_path(Path(unquote(parsed.path)))
    return _git_commit(root) if root is not None else None


def _xrd_source_root_from_file_url(url: Any) -> Path | None:
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return _xrd_source_root_from_path(Path(unquote(parsed.path)))


def _xrd_source_root_from_package_source() -> Path | None:
    try:
        import xrd_preprocessing
    except ImportError:
        return None
    return _xrd_source_root_from_path(Path(xrd_preprocessing.__file__).resolve())


def _xrd_source_root_from_path(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if not (candidate / ".git").exists():
            continue
        if _package_name_from_source(candidate) == "xrd-preprocessing":
            return candidate
    return None


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _require_clean_git_source(root: Path) -> None:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"Cannot verify XRD-preprocessing source checkout: {root}"
        ) from exc
    if status.strip():
        raise ValueError(
            "Aramina refuses to create product lineage from a dirty "
            f"XRD-preprocessing checkout: {root}"
        )


def _package_version_from_source(root: Path) -> str:
    payload = _package_metadata_from_source(root)
    value = payload.get("version")
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Invalid XRD-preprocessing package version in {root / 'pyproject.toml'}."
        )
    return value


def _package_name_from_source(root: Path) -> str | None:
    try:
        value = _package_metadata_from_source(root).get("name")
    except ValueError:
        return None
    return value.lower() if isinstance(value, str) else None


def _package_metadata_from_source(root: Path) -> dict[str, Any]:
    pyproject = root / "pyproject.toml"
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = payload["project"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        if pyproject.exists():
            raise ValueError(
                f"Cannot read package metadata from {pyproject}."
            ) from exc
        return {}
    return project if isinstance(project, dict) else {}


def _installed_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unavailable"


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    child = value.get(key)
    if not isinstance(child, dict):
        raise ValueError(f"Aramina preprocessing requires mapping {key}.")
    return child


def _nonempty_string(value: dict[str, Any], key: str, where: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Aramina preprocessing requires non-empty {where}.{key}.")
    return item
