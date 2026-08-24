"""Shared file and runtime identity helpers."""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import tomllib
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 digest of one file."""
    return file_hashes(path, algorithms=("sha256",))["sha256"]


def file_hashes(
    path: str | Path,
    *,
    algorithms: tuple[str, ...],
) -> dict[str, str]:
    """Return requested file digests after one sequential read."""
    if not algorithms:
        raise ValueError("At least one hash algorithm is required.")
    digests = {
        algorithm: hashlib.new(algorithm, usedforsecurity=False)
        for algorithm in algorithms
    }
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            for digest in digests.values():
                digest.update(chunk)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def safe_stem(value: str, *, fallback: str = "") -> str:
    """Convert a user- or model-supplied value into a portable file stem."""
    stem = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    ).strip("_")
    return stem or fallback


def aramina_version() -> str:
    """Return source-tree or installed-package version."""
    pyproject_path = _repo_root() / "pyproject.toml"
    if pyproject_path.exists():
        with pyproject_path.open("rb") as handle:
            pyproject = tomllib.load(handle)
        return str(pyproject.get("project", {}).get("version", "unknown"))
    try:
        return version("aramina")
    except PackageNotFoundError:
        return "unknown"


def aramina_git_sha() -> str:
    """Return source-tree commit or ``unavailable`` outside a Git checkout."""
    embedded = os.environ.get("ARAMINA_GIT_SHA")
    if embedded:
        return _validated_git_sha(embedded, "ARAMINA_GIT_SHA")
    repo_root = _repo_root()
    if not (repo_root / ".git").exists():
        return "unavailable"
    return _validated_git_sha(
        subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "Aramina checkout",
    )


def xrd_preprocessing_git_sha() -> str:
    """Return the installed XRD-preprocessing source commit."""
    embedded = os.environ.get("XRD_PREPROCESSING_GIT_SHA")
    if embedded:
        return _validated_git_sha(embedded, "XRD_PREPROCESSING_GIT_SHA")
    try:
        payload = distribution("xrd-preprocessing").read_text("direct_url.json")
    except PackageNotFoundError:
        return "unavailable"
    if payload is None:
        return "unavailable"
    direct_url = json.loads(payload)
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict) or not vcs_info.get("commit_id"):
        return "unavailable"
    return _validated_git_sha(
        str(vcs_info["commit_id"]),
        "xrd-preprocessing direct_url.json",
    )


def _validated_git_sha(value: str, where: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", normalized):
        raise ValueError(f"{where} must provide a full 40-character Git SHA.")
    return normalized


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
