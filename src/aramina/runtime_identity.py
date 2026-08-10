"""Shared file and runtime identity helpers."""

from __future__ import annotations

import subprocess
import tomllib
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    repo_root = _repo_root()
    if not (repo_root / ".git").exists():
        return "unavailable"
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
