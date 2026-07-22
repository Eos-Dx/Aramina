"""Shared resolution rules for paths declared in Aramis YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def config_reference_root(config_path: str | Path) -> Path:
    """Return the documented root for relative paths declared by one config."""
    source = Path(config_path).expanduser().resolve()
    for parent in source.parents:
        if parent.name == "config" and (parent.parent / "pyproject.toml").is_file():
            return parent.parent
    for parent in source.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return source.parent


def resolve_config_path(value: Any, config_path: str | Path) -> Path:
    """Resolve a declared config path under the shared root policy."""
    path = Path(str(value)).expanduser()
    return (
        path
        if path.is_absolute()
        else (config_reference_root(config_path) / path).resolve()
    )
