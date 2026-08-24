"""DVC identity checks for the internal Aramina training H5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config_paths import resolve_config_path
from .runtime_identity import file_hashes


DVC_DATA_CONTRACT = "aramina_dvc_input_v0_1"


def verify_dvc_input(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    input_h5_path: str | Path,
) -> dict[str, Any] | None:
    """Verify the configured H5 against its tracked DVC pointer."""
    declared = config.get("data_version")
    if declared is None:
        return None
    if not isinstance(declared, dict):
        raise ValueError("data_version must be a mapping.")

    pointer_value = _required_string(declared, "pointer_path")
    pointer_path = resolve_config_path(pointer_value, config_path)
    pointer = _load_pointer(pointer_path)
    output = pointer["outs"][0]
    algorithm = str(output.get("hash", "md5"))
    if algorithm != "md5" or not isinstance(output.get("md5"), str):
        raise ValueError("Aramina DVC input pointer must contain one MD5 output hash.")

    tracked_path = (pointer_path.parent / _required_string(output, "path")).resolve()
    actual_path = Path(input_h5_path).expanduser().resolve()
    if tracked_path != actual_path:
        raise ValueError(
            "Configured H5 does not match data_version.pointer_path output: "
            f"{actual_path} != {tracked_path}."
        )
    if not actual_path.is_file():
        raise FileNotFoundError(
            f"DVC-tracked H5 is not materialized: {actual_path}. Run `dvc pull`."
        )

    expected_size = output.get("size")
    if not isinstance(expected_size, int) or expected_size < 1:
        raise ValueError("Aramina DVC input pointer requires a positive output size.")
    actual_size = actual_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"DVC-tracked H5 size mismatch: expected {expected_size}, got {actual_size}."
        )

    hashes = file_hashes(actual_path, algorithms=("sha256", "md5"))
    expected_hash = str(output["md5"])
    if hashes["md5"] != expected_hash:
        raise ValueError(
            "DVC-tracked H5 hash mismatch: "
            f"expected md5:{expected_hash}, got md5:{hashes['md5']}."
        )
    return {
        "contract": _required_string(declared, "contract"),
        "system": _required_string(declared, "system"),
        "dataset_id": _required_string(declared, "dataset_id"),
        "dvc_version": _required_string(declared, "dvc_version"),
        "pointer_path": pointer_value,
        "output_path": str(output["path"]),
        "hash_algorithm": algorithm,
        "hash": expected_hash,
        "size_bytes": expected_size,
        "input_h5_sha256": hashes["sha256"],
    }


def _load_pointer(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"DVC pointer is missing: {path}.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"DVC pointer must contain a mapping: {path}.")
    outputs = payload.get("outs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("Aramina DVC input pointer must contain exactly one output.")
    if not isinstance(outputs[0], dict):
        raise ValueError("Aramina DVC input pointer output must be a mapping.")
    return payload


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty {key} in DVC data contract.")
    return value.strip()
