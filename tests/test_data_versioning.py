from __future__ import annotations

from hashlib import md5
from pathlib import Path

import pytest
import yaml

from aramina.data_versioning import DVC_DATA_CONTRACT, verify_dvc_input


def test_verify_dvc_input_returns_portable_identity(tmp_path: Path):
    input_path = tmp_path / "input.h5"
    input_path.write_bytes(b"versioned-h5")
    pointer_path = tmp_path / "input.h5.dvc"
    _write_pointer(pointer_path, input_path)
    config = _config(pointer_path.name)

    identity = verify_dvc_input(
        config,
        config_path=tmp_path / "preprocessing.yaml",
        input_h5_path=input_path,
    )

    assert identity == {
        "contract": DVC_DATA_CONTRACT,
        "system": "dvc",
        "dataset_id": "synthetic_h5",
        "dvc_version": "3.67.1",
        "pointer_path": "input.h5.dvc",
        "output_path": "input.h5",
        "hash_algorithm": "md5",
        "hash": md5(b"versioned-h5", usedforsecurity=False).hexdigest(),
        "size_bytes": len(b"versioned-h5"),
        "input_h5_sha256": (
            "1b26b9b2a5a0faf709c844025a61e9b75ddb0298e98e14a8b554c0b163025cb0"
        ),
    }


def test_verify_dvc_input_rejects_content_change(tmp_path: Path):
    input_path = tmp_path / "input.h5"
    input_path.write_bytes(b"tracked")
    pointer_path = tmp_path / "input.h5.dvc"
    _write_pointer(pointer_path, input_path)
    input_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_dvc_input(
            _config(pointer_path.name),
            config_path=tmp_path / "preprocessing.yaml",
            input_h5_path=input_path,
        )


def test_verify_dvc_input_rejects_different_materialized_path(tmp_path: Path):
    input_path = tmp_path / "input.h5"
    other_path = tmp_path / "other.h5"
    input_path.write_bytes(b"tracked")
    other_path.write_bytes(b"tracked")
    pointer_path = tmp_path / "input.h5.dvc"
    _write_pointer(pointer_path, input_path)

    with pytest.raises(ValueError, match="does not match"):
        verify_dvc_input(
            _config(pointer_path.name),
            config_path=tmp_path / "preprocessing.yaml",
            input_h5_path=other_path,
        )


def test_verify_dvc_input_requires_one_output(tmp_path: Path):
    pointer_path = tmp_path / "input.h5.dvc"
    pointer_path.write_text("outs: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one output"):
        verify_dvc_input(
            _config(pointer_path.name),
            config_path=tmp_path / "preprocessing.yaml",
            input_h5_path=tmp_path / "input.h5",
        )


def _config(pointer_path: str) -> dict[str, object]:
    return {
        "data_version": {
            "contract": DVC_DATA_CONTRACT,
            "system": "dvc",
            "dataset_id": "synthetic_h5",
            "dvc_version": "3.67.1",
            "pointer_path": pointer_path,
        }
    }


def _write_pointer(pointer_path: Path, input_path: Path) -> None:
    content = input_path.read_bytes()
    pointer_path.write_text(
        yaml.safe_dump(
            {
                "outs": [
                    {
                        "md5": md5(content, usedforsecurity=False).hexdigest(),
                        "size": len(content),
                        "hash": "md5",
                        "path": input_path.name,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
