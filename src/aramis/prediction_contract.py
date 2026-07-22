"""Prediction YAML, H5, model-identity, and output-path contracts."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import h5py
import yaml

from .config_paths import config_reference_root, resolve_config_path

def _model_info(model_artifact: dict[str, Any], model_name: str) -> dict[str, Any]:
    if model_artifact.get("kind") != "aramis_training_artifact":
        raise ValueError("Prediction model joblib is not an Aramis training artifact.")
    models = model_artifact.get("models")
    if not isinstance(models, dict):
        raise ValueError("Prediction model artifact has no models dictionary.")
    if model_name not in models:
        raise ValueError(
            f"Model {model_name!r} not found in artifact; available: {sorted(models)}"
        )
    return models[model_name]


def _model_identity(
    model_artifact: dict[str, Any], model_path: Path
) -> tuple[str, str, str]:
    """Read the immutable one-model identity from a prediction artifact."""
    identity = model_artifact.get("model_identity", {})
    artifact_model_name = identity.get("name")
    artifact_model_version = identity.get("version")
    if not artifact_model_name or artifact_model_version in {None, ""}:
        raise ValueError("Prediction model artifact has no model_identity.")
    model_names = sorted(model_artifact.get("models", {}))
    if len(model_names) != 1:
        raise ValueError(
            "Prediction model artifact must contain exactly one selected model; "
            f"available={model_names}"
        )
    artifact_sha = _file_sha256(model_path)
    model_id = _safe_stem(
        f"{artifact_model_name}_{artifact_model_version}_{artifact_sha[:12]}"
    )
    return model_id, model_names[0], str(artifact_model_version)


def _model_threshold(
    model_info: dict[str, Any],
    threshold_key: str,
    model_route: str | None,
) -> float:
    thresholds = (
        model_info.get("routes", {}).get(model_route, model_info).get("thresholds", {})
    )
    if threshold_key not in thresholds:
        raise ValueError(
            f"Threshold {threshold_key!r} not found; available: {sorted(thresholds)}"
        )
    return float(thresholds[threshold_key])


def _prediction_output_paths(
    config: dict[str, Any],
    config_path: Path,
    *,
    model_id: str,
) -> dict[str, Path]:
    folder = _config_path(config, config_path, section="io", key="output_folder")
    report_id = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y%m%dT%H%M%S")
    report_id = f"{report_id}_{uuid4().hex[:8]}"
    stem = _safe_stem(f"{config['patient']['patient_id']}_{model_id}_{report_id}")
    return {
        "folder": folder,
        "report_id": report_id,
        "dataframe_joblib": folder / f"{stem}_prediction_dataframe.joblib",
        "external_json": folder / f"{stem}_external_report.json",
        "external_yaml": folder / f"{stem}_external_report.yaml",
        "internal_json": folder / f"{stem}_internal_report.json",
        "internal_yaml": folder / f"{stem}_internal_report.yaml",
    }


def _safe_stem(value: str) -> str:
    out = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "aramis_prediction"


def _validate_prediction_config(config: dict[str, Any], config_path: Path) -> None:
    if not isinstance(config, dict):
        raise TypeError(f"Prediction config must be a mapping: {config_path}")
    missing = [section for section in ("run", "io", "patient") if section not in config]
    if missing:
        raise ValueError(f"Missing prediction config sections: {missing}")
    for key in ("input_model_joblib_path", "output_folder"):
        if not config.get("io", {}).get(key):
            raise ValueError(f"Missing io.{key} in {config_path}")
    forbidden = sorted(set(config).difference({"run", "io", "patient"}))
    if forbidden:
        raise ValueError(
            f"Predict YAML cannot override model-held sections: {forbidden}"
        )
    _reject_unknown_fields(
        config["run"],
        allowed={"analysis_author", "prediction_comment", "synthetic_test_mode"},
        where="run",
    )
    _reject_unknown_fields(
        config["io"],
        allowed={
            "input_h5_path",
            "input_dataframe_joblib_path",
            "input_model_joblib_path",
            "output_folder",
        },
        where="io",
    )
    _reject_unknown_fields(
        config["patient"],
        allowed={"patient_id", "target_side"},
        where="patient",
    )
    has_h5 = bool(config.get("io", {}).get("input_h5_path"))
    has_dataframe = bool(config.get("io", {}).get("input_dataframe_joblib_path"))
    if has_h5 == has_dataframe:
        raise ValueError(
            "Set exactly one input: io.input_h5_path or io.input_dataframe_joblib_path."
        )
    if has_dataframe and not config.get("run", {}).get("synthetic_test_mode"):
        raise ValueError(
            "io.input_dataframe_joblib_path is allowed only with "
            "run.synthetic_test_mode: true."
        )
    if "synthetic_test_mode" in config["run"] and not isinstance(
        config["run"]["synthetic_test_mode"], bool
    ):
        raise TypeError(f"run.synthetic_test_mode must be boolean in {config_path}")
    _require_nonempty_string(config["run"], "analysis_author", config_path, "run")
    if not isinstance(config.get("run", {}).get("prediction_comment", ""), str):
        raise TypeError(f"run.prediction_comment must be a string in {config_path}")
    for key in ("input_model_joblib_path", "output_folder"):
        _require_nonempty_string(config["io"], key, config_path, "io")
    if has_h5:
        _require_nonempty_string(config["io"], "input_h5_path", config_path, "io")
    if has_dataframe:
        _require_nonempty_string(
            config["io"], "input_dataframe_joblib_path", config_path, "io"
        )
    _require_nonempty_string(config["patient"], "patient_id", config_path, "patient")
    target_side = _require_nonempty_string(
        config["patient"], "target_side", config_path, "patient"
    )
    if target_side.casefold() not in {"left", "right"}:
        raise ValueError(f"patient.target_side must be left or right in {config_path}")


def _validate_h5_container_contract(
    model_artifact: dict[str, Any],
    h5_path: Path,
    *,
    expected_patient_id: str,
) -> None:
    container = _prediction_contract(model_artifact)["container"]

    expected_schema_version = str(
        _required_container_value(container, "schema_version")
    )
    expected_format = str(_required_container_value(container, "format"))

    with h5py.File(h5_path, "r") as h5:
        actual_schema_version = _h5_attr_text(h5, "schema_version")
        actual_format = _h5_attr_text(h5, "format")
        if actual_schema_version != expected_schema_version:
            raise ValueError(
                "Prediction H5 schema_version does not match model-held contract: "
                f"expected={expected_schema_version!r}, actual={actual_schema_version!r}"
            )
        if actual_format != expected_format:
            raise ValueError(
                "Prediction H5 format does not match model-held contract: "
                f"expected={expected_format!r}, actual={actual_format!r}"
            )
        if actual_schema_version != "0.3":
            raise ValueError(
                "Unsupported prediction H5 schema_version validator: "
                f"{actual_schema_version!r}. Add a version-specific validator first."
            )
        patient_ids = _v0_3_patient_ids(h5)

    if not patient_ids:
        raise ValueError("Prediction H5 contains no patientId values.")
    if len(patient_ids) != 1:
        raise ValueError(
            "Aramis prediction requires exactly one patient per H5 container; "
            f"found {len(patient_ids)} patients: {patient_ids}"
        )
    only_patient_id = patient_ids[0]
    if only_patient_id != expected_patient_id:
        raise ValueError(
            "Prediction patient.patient_id does not match H5 patientId: "
            f"expected={expected_patient_id!r}, h5={only_patient_id!r}"
        )


def _prediction_contract(model_artifact: dict[str, Any]) -> dict[str, Any]:
    contract_yaml = model_artifact.get("prediction_contract_yaml")
    contract = yaml.safe_load(contract_yaml) if isinstance(contract_yaml, str) else None
    if not isinstance(contract, dict):
        raise ValueError(
            "Model artifact has no prediction_contract_yaml. Retrain model."
        )
    for key in ("container", "reporting", "decision"):
        if not isinstance(contract.get(key), dict):
            raise ValueError(f"Model prediction_contract has no {key} section.")
    return contract


def _reject_unknown_fields(
    value: Any,
    *,
    allowed: set[str],
    where: str,
) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"Prediction {where} must be a mapping.")
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown prediction {where} fields: {unknown}")


def _require_nonempty_string(
    section: dict[str, Any],
    key: str,
    config_path: Path,
    where: str,
) -> str:
    value = section.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing {where}.{key} in {config_path}")
    if not isinstance(value, str):
        raise TypeError(f"{where}.{key} must be a string in {config_path}")
    if not value.strip():
        raise ValueError(f"Missing {where}.{key} in {config_path}")
    return value


def _required_container_value(container: dict[str, Any], key: str) -> Any:
    value = container.get(key)
    if value in {None, ""}:
        raise ValueError(
            f"Missing prediction_contract.container.{key} in model artifact."
        )
    return value


def _h5_attr_text(group: h5py.Group | h5py.File, key: str) -> str | None:
    value = group.attrs.get(key)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None:
        return None
    return str(value)


def _h5_text_dataset(group: h5py.Group, path: str) -> str | None:
    if path not in group:
        return None
    value = group[path][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _v0_3_patient_ids(h5: h5py.File) -> list[str]:
    session = h5.get("session")
    if not isinstance(session, h5py.Group):
        raise ValueError("Prediction H5 v0.3 is missing /session group.")
    fallback_patient_id = _h5_text_dataset(session, "sample/patient_name")
    sets = session.get("sets")
    if not isinstance(sets, h5py.Group):
        raise ValueError("Prediction H5 v0.3 is missing /session/sets group.")

    patient_ids: set[str] = set()
    for set_name in sorted(sets):
        set_group = sets[set_name]
        if not isinstance(set_group, h5py.Group):
            continue
        patient_id = _h5_attr_text(set_group, "patientId") or fallback_patient_id
        if patient_id not in {None, ""}:
            patient_ids.add(str(patient_id))
    return sorted(patient_ids)


def _config_path(
    config: dict[str, Any],
    config_path: Path,
    *,
    section: str,
    key: str,
) -> Path:
    value = config.get(section, {}).get(key)
    if value in {None, ""}:
        raise ValueError(f"Missing {section}.{key} in {config_path}")
    return resolve_config_path(value, config_path)


def _config_root(config_path: Path) -> Path:
    """Return the project root for a config or a runnable example YAML."""
    return config_reference_root(config_path)


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=False) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return round(value, 5) if isfinite(value) else None
    return value
