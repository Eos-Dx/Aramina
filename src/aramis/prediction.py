"""Prediction entrypoint for Aramis research-draft decision support."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import h5py
import joblib
import yaml
from xrd_preprocessing import (
    load_preprocessing_artifact,
    load_preprocessing_config,
    load_preprocessing_dataframe,
)

from .pipelines import run_preprocessing_pipeline
from .training import build_patient_prediction_feature_row


def run_prediction_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run one-patient prediction from a YAML config."""
    config_path = Path(config_path)
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    _validate_prediction_config(config, config_path)

    model_path = _config_path(
        config,
        config_path,
        section="io",
        key="input_model_joblib_path",
    )
    output_json_path = _config_path(
        config,
        config_path,
        section="io",
        key="output_json_path",
    )
    output_yaml_path = _config_path(
        config,
        config_path,
        section="io",
        key="output_yaml_path",
    )

    model_artifact = joblib.load(model_path)
    model_id = str(config["model"]["model_id"])
    _validate_model_id(model_artifact, model_id)
    model_name = str(config["model"]["selected_model"]).upper()
    model_info = _model_info(model_artifact, model_name)
    df, dataframe_path, preprocessing_artifact = _prediction_dataframe(
        config,
        config_path,
        model_artifact,
    )
    patient_id = str(config["patient"]["patient_id"])
    target_side = _prediction_target_side(config, patient_id)
    feature_row = build_patient_prediction_feature_row(
        df,
        model_info,
        patient_id=patient_id,
        target_side=target_side,
        **_prediction_columns(config, model_artifact),
    )
    p_cancer = _score_model(feature_row, model_name, model_info)
    threshold_key = str(config.get("decision", {}).get("threshold_key", "threshold_target"))
    threshold = _model_threshold(model_info, threshold_key)
    suggested_class = "CANCER" if p_cancer >= threshold else "BENIGN"
    report = _prediction_report(
        config=config,
        config_path=config_path,
        config_text=config_text,
        dataframe_path=dataframe_path,
        preprocessing_artifact=preprocessing_artifact,
        model_path=model_path,
        model_artifact=model_artifact,
        model_id=model_id,
        model_name=model_name,
        feature_row=feature_row,
        p_cancer=p_cancer,
        threshold_key=threshold_key,
        threshold=threshold,
        suggested_class=suggested_class,
    )
    _write_text(output_json_path, _json_dumps(report))
    _write_text(output_yaml_path, yaml.safe_dump(report, sort_keys=False))
    return report


def _prediction_dataframe(
    config: dict[str, Any],
    config_path: Path,
    model_artifact: dict[str, Any],
) -> tuple[Any, Path, dict[str, Any] | None]:
    if "preprocessing" not in config and not config.get("io", {}).get("input_h5_path"):
        dataframe_path = _config_path(
            config,
            config_path,
            section="io",
            key="input_dataframe_joblib_path",
        )
        return load_preprocessing_dataframe(dataframe_path), dataframe_path, None

    preprocessing_config = _prediction_preprocessing_config(
        config,
        config_path,
        model_artifact,
    )
    h5_path = _config_path(
        config,
        config_path,
        section="io",
        key="input_h5_path",
    )
    _validate_h5_container_contract(config, config_path, h5_path)
    output_dataframe_path = _config_path(
        config,
        config_path,
        section="io",
        key="output_dataframe_joblib_path",
    )
    preprocessing_config.setdefault("io", {})
    preprocessing_config["io"]["input_h5_path"] = str(h5_path)
    preprocessing_config["io"]["output_joblib_path"] = str(output_dataframe_path)
    branch = preprocessing_config.get("aramis_preprocessing", {}).get("branch")
    if branch != "one_to_many":
        raise ValueError(f"Unsupported prediction preprocessing branch: {branch!r}")
    df = run_preprocessing_pipeline(
        h5_path,
        preprocessing_config,
        output_joblib_path=output_dataframe_path,
    )
    return df, output_dataframe_path, load_preprocessing_artifact(output_dataframe_path)


def _prediction_preprocessing_config(
    config: dict[str, Any],
    config_path: Path,
    model_artifact: dict[str, Any],
) -> dict[str, Any]:
    if "preprocessing" in config:
        preprocessing_config_path = _config_path(
            config,
            config_path,
            section="preprocessing",
            key="config_path",
        )
        return load_preprocessing_config(preprocessing_config_path)
    model_config = model_artifact.get("prediction_preprocessing_config")
    if not isinstance(model_config, dict):
        raise ValueError(
            "Model artifact has no prediction_preprocessing_config. Retrain model "
            "with io.prediction_preprocessing_config_path or pass an explicit "
            "preprocessing.config_path override in predict YAML."
        )
    return dict(model_config)


def _prediction_target_side(
    config: dict[str, Any],
    patient_id: str,
) -> str:
    target_side = config.get("patient", {}).get("target_side")
    if target_side not in {None, ""}:
        return str(target_side)
    raise ValueError(
        "Prediction target side is missing: set patient.target_side in predict YAML. "
        f"Target side is not inferred from H5 metadata for patient {patient_id!r}."
    )


def _score_model(feature_row, model_name: str, model_info: dict[str, Any]) -> float:
    columns = list(model_info["feature_columns"])
    missing = [column for column in columns if column not in feature_row.columns]
    if missing:
        raise ValueError(f"Prediction feature row is missing columns: {missing}")
    if model_name == "M0":
        return float(feature_row["profile_p_cancer_logit_average"].iloc[0])
    final_model = model_info.get("final_model")
    if final_model is None:
        raise ValueError(f"Model {model_name} is missing final_model.")
    return float(final_model.predict_proba(feature_row[columns])[:, 1][0])


def _prediction_report(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_text: str,
    dataframe_path: Path,
    preprocessing_artifact: dict[str, Any] | None,
    model_path: Path,
    model_artifact: dict[str, Any],
    model_id: str,
    model_name: str,
    feature_row,
    p_cancer: float,
    threshold_key: str,
    threshold: float,
    suggested_class: str,
) -> dict[str, Any]:
    row = feature_row.iloc[0].to_dict()
    return _json_safe(
        {
            "kind": "aramis_prediction_report",
            "version": "0.1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "clinical_stage": config["prediction"].get(
                "clinical_stage",
                "research draft",
            ),
            "intended_use": config["prediction"].get(
                "intended_use",
                "decision-support p_cancer research draft; requires radiologist review",
            ),
            "decision_support_only": True,
            "requires_radiologist_review": True,
            "patient_id": str(config["patient"]["patient_id"]),
            "target_side": row["target_side"],
            "contralateral_side": row["contralateral_side"],
            "model_id": model_id,
            "model_name": model_name,
            "p_cancer": p_cancer,
            "threshold_key": threshold_key,
            "threshold": threshold,
            "suggested_class": suggested_class,
            "risk_level": "high" if suggested_class == "CANCER" else "low",
            "reliability": row["result_reliability"],
            "reliability_reason": row["result_reliability_reason"],
            "feature_row": row,
            "provenance": {
                "prediction_config_path": str(config_path),
                "prediction_config_sha256": sha256(
                    config_text.encode("utf-8")
                ).hexdigest(),
                "input_dataframe_joblib_path": str(dataframe_path),
                "input_dataframe_joblib_sha256": _file_sha256(dataframe_path),
                "input_h5_path": _optional_config_path(config, config_path, "input_h5_path"),
                "input_h5_sha256": _optional_file_sha256(
                    _optional_config_path(config, config_path, "input_h5_path")
                ),
                "prediction_preprocessing_config_path": _optional_section_config_path(
                    config,
                    config_path,
                    section="preprocessing",
                    key="config_path",
                ),
                "model_prediction_preprocessing_config_path": model_artifact.get(
                    "prediction_preprocessing_config_path"
                ),
                "model_prediction_preprocessing_config_sha256": model_artifact.get(
                    "prediction_preprocessing_config_sha256"
                ),
                "prediction_preprocessing_config_sha256": (
                    preprocessing_artifact or {}
                ).get("preprocessing_config_sha256"),
                "input_model_joblib_path": str(model_path),
                "input_model_joblib_sha256": _file_sha256(model_path),
                "input_model_id": model_id,
                "training_config_sha256": model_artifact.get(
                    "training_config_sha256"
                ),
                "preprocessing_config_sha256": model_artifact.get(
                    "preprocessing_config_sha256"
                ),
                "model_metadata": model_artifact.get("metadata", {}),
            },
            "limitations": [
                "research draft",
                "not for autonomous diagnosis",
                "requires breast-imaging clinician review",
            ],
        }
    )


def _prediction_columns(
    config: dict[str, Any],
    model_artifact: dict[str, Any],
) -> dict[str, str]:
    model_config = dict(model_artifact.get("training_config", {}).get("model", {}))
    model_config.update(config.get("model", {}))
    return {
        "profile_column": str(model_config.get("profile_column", "radial_profile_data")),
        "group_column": str(model_config.get("group_column", "patientId")),
        "specimen_column": str(model_config.get("specimen_column", "specimenId")),
        "side_column": str(model_config.get("side_column", "side")),
        "q_column": str(model_config.get("q_column", "q_range")),
        "age_column": str(model_config.get("age_column", "age")),
    }


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


def _validate_model_id(model_artifact: dict[str, Any], requested_model_id: str) -> None:
    artifact_model_id = (
        model_artifact.get("training_config", {})
        .get("training", {})
        .get("name")
    )
    if not artifact_model_id:
        raise ValueError("Prediction model artifact has no training.name model_id.")
    if str(artifact_model_id) != str(requested_model_id):
        raise ValueError(
            "Prediction model_id does not match artifact training.name: "
            f"requested={requested_model_id!r}, artifact={artifact_model_id!r}"
        )


def _model_threshold(model_info: dict[str, Any], threshold_key: str) -> float:
    thresholds = model_info.get("thresholds", {})
    if threshold_key not in thresholds:
        raise ValueError(
            f"Threshold {threshold_key!r} not found; available: {sorted(thresholds)}"
        )
    return float(thresholds[threshold_key])


def _validate_prediction_config(config: dict[str, Any], config_path: Path) -> None:
    if not isinstance(config, dict):
        raise TypeError(f"Prediction config must be a mapping: {config_path}")
    missing = [
        section
        for section in ("prediction", "io", "patient", "model")
        if section not in config
    ]
    if missing:
        raise ValueError(f"Missing prediction config sections: {missing}")
    for key in (
        "input_model_joblib_path",
        "output_json_path",
        "output_yaml_path",
    ):
        if not config.get("io", {}).get(key):
            raise ValueError(f"Missing io.{key} in {config_path}")
    if "preprocessing" in config:
        if not config.get("preprocessing", {}).get("config_path"):
            raise ValueError(f"Missing preprocessing.config_path in {config_path}")
        for key in ("input_h5_path", "output_dataframe_joblib_path"):
            if not config.get("io", {}).get(key):
                raise ValueError(f"Missing io.{key} in {config_path}")
    elif config.get("io", {}).get("input_h5_path"):
        if not config.get("io", {}).get("output_dataframe_joblib_path"):
            raise ValueError(f"Missing io.output_dataframe_joblib_path in {config_path}")
    elif not config.get("io", {}).get("input_dataframe_joblib_path"):
        raise ValueError(f"Missing io.input_dataframe_joblib_path in {config_path}")
    if not config.get("patient", {}).get("patient_id"):
        raise ValueError(f"Missing patient.patient_id in {config_path}")
    if not config.get("patient", {}).get("target_side"):
        raise ValueError(f"Missing patient.target_side in {config_path}")
    if not config.get("model", {}).get("model_id"):
        raise ValueError(f"Missing model.model_id in {config_path}")
    if not config.get("model", {}).get("selected_model"):
        raise ValueError(f"Missing model.selected_model in {config_path}")
    if config.get("io", {}).get("input_h5_path") and "container" not in config:
        raise ValueError(
            "Missing container section in prediction config with io.input_h5_path: "
            f"{config_path}"
        )


def _validate_h5_container_contract(
    config: dict[str, Any],
    config_path: Path,
    h5_path: Path,
) -> None:
    container = config.get("container")
    if not isinstance(container, dict):
        raise ValueError(f"Missing container section in prediction config: {config_path}")

    expected_schema_version = str(_required_container_value(container, "schema_version"))
    expected_format = str(_required_container_value(container, "format"))
    max_patients = int(container.get("max_patients", 1))
    if max_patients != 1:
        raise ValueError("Aramis prediction currently requires container.max_patients=1.")

    with h5py.File(h5_path, "r") as h5:
        actual_schema_version = _h5_attr_text(h5, "schema_version")
        actual_format = _h5_attr_text(h5, "format")
        if actual_schema_version != expected_schema_version:
            raise ValueError(
                "Prediction H5 schema_version does not match YAML container contract: "
                f"expected={expected_schema_version!r}, actual={actual_schema_version!r}"
            )
        if actual_format != expected_format:
            raise ValueError(
                "Prediction H5 format does not match YAML container contract: "
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
    if len(patient_ids) > max_patients:
        raise ValueError(
            "Aramis prediction requires exactly one patient per H5 container; "
            f"found {len(patient_ids)} patients: {patient_ids}"
        )
    expected_patient_id = str(config["patient"]["patient_id"])
    only_patient_id = patient_ids[0]
    if only_patient_id != expected_patient_id:
        raise ValueError(
            "Prediction patient.patient_id does not match H5 patientId: "
            f"expected={expected_patient_id!r}, h5={only_patient_id!r}"
        )


def _required_container_value(container: dict[str, Any], key: str) -> Any:
    value = container.get(key)
    if value in {None, ""}:
        raise ValueError(f"Missing container.{key} in prediction config.")
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
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _optional_config_path(config: dict[str, Any], config_path: Path, key: str) -> str | None:
    value = config.get("io", {}).get(key)
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return str(path)


def _optional_section_config_path(
    config: dict[str, Any],
    config_path: Path,
    *,
    section: str,
    key: str,
) -> str | None:
    value = config.get(section, {}).get(key)
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return str(path)


def _optional_file_sha256(path: str | None) -> str | None:
    return None if path is None else _file_sha256(path)


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
    if hasattr(value, "item"):
        return value.item()
    return value
