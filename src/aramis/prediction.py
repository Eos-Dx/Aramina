"""Prediction entrypoint for Aramis research-draft decision support."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml

from .model_utils import profile_matrix
from .pipelines import run_preprocessing_pipeline
from .training import build_patient_prediction_feature_row
from .training_config import PRODUCT_MODEL_NAME


def run_prediction_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run one-patient prediction from a YAML config."""
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_prediction_config(config, config_path)

    model_path = _config_path(
        config,
        config_path,
        section="io",
        key="input_model_joblib_path",
    )
    model_artifact = joblib.load(model_path)
    model_id, model_name, model_version = _model_identity(model_artifact, model_path)
    output_paths = _prediction_output_paths(config, config_path, model_id=model_id)
    model_info = _model_info(model_artifact, model_name)
    df = _prediction_dataframe(
        config,
        config_path,
        model_artifact,
        output_paths["dataframe_joblib"],
    )
    patient_id = str(config["patient"]["patient_id"])
    target_side = _prediction_target_side(config, patient_id)
    threshold_key = _prediction_contract(model_artifact)["decision"]["threshold_key"]
    target_prediction = _side_prediction(
        df,
        model_info,
        patient_id=patient_id,
        target_side=target_side,
        columns=_prediction_columns(model_artifact),
        model_name=model_name,
        threshold_key=threshold_key,
    )
    target_prediction["is_target"] = True
    contralateral_side = target_prediction["feature_row"].get("contralateral_side")
    contralateral_prediction = (
        _side_prediction(
            df,
            model_info,
            patient_id=patient_id,
            target_side=contralateral_side,
            columns=_prediction_columns(model_artifact),
            model_name=model_name,
            threshold_key=threshold_key,
        )
        if _normalize_side(contralateral_side) is not None
        else _unavailable_side_prediction()
    )
    contralateral_prediction["is_target"] = False
    reports = _prediction_reports(
        config=config,
        output_paths=output_paths,
        model_path=model_path,
        model_artifact=model_artifact,
        model_id=model_id,
        model_name=model_name,
        model_version=model_version,
        target_prediction=target_prediction,
        contralateral_prediction=contralateral_prediction,
    )
    _write_text(output_paths["external_json"], _json_dumps(reports["external_report"]))
    _write_text(
        output_paths["external_yaml"],
        yaml.safe_dump(reports["external_report"], sort_keys=False),
    )
    _write_text(output_paths["internal_json"], _json_dumps(reports["internal_report"]))
    _write_text(
        output_paths["internal_yaml"],
        yaml.safe_dump(reports["internal_report"], sort_keys=False),
    )
    return reports


def _prediction_dataframe(
    config: dict[str, Any],
    config_path: Path,
    model_artifact: dict[str, Any],
    output_dataframe_path: Path,
) -> pd.DataFrame:
    dataframe_value = config.get("io", {}).get("input_dataframe_joblib_path")
    if dataframe_value not in {None, ""}:
        dataframe_path = _config_path(
            config,
            config_path,
            section="io",
            key="input_dataframe_joblib_path",
        )
        from xrd_preprocessing import load_preprocessing_dataframe

        return load_preprocessing_dataframe(dataframe_path)

    preprocessing_config = _prediction_preprocessing_config(model_artifact)
    h5_path = _config_path(
        config,
        config_path,
        section="io",
        key="input_h5_path",
    )
    _validate_h5_container_contract(
        model_artifact,
        h5_path,
        expected_patient_id=str(config["patient"]["patient_id"]),
    )
    preprocessing_config.setdefault("io", {})
    preprocessing_config["io"]["input_h5_path"] = str(h5_path)
    preprocessing_config["io"]["output_joblib_path"] = str(output_dataframe_path)
    df = run_preprocessing_pipeline(
        h5_path,
        preprocessing_config,
        output_joblib_path=output_dataframe_path,
    )
    return df


def _prediction_preprocessing_config(model_artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the preprocessing contract embedded at model training time."""
    config_yaml = model_artifact.get("prediction_preprocessing_yaml")
    if not isinstance(config_yaml, str):
        raise ValueError(
            "Model artifact has no prediction_preprocessing_yaml. Retrain model."
        )
    model_config = yaml.safe_load(config_yaml)
    if not isinstance(model_config, dict):
        raise ValueError("Model prediction_preprocessing_yaml is not a YAML mapping.")
    return deepcopy(model_config)


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


def _prediction_model_route(feature_row: pd.DataFrame, model_info: dict[str, Any]) -> str | None:
    if model_info.get("symmetry_policy") == "single_model_gated_optional_refinement":
        return None
    if "routes" not in model_info:
        return None
    return "paired" if int(feature_row["symmetry_available"].iloc[0]) == 1 else "fallback_no_symmetry"


def _score_model(
    feature_row,
    model_name: str,
    model_info: dict[str, Any],
    model_route: str | None,
) -> float:
    route_info = model_info.get("routes", {}).get(model_route, model_info)
    columns = list(route_info["feature_columns"])
    missing = [column for column in columns if column not in feature_row.columns]
    if missing:
        raise ValueError(f"Prediction feature row is missing columns: {missing}")
    if model_name != PRODUCT_MODEL_NAME:
        raise ValueError(f"Unsupported product model: {model_name!r}")
    final_model = route_info.get("final_model")
    if final_model is None:
        raise ValueError(f"Model {model_name} is missing final_model.")
    return float(final_model.predict_proba(feature_row[columns])[:, 1][0])


def _side_prediction(
    df: pd.DataFrame,
    model_info: dict[str, Any],
    *,
    patient_id: str,
    target_side: Any,
    columns: dict[str, str],
    model_name: str,
    threshold_key: str,
) -> dict[str, Any]:
    """Score one requested side with the fixed final product artifact."""
    feature_table = build_patient_prediction_feature_row(
        df,
        model_info,
        patient_id=patient_id,
        target_side=str(target_side),
        **columns,
    )
    feature_row = feature_table.iloc[0].to_dict()
    model_route = _prediction_model_route(feature_table, model_info)
    p_cancer = _score_model(feature_table, model_name, model_info, model_route)
    threshold = _model_threshold(model_info, threshold_key, model_route)
    profile = _side_profile_score(
        df,
        model_info,
        patient_id=patient_id,
        side=target_side,
        columns=columns,
    )
    return {
        "available": True,
        "reason": None,
        "feature_row": feature_row,
        "model_route": model_route,
        "p_cancer": p_cancer,
        "xrd_profile": profile,
        "threshold_key": threshold_key,
        "threshold": threshold,
        "suggested_class": "CANCER" if p_cancer >= threshold else "BENIGN",
        "quantiles": _prediction_quantiles(model_info, p_cancer),
    }


def _unavailable_side_prediction() -> dict[str, Any]:
    return {
        "available": False,
        "reason": "contralateral breast is unavailable after preprocessing",
    }


def _prediction_quantiles(model_info: dict[str, Any], p_cancer: float) -> dict[str, float]:
    """Locate a score in frozen train-on-all empirical score distributions."""
    reference = model_info.get("prediction_reference_scores")
    if not isinstance(reference, dict):
        raise ValueError(
            "Model artifact has no prediction_reference_scores. Retrain model."
        )
    keys = {
        "all_training_patients": "all_target_cases",
        "benign_training_patients": "benign_target_cases",
        "cancer_training_patients": "cancer_target_cases",
    }
    out: dict[str, float] = {}
    for report_key, reference_key in keys.items():
        values = np.asarray(reference.get(reference_key, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                f"Model prediction reference has no values for {reference_key!r}."
            )
        out[report_key] = float(np.searchsorted(np.sort(values), p_cancer, side="right") / values.size)
    return out


def _prediction_reports(
    *,
    config: dict[str, Any],
    output_paths: dict[str, Path],
    model_path: Path,
    model_artifact: dict[str, Any],
    model_id: str,
    model_name: str,
    model_version: str,
    target_prediction: dict[str, Any],
    contralateral_prediction: dict[str, Any],
) -> dict[str, Any]:
    row = target_prediction["feature_row"]
    created_at = _report_timestamp()
    reporting = _prediction_contract(model_artifact)["reporting"]
    common = {
        "created_at": created_at,
        "analysis_author": config["run"]["analysis_author"],
        "prediction_comment": config["run"].get("prediction_comment", ""),
        "patient_id": str(config["patient"]["patient_id"]),
        "target_side": row["target_side"],
        "model_id": model_id,
        "model_name": model_name,
        "model_version": model_version,
        "report_id": output_paths["report_id"],
    }
    external = _external_report(
        common=common,
        version=str(reporting["external_report"]["version"]),
        suggested_class=target_prediction["suggested_class"],
        reliability=row["result_reliability"],
        reliability_reason=row["result_reliability_reason"],
        model_performance=_external_model_performance(model_artifact),
        scan_metadata=_scan_metadata(
            row,
            patient_id=common["patient_id"],
            target_side=common["target_side"],
            age_available=bool(row.get("age_available", False)),
        ),
    )
    internal = _internal_report(
        common=common,
        version=str(reporting["internal_report"]["version"]),
        reference_doc=reporting["internal_report"].get("reference_doc"),
        target_prediction=target_prediction,
        contralateral_prediction=contralateral_prediction,
        model_artifact_sha256=_file_sha256(model_path),
    )
    return _json_safe({"external_report": external, "internal_report": internal})


def _external_report(
    *,
    common: dict[str, Any],
    version: str,
    suggested_class: str,
    reliability: str,
    reliability_reason: str,
    model_performance: dict[str, Any],
    scan_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "output_type": "aramis_external_report",
        "report_version": version,
        "report_id": common["report_id"],
        "created_at": common["created_at"],
        "analysis_author": common["analysis_author"],
        "prediction_comment": common["prediction_comment"],
        "patient_id": common["patient_id"],
        "patient_age": scan_metadata["patient_age"],
        "target_side": _lower_side(common["target_side"]),
        "mammography_suspicious_field": scan_metadata[
            "mammography_suspicious_field"
        ],
        "scan_date_time": scan_metadata["scan_date_time"],
        "operator_id": scan_metadata["operator_id"],
        "hardware_version": scan_metadata["hardware_version"],
        "eoscan_version": scan_metadata["eoscan_version"],
        "model_name": common["model_name"],
        "model_version": common["model_version"],
        "method_performance": model_performance,
        "suggested_class": suggested_class,
        "reliability": reliability,
        "reliability_reason": reliability_reason,
    }


def _external_model_performance(model_artifact: dict[str, Any]) -> dict[str, Any]:
    """Expose the frozen validation summary without patient-level model internals."""
    performance = model_artifact.get("model_performance", {})
    metrics = performance.get("held_out_metrics", {})
    sensitivity = metrics.get("sensitivity", {})
    specificity = metrics.get("specificity", {})
    return {
        "evaluation_available": bool(performance.get("evaluation_available", False)),
        "evaluation_method": performance.get("evaluation_method", "unknown"),
        "folds": performance.get("folds", "unknown"),
        "repeats": performance.get("repeats", "unknown"),
        "sensitivity": sensitivity.get("mean", "unknown"),
        "sensitivity_std": sensitivity.get("std", "unknown"),
        "specificity": specificity.get("mean", "unknown"),
        "specificity_std": specificity.get("std", "unknown"),
    }


def _internal_report(
    *,
    common: dict[str, Any],
    version: str,
    reference_doc: str | None,
    target_prediction: dict[str, Any],
    contralateral_prediction: dict[str, Any],
    model_artifact_sha256: str,
) -> dict[str, Any]:
    feature_row = target_prediction["feature_row"]
    age_available = bool(feature_row.get("age_available", False))
    return {
        "output_type": "aramis_internal_clinical_report",
        "report_version": version,
        "reference_doc": _project_reference_doc(reference_doc),
        "report_id": common["report_id"],
        "created_at": common["created_at"],
        "analysis_author": common["analysis_author"],
        "prediction_comment": common["prediction_comment"],
        "model": {
            "id": common["model_id"],
            "name": common["model_name"],
            "version": common["model_version"],
            "artifact_sha256": model_artifact_sha256,
        },
        "scan_metadata": _scan_metadata(
            feature_row,
            patient_id=common["patient_id"],
            target_side=common["target_side"],
            age_available=age_available,
        ),
        "breast_predictions": {
            "target": _breast_prediction_report(target_prediction),
            "contralateral": _breast_prediction_report(contralateral_prediction),
        },
    }


def _breast_prediction_report(prediction: dict[str, Any]) -> dict[str, Any]:
    profile_key = (
        "azimuthal_integration_target_profile"
        if prediction.get("is_target", False)
        else "azimuthal_integration_contralateral_profile"
    )
    if not prediction["available"]:
        return {
            "available": False,
            "side": "unknown",
            profile_key: {
                "available": False,
                "p_cancer": "unknown",
                "per_measurement_p_cancer": [],
            },
            "final_prediction": {
                "p_cancer": "unknown",
                "decision_threshold_id": "unknown",
                "decision_threshold": "unknown",
                "suggested_class": "unknown",
                "score_percentiles": {
                    "all_training_patients": "unknown",
                    "benign_training_patients": "unknown",
                    "cancer_training_patients": "unknown",
                },
                "reliability": {"level": "unknown", "reason": "unknown"},
            },
            "symmetry": {"available": False, "status": "not_available"},
            "reason": prediction["reason"],
        }
    row = prediction["feature_row"]
    return {
        "available": True,
        "side": _lower_side(row["target_side"]),
        profile_key: {
            "available": bool(prediction["xrd_profile"]["available"]),
            "p_cancer": prediction["xrd_profile"]["profile_p_cancer"],
            "per_measurement_p_cancer": prediction["xrd_profile"][
                "measurement_p_cancer"
            ],
        },
        "final_prediction": {
            "p_cancer": prediction["p_cancer"],
            "decision_threshold_id": _decision_threshold_id(prediction["threshold_key"]),
            "decision_threshold": prediction["threshold"],
            "suggested_class": prediction["suggested_class"],
            "score_percentiles": prediction["quantiles"],
            "reliability": {
                "level": row["result_reliability"],
                "reason": row["result_reliability_reason"],
            },
        },
        "symmetry": {
            "available": bool(row.get("symmetry_available", 0)),
            "status": (
                "applied"
                if bool(row.get("symmetry_available", 0))
                else "not_available"
            ),
        },
        "model_execution": {
            "scoring_path": (
                "profile_age_with_symmetry"
                if bool(row.get("symmetry_available", 0))
                else "profile_age_with_neutral_symmetry_gate"
            ),
            "symmetry_refinement": (
                "applied"
                if bool(row.get("symmetry_available", 0))
                else "not_applied"
            ),
        },
    }


def _scan_metadata(
    feature_row: dict[str, Any],
    *,
    patient_id: str,
    target_side: str,
    age_available: bool,
) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "target_side": _lower_side(target_side),
        "patient_age": feature_row.get("age") if age_available else "unknown",
        "patient_age_available": age_available,
        "session_id": _metadata_value(feature_row, "session_uid", "session_id"),
        "scan_date_time": _metadata_value(feature_row, "scan_date_time", "started_at"),
        "operator_id": _metadata_value(feature_row, "operator_id"),
        "hardware_version": _metadata_value(feature_row, "hardware_version"),
        "eoscan_version": _metadata_value(feature_row, "eoscan_version"),
        "experimental_protocol_version": _metadata_value(
            feature_row, "experimental_protocol_version", "product_protocol_version"
        ),
        "mammography_suspicious_field": _metadata_value(
            feature_row, "mammography_suspicious_field"
        ),
        "mammography_conclusion": _metadata_value(
            feature_row, "mammography_conclusion"
        ),
        "measurement_summary": {
            "target_valid_measurements": feature_row.get("target_measurements", 0),
            "contralateral_present": bool(
                feature_row.get("contralateral_measurements", 0)
            ),
            "contralateral_valid_measurements": feature_row.get(
                "contralateral_measurements", 0
            ),
        },
    }


def _metadata_value(feature_row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = feature_row.get(key)
        if value is not None and not pd.isna(value):
            return value
    return "unknown"


def _report_timestamp() -> str:
    return datetime.now(ZoneInfo("Europe/Paris")).isoformat(timespec="seconds")


def _project_reference_doc(reference_doc: str | None) -> str | None:
    if reference_doc is None:
        return None
    text = str(reference_doc)
    marker = "docs/"
    return f"./{text[text.index(marker):]}" if marker in text else text


def _lower_side(value: Any) -> str | None:
    return None if value in {None, ""} else str(value).lower()


def _decision_threshold_id(threshold_key: str) -> str:
    return "target_sensitivity_0.95" if threshold_key == "threshold_target" else threshold_key


def _prediction_columns(model_artifact: dict[str, Any]) -> dict[str, str]:
    model_config = dict(model_artifact.get("model_columns", {}))
    return {
        "profile_column": str(model_config.get("profile_column", "radial_profile_data")),
        "group_column": str(model_config.get("group_column", "patientId")),
        "specimen_column": str(model_config.get("specimen_column", "specimenId")),
        "side_column": str(model_config.get("side_column", "side")),
        "age_column": str(model_config.get("age_column", "age")),
    }


def _side_profile_scores(
    df: pd.DataFrame,
    model_info: dict[str, Any],
    *,
    patient_id: str,
    target_side: str,
    columns: dict[str, str],
    feature_row: pd.DataFrame,
) -> dict[str, Any]:
    row = feature_row.iloc[0]
    target = _side_profile_score(
        df,
        model_info,
        patient_id=patient_id,
        side=target_side,
        columns=columns,
    )
    contralateral = _side_profile_score(
        df,
        model_info,
        patient_id=patient_id,
        side=row.get("contralateral_side"),
        columns=columns,
    )
    return {"target": target, "contralateral": contralateral}


def _side_profile_score(
    df: pd.DataFrame,
    model_info: dict[str, Any],
    *,
    patient_id: str,
    side: Any,
    columns: dict[str, str],
) -> dict[str, Any]:
    side_norm = _normalize_side(side)
    if side_norm is None:
        return {
            "available": False,
            "profile_p_cancer": None,
            "measurement_p_cancer": [],
        }
    group_column = columns["group_column"]
    side_column = columns["side_column"]
    profile_column = columns["profile_column"]
    patient_df = df[df[group_column].astype(str) == str(patient_id)].copy()
    side_df = patient_df[patient_df[side_column].map(_normalize_side) == side_norm].copy()
    if side_df.empty:
        return {
            "available": False,
            "profile_p_cancer": None,
            "measurement_p_cancer": [],
        }
    lr1_model = model_info.get("lr1_model")
    if lr1_model is None:
        raise ValueError("Model artifact is missing lr1_model.")
    scores = lr1_model.predict_proba(profile_matrix(side_df, profile_column))[:, 1]
    profile_p_cancer = _logit_average_probability(scores)
    return {
        "available": True,
        "profile_p_cancer": profile_p_cancer,
        "measurement_p_cancer": [float(value) for value in scores],
    }


def _logit_average_probability(scores: Any) -> float:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return 0.5
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return float(1.0 / (1.0 + np.exp(-float(np.mean(logits)))))


def _normalize_side(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value).strip().lower()
    if text in {"left", "l"}:
        return "left"
    if text in {"right", "r"}:
        return "right"
    return None


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
    thresholds = model_info.get("routes", {}).get(model_route, model_info).get(
        "thresholds", {}
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
    stem = _safe_stem(
        f"{config['patient']['patient_id']}_{model_id}_{report_id}"
    )
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
    missing = [
        section
        for section in ("run", "io", "patient")
        if section not in config
    ]
    if missing:
        raise ValueError(f"Missing prediction config sections: {missing}")
    for key in ("input_model_joblib_path", "output_folder"):
        if not config.get("io", {}).get(key):
            raise ValueError(f"Missing io.{key} in {config_path}")
    forbidden = sorted(
        set(config).difference({"run", "io", "patient"})
    )
    if forbidden:
        raise ValueError(
            "Predict YAML cannot override model-held sections: " f"{forbidden}"
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
    if not config.get("run", {}).get("analysis_author"):
        raise ValueError(f"Missing run.analysis_author in {config_path}")
    if not isinstance(config.get("run", {}).get("prediction_comment", ""), str):
        raise TypeError(f"run.prediction_comment must be a string in {config_path}")
    if not config.get("patient", {}).get("patient_id"):
        raise ValueError(f"Missing patient.patient_id in {config_path}")
    if not config.get("patient", {}).get("target_side"):
        raise ValueError(f"Missing patient.target_side in {config_path}")


def _validate_h5_container_contract(
    model_artifact: dict[str, Any],
    h5_path: Path,
    *,
    expected_patient_id: str,
) -> None:
    container = _prediction_contract(model_artifact)["container"]

    expected_schema_version = str(_required_container_value(container, "schema_version"))
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


def _required_container_value(container: dict[str, Any], key: str) -> Any:
    value = container.get(key)
    if value in {None, ""}:
        raise ValueError(f"Missing prediction_contract.container.{key} in model artifact.")
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
    return (Path(__file__).resolve().parents[2] / path).resolve()


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
