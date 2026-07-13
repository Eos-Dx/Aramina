"""Prediction entrypoint for Aramis research-draft decision support."""

from __future__ import annotations

from datetime import datetime, timezone
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
from xrd_preprocessing import (
    load_preprocessing_artifact,
    load_preprocessing_dataframe,
)

from .modeling import profile_matrix
from .pipelines import run_preprocessing_pipeline
from .training import build_patient_prediction_feature_row


def run_prediction_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run one-patient prediction from a YAML config."""
    config_path = Path(config_path).expanduser().resolve()
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    _validate_prediction_config(config, config_path)

    model_path = _config_path(
        config,
        config_path,
        section="io",
        key="input_model_joblib_path",
    )
    model_artifact = joblib.load(model_path)
    model_id, model_name, model_version = _validate_model_identity(
        model_artifact,
        config["model"],
    )
    output_paths = _prediction_output_paths(config, config_path)
    model_info = _model_info(model_artifact, model_name)
    df, dataframe_path, preprocessing_artifact = _prediction_dataframe(
        config,
        config_path,
        model_artifact,
        output_paths["dataframe_joblib"],
    )
    patient_id = str(config["patient"]["patient_id"])
    target_side = _prediction_target_side(config, patient_id)
    feature_row = build_patient_prediction_feature_row(
        df,
        model_info,
        patient_id=patient_id,
        target_side=target_side,
        **_prediction_columns(model_artifact),
    )
    model_route = _prediction_model_route(feature_row, model_info)
    p_cancer = _score_model(feature_row, model_name, model_info, model_route)
    threshold_key = _prediction_contract(model_artifact)["decision"]["threshold_key"]
    threshold = _model_threshold(model_info, threshold_key, model_route)
    suggested_class = "CANCER" if p_cancer >= threshold else "BENIGN"
    side_profile_scores = _side_profile_scores(
        df,
        model_info,
        patient_id=patient_id,
        target_side=target_side,
        columns=_prediction_columns(model_artifact),
        feature_row=feature_row,
    )
    reports = _prediction_reports(
        config=config,
        config_path=config_path,
        config_text=config_text,
        dataframe_path=dataframe_path,
        output_paths=output_paths,
        preprocessing_artifact=preprocessing_artifact,
        model_path=model_path,
        model_artifact=model_artifact,
        model_id=model_id,
        model_name=model_name,
        model_version=model_version,
        model_info=model_info,
        model_route=model_route,
        feature_row=feature_row,
        p_cancer=p_cancer,
        threshold_key=threshold_key,
        threshold=threshold,
        suggested_class=suggested_class,
        side_profile_scores=side_profile_scores,
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
) -> tuple[Any, Path, dict[str, Any] | None]:
    if not config.get("io", {}).get("input_h5_path"):
        dataframe_path = _config_path(
            config,
            config_path,
            section="io",
            key="input_dataframe_joblib_path",
        )
        return load_preprocessing_dataframe(dataframe_path), dataframe_path, None

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
    branch = preprocessing_config.get("aramis_preprocessing", {}).get("branch")
    if branch != "one_to_many":
        raise ValueError(f"Unsupported prediction preprocessing branch: {branch!r}")
    df = run_preprocessing_pipeline(
        h5_path,
        preprocessing_config,
        output_joblib_path=output_dataframe_path,
    )
    return df, output_dataframe_path, load_preprocessing_artifact(output_dataframe_path)


def _prediction_preprocessing_config(model_artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the preprocessing contract embedded at model training time."""
    model_config = model_artifact.get("prediction_preprocessing_config")
    if not isinstance(model_config, dict):
        raise ValueError(
            "Model artifact has no prediction_preprocessing_config. Retrain model "
            "with io.prediction_preprocessing_config_path."
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
    if model_name in {"M0", "M0Q"}:
        return float(feature_row["profile_p_cancer_logit_average"].iloc[0])
    final_model = route_info.get("final_model")
    if final_model is None:
        raise ValueError(f"Model {model_name} is missing final_model.")
    return float(final_model.predict_proba(feature_row[columns])[:, 1][0])


def _prediction_reports(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_text: str,
    dataframe_path: Path,
    output_paths: dict[str, Path],
    preprocessing_artifact: dict[str, Any] | None,
    model_path: Path,
    model_artifact: dict[str, Any],
    model_id: str,
    model_name: str,
    model_version: str,
    model_info: dict[str, Any],
    model_route: str | None,
    feature_row,
    p_cancer: float,
    threshold_key: str,
    threshold: float,
    suggested_class: str,
    side_profile_scores: dict[str, Any],
) -> dict[str, Any]:
    row = feature_row.iloc[0].to_dict()
    created_at = datetime.now(timezone.utc).isoformat()
    reporting = _prediction_contract(model_artifact)["reporting"]
    provenance = _prediction_provenance(
        config=config,
        config_path=config_path,
        config_text=config_text,
        dataframe_path=dataframe_path,
        preprocessing_artifact=preprocessing_artifact,
        model_path=model_path,
        model_artifact=model_artifact,
        model_id=model_id,
    )
    common = {
        "created_at": created_at,
        "author": config["prediction"].get("author"),
        "clinical_stage": config["prediction"].get("clinical_stage", "research draft"),
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
        "model_version": model_version,
        "prediction_run_id": output_paths["run_id"],
        "limitations": [
            "research draft",
            "not for autonomous diagnosis",
            "requires breast-imaging clinician review",
        ],
        "provenance": provenance,
    }
    external = _external_report(
        common=common,
        version=str(reporting["external_report"]["version"]),
        reference_doc=reporting["external_report"].get("reference_doc"),
        p_cancer=p_cancer,
        threshold_key=threshold_key,
        threshold=threshold,
        suggested_class=suggested_class,
        reliability=row["result_reliability"],
        reliability_reason=row["result_reliability_reason"],
    )
    internal = _internal_report(
        common=common,
        version=str(reporting["internal_report"]["version"]),
        reference_doc=reporting["internal_report"].get("reference_doc"),
        p_cancer=p_cancer,
        threshold_key=threshold_key,
        threshold=threshold,
        suggested_class=suggested_class,
        reliability=row["result_reliability"],
        reliability_reason=row["result_reliability_reason"],
        feature_row=row,
        model_route=model_route,
        model_artifact=model_artifact,
        provenance=provenance,
        side_profile_scores=side_profile_scores,
        output_paths=output_paths,
    )
    return _json_safe({"external_report": external, "internal_report": internal})


def _external_report(
    *,
    common: dict[str, Any],
    version: str,
    reference_doc: str | None,
    p_cancer: float,
    threshold_key: str,
    threshold: float,
    suggested_class: str,
    reliability: str,
    reliability_reason: str,
) -> dict[str, Any]:
    return {
        "kind": "aramis_external_prediction_report",
        "version": version,
        "reference_doc": reference_doc,
        **common,
        "p_cancer": p_cancer,
        "threshold_key": threshold_key,
        "threshold": threshold,
        "suggested_class": suggested_class,
        "risk_level": "high" if suggested_class == "CANCER" else "low",
        "reliability": reliability,
        "reliability_reason": reliability_reason,
    }


def _internal_report(
    *,
    common: dict[str, Any],
    version: str,
    reference_doc: str | None,
    p_cancer: float,
    threshold_key: str,
    threshold: float,
    suggested_class: str,
    reliability: str,
    reliability_reason: str,
    feature_row: dict[str, Any],
    model_route: str | None,
    model_artifact: dict[str, Any],
    provenance: dict[str, Any],
    side_profile_scores: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    training = model_artifact.get("training_config", {}).get("training", {})
    age_available = bool(feature_row.get("age_available", False))
    final_prediction = {
        "p_cancer": p_cancer,
        "decision_threshold_id": _decision_threshold_id(threshold_key),
        "threshold": threshold,
        "suggested_class": suggested_class,
    }
    if model_route is not None:
        final_prediction["model_route"] = model_route
    return {
        "output_type": "aramis_internal_clinical_report",
        "version": version,
        "reference_doc": _project_reference_doc(reference_doc),
        "created_at": _internal_report_timestamp(),
        "analysis_author": common.get("author"),
        "clinical_stage": training.get("clinical_stage", common["clinical_stage"]),
        "intended_use": training.get("intended_use", common["intended_use"]),
        "research_only": str(
            training.get("clinical_stage", common["clinical_stage"])
        ).lower()
        != "production",
        "patient_id": common["patient_id"],
        "target_side": _lower_side(common["target_side"]),
        "contralateral_side": _lower_side(common["contralateral_side"]),
        "model_id": common["model_id"],
        "model_component": common["model_name"],
        "provenance": _internal_provenance(provenance),
        "xrd_scan_information": {
            "target_measurements": feature_row.get("target_measurements"),
            "contralateral_measurements": feature_row.get("contralateral_measurements"),
            "patient_age": feature_row.get("age") if age_available else None,
            "patient_age_available": age_available,
        },
        "features": {
            "azimuthal_integration": {
                "target_profile": _internal_profile_score(side_profile_scores["target"]),
                "contralateral_profile": _internal_profile_score(
                    side_profile_scores["contralateral"]
                ),
                "healthy_reference_distance": {
                    "status": "not_implemented",
                    "reason": "average healthy reference profile is not fixed in model artifact",
                },
            },
            "symmetry": _symmetry_report_block(feature_row),
            "reliability": {
                "result_reliability": reliability,
                "result_reliability_reason": reliability_reason,
            },
        },
        "final_prediction": final_prediction,
        "outputs": {
            "output_folder": str(output_paths["folder"]),
            "prediction_dataframe_joblib": str(output_paths["dataframe_joblib"]),
            "external_json": str(output_paths["external_json"]),
            "external_yaml": str(output_paths["external_yaml"]),
            "internal_json": str(output_paths["internal_json"]),
            "internal_yaml": str(output_paths["internal_yaml"]),
        },
    }


def _prediction_provenance(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_text: str,
    dataframe_path: Path,
    preprocessing_artifact: dict[str, Any] | None,
    model_path: Path,
    model_artifact: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    input_h5_path = _optional_config_path(config, config_path, "input_h5_path")
    return {
        "prediction_config_path": str(config_path),
        "prediction_config_sha256": sha256(config_text.encode("utf-8")).hexdigest(),
        "input_dataframe_joblib_path": str(dataframe_path),
        "input_dataframe_joblib_sha256": _file_sha256(dataframe_path),
        "input_h5_path": input_h5_path,
        "input_h5_sha256": _optional_file_sha256(input_h5_path),
        "prediction_preprocessing_config_path": model_artifact.get(
            "prediction_preprocessing_config_path"
        ),
        "model_prediction_preprocessing_config_path": model_artifact.get(
            "prediction_preprocessing_config_path"
        ),
        "model_prediction_preprocessing_config_sha256": model_artifact.get(
            "prediction_preprocessing_config_sha256"
        ),
        "prediction_preprocessing_config_sha256": (preprocessing_artifact or {}).get(
            "preprocessing_config_sha256"
        ) or model_artifact.get("prediction_preprocessing_config_sha256"),
        "prediction_contract_sha256": model_artifact.get("prediction_contract_sha256"),
        "input_model_joblib_path": str(model_path),
        "input_model_joblib_sha256": _file_sha256(model_path),
        "input_model_id": model_id,
        "author": config["prediction"].get("author"),
        "training_config_sha256": model_artifact.get("training_config_sha256"),
        "preprocessing_config_sha256": model_artifact.get(
            "preprocessing_config_sha256"
        ),
        "model_metadata": model_artifact.get("metadata", {}),
    }


def _internal_profile_score(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(score.get("available", False)),
        "side": _lower_side(score.get("side")),
        "profile_p_cancer": score.get("profile_p_cancer"),
        "measurement_p_cancer": score.get("measurement_p_cancer", []),
        "profile_statistics": score.get("profile_statistics"),
    }


def _internal_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction_config": {
            "path": provenance.get("prediction_config_path"),
            "sha256": provenance.get("prediction_config_sha256"),
        },
        "input": {
            "prediction_dataframe_joblib": {
                "path": provenance.get("input_dataframe_joblib_path"),
                "sha256": provenance.get("input_dataframe_joblib_sha256"),
            },
            "h5_container": {
                "path": provenance.get("input_h5_path"),
                "sha256": provenance.get("input_h5_sha256"),
            },
        },
        "prediction_preprocessing": {
            "config_path": (
                provenance.get("prediction_preprocessing_config_path")
                or provenance.get("model_prediction_preprocessing_config_path")
            ),
            "config_sha256": provenance.get(
                "prediction_preprocessing_config_sha256"
            )
            or provenance.get("model_prediction_preprocessing_config_sha256"),
        },
        "model_artifact": {
            "path": provenance.get("input_model_joblib_path"),
            "sha256": provenance.get("input_model_joblib_sha256"),
            "model_id": provenance.get("input_model_id"),
            "aramis_version": provenance.get("model_metadata", {}).get(
                "aramis_version"
            ),
            "aramis_git_sha": provenance.get("model_metadata", {}).get(
                "aramis_git_sha"
            ),
        },
    }


def _internal_report_timestamp() -> str:
    return datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%dT%H:%M:%S%Z")


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
    model_config = dict(model_artifact.get("training_config", {}).get("model", {}))
    return {
        "profile_column": str(model_config.get("profile_column", "radial_profile_data")),
        "group_column": str(model_config.get("group_column", "patientId")),
        "specimen_column": str(model_config.get("specimen_column", "specimenId")),
        "side_column": str(model_config.get("side_column", "side")),
        "q_column": str(model_config.get("q_column", "q_range")),
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
            "side": None,
            "measurements": 0,
            "profile_p_cancer": None,
            "p_cancer_probability_mean": None,
            "profile_p_cancer_probability_mean": None,
            "measurement_p_cancer": [],
            "profile_statistics": None,
        }
    group_column = columns["group_column"]
    side_column = columns["side_column"]
    profile_column = columns["profile_column"]
    patient_df = df[df[group_column].astype(str) == str(patient_id)].copy()
    side_df = patient_df[patient_df[side_column].map(_normalize_side) == side_norm].copy()
    if side_df.empty:
        return {
            "available": False,
            "side": _display_side(side_norm),
            "measurements": 0,
            "profile_p_cancer": None,
            "p_cancer_probability_mean": None,
            "profile_p_cancer_probability_mean": None,
            "measurement_p_cancer": [],
            "profile_statistics": None,
        }
    lr1_model = model_info.get("lr1_model")
    if lr1_model is None:
        raise ValueError("Model artifact is missing lr1_model.")
    scores = lr1_model.predict_proba(profile_matrix(side_df, profile_column))[:, 1]
    profile_p_cancer = _logit_average_probability(scores)
    return {
        "available": True,
        "side": _display_side(side_norm),
        "measurements": int(len(side_df)),
        "profile_p_cancer": profile_p_cancer,
        "p_cancer_probability_mean": float(np.mean(scores)),
        "profile_p_cancer_probability_mean": float(np.mean(scores)),
        "measurement_p_cancer": [float(value) for value in scores],
        "profile_statistics": _profile_statistics(
            side_df,
            profile_column=profile_column,
            q_column=columns["q_column"],
        ),
    }


def _profile_statistics(
    df: pd.DataFrame,
    *,
    profile_column: str,
    q_column: str,
) -> dict[str, Any]:
    q_values = [np.asarray(value, dtype=float) for value in df[q_column]]
    profiles = [np.asarray(value, dtype=float) for value in df[profile_column]]
    q = q_values[0]
    profile = np.mean(np.vstack(profiles), axis=0)
    peak_index = int(np.argmax(profile))
    return {
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
        "q_mean": float(np.mean(q)),
        "q_median": float(np.median(q)),
        "q_first_quartile": float(np.quantile(q, 0.25)),
        "q_third_quartile": float(np.quantile(q, 0.75)),
        "q_mode_peak_intensity": float(q[peak_index]),
        "highest_intensity": float(profile[peak_index]),
    }


def _symmetry_report_block(feature_row: dict[str, Any]) -> dict[str, Any]:
    keys = [key for key in feature_row if key.startswith("sk_")]
    return {
        "available": bool(feature_row.get("symmetry_available", 0)),
        "features": {key: feature_row.get(key) for key in keys},
        "symmetry_only_p_cancer": {
            "status": "not_implemented",
            "reason": "standalone symmetry-only model is not fixed in prediction artifact",
        },
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


def _display_side(side_norm: str | None) -> str | None:
    if side_norm is None:
        return None
    return {"left": "Left", "right": "Right"}[side_norm]


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


def _validate_model_identity(
    model_artifact: dict[str, Any], requested: dict[str, Any]
) -> tuple[str, str, str]:
    """Check all caller-supplied model identifiers against the loaded artifact."""
    artifact_model_id = (
        model_artifact.get("training_config", {})
        .get("training", {})
        .get("name")
    )
    artifact_model_version = (
        model_artifact.get("training_config", {}).get("training", {}).get("version")
    )
    if not artifact_model_id or artifact_model_version in {None, ""}:
        raise ValueError("Prediction model artifact has no training.name model_id.")
    requested_model_id = str(requested["model_id"])
    requested_model_name = str(requested["model_name"]).upper()
    requested_model_version = str(requested["model_version"])
    if str(artifact_model_id) != requested_model_id:
        raise ValueError(
            "Prediction model_id does not match artifact training.name: "
            f"requested={requested_model_id!r}, artifact={artifact_model_id!r}"
        )
    if requested_model_name not in model_artifact.get("models", {}):
        raise ValueError(
            "Prediction model_name does not match artifact models: "
            f"requested={requested_model_name!r}, "
            f"available={sorted(model_artifact.get('models', {}))}"
        )
    if str(artifact_model_version) != requested_model_version:
        raise ValueError(
            "Prediction model_version does not match artifact training.version: "
            f"requested={requested_model_version!r}, artifact={artifact_model_version!r}"
        )
    return str(artifact_model_id), requested_model_name, str(artifact_model_version)


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
) -> dict[str, Path]:
    folder = _config_path(config, config_path, section="io", key="output_folder")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    stem = _safe_stem(f"{config['prediction']['name']}_{config['patient']['patient_id']}_{run_id}")
    return {
        "folder": folder,
        "run_id": run_id,
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
        for section in ("prediction", "io", "patient", "model")
        if section not in config
    ]
    if missing:
        raise ValueError(f"Missing prediction config sections: {missing}")
    for key in ("input_model_joblib_path", "output_folder"):
        if not config.get("io", {}).get(key):
            raise ValueError(f"Missing io.{key} in {config_path}")
    forbidden = sorted(
        set(config).intersection({"preprocessing", "reporting", "container", "decision"})
    )
    if forbidden:
        raise ValueError(
            "Predict YAML cannot override model-held sections: " f"{forbidden}"
        )
    has_h5 = bool(config.get("io", {}).get("input_h5_path"))
    has_dataframe = bool(config.get("io", {}).get("input_dataframe_joblib_path"))
    if has_h5 == has_dataframe:
        raise ValueError(
            "Set exactly one input: io.input_h5_path or io.input_dataframe_joblib_path."
        )
    if not has_h5 and not has_dataframe:
        raise ValueError(f"Missing io.input_dataframe_joblib_path in {config_path}")
    if not config.get("patient", {}).get("patient_id"):
        raise ValueError(f"Missing patient.patient_id in {config_path}")
    if not config.get("patient", {}).get("target_side"):
        raise ValueError(f"Missing patient.target_side in {config_path}")
    if not config.get("model", {}).get("model_id"):
        raise ValueError(f"Missing model.model_id in {config_path}")
    for key in ("model_name", "model_version"):
        if not config.get("model", {}).get(key):
            raise ValueError(f"Missing model.{key} in {config_path}")
    extra_model_keys = sorted(
        set(config.get("model", {})).difference({"model_id", "model_name", "model_version"})
    )
    if extra_model_keys:
        raise ValueError(
            "Predict YAML model section only identifies the artifact: "
            f"unexpected keys={extra_model_keys}"
        )


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
    contract = model_artifact.get("prediction_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            "Model artifact has no prediction_contract. Retrain with prediction_contract."
        )
    for key in ("container", "reporting", "decision"):
        if not isinstance(contract.get(key), dict):
            raise ValueError(f"Model prediction_contract has no {key} section.")
    return contract


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
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return round(value, 5) if isfinite(value) else None
    return value
