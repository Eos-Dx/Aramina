"""Internal and external Aramis prediction report builders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .prediction_contract import _file_sha256, _json_safe, _prediction_contract


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
        risk_probability=target_prediction["p_cancer"],
        biopsy_required=target_prediction["biopsy_required"],
        decision_threshold=target_prediction["threshold"],
        reliability=row["result_reliability"],
        reliability_reason=row["result_reliability_reason"],
        model_metrics=_final_model_metrics(model_artifact),
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
        model_metrics=_final_model_metrics(model_artifact),
    )
    return _json_safe({"external_report": external, "internal_report": internal})


def _external_report(
    *,
    common: dict[str, Any],
    version: str,
    risk_probability: float,
    biopsy_required: bool,
    decision_threshold: float,
    reliability: str,
    reliability_reason: str,
    model_metrics: dict[str, Any],
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
        "mammography_suspicious_field": scan_metadata["mammography_suspicious_field"],
        "scan_date_time": scan_metadata["scan_date_time"],
        "operator_id": scan_metadata["operator_id"],
        "hardware_version": scan_metadata["hardware_version"],
        "eoscan_version": scan_metadata["eoscan_version"],
        "model_name": common["model_name"],
        "model_version": common["model_version"],
        "model_metrics": model_metrics,
        "risk_probability": risk_probability,
        "target_class_risk_level": _target_class_risk_level(biopsy_required),
        "decision_threshold": decision_threshold,
        "biopsy_required": biopsy_required,
        "reliability": reliability,
        "reliability_reason": reliability_reason,
    }


def _final_model_metrics(model_artifact: dict[str, Any]) -> dict[str, Any]:
    """Expose the selected final model's descriptive train-on-all metrics."""
    metrics = model_artifact.get("final_fit_training_metrics", {})
    return {
        "dataset": "train_on_all_target_breast_cases",
        "validation": "not_performed",
        "sensitivity": metrics.get("sensitivity", "unknown"),
        "specificity": metrics.get("specificity", "unknown"),
    }


def _internal_report(
    *,
    common: dict[str, Any],
    version: str,
    reference_doc: str | None,
    target_prediction: dict[str, Any],
    contralateral_prediction: dict[str, Any],
    model_artifact_sha256: str,
    model_metrics: dict[str, Any],
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
        "model_metrics": model_metrics,
        "decision_threshold": _decision_policy(target_prediction),
        "scan_metadata": _scan_metadata(
            feature_row,
            patient_id=common["patient_id"],
            target_side=common["target_side"],
            age_available=age_available,
        ),
        "breast_predictions": {
            "target": _target_breast_prediction_report(target_prediction),
            "contralateral": _contralateral_breast_prediction_report(
                contralateral_prediction
            ),
        },
    }


def _decision_policy(prediction: dict[str, Any]) -> dict[str, Any]:
    """Return the target-side decision threshold held by the frozen model."""
    return {
        "threshold_id": _decision_threshold_id(prediction["threshold_key"]),
        "threshold": prediction["threshold"],
        "applies_to": ["target.final_prediction"],
    }


def _target_breast_prediction_report(prediction: dict[str, Any]) -> dict[str, Any]:
    """Render the formally validated target-breast decision-support result."""
    row = prediction["feature_row"]
    return {
        "available": True,
        "side": _lower_side(row["target_side"]),
        "azimuthal_integration_profile": {
            "p_cancer": prediction["xrd_profile"]["profile_p_cancer"],
            "per_measurement_p_cancer": prediction["xrd_profile"][
                "measurement_p_cancer"
            ],
        },
        "final_prediction": {
            "p_cancer": prediction["p_cancer"],
            "reference_class": prediction["class_definition"]["reference_class"],
            "target_class": prediction["class_definition"]["target_class"],
            "target_class_risk_level": _target_class_risk_level(
                prediction["biopsy_required"]
            ),
            "biopsy_required": prediction["biopsy_required"],
            "level": prediction["tissue_risk_assessment"]["level"],
            "score_percentiles": prediction["quantiles"],
        },
        "reliability": {
            "level": row["result_reliability"],
            "reason": row["result_reliability_reason"],
        },
        "symmetry": {
            "available": bool(row.get("symmetry_available", 0)),
        },
        "model_execution": {
            "scoring_path": (
                "azimuthal_integration_age_with_symmetry"
                if bool(row.get("symmetry_available", 0))
                else "azimuthal_integration_age"
            ),
        },
    }


def _contralateral_breast_prediction_report(
    prediction: dict[str, Any]
) -> dict[str, Any]:
    """Render internal contralateral evidence without a biopsy action."""
    if not prediction["available"]:
        return {
            "available": False,
            "side": "unknown",
            "azimuthal_integration_profile": {
                "p_cancer": "unknown",
                "per_measurement_p_cancer": [],
            },
            "final_prediction": {
                "p_cancer": "unknown",
                "reference_class": prediction["class_definition"]["reference_class"],
                "target_class": prediction["class_definition"]["target_class"],
                "level": "unknown",
                "score_percentiles": _unknown_score_percentiles(),
            },
            "reliability": {"level": "unknown", "reason": "unknown"},
            "reason": prediction["reason"],
        }
    profile = prediction["xrd_profile"]
    return {
        "available": True,
        "side": _lower_side(prediction["feature_row"]["target_side"]),
        "azimuthal_integration_profile": {
            "p_cancer": profile["profile_p_cancer"],
            "per_measurement_p_cancer": profile["measurement_p_cancer"],
        },
        "final_prediction": {
            "p_cancer": prediction["p_cancer"],
            "reference_class": prediction["class_definition"]["reference_class"],
            "target_class": prediction["class_definition"]["target_class"],
            "level": prediction["tissue_risk_assessment"]["level"],
            "score_percentiles": prediction["quantiles"],
        },
        "reliability": {
            "level": prediction["feature_row"]["result_reliability"],
            "reason": prediction["feature_row"]["result_reliability_reason"],
        },
        "symmetry": {"available": False},
        "model_execution": {
            "scoring_path": "azimuthal_integration_age",
        },
    }


def _unknown_score_percentiles() -> dict[str, str]:
    return {
        "reference_population": "unknown",
        "all": "unknown",
        "reference_class": "unknown",
        "target_class": "unknown",
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
            "contralateral_available": bool(
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
        if value is None or pd.isna(value):
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        return value
    return "unknown"


def _report_timestamp() -> str:
    return datetime.now(ZoneInfo("Europe/Paris")).isoformat(timespec="seconds")


def _project_reference_doc(reference_doc: str | None) -> str | None:
    if reference_doc is None:
        return None
    text = str(reference_doc)
    marker = "docs/"
    return f"./{text[text.index(marker) :]}" if marker in text else text


def _lower_side(value: Any) -> str | None:
    return None if value in {None, ""} else str(value).lower()


def _decision_threshold_id(threshold_key: str) -> str:
    return (
        "target_sensitivity_0.95"
        if threshold_key == "threshold_target"
        else threshold_key
    )


def _target_class_risk_level(biopsy_required: bool) -> str:
    """Render the threshold-derived external and target-side risk level."""
    return "high" if biopsy_required else "low"
