"""Target and contralateral breast scoring for a frozen Aramis model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .model_utils import profile_matrix
from .patient_features import build_patient_prediction_feature_row
from .prediction_contract import _model_threshold
from .training_config import PRODUCT_MODEL_NAME


_DEFAULT_TRA_POLICY = {
    "contract": "aramis_tra_v0_1",
    "reference_score": "final_prediction.p_cancer",
    "reference_population": "train_on_all_target-breast_cases",
    "levels": [
        {"level": "TRA 1", "minimum_percentile": 0.0, "maximum_percentile": 20.0},
        {"level": "TRA 2", "minimum_percentile": 20.0, "maximum_percentile": 50.0},
        {"level": "TRA 3", "minimum_percentile": 50.0, "maximum_percentile": 80.0},
        {"level": "TRA 4", "minimum_percentile": 80.0, "maximum_percentile": 90.0},
        {"level": "TRA 5", "minimum_percentile": 90.0, "maximum_percentile": 100.0},
    ],
}


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


def _prediction_model_route(
    feature_row: pd.DataFrame, model_info: dict[str, Any]
) -> str | None:
    if str(model_info.get("symmetry_policy", "")).startswith(
        "single_model_gated_optional_refinement"
    ):
        return None
    if "routes" not in model_info:
        return None
    return (
        "paired"
        if int(feature_row["symmetry_available"].iloc[0]) == 1
        else "fallback_no_symmetry"
    )


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
    force_no_symmetry: bool = False,
) -> dict[str, Any]:
    """Score one requested side with the fixed final product artifact."""
    feature_table = build_patient_prediction_feature_row(
        df,
        model_info,
        patient_id=patient_id,
        target_side=str(target_side),
        **columns,
    )
    if force_no_symmetry:
        feature_table = _with_neutral_symmetry_gate(feature_table)
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
        "quantiles": _prediction_quantiles(
            model_info,
            p_cancer,
            score_kind="final_prediction",
        ),
        "tissue_risk_assessment": _tissue_risk_assessment(model_info, p_cancer),
    }


def _with_neutral_symmetry_gate(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Disable paired-breast refinement while retaining profile and age evidence."""
    out = feature_table.copy()
    out["symmetry_available"] = 0
    out["result_reliability"] = "low"
    out["result_reliability_reason"] = (
        "SK symmetry refinement is intentionally unavailable for contralateral scoring"
    )
    return out


def _unavailable_side_prediction() -> dict[str, Any]:
    return {
        "available": False,
        "reason": "contralateral breast is unavailable after preprocessing",
    }


def _prediction_quantiles(
    model_info: dict[str, Any], p_cancer: float, *, score_kind: str
) -> dict[str, Any]:
    """Locate a score in its frozen train-on-all reference distribution."""
    references = model_info.get("prediction_reference_scores")
    if not isinstance(references, dict):
        raise ValueError(
            "Model artifact has no prediction_reference_scores. Retrain model."
        )
    reference = references.get(score_kind)
    if not isinstance(reference, dict):
        raise ValueError(
            "Model artifact has no frozen "
            f"{score_kind!r} score reference. Retrain model."
        )
    keys = {
        "all_training_target_cases": "all_target_cases",
        "benign_training_target_cases": "benign_target_cases",
        "cancer_training_target_cases": "cancer_target_cases",
    }
    out: dict[str, Any] = {
        "reference_score": str(reference.get("score", "unknown")),
        "reference_population": _report_identifier(
            reference.get("population", "unknown")
        ),
    }
    for report_key, reference_key in keys.items():
        values = np.asarray(reference.get(reference_key, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                f"Model prediction reference has no values for {reference_key!r}."
            )
        out[report_key] = float(
            np.searchsorted(np.sort(values), p_cancer, side="right") / values.size
        )
    return out


def _report_identifier(value: Any) -> str:
    """Render a frozen identifier with spaces normalized to underscores."""
    return "_".join(str(value).split())


def _tissue_risk_assessment(model_info: dict[str, Any], p_cancer: float) -> dict[str, Any]:
    """Map a final score to the model-held ordinal TRA reference scale."""
    policy = model_info.get("tissue_risk_assessment", _DEFAULT_TRA_POLICY)
    if policy.get("contract") != "aramis_tra_v0_1":
        raise ValueError("Unsupported tissue risk assessment policy in model artifact.")

    references = model_info.get("prediction_reference_scores", {})
    reference = references.get("final_prediction", {})
    values = np.asarray(reference.get("all_target_cases", []), dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        raise ValueError("Model prediction reference has no final target-case scores.")

    index = 100.0 * np.searchsorted(values, p_cancer, side="right") / values.size
    return {
        "index": float(index),
        "level": _tra_level(policy, index),
        "reference_population": str(
            policy.get("reference_population", reference.get("population", "unknown"))
        ),
    }


def _tra_level(policy: dict[str, Any], index: float) -> str:
    """Return the sole TRA level whose upper boundary contains ``index``."""
    levels = policy.get("levels")
    if not isinstance(levels, list) or len(levels) != 5:
        raise ValueError("TRA policy must define exactly five ordered levels.")
    for item in levels:
        upper = float(item["maximum_percentile"])
        if index < upper or upper == 100.0:
            return str(item["level"]).replace("_", " ")
    raise ValueError("TRA index is outside the frozen policy range.")


def _prediction_columns(model_artifact: dict[str, Any]) -> dict[str, str]:
    model_config = dict(model_artifact.get("model_columns", {}))
    return {
        "profile_column": str(
            model_config.get("profile_column", "radial_profile_data")
        ),
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
    side_df = patient_df[
        patient_df[side_column].map(_normalize_side) == side_norm
    ].copy()
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
