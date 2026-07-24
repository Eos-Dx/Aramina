"""Threshold-centred Tissue Risk Assessment (TRA) policy calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TRA_POLICY_CONTRACT = "aramis_tra_v0_2"
_DECISION_STABILITY_CUTOFF = 1.0
_MIN_CASES_FOR_OOF_CALIBRATION = 5
_MARGIN_ROUNDING = 0.1
_MIN_BORDERLINE_MARGIN = 0.1
_MAX_BORDERLINE_MARGIN = 2.0
_OUTER_MARGIN_MULTIPLIER = 3.0


def derive_tra_policy(
    split_predictions: pd.DataFrame,
    *,
    decision_threshold: float,
) -> dict[str, Any]:
    """Freeze a threshold-centred TRA policy for one final model.

    TRA 2/3 always meets at the final probability decision threshold. The
    borderline half-width is calibrated from cases whose patient-safe OOF
    decisions vary across repeated folds. If evaluation is not available, a
    documented deterministic fallback retains the same threshold-centred
    semantics without claiming OOF calibration.
    """
    threshold = _validate_probability(decision_threshold, "decision_threshold")
    case_table = _oof_case_table(split_predictions, decision_threshold=threshold)
    borderline_margin, calibration = _borderline_margin(case_table)
    high_margin = _round_margin(_OUTER_MARGIN_MULTIPLIER * borderline_margin)
    threshold_logit = _logit(threshold)
    return {
        "contract": TRA_POLICY_CONTRACT,
        "reference_population": calibration["reference_population"],
        "decision_threshold": threshold,
        "calibration": calibration,
        "logit_margin_boundaries": {
            "tra_1_to_2": -borderline_margin,
            "tra_2_to_3": 0.0,
            "tra_3_to_4": borderline_margin,
            "tra_4_to_5": high_margin,
        },
        "probability_boundaries": {
            "tra_1_to_2": _sigmoid(threshold_logit - borderline_margin),
            "tra_2_to_3": threshold,
            "tra_3_to_4": _sigmoid(threshold_logit + borderline_margin),
            "tra_4_to_5": _sigmoid(threshold_logit + high_margin),
        },
        "levels": [
            {
                "level": "TRA 1",
                "decision_support_class": "BENIGN",
                "biopsy_required": False,
                "requires_radiologist_review": False,
                "interpretation": "below decision threshold",
            },
            {
                "level": "TRA 2",
                "decision_support_class": "BENIGN",
                "biopsy_required": False,
                "requires_radiologist_review": False,
                "interpretation": "near decision threshold, below threshold",
            },
            {
                "level": "TRA 3",
                "decision_support_class": "CANCER",
                "biopsy_required": True,
                "requires_radiologist_review": True,
                "interpretation": "borderline, above decision threshold",
            },
            {
                "level": "TRA 4",
                "decision_support_class": "CANCER",
                "biopsy_required": True,
                "requires_radiologist_review": True,
                "interpretation": "high score above decision threshold",
            },
            {
                "level": "TRA 5",
                "decision_support_class": "CANCER",
                "biopsy_required": True,
                "requires_radiologist_review": True,
                "interpretation": "very high score above decision threshold",
            },
        ],
    }


def tra_level(policy: dict[str, Any], p_cancer: float) -> str:
    """Return the threshold-centred TRA level for one final probability."""
    if policy.get("contract") != TRA_POLICY_CONTRACT:
        raise ValueError("Unsupported tissue risk assessment policy in model artifact.")
    threshold = _validate_probability(policy.get("decision_threshold"), "decision_threshold")
    boundaries = policy.get("logit_margin_boundaries")
    if not isinstance(boundaries, dict):
        raise ValueError("TRA policy is missing logit_margin_boundaries.")
    required = ("tra_1_to_2", "tra_2_to_3", "tra_3_to_4", "tra_4_to_5")
    if any(key not in boundaries for key in required):
        raise ValueError("TRA policy has incomplete logit_margin_boundaries.")
    lower = float(boundaries["tra_1_to_2"])
    decision = float(boundaries["tra_2_to_3"])
    upper = float(boundaries["tra_3_to_4"])
    high = float(boundaries["tra_4_to_5"])
    if not lower < decision == 0.0 < upper < high:
        raise ValueError("TRA policy has invalid threshold-centred boundaries.")
    margin = _logit(_validate_probability(p_cancer, "p_cancer")) - _logit(threshold)
    if margin < lower:
        return "TRA 1"
    if margin < decision:
        return "TRA 2"
    if margin < upper:
        return "TRA 3"
    if margin < high:
        return "TRA 4"
    return "TRA 5"


def _oof_case_table(
    predictions: pd.DataFrame,
    *,
    decision_threshold: float,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["p_cancer", "decision_stability", "margin"])
    required = {"target_case_id", "p_cancer", "y_pred_target"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing TRA calibration columns: {missing}")
    cases = (
        predictions.groupby("target_case_id", sort=False)
        .agg(
            p_cancer=("p_cancer", "median"),
            decision_stability=(
                "y_pred_target",
                lambda value: max(float(np.mean(value)), 1.0 - float(np.mean(value))),
            ),
        )
        .reset_index(drop=True)
    )
    cases["margin"] = _logit(cases["p_cancer"].to_numpy(dtype=float)) - _logit(
        decision_threshold
    )
    return cases


def _borderline_margin(cases: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    if not cases.empty:
        unstable = cases[
            cases["decision_stability"] < _DECISION_STABILITY_CUTOFF
        ]
        if len(unstable) >= _MIN_CASES_FOR_OOF_CALIBRATION:
            margin = _round_margin(
                float(np.median(np.abs(unstable["margin"].to_numpy(dtype=float))))
            )
            return margin, {
                "method": "patient_safe_oof_decision_stability",
                "reference_population": "patient_safe_oof_target-breast_cases",
                "target_cases": int(len(cases)),
                "unstable_target_cases": int(len(unstable)),
                "decision_stability_cutoff": _DECISION_STABILITY_CUTOFF,
                "borderline_margin_logit": margin,
            }
    return 0.5, {
        "method": "fixed_logit_margin_without_oof_calibration",
        "reference_population": "not_available",
        "target_cases": int(len(cases)),
        "unstable_target_cases": 0,
        "decision_stability_cutoff": _DECISION_STABILITY_CUTOFF,
        "borderline_margin_logit": 0.5,
    }


def _round_margin(value: float) -> float:
    rounded = round(float(value) / _MARGIN_ROUNDING) * _MARGIN_ROUNDING
    clipped = float(np.clip(rounded, _MIN_BORDERLINE_MARGIN, _MAX_BORDERLINE_MARGIN))
    return float(round(clipped, 1))


def _logit(value: float | np.ndarray) -> float | np.ndarray:
    clipped = np.clip(value, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _validate_probability(value: Any, name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TRA policy {name} must be a finite probability.") from exc
    if not np.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(f"TRA policy {name} must be strictly between 0 and 1.")
    return probability
