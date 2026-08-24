"""Shared contracts for the research-only attenuation experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any

import pandas as pd


STANDARDIZED_ATTENUATION_POSITIONS = ("P1", "P2", "P3")
VALIDATED_ATTENUATION_STATUS = "measured_validated"
ATTENUATION_VALUE_COLUMNS = (
    "attenuation_p1",
    "attenuation_p2",
    "attenuation_p3",
    "attenuation_mean",
    "attenuation_std",
    "attenuation_range",
)
ATTENUATION_SYMMETRY_COLUMNS = (
    "attenuation_delta_p1",
    "attenuation_delta_p2",
    "attenuation_delta_p3",
    "attenuation_abs_delta_p1",
    "attenuation_abs_delta_p2",
    "attenuation_abs_delta_p3",
    "attenuation_mean_delta",
    "attenuation_mean_abs_delta",
    "attenuation_rms_delta",
)
DEFAULT_ATTENUATION_EVALUATION_COLUMNS = (
    *ATTENUATION_VALUE_COLUMNS,
    *ATTENUATION_SYMMETRY_COLUMNS,
)


class AttenuationExperimentUnavailable(ValueError):
    """Raised when audited attenuation data cannot support this experiment."""


@dataclass(frozen=True)
class AttenuationFeatureResult:
    """Feature table plus explicit breast-level availability accounting."""

    features: pd.DataFrame
    coverage: pd.DataFrame
    status: str
    unavailable_reason: str = ""


@dataclass(frozen=True)
class ArchiveTransmissionAudit:
    """Raw archive inventory; it does not claim an attenuation measurement."""

    inventory: pd.DataFrame
    coverage: pd.DataFrame
    status: str
    unavailable_reason: str


@dataclass(frozen=True)
class PairedAttenuationEvaluation:
    """Patient-safe comparison of a baseline and attenuation-augmented model."""

    eligible_cases: pd.DataFrame
    coverage: pd.DataFrame
    split_metrics: pd.DataFrame
    predictions: pd.DataFrame
    paired_deltas: pd.DataFrame


def _normalize_side(value: Any) -> str | None:
    normalized = _normalized_text(value)
    if normalized.startswith("left"):
        return "LEFT"
    if normalized.startswith("right"):
        return "RIGHT"
    return None


def _normalize_position(value: Any) -> str | None:
    match = re.fullmatch(r"p([0-9]+)", _normalized_text(value))
    return f"P{match.group(1)}" if match else None


def _canonical_kind(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalized_text(value))


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", "", _text(value).strip().lower())


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _has_text(value: Any) -> bool:
    return bool(_text(value).strip())


def _numeric_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalized_text(value) in {"1", "true", "yes"}


def _first_present(
    attrs: Any,
    metadata: dict[str, Any],
    processing: dict[str, Any],
    *,
    keys: Sequence[str],
) -> Any:
    for source in (attrs, metadata, processing):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None
