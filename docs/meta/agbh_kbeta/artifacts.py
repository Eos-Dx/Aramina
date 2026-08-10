"""Controlled preprocessing-config artifact builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(_key): _json_ready(_value) for _key, _value in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(_item) for _item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _records_for_config(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    existing_columns = [column for column in columns if column in df.columns]
    out = df.loc[:, existing_columns].copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return [_json_ready(record) for record in out.to_dict(orient="records")]


def build_aramina_preprocessing_config(
    *,
    scored_df: pd.DataFrame,
    qc_stats: dict[str, Any],
    calibration_dir: Path,
    selected_batches: list[str],
    reference_manifest_df: pd.DataFrame,
    output_paths: dict[str, str],
    max_score: float,
    q_min: float,
    q_max: float,
    beta_ratio: float,
    beta_half_width: float,
    npt: int,
    max_agbh_distance_mm: float | None = None,
    distance_reference_batch: str | None = None,
    distance_filter_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scored = scored_df.copy()
    scored["started_at"] = pd.to_datetime(scored["started_at"], errors="coerce")
    scored["calibration_day"] = scored["started_at"].dt.date.astype(str)
    accepted = scored.loc[scored["agbh_monochromaticity_pass"].astype(bool)].copy()
    rejected = scored.loc[~scored["agbh_monochromaticity_pass"].astype(bool)].copy()
    accepted_dates = sorted(accepted["calibration_day"].dropna().astype(str).unique())
    rejected_dates = sorted(rejected["calibration_day"].dropna().astype(str).unique())
    accepted_ids = sorted(accepted["session_uid"].dropna().astype(str).unique())
    rejected_ids = sorted(rejected["session_uid"].dropna().astype(str).unique())
    thickness_rules = (
        sorted(scored["agbh_calibrant_thickness_rule"].dropna().astype(str).unique())
        if "agbh_calibrant_thickness_rule" in scored.columns
        else []
    )
    columns = [
        "session_uid",
        "started_at",
        "calibration_day",
        "file_name",
        "data_batch",
        "source_line",
        "nova_range",
        "distance_mm",
        "calculated_distance",
        "agbh_calibrant_thickness_mm",
        "agbh_calibrant_thickness_rule",
        "agbh_monochromaticity_score",
        "agbh_monochromaticity_pass",
        "agbh_monochromaticity_status",
        "agbh_kbeta_left_net_area",
        "agbh_kbeta_left_positive_area",
        "agbh_kbeta_right_control_positive_area",
        "agbh_kbeta_n_windows",
        "agbh_kbeta_window_orders",
    ]
    return {
        "schema_version": "aramina_preprocessing_v0_1",
        "product": "Aramina",
        "clinical_stage": "research draft",
        "purpose": (
            "AgBH monochromaticity QC config for downstream Aramina data selection; "
            "not for autonomous diagnosis."
        ),
        "source": {
            "calibration_dir": str(calibration_dir),
            "data_preference": "gfrm",
            "integration": "xrd_preprocessing.AzimuthalIntegration",
            "monochromaticity_qc": "xrd_preprocessing.AgBHMonochromaticityQualityControl",
        },
        "parameters": {
            "selected_batches": [str(batch) for batch in selected_batches],
            "max_score": float(max_score),
            "q_min_nm_inv": float(q_min),
            "q_max_nm_inv": float(q_max),
            "beta_ratio": float(beta_ratio),
            "beta_half_width_nm_inv": float(beta_half_width),
            "npt": int(npt),
            "max_agbh_distance_mm": (
                None if max_agbh_distance_mm is None else float(max_agbh_distance_mm)
            ),
        },
        "product_distance_q_range_policy": {
            "required_q_max_nm_inv": 20.0,
            "reference_batch": distance_reference_batch,
            "max_agbh_distance_mm": (
                None if max_agbh_distance_mm is None else float(max_agbh_distance_mm)
            ),
            "distance_filter_stats": _json_ready(distance_filter_stats or {}),
            "excluded_batches_by_distance": (
                (distance_filter_stats or {}).get("dropped_batches", [])
            ),
            "reason": (
                "Aramina product preprocessing requires detector distance short enough "
                "to provide q-range coverage above 20 nm^-1. Batch 7 distance is "
                "used as the current acceptable reference; longer-distance AgBH days "
                "are excluded before downstream measurement selection."
            ),
        },
        "agbh_calibrant_thickness": {
            "unit": "mm",
            "rules": thickness_rules,
            "field": "agbh_calibrant_thickness_mm",
        },
        "reference_agbh": _records_for_config(
            reference_manifest_df,
            [
                "session_uid",
                "started_at",
                "file_name",
                "data_batch",
                "source_line",
                "nova_range",
                "distance_mm",
                "agbh_calibrant_thickness_mm",
                "agbh_calibrant_thickness_rule",
            ],
        ),
        "selection": {
            "rows_total": int(len(scored)),
            "rows_accepted": int(len(accepted)),
            "rows_rejected": int(len(rejected)),
            "accepted_dates": accepted_dates,
            "rejected_dates": rejected_dates,
            "accepted_session_uids": accepted_ids,
            "rejected_session_uids": rejected_ids,
            "h5_date_filter": {
                "column": "started_at",
                "op": "date in",
                "values": accepted_dates,
            },
            "h5_id_filter": {
                "column": "linked_agbh_session_uid",
                "op": "in",
                "values": accepted_ids,
            },
            "measurement_mapping_policy": (
                "Prefer explicit linked AgBH/session ID if present in product H5; "
                "otherwise use calibration-day date filter."
            ),
        },
        "qc_stats": _json_ready(qc_stats),
        "accepted_calibrations": _records_for_config(accepted, columns),
        "rejected_calibrations": _records_for_config(rejected, columns),
        "artifacts": output_paths,
    }


def write_aramina_preprocessing_config(config: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(path)
