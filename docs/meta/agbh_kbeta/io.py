"""Calibration discovery, H5 loading, and AgBH integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from xrd_preprocessing import AzimuthalIntegration, h5_to_df


AGBH_D_SPACING_NM = 5.838
AG_KBETA_TO_KALPHA_Q_RATIO = 0.886
DEFAULT_AGBH_THICKNESS_CHANGE_DATE = "2026-05-01"
DEFAULT_AGBH_THICKNESS_BEFORE_MM = 40.0
DEFAULT_AGBH_THICKNESS_AFTER_MM = 10.0
DEFAULT_AGBH_MAX_DISTANCE_MM = 500.0
DEFAULT_AGBH_DISTANCE_REFERENCE_BATCH = "7"


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_dataset(group: h5py.Group, name: str) -> dict[str, Any]:
    if name not in group:
        return {}
    value = _decode(group[name][()])
    return json.loads(value)


def _scalar_fields(group: h5py.Group) -> dict[str, Any]:
    out = {}
    for _name, _obj in group.items():
        if isinstance(_obj, h5py.Dataset) and _obj.shape == ():
            out[_name] = _decode(_obj[()])
    return out


def load_product_versioning(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scan_calibration_manifest(calibration_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _path in sorted(Path(calibration_dir).glob("*.h5")):
        with h5py.File(_path, "r") as _h5:
            _session = _h5["session"]
            for _set_name in sorted(_session["sets"]):
                _set_group = _session["sets"][_set_name]
                _metadata = _json_dataset(_set_group, "metadata")
                _provenance = _metadata.get("backfill_provenance", {})
                _acquisition = (
                    _scalar_fields(_set_group["acquisition"])
                    if "acquisition" in _set_group
                    else {}
                )
                rows.append(
                    {
                        "file_path": str(_path),
                        "file_name": _path.name,
                        "session_uid": _decode(_session.attrs.get("session_uid")),
                        "started_at": _decode(_session.attrs.get("started_at")),
                        "set_name": _set_name,
                        "measurement_type_name": _decode(
                            _set_group.attrs.get("measurement_type_name")
                        ),
                        "measurement_type_category": _decode(
                            _set_group.attrs.get("measurement_type_category")
                        ),
                        "data_batch": _provenance.get("data_batch"),
                        "source_line": _provenance.get("source_line"),
                        "nova_range": _provenance.get("nova_range"),
                        "human1_version": _provenance.get("human1_version"),
                        "protocol_version": _provenance.get("protocol_version"),
                        "distance_mm": _acquisition.get("distance"),
                        "exposure_time_s": _acquisition.get("exposure_time"),
                        "has_poni": "artifacts" in _set_group
                        and "poni" in _set_group["artifacts"],
                        "has_raw_file": "measurements" in _set_group
                        and any(
                            "raw_file" in _set_group["measurements"][_name]
                            for _name in _set_group["measurements"]
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    out["started_at"] = pd.to_datetime(out["started_at"], errors="coerce")
    return out.sort_values(["started_at", "file_name", "set_name"]).reset_index(
        drop=True
    )


def add_agbh_calibrant_thickness(
    df: pd.DataFrame,
    *,
    change_date: str = DEFAULT_AGBH_THICKNESS_CHANGE_DATE,
    before_mm: float = DEFAULT_AGBH_THICKNESS_BEFORE_MM,
    after_mm: float = DEFAULT_AGBH_THICKNESS_AFTER_MM,
) -> pd.DataFrame:
    out = df.copy()
    started_at = pd.to_datetime(out["started_at"], errors="coerce")
    cutoff = pd.Timestamp(change_date)
    out["agbh_calibrant_thickness_mm"] = np.where(
        started_at < cutoff,
        float(before_mm),
        float(after_mm),
    )
    out.loc[started_at.isna(), "agbh_calibrant_thickness_mm"] = np.nan
    out["agbh_calibrant_thickness_rule"] = (
        f"started_at < {cutoff.date().isoformat()}: {float(before_mm)} mm; "
        f"started_at >= {cutoff.date().isoformat()}: {float(after_mm)} mm"
    )
    return out


def agbh_manifest(
    manifest_df: pd.DataFrame,
    *,
    batches: list[str],
    max_files_per_batch: int | None = None,
) -> pd.DataFrame:
    out = manifest_df.copy()
    out = out.loc[
        out["measurement_type_name"]
        .astype(str)
        .str.contains("Silver Behenate", case=False, na=False)
    ]
    out = out.loc[out["data_batch"].astype(str).isin([str(_b) for _b in batches])]
    out = out.loc[out["has_poni"] & out["has_raw_file"]]
    out = out.sort_values(["data_batch", "started_at", "file_name"])
    if max_files_per_batch is not None:
        out = out.groupby("data_batch", group_keys=False).head(max_files_per_batch)
    return out.reset_index(drop=True)


def filter_agbh_distance(
    agbh_manifest_df: pd.DataFrame,
    *,
    max_distance_mm: float = DEFAULT_AGBH_MAX_DISTANCE_MM,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = agbh_manifest_df.copy()
    distance = pd.to_numeric(out["distance_mm"], errors="coerce")
    mask = (
        distance.notna() & np.isfinite(distance) & (distance <= float(max_distance_mm))
    )
    filtered = out.loc[mask].copy()
    stats = {
        "rows_in": int(len(out)),
        "rows_pass": int(len(filtered)),
        "rows_dropped": int(len(out) - len(filtered)),
        "max_distance_mm": float(max_distance_mm),
        "dropped_batches": sorted(
            out.loc[~mask, "data_batch"].dropna().astype(str).unique().tolist()
        ),
    }
    return filtered.reset_index(drop=True), stats


def latest_reference_agbh_manifest(
    manifest_df: pd.DataFrame,
    *,
    batch: str,
    date_max: str,
    count: int,
) -> pd.DataFrame:
    out = agbh_manifest(
        manifest_df,
        batches=[str(batch)],
        max_files_per_batch=None,
    )
    max_day = pd.to_datetime(date_max, errors="raise").normalize()
    out = out.loc[out["started_at"].dt.normalize() <= max_day]
    out = out.sort_values(["started_at", "file_name"]).tail(int(count))
    return out.reset_index(drop=True)


def load_agbh_frames(agbh_manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    manifest_by_file = agbh_manifest_df.set_index("file_path", drop=False)
    for _file_path in agbh_manifest_df["file_path"].tolist():
        _calibration_df, _measurement_df = h5_to_df(
            _file_path,
            data_preference="gfrm",
            require_clinical_ids=False,
            set_category="CALIBRATION",
        )
        _ = _measurement_df
        _calibration_df = _calibration_df.loc[
            _calibration_df["measurement_type_name"]
            .astype(str)
            .str.contains("Silver Behenate", case=False, na=False)
        ].copy()
        _manifest_row = manifest_by_file.loc[_file_path]
        for _column in [
            "data_batch",
            "source_line",
            "nova_range",
            "human1_version",
            "protocol_version",
            "file_name",
            "distance_mm",
            "agbh_calibrant_thickness_mm",
            "agbh_calibrant_thickness_rule",
        ]:
            if _column in _manifest_row.index:
                _calibration_df[_column] = _manifest_row[_column]
        rows.append(_calibration_df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def integrate_agbh_profiles(
    agbh_df: pd.DataFrame,
    *,
    npt: int,
) -> pd.DataFrame:
    integrator = AzimuthalIntegration(
        npt=int(npt),
        calibration_mode="poni",
        error_model="poisson",
        thickness_adjustment=False,
        require_thickness_adjustment=False,
    )
    return integrator.fit_transform(agbh_df)
