from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xrd_preprocessing import (
    AgBHMonochromaticityQualityControl,
    AzimuthalIntegration,
    h5_to_df,
)


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
        out["measurement_type_name"].astype(str).str.contains(
            "Silver Behenate", case=False, na=False
        )
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
    mask = distance.notna() & np.isfinite(distance) & (distance <= float(max_distance_mm))
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
            _calibration_df["measurement_type_name"].astype(str).str.contains(
                "Silver Behenate", case=False, na=False
            )
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


def agbh_alpha_peaks(q_max: float, *, d_spacing_nm: float = AGBH_D_SPACING_NM):
    q1 = 2.0 * np.pi / float(d_spacing_nm)
    orders = np.arange(1, int(np.floor(q_max / q1)) + 1)
    return orders * q1


def agbh_beta_peaks(
    q_max: float,
    *,
    beta_ratio: float = AG_KBETA_TO_KALPHA_Q_RATIO,
    d_spacing_nm: float = AGBH_D_SPACING_NM,
):
    return agbh_alpha_peaks(q_max, d_spacing_nm=d_spacing_nm) * float(beta_ratio)


def _peak_height(q: np.ndarray, y: np.ndarray, center: float, window: float) -> float:
    peak_mask = (q >= center - window) & (q <= center + window)
    side_mask = ((q >= center - 3 * window) & (q < center - 1.5 * window)) | (
        (q > center + 1.5 * window) & (q <= center + 3 * window)
    )
    if not np.any(peak_mask):
        return np.nan
    baseline = np.nanmedian(y[side_mask]) if np.any(side_mask) else np.nanmedian(y)
    return float(np.nanmax(y[peak_mask] - baseline))


def summary_text(
    manifest_df: pd.DataFrame,
    agbh_manifest_df: pd.DataFrame,
    integrated_df: pd.DataFrame,
    reference_manifest_df: pd.DataFrame | None = None,
) -> str:
    lines = [
        f"calibration_sets_total={len(manifest_df)}",
        f"selected_agbh_sets={len(agbh_manifest_df)}",
        f"integrated_agbh_sets={len(integrated_df)}",
    ]
    for _batch, _rows in integrated_df.groupby("data_batch"):
        lines.append(f"batch={_batch}: integrated_agbh_sets={len(_rows)}")
    if reference_manifest_df is not None:
        names = ", ".join(reference_manifest_df["file_name"].astype(str).tolist())
        lines.append(f"reference_agbh_sets={len(reference_manifest_df)}")
        lines.append(f"reference_files={names}")
    return "\n".join(lines)


def plot_profiles_by_batch(
    integrated_df: pd.DataFrame,
    *,
    alpha_peaks: np.ndarray,
    beta_peaks: np.ndarray,
    q_min: float,
    q_max: float,
    max_profiles: int,
    normalize: bool,
    reference_df: pd.DataFrame | None = None,
):
    batches = sorted(integrated_df["data_batch"].astype(str).unique())
    fig, axes = plt.subplots(
        len(batches),
        1,
        figsize=(10.5, 4.0 * max(1, len(batches))),
        sharex=True,
        squeeze=False,
    )
    for _ax, _batch in zip(axes[:, 0], batches, strict=False):
        _df = integrated_df.loc[integrated_df["data_batch"].astype(str).eq(_batch)]
        for _row in _df.head(max_profiles).itertuples(index=False):
            _q = np.asarray(getattr(_row, "q_range"), dtype=float)
            _y = np.asarray(getattr(_row, "radial_profile_data"), dtype=float)
            _mask = np.isfinite(_q) & np.isfinite(_y) & (_q >= q_min) & (_q <= q_max)
            _q = _q[_mask]
            _y = _y[_mask]
            if normalize and np.nanmax(_y) > 0:
                _y = _y / np.nanmax(_y)
            _ax.plot(_q, _y, alpha=0.22, linewidth=0.8)
        if reference_df is not None and not reference_df.empty:
            for _ref_idx, _row in enumerate(
                reference_df.sort_values("started_at").itertuples(index=False),
                start=1,
            ):
                _q = np.asarray(getattr(_row, "q_range"), dtype=float)
                _y = np.asarray(getattr(_row, "radial_profile_data"), dtype=float)
                _mask = (
                    np.isfinite(_q)
                    & np.isfinite(_y)
                    & (_q >= q_min)
                    & (_q <= q_max)
                )
                _q = _q[_mask]
                _y = _y[_mask]
                if normalize and np.nanmax(_y) > 0:
                    _y = _y / np.nanmax(_y)
                _ax.plot(
                    _q,
                    _y,
                    alpha=0.95,
                    linewidth=3.0,
                    color="#16a34a" if _ref_idx == 1 else "#84cc16",
                    label=f"good K-alpha reference #{_ref_idx}",
                )
        for _peak in alpha_peaks:
            if q_min <= _peak <= q_max:
                _ax.axvline(_peak, color="#2f6fbb", alpha=0.22, linewidth=0.8)
        for _peak in beta_peaks:
            if q_min <= _peak <= q_max:
                _ax.axvline(_peak, color="#c43b3b", alpha=0.35, linewidth=0.8)
        _ax.set_title(f"AgBH integrated profiles, batch {_batch}")
        _ax.set_ylabel("normalized intensity" if normalize else "intensity")
        _ax.grid(alpha=0.18)
        if reference_df is not None and not reference_df.empty:
            _ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("q, nm^-1")
    fig.tight_layout()
    return fig


def _heatmap_matrix(
    df: pd.DataFrame,
    *,
    q_min: float,
    q_max: float,
    q_points: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grid = np.linspace(q_min, q_max, q_points)
    matrix = []
    labels = []
    for _row in df.sort_values("started_at").itertuples(index=False):
        _q = np.asarray(getattr(_row, "q_range"), dtype=float)
        _y = np.asarray(getattr(_row, "radial_profile_data"), dtype=float)
        _mask = np.isfinite(_q) & np.isfinite(_y)
        _interp = np.interp(grid, _q[_mask], _y[_mask])
        if np.nanmax(_interp) > 0:
            _interp = _interp / np.nanmax(_interp)
        matrix.append(_interp)
        labels.append(pd.Timestamp(getattr(_row, "started_at")).strftime("%Y-%m-%d"))
    return grid, np.asarray(matrix), labels


def plot_profile_heatmaps(
    integrated_df: pd.DataFrame,
    *,
    alpha_peaks: np.ndarray,
    beta_peaks: np.ndarray,
    q_min: float,
    q_max: float,
):
    batches = sorted(integrated_df["data_batch"].astype(str).unique())
    fig, axes = plt.subplots(
        len(batches),
        1,
        figsize=(10.5, 4.2 * max(1, len(batches))),
        sharex=True,
        squeeze=False,
    )
    for _ax, _batch in zip(axes[:, 0], batches, strict=False):
        _df = integrated_df.loc[integrated_df["data_batch"].astype(str).eq(_batch)]
        _grid, _matrix, _labels = _heatmap_matrix(
            _df,
            q_min=q_min,
            q_max=q_max,
            q_points=900,
        )
        _ax.imshow(
            _matrix,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=[q_min, q_max, 0, len(_labels)],
            cmap="magma",
        )
        for _peak in alpha_peaks:
            if q_min <= _peak <= q_max:
                _ax.axvline(_peak, color="#8ecae6", alpha=0.35, linewidth=0.8)
        for _peak in beta_peaks:
            if q_min <= _peak <= q_max:
                _ax.axvline(_peak, color="#ffb703", alpha=0.55, linewidth=0.8)
        _ax.set_title(f"AgBH normalized profile heatmap, batch {_batch}")
        _ax.set_ylabel("calibrations")
    axes[-1, 0].set_xlabel("q, nm^-1")
    fig.tight_layout()
    return fig


def _profile_on_grid(row, grid: np.ndarray) -> np.ndarray:
    q = np.asarray(getattr(row, "q_range"), dtype=float)
    y = np.asarray(getattr(row, "radial_profile_data"), dtype=float)
    mask = np.isfinite(q) & np.isfinite(y)
    return np.interp(grid, q[mask], y[mask])


def _normalize_profile(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float)
    out = out - np.nanmin(out)
    max_value = np.nanmax(out)
    if max_value > 0:
        out = out / max_value
    return out


def shoulder_metric_table(
    integrated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    q_min: float,
    q_max: float,
    shoulder_min: float,
    shoulder_max: float,
    q_points: int = 500,
) -> pd.DataFrame:
    grid = np.linspace(q_min, q_max, q_points)
    reference_matrix = [
        _normalize_profile(_profile_on_grid(_row, grid))
        for _row in reference_df.itertuples(index=False)
    ]
    reference_curve = np.nanmedian(np.asarray(reference_matrix), axis=0)
    shoulder_mask = (grid >= shoulder_min) & (grid <= shoulder_max)
    fit_mask = ~shoulder_mask

    rows = []
    for _row in integrated_df.sort_values(["data_batch", "started_at"]).itertuples(
        index=False
    ):
        y = _normalize_profile(_profile_on_grid(_row, grid))
        design = np.column_stack(
            [reference_curve[fit_mask], np.ones(int(np.sum(fit_mask)))]
        )
        scale, offset = np.linalg.lstsq(design, y[fit_mask], rcond=None)[0]
        fitted_reference = scale * reference_curve + offset
        residual = y - fitted_reference
        positive_residual = np.clip(residual[shoulder_mask], 0.0, None)
        shoulder_width = float(shoulder_max - shoulder_min)
        positive_area = float(np.trapezoid(positive_residual, grid[shoulder_mask]))
        mean_residual = positive_area / shoulder_width
        peak_residual = float(np.nanmax(residual[shoulder_mask]))
        rows.append(
            {
                "session_uid": getattr(_row, "session_uid"),
                "started_at": getattr(_row, "started_at"),
                "file_name": getattr(_row, "file_name"),
                "data_batch": str(getattr(_row, "data_batch")),
                "source_line": getattr(_row, "source_line"),
                "nova_range": getattr(_row, "nova_range"),
                "shoulder_q_min": float(shoulder_min),
                "shoulder_q_max": float(shoulder_max),
                "shoulder_positive_area": positive_area,
                "shoulder_mean_positive_residual": float(mean_residual),
                "shoulder_peak_residual": peak_residual,
                "reference_fit_scale": float(scale),
                "reference_fit_offset": float(offset),
            }
        )
    out = pd.DataFrame(rows)
    out["started_at"] = pd.to_datetime(out["started_at"], errors="coerce")
    return out


def _window_mask(grid: np.ndarray, center: float, half_width: float) -> np.ndarray:
    return (grid >= center - half_width) & (grid <= center + half_width)


def _agbh_peak_windows(
    q_min: float,
    q_max: float,
    *,
    beta_ratio: float,
    beta_half_width: float,
    min_order: int,
) -> list[dict[str, float | int]]:
    windows: list[dict[str, float | int]] = []
    alpha_peaks = agbh_alpha_peaks(q_max)
    for _order, _alpha_q in enumerate(alpha_peaks, start=1):
        if _order < min_order:
            continue
        _beta_q = float(_alpha_q * beta_ratio)
        _right_control_q = float(_alpha_q + (_alpha_q - _beta_q))
        if _beta_q - beta_half_width < q_min:
            continue
        if _right_control_q + beta_half_width > q_max:
            continue
        windows.append(
            {
                "order": _order,
                "alpha_q": float(_alpha_q),
                "beta_q": _beta_q,
                "right_control_q": _right_control_q,
            }
        )
    return windows


def _alpha_fit_mask(
    grid: np.ndarray,
    *,
    q_max: float,
    left_width: float,
    right_width: float,
    min_order: int,
) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    alpha_peaks = agbh_alpha_peaks(q_max)
    for _order, _alpha_q in enumerate(alpha_peaks, start=1):
        if _order >= min_order:
            mask |= (grid >= float(_alpha_q) - left_width) & (
                grid <= float(_alpha_q) + right_width
            )
    return mask


def _shift_curve(curve: np.ndarray, grid: np.ndarray, q_shift: float) -> np.ndarray:
    return np.interp(
        grid - float(q_shift),
        grid,
        curve,
        left=np.nan,
        right=np.nan,
    )


def _fit_reference_with_q_shift(
    y: np.ndarray,
    reference_curve: np.ndarray,
    grid: np.ndarray,
    fit_mask: np.ndarray,
    centered_q: np.ndarray,
    *,
    q_shift_min: float,
    q_shift_max: float,
    q_shift_steps: int,
) -> tuple[np.ndarray, float, float, float, float]:
    best: tuple[float, np.ndarray, float, float, float, float] | None = None
    for _shift in np.linspace(q_shift_min, q_shift_max, q_shift_steps):
        shifted_reference = _shift_curve(reference_curve, grid, float(_shift))
        valid_mask = fit_mask & np.isfinite(shifted_reference) & np.isfinite(y)
        if int(np.sum(valid_mask)) < 10:
            continue
        design = np.column_stack(
            [
                shifted_reference[valid_mask],
                np.ones(int(np.sum(valid_mask))),
                centered_q[valid_mask],
            ]
        )
        scale, offset, slope = np.linalg.lstsq(design, y[valid_mask], rcond=None)[0]
        fitted = scale * shifted_reference + offset + slope * centered_q
        residual = y[valid_mask] - fitted[valid_mask]
        sse = float(np.nanmean(residual**2))
        if best is None or sse < best[0]:
            best = (sse, fitted, float(_shift), float(scale), float(offset), float(slope))
    if best is None:
        raise ValueError("Could not fit reference with q shift")
    _, fitted, q_shift, scale, offset, slope = best
    return fitted, q_shift, scale, offset, slope


def kbeta_left_shoulder_metric_table(
    integrated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    q_min: float,
    q_max: float,
    beta_ratio: float = AG_KBETA_TO_KALPHA_Q_RATIO,
    beta_half_width: float = 0.12,
    min_order: int = 3,
    alpha_fit_left_width: float = 0.03,
    alpha_fit_right_width: float = 0.18,
    q_shift_min: float = 0.0,
    q_shift_max: float = 0.0,
    q_shift_steps: int = 1,
    q_points: int = 1800,
) -> pd.DataFrame:
    grid = np.linspace(q_min, q_max, q_points)
    reference_matrix = [
        _normalize_profile(_profile_on_grid(_row, grid))
        for _row in reference_df.itertuples(index=False)
    ]
    reference_curve = np.nanmedian(np.asarray(reference_matrix), axis=0)
    windows = _agbh_peak_windows(
        q_min,
        q_max,
        beta_ratio=beta_ratio,
        beta_half_width=beta_half_width,
        min_order=min_order,
    )
    beta_mask = np.zeros_like(grid, dtype=bool)
    for _window in windows:
        beta_mask |= _window_mask(grid, float(_window["beta_q"]), beta_half_width)

    fit_mask = _alpha_fit_mask(
        grid,
        q_max=q_max,
        left_width=alpha_fit_left_width,
        right_width=alpha_fit_right_width,
        min_order=min_order,
    )
    fit_mask &= ~beta_mask
    centered_q = grid - float(np.nanmean(grid))

    rows = []
    for _row in integrated_df.sort_values(["data_batch", "started_at"]).itertuples(
        index=False
    ):
        y = _normalize_profile(_profile_on_grid(_row, grid))
        fitted_reference, q_shift, scale, offset, slope = _fit_reference_with_q_shift(
            y,
            reference_curve,
            grid,
            fit_mask,
            centered_q,
            q_shift_min=q_shift_min,
            q_shift_max=q_shift_max,
            q_shift_steps=q_shift_steps,
        )
        residual = y - fitted_reference
        left_area_sum = 0.0
        right_control_area_sum = 0.0
        peak_rows = []
        for _window in windows:
            _left_mask = _window_mask(
                grid,
                float(_window["beta_q"]),
                beta_half_width,
            )
            _right_mask = _window_mask(
                grid,
                float(_window["right_control_q"]),
                beta_half_width,
            )
            _left_residual = np.clip(residual[_left_mask], 0.0, None)
            _right_residual = np.clip(residual[_right_mask], 0.0, None)
            _left_area = float(np.trapezoid(_left_residual, grid[_left_mask]))
            _right_area = float(np.trapezoid(_right_residual, grid[_right_mask]))
            left_area_sum += _left_area
            right_control_area_sum += _right_area
            peak_rows.append(
                {
                    "order": int(_window["order"]),
                    "alpha_q": float(_window["alpha_q"]),
                    "beta_q": float(_window["beta_q"]),
                    "right_control_q": float(_window["right_control_q"]),
                    "left_area": _left_area,
                    "right_control_area": _right_area,
                }
            )
        net_area = max(left_area_sum - right_control_area_sum, 0.0)
        window_width_total = 2.0 * beta_half_width * max(1, len(windows))
        rows.append(
            {
                "session_uid": getattr(_row, "session_uid"),
                "started_at": getattr(_row, "started_at"),
                "file_name": getattr(_row, "file_name"),
                "data_batch": str(getattr(_row, "data_batch")),
                "source_line": getattr(_row, "source_line"),
                "nova_range": getattr(_row, "nova_range"),
                "kbeta_left_net_area": float(net_area),
                "kbeta_left_mean_net_residual": float(net_area / window_width_total),
                "kbeta_left_positive_area": float(left_area_sum),
                "right_control_positive_area": float(right_control_area_sum),
                "n_kbeta_windows": len(windows),
                "window_orders": ",".join(str(_item["order"]) for _item in windows),
                "beta_ratio": float(beta_ratio),
                "beta_half_width": float(beta_half_width),
                "alpha_fit_left_width": float(alpha_fit_left_width),
                "alpha_fit_right_width": float(alpha_fit_right_width),
                "reference_q_shift_nm_inv": float(q_shift),
                "reference_fit_scale": float(scale),
                "reference_fit_offset": float(offset),
                "reference_fit_linear_background": float(slope),
                "peak_window_details": json.dumps(peak_rows),
            }
        )
    out = pd.DataFrame(rows)
    out["started_at"] = pd.to_datetime(out["started_at"], errors="coerce")
    return out


def score_agbh_monochromaticity(
    integrated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    max_score: float,
    q_min: float,
    q_max: float,
    beta_ratio: float = AG_KBETA_TO_KALPHA_Q_RATIO,
    beta_half_width: float = 0.12,
    min_order: int = 3,
    q_points: int = 1800,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    qc = AgBHMonochromaticityQualityControl(
        reference_df=reference_df,
        id_column="session_uid",
        date_column="started_at",
        max_score=float(max_score),
        q_min=float(q_min),
        q_max=float(q_max),
        q_points=int(q_points),
        beta_ratio=float(beta_ratio),
        beta_half_width=float(beta_half_width),
        min_order=int(min_order),
    )
    scored_df = qc.fit_transform(integrated_df)
    manifest_df = qc.selection_.manifest_columns()
    return scored_df, qc.stats_, manifest_df


def agbh_score_summary_by_batch(scored_df: pd.DataFrame) -> pd.DataFrame:
    out = scored_df.copy()
    out["data_batch"] = out["data_batch"].astype(str)
    grouped = out.groupby("data_batch", dropna=False)
    return grouped.agg(
        n=("agbh_monochromaticity_score", "size"),
        accepted=("agbh_monochromaticity_pass", "sum"),
        rejected=("agbh_monochromaticity_pass", lambda _x: int((~_x.astype(bool)).sum())),
        score_min=("agbh_monochromaticity_score", "min"),
        score_median=("agbh_monochromaticity_score", "median"),
        score_mean=("agbh_monochromaticity_score", "mean"),
        score_max=("agbh_monochromaticity_score", "max"),
    ).reset_index()


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


def build_aramis_preprocessing_config(
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
        "schema_version": "aramis_preprocessing_v0_1",
        "product": "Aramis",
        "clinical_stage": "research draft",
        "purpose": (
            "AgBH monochromaticity QC config for downstream Aramis data selection; "
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
                "Aramis product preprocessing requires detector distance short enough "
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


def write_aramis_preprocessing_config(config: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(path)


def plot_shoulder_residuals(
    integrated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    q_min: float,
    q_max: float,
    shoulder_min: float,
    shoulder_max: float,
    q_points: int = 500,
):
    grid = np.linspace(q_min, q_max, q_points)
    reference_matrix = [
        _normalize_profile(_profile_on_grid(_row, grid))
        for _row in reference_df.itertuples(index=False)
    ]
    reference_curve = np.nanmedian(np.asarray(reference_matrix), axis=0)
    shoulder_mask = (grid >= shoulder_min) & (grid <= shoulder_max)
    fit_mask = ~shoulder_mask

    batches = sorted(integrated_df["data_batch"].astype(str).unique())
    fig, axes = plt.subplots(
        len(batches),
        1,
        figsize=(10.5, 3.8 * max(1, len(batches))),
        sharex=True,
        squeeze=False,
    )
    for _ax, _batch in zip(axes[:, 0], batches, strict=False):
        _df = integrated_df.loc[integrated_df["data_batch"].astype(str).eq(_batch)]
        for _row in _df.sort_values("started_at").itertuples(index=False):
            y = _normalize_profile(_profile_on_grid(_row, grid))
            design = np.column_stack(
                [reference_curve[fit_mask], np.ones(int(np.sum(fit_mask)))]
            )
            scale, offset = np.linalg.lstsq(design, y[fit_mask], rcond=None)[0]
            residual = y - (scale * reference_curve + offset)
            _ax.plot(grid, residual, alpha=0.28, linewidth=0.9)
        _ax.axhline(0.0, color="#111827", alpha=0.35, linewidth=0.8)
        _ax.axvspan(shoulder_min, shoulder_max, color="#e11d48", alpha=0.12)
        _ax.set_title(f"Residual vs trusted K-alpha reference, batch {_batch}")
        _ax.set_ylabel("residual")
        _ax.grid(alpha=0.18)
    axes[-1, 0].set_xlabel("q, nm^-1")
    fig.tight_layout()
    return fig


def plot_shoulder_metric(
    metric_df: pd.DataFrame,
    *,
    good_metric_df: pd.DataFrame | None = None,
):
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    for _batch, _df in metric_df.groupby("data_batch"):
        ax.scatter(
            _df["started_at"],
            _df["shoulder_mean_positive_residual"],
            label=f"batch {_batch}",
            s=34,
            alpha=0.86,
        )
        ax.plot(
            _df["started_at"],
            _df["shoulder_mean_positive_residual"],
            alpha=0.45,
        )
    if good_metric_df is not None and not good_metric_df.empty:
        good_values = good_metric_df["shoulder_mean_positive_residual"]
        ax.scatter(
            good_metric_df["started_at"],
            good_values,
            marker="*",
            s=150,
            color="#16a34a",
            edgecolor="#064e3b",
            linewidth=0.8,
            label="good K-alpha reference",
            zorder=5,
        )
        ax.axhline(
            float(good_values.median()),
            color="#16a34a",
            linewidth=1.8,
            linestyle="--",
            alpha=0.85,
        )
    ax.set_title("AgBH K-beta shoulder metric")
    ax.set_xlabel("calibration date")
    ax.set_ylabel("mean positive residual in shoulder window")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_shoulder_metric_by_batch_panels(
    metric_df: pd.DataFrame,
    *,
    good_metric_df: pd.DataFrame | None = None,
    reference_manifest_df: pd.DataFrame | None = None,
    reference_label: str | None = None,
    threshold: float | None = None,
    metric_column: str = "shoulder_mean_positive_residual",
    metric_label: str = "mean positive residual, q=2.55..3.05 nm^-1",
):
    out = metric_df.copy()
    out["started_at"] = pd.to_datetime(out["started_at"], errors="coerce")
    out["data_batch"] = out["data_batch"].astype(str)
    batches = sorted(
        out["data_batch"].dropna().unique(),
        key=lambda _x: (_x == "None", _x),
    )
    fig, axes = plt.subplots(
        len(batches),
        1,
        figsize=(13.5, 2.2 * max(1, len(batches))),
        sharey=True,
        squeeze=False,
    )
    good_min = None
    good_max = None
    if good_metric_df is not None and not good_metric_df.empty:
        good_values = good_metric_df[metric_column].astype(float)
        good_min = float(good_values.min())
        good_max = float(good_values.max())
    reference_text = "Reference: not provided"
    if reference_manifest_df is not None and not reference_manifest_df.empty:
        _reference = reference_manifest_df.sort_values("started_at").copy()
        _reference["started_at"] = pd.to_datetime(
            _reference["started_at"],
            errors="coerce",
        )
        _dates = ", ".join(_reference["started_at"].dt.strftime("%Y-%m-%d %H:%M"))
        _files = ", ".join(_reference["file_name"].astype(str).tolist())
        _batches = ", ".join(_reference["data_batch"].astype(str).unique())
        _label = reference_label or f"batch {_batches}"
        reference_text = (
            f"Reference green band: {_label}; dates: {_dates}; files: {_files}"
        )
    if good_min is not None and good_max is not None:
        reference_text = (
            f"{reference_text}; metric range={good_min:.4f}..{good_max:.4f}"
        )
    for _ax, _batch in zip(axes[:, 0], batches, strict=False):
        _df = out.loc[out["data_batch"].eq(_batch)].sort_values("started_at")
        _x = np.arange(len(_df))
        _y = _df[metric_column].astype(float).to_numpy()
        if good_min is not None and good_max is not None:
            _ax.axhspan(good_min, good_max, color="#16a34a", alpha=0.16)
        if threshold is not None:
            _ax.axhline(
                float(threshold),
                color="#dc2626",
                linewidth=1.2,
                linestyle="--",
                alpha=0.82,
            )
        _ax.plot(_x, _y, color="#2563eb", linewidth=1.2, alpha=0.72)
        _ax.scatter(_x, _y, color="#1d4ed8", s=28, zorder=3)
        _labels = _df["started_at"].dt.strftime("%Y-%m-%d").tolist()
        _tick_step = max(1, int(np.ceil(len(_labels) / 14)))
        _ticks = _x[::_tick_step]
        _ax.set_xticks(_ticks)
        _ax.set_xticklabels(
            [_labels[int(_idx)] for _idx in _ticks],
            rotation=35,
            ha="right",
        )
        _source_lines = ", ".join(
            sorted(_df["source_line"].dropna().astype(str).unique())
        )
        _line_label = (
            "k-alpha (product JSON)"
            if _batch in {"7", "7.0"}
            else _source_lines
        )
        _ax.set_ylabel(f"batch {_batch}\n{_line_label}")
        _ax.grid(axis="y", alpha=0.18)
        _ax.grid(axis="x", alpha=0.08)
    axes[0, 0].set_title(f"AgBH K-beta shoulder metric by batch\n{reference_text}")
    axes[-1, 0].set_xlabel("calibration day within batch")
    fig.supylabel(metric_label)
    fig.tight_layout()
    return fig


def kbeta_left_metric_diagnostic_figures(
    integrated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    *,
    q_min: float = 2.0,
    q_max: float = 6.3,
    beta_ratio: float = AG_KBETA_TO_KALPHA_Q_RATIO,
    beta_half_width: float = 0.12,
    min_order: int = 3,
    alpha_fit_left_width: float = 0.03,
    alpha_fit_right_width: float = 0.18,
    q_points: int = 1200,
):
    grid = np.linspace(q_min, q_max, q_points)
    reference_matrix = [
        _normalize_profile(_profile_on_grid(_row, grid))
        for _row in reference_df.itertuples(index=False)
    ]
    reference_curve = np.nanmedian(np.asarray(reference_matrix), axis=0)
    windows = _agbh_peak_windows(
        q_min,
        q_max,
        beta_ratio=beta_ratio,
        beta_half_width=beta_half_width,
        min_order=min_order,
    )
    beta_mask = np.zeros_like(grid, dtype=bool)
    right_mask = np.zeros_like(grid, dtype=bool)
    for _window in windows:
        beta_mask |= _window_mask(grid, float(_window["beta_q"]), beta_half_width)
        right_mask |= _window_mask(
            grid,
            float(_window["right_control_q"]),
            beta_half_width,
        )
    fit_mask = _alpha_fit_mask(
        grid,
        q_max=q_max,
        left_width=alpha_fit_left_width,
        right_width=alpha_fit_right_width,
        min_order=min_order,
    )
    fit_mask &= ~beta_mask
    centered_q = grid - float(np.nanmean(grid))

    bad_metric = metric_df.sort_values(
        "kbeta_left_mean_net_residual",
        ascending=False,
    ).iloc[0]
    selected_rows = []
    bad_rows = integrated_df.loc[
        integrated_df["session_uid"].astype(str).eq(str(bad_metric["session_uid"]))
    ]
    selected_rows.append(bad_rows.iloc[0])
    selected_rows.append(reference_df.sort_values("started_at").iloc[-1])

    profiles = []
    for _row in selected_rows:
        y = _normalize_profile(_profile_on_grid(_row, grid))
        fitted, q_shift, _scale, _offset, _slope = _fit_reference_with_q_shift(
            y,
            reference_curve,
            grid,
            fit_mask,
            centered_q,
            q_shift_min=0.0,
            q_shift_max=0.0,
            q_shift_steps=1,
        )
        residual = y - fitted
        left_area = float(np.trapezoid(np.clip(residual[beta_mask], 0, None), grid[beta_mask]))
        right_area = float(
            np.trapezoid(np.clip(residual[right_mask], 0, None), grid[right_mask])
        )
        profiles.append(
            {
                "row": _row,
                "y": y,
                "fitted": fitted,
                "residual": residual,
                "q_shift": q_shift,
                "left_area": left_area,
                "right_area": right_area,
                "net": max(left_area - right_area, 0.0),
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharex="col")
    titles = ["High left residual / likely K-beta", "Batch 7 reference-like"]
    for _ax_col, _profile, _title in zip(axes.T, profiles, titles, strict=False):
        _row = _profile["row"]
        _ax = _ax_col[0]
        _ax.plot(grid, _profile["y"], color="#2563eb", linewidth=2.0, label="measured profile")
        _ax.plot(
            grid,
            reference_curve,
            color="#16a34a",
            linewidth=1.4,
            alpha=0.7,
            label="raw batch-7 K-alpha reference",
        )
        _ax.plot(
            grid,
            _profile["fitted"],
            color="#111827",
            linewidth=1.8,
            linestyle="--",
            label="fitted K-alpha reference, q fixed",
        )
        for _window in windows:
            _alpha_q = float(_window["alpha_q"])
            _beta_q = float(_window["beta_q"])
            _right_q = float(_window["right_control_q"])
            _ax.axvline(_alpha_q, color="#64748b", linewidth=0.8, alpha=0.35)
            _ax.axvspan(
                _beta_q - beta_half_width,
                _beta_q + beta_half_width,
                color="#ef4444",
                alpha=0.16,
            )
            _ax.axvspan(
                _right_q - beta_half_width,
                _right_q + beta_half_width,
                color="#22c55e",
                alpha=0.12,
            )
            _ax.text(
                _beta_q,
                1.02,
                f"β{int(_window['order'])}",
                color="#b91c1c",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        _started_at = pd.Timestamp(getattr(_row, "started_at")).strftime("%Y-%m-%d %H:%M")
        _ax.set_title(
            f"{_title}\n"
            f"batch={getattr(_row, 'data_batch')} date={_started_at} "
            f"q_shift={_profile['q_shift']:+.4f}"
        )
        _ax.set_ylabel("normalized intensity")
        _ax.grid(alpha=0.16)
        _ax.legend(fontsize=8, loc="upper right")

        _residual_ax = _ax_col[1]
        _residual_ax.plot(grid, _profile["residual"], color="#7c3aed", linewidth=1.4)
        _residual_ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.45)
        for _window in windows:
            _beta_q = float(_window["beta_q"])
            _right_q = float(_window["right_control_q"])
            _residual_ax.axvspan(
                _beta_q - beta_half_width,
                _beta_q + beta_half_width,
                color="#ef4444",
                alpha=0.16,
            )
            _residual_ax.axvspan(
                _right_q - beta_half_width,
                _right_q + beta_half_width,
                color="#22c55e",
                alpha=0.12,
            )
        _residual_ax.set_title(
            f"residual: left area={_profile['left_area']:.4f}, "
            f"right control={_profile['right_area']:.4f}, net={_profile['net']:.4f}"
        )
        _residual_ax.set_xlabel("q, nm^-1")
        _residual_ax.set_ylabel("measured - fitted")
        _residual_ax.grid(alpha=0.16)
    fig.suptitle(
        "K-beta left metric: red=smaller-q K-beta windows, "
        "green=right-side control, K-alpha q fixed",
        y=1.02,
    )
    fig.tight_layout()

    zoom_fig, zoom_ax = plt.subplots(figsize=(10.5, 4.2))
    bad_profile = profiles[0]
    zoom_ax.plot(
        grid,
        bad_profile["y"],
        color="#2563eb",
        linewidth=2.4,
        label="measured high-residual profile",
    )
    zoom_ax.plot(
        grid,
        reference_curve,
        color="#16a34a",
        linewidth=1.6,
        alpha=0.8,
        label="raw batch-7 K-alpha reference",
    )
    zoom_ax.plot(
        grid,
        bad_profile["fitted"],
        color="#111827",
        linewidth=2.0,
        linestyle="--",
        label="fitted K-alpha reference, q fixed",
    )
    for _window in windows:
        _alpha_q = float(_window["alpha_q"])
        _beta_q = float(_window["beta_q"])
        _right_q = float(_window["right_control_q"])
        zoom_ax.axvline(_alpha_q, color="#64748b", linewidth=0.8, alpha=0.35)
        zoom_ax.axvspan(
            _beta_q - beta_half_width,
            _beta_q + beta_half_width,
            color="#ef4444",
            alpha=0.16,
        )
        zoom_ax.axvspan(
            _right_q - beta_half_width,
            _right_q + beta_half_width,
            color="#22c55e",
            alpha=0.12,
        )
        zoom_ax.text(
            _beta_q,
            1.02,
            f"β{int(_window['order'])}",
            color="#b91c1c",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    zoom_ax.set_xlim(q_min, q_max)
    zoom_ax.set_ylim(-0.03, 1.08)
    zoom_ax.set_title(
        "Zoom: K-alpha q fixed; left broadening/left shift is counted as K-beta evidence"
    )
    zoom_ax.set_xlabel("q, nm^-1")
    zoom_ax.set_ylabel("normalized intensity")
    zoom_ax.grid(alpha=0.16)
    zoom_ax.legend(fontsize=8, loc="upper right")
    zoom_fig.tight_layout()
    return fig, zoom_fig


def write_outputs(
    *,
    out_dir: Path,
    manifest_df: pd.DataFrame,
    agbh_manifest_df: pd.DataFrame,
    integrated_df: pd.DataFrame,
    reference_manifest_df: pd.DataFrame | None = None,
    shoulder_metric_df: pd.DataFrame | None = None,
    good_signal_metric_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": out_dir / "calibration_manifest.csv",
        "selected_agbh": out_dir / "selected_agbh.csv",
        "integrated_agbh": out_dir / "integrated_agbh.csv",
        "reference_agbh": out_dir / "reference_agbh_batch7_latest_april.csv",
        "shoulder_metric": out_dir / "agbh_kbeta_shoulder_metric.csv",
        "good_signal_metric": out_dir / "good_signal_reference_shoulder_metric.csv",
    }
    integrated_columns = [
        "session_uid",
        "started_at",
        "file_name",
        "data_batch",
        "source_line",
        "nova_range",
        "measurement_type_name",
        "measurement_data_source",
        "calculated_distance",
    ]
    manifest_df.to_csv(paths["manifest"], index=False)
    agbh_manifest_df.to_csv(paths["selected_agbh"], index=False)
    integrated_df[integrated_columns].to_csv(paths["integrated_agbh"], index=False)
    if reference_manifest_df is not None:
        reference_manifest_df.to_csv(paths["reference_agbh"], index=False)
    if shoulder_metric_df is not None:
        shoulder_metric_df.to_csv(paths["shoulder_metric"], index=False)
    if good_signal_metric_df is not None:
        good_signal_metric_df.to_csv(paths["good_signal_metric"], index=False)
    return {key: str(value) for key, value in paths.items()}
