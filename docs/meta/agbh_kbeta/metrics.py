"""AgBH monochromaticity metrics and primary diagnostic plots."""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xrd_preprocessing import AgBHMonochromaticityQualityControl

from .io import AGBH_D_SPACING_NM, AG_KBETA_TO_KALPHA_Q_RATIO


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
                    np.isfinite(_q) & np.isfinite(_y) & (_q >= q_min) & (_q <= q_max)
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
            best = (
                sse,
                fitted,
                float(_shift),
                float(scale),
                float(offset),
                float(slope),
            )
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
        rejected=(
            "agbh_monochromaticity_pass",
            lambda _x: int((~_x.astype(bool)).sum()),
        ),
        score_min=("agbh_monochromaticity_score", "min"),
        score_median=("agbh_monochromaticity_score", "median"),
        score_mean=("agbh_monochromaticity_score", "mean"),
        score_max=("agbh_monochromaticity_score", "max"),
    ).reset_index()
