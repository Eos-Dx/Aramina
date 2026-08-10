"""Secondary AgBH diagnostics and exported audit tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import AG_KBETA_TO_KALPHA_Q_RATIO
from .metrics import (
    _agbh_peak_windows,
    _alpha_fit_mask,
    _fit_reference_with_q_shift,
    _normalize_profile,
    _profile_on_grid,
    _window_mask,
)


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
            "k-alpha (product JSON)" if _batch in {"7", "7.0"} else _source_lines
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
        left_area = float(
            np.trapezoid(np.clip(residual[beta_mask], 0, None), grid[beta_mask])
        )
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
        _ax.plot(
            grid,
            _profile["y"],
            color="#2563eb",
            linewidth=2.0,
            label="measured profile",
        )
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
        _started_at = pd.Timestamp(getattr(_row, "started_at")).strftime(
            "%Y-%m-%d %H:%M"
        )
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
