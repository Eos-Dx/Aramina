import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import importlib
    from pathlib import Path

    import aramina_agbh_kbeta_helpers as agbh_helpers

    agbh_helpers = importlib.reload(agbh_helpers)

    PRODUCT_DIR = Path(__file__).resolve().parent
    BASE_DIR = PRODUCT_DIR.parents[1]
    DATA_DIR = BASE_DIR / "data" / "product-aramina-data"
    CALIBRATION_DIR = DATA_DIR / "calibration"
    OUT_DIR = BASE_DIR / "analysis" / "aramina_agbh_kbeta_shoulder_metric"
    LEFT_OUT_DIR = BASE_DIR / "analysis" / "aramina_agbh_kbeta_left_metric"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEFT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    DEFAULT_BATCHES = "1,2,3,4,5,6,7"
    DEFAULT_REFERENCE_BATCH = "7"
    DEFAULT_REFERENCE_DATE_MAX = "2026-04-30"
    DEFAULT_REFERENCE_COUNT = 2
    DEFAULT_NPT = 100
    DEFAULT_Q_MIN = 2.0
    DEFAULT_Q_MAX = 18.0
    DEFAULT_SHOULDER_Q_MIN = 2.55
    DEFAULT_SHOULDER_Q_MAX = 3.05
    DEFAULT_BETA_RATIO = agbh_helpers.AG_KBETA_TO_KALPHA_Q_RATIO
    return (
        CALIBRATION_DIR,
        DEFAULT_BATCHES,
        DEFAULT_BETA_RATIO,
        DEFAULT_NPT,
        DEFAULT_Q_MAX,
        DEFAULT_Q_MIN,
        DEFAULT_REFERENCE_BATCH,
        DEFAULT_REFERENCE_COUNT,
        DEFAULT_REFERENCE_DATE_MAX,
        DEFAULT_SHOULDER_Q_MAX,
        DEFAULT_SHOULDER_Q_MIN,
        LEFT_OUT_DIR,
        OUT_DIR,
        Path,
        agbh_helpers,
    )


@app.cell(hide_code=True)
def _(
    CALIBRATION_DIR,
    DEFAULT_BATCHES,
    DEFAULT_BETA_RATIO,
    DEFAULT_NPT,
    DEFAULT_Q_MAX,
    DEFAULT_Q_MIN,
    DEFAULT_REFERENCE_BATCH,
    DEFAULT_REFERENCE_COUNT,
    DEFAULT_REFERENCE_DATE_MAX,
    DEFAULT_SHOULDER_Q_MAX,
    DEFAULT_SHOULDER_Q_MIN,
    mo,
):
    calibration_dir_input = mo.ui.text(
        value=str(CALIBRATION_DIR),
        label="technical H5 calibration containers",
    )
    batches_input = mo.ui.text(value=DEFAULT_BATCHES, label="data batches")
    reference_batch_input = mo.ui.text(
        value=DEFAULT_REFERENCE_BATCH,
        label="reference batch",
    )
    reference_date_max_input = mo.ui.text(
        value=DEFAULT_REFERENCE_DATE_MAX,
        label="reference latest date <=",
    )
    reference_count_input = mo.ui.number(
        value=DEFAULT_REFERENCE_COUNT,
        start=1,
        stop=10,
        step=1,
        label="reference AgBH count",
    )
    max_files_per_batch_input = mo.ui.number(
        value=0,
        start=0,
        stop=100,
        step=1,
        label="max AgBH files per batch, 0 = all",
    )
    npt_input = mo.ui.number(
        value=DEFAULT_NPT,
        start=100,
        stop=5000,
        step=100,
        label="integration npt",
    )
    q_min_input = mo.ui.number(
        value=DEFAULT_Q_MIN,
        start=0.0,
        stop=25.0,
        step=0.1,
        label="q min",
    )
    q_max_input = mo.ui.number(
        value=DEFAULT_Q_MAX,
        start=1.0,
        stop=25.0,
        step=0.1,
        label="q max",
    )
    beta_ratio_input = mo.ui.number(
        value=DEFAULT_BETA_RATIO,
        start=0.80,
        stop=0.95,
        step=0.001,
        label="K-beta / K-alpha q ratio",
    )
    shoulder_q_min_input = mo.ui.number(
        value=DEFAULT_SHOULDER_Q_MIN,
        start=2.0,
        stop=4.0,
        step=0.01,
        label="shoulder q min",
    )
    shoulder_q_max_input = mo.ui.number(
        value=DEFAULT_SHOULDER_Q_MAX,
        start=2.0,
        stop=4.0,
        step=0.01,
        label="shoulder q max",
    )
    normalize_profiles_input = mo.ui.checkbox(
        value=True,
        label="normalize profiles",
    )
    controls = mo.vstack(
        [
            calibration_dir_input,
            batches_input,
            reference_batch_input,
            reference_date_max_input,
            reference_count_input,
            max_files_per_batch_input,
            npt_input,
            q_min_input,
            q_max_input,
            beta_ratio_input,
            shoulder_q_min_input,
            shoulder_q_max_input,
            normalize_profiles_input,
        ]
    )
    controls
    return (
        batches_input,
        beta_ratio_input,
        calibration_dir_input,
        max_files_per_batch_input,
        normalize_profiles_input,
        npt_input,
        q_max_input,
        q_min_input,
        reference_batch_input,
        reference_count_input,
        reference_date_max_input,
        shoulder_q_max_input,
        shoulder_q_min_input,
    )


@app.cell(hide_code=True)
def _(
    Path,
    batches_input,
    beta_ratio_input,
    calibration_dir_input,
    max_files_per_batch_input,
    normalize_profiles_input,
    npt_input,
    q_max_input,
    q_min_input,
    reference_batch_input,
    reference_count_input,
    reference_date_max_input,
    shoulder_q_max_input,
    shoulder_q_min_input,
):
    calibration_dir = Path(calibration_dir_input.value)
    selected_batches = [
        _part.strip()
        for _part in str(batches_input.value).split(",")
        if _part.strip()
    ]
    max_files_per_batch_value = int(max_files_per_batch_input.value)
    max_files_per_batch = (
        None if max_files_per_batch_value == 0 else max_files_per_batch_value
    )
    npt = int(npt_input.value)
    q_min = float(q_min_input.value)
    q_max = float(q_max_input.value)
    beta_ratio = float(beta_ratio_input.value)
    normalize_profiles = bool(normalize_profiles_input.value)
    reference_batch = str(reference_batch_input.value).strip()
    reference_count = int(reference_count_input.value)
    reference_date_max = str(reference_date_max_input.value).strip()
    shoulder_q_min = float(shoulder_q_min_input.value)
    shoulder_q_max = float(shoulder_q_max_input.value)
    return (
        beta_ratio,
        calibration_dir,
        max_files_per_batch,
        normalize_profiles,
        npt,
        q_max,
        q_min,
        reference_batch,
        reference_count,
        reference_date_max,
        selected_batches,
        shoulder_q_max,
        shoulder_q_min,
    )


@app.cell
def _(agbh_helpers, calibration_dir):
    calibration_manifest_df = agbh_helpers.scan_calibration_manifest(calibration_dir)
    return (calibration_manifest_df,)


@app.cell(hide_code=True)
def _(
    agbh_helpers,
    calibration_manifest_df,
    max_files_per_batch,
    selected_batches,
):
    selected_agbh_manifest_df = agbh_helpers.agbh_manifest(
        calibration_manifest_df,
        batches=selected_batches,
        max_files_per_batch=max_files_per_batch,
    )
    return (selected_agbh_manifest_df,)


@app.cell
def _(
    agbh_helpers,
    calibration_manifest_df,
    reference_batch,
    reference_count,
    reference_date_max,
):
    reference_agbh_manifest_df = agbh_helpers.latest_reference_agbh_manifest(
        calibration_manifest_df,
        batch=reference_batch,
        date_max=reference_date_max,
        count=reference_count,
    )
    return (reference_agbh_manifest_df,)


@app.cell(hide_code=True)
def _(agbh_helpers, npt, selected_agbh_manifest_df):
    agbh_frame_df = agbh_helpers.load_agbh_frames(selected_agbh_manifest_df)
    integrated_agbh_df = agbh_helpers.integrate_agbh_profiles(
        agbh_frame_df,
        npt=npt,
    )
    return (integrated_agbh_df,)


@app.cell(hide_code=True)
def _(agbh_helpers, npt, reference_agbh_manifest_df):
    reference_agbh_frame_df = agbh_helpers.load_agbh_frames(reference_agbh_manifest_df)
    reference_integrated_agbh_df = agbh_helpers.integrate_agbh_profiles(
        reference_agbh_frame_df,
        npt=npt,
    )
    return (reference_integrated_agbh_df,)


@app.cell(hide_code=True)
def _(agbh_helpers, beta_ratio, q_max, q_min):
    alpha_peaks = agbh_helpers.agbh_alpha_peaks(q_max)
    beta_peaks = agbh_helpers.agbh_beta_peaks(q_max, beta_ratio=beta_ratio)
    alpha_peaks = alpha_peaks[(alpha_peaks >= q_min) & (alpha_peaks <= q_max)]
    beta_peaks = beta_peaks[(beta_peaks >= q_min) & (beta_peaks <= q_max)]
    return alpha_peaks, beta_peaks


@app.cell(hide_code=True)
def _(
    agbh_helpers,
    integrated_agbh_df,
    reference_integrated_agbh_df,
    shoulder_q_max,
    shoulder_q_min,
):
    shoulder_metric_df = agbh_helpers.shoulder_metric_table(
        integrated_agbh_df,
        reference_integrated_agbh_df,
        q_min=2.0,
        q_max=4.0,
        shoulder_min=shoulder_q_min,
        shoulder_max=shoulder_q_max,
    )
    return (shoulder_metric_df,)


@app.cell(hide_code=True)
def _(
    agbh_helpers,
    reference_integrated_agbh_df,
    shoulder_q_max,
    shoulder_q_min,
):
    good_signal_metric_df = agbh_helpers.shoulder_metric_table(
        reference_integrated_agbh_df,
        reference_integrated_agbh_df,
        q_min=2.0,
        q_max=4.0,
        shoulder_min=shoulder_q_min,
        shoulder_max=shoulder_q_max,
    )
    return (good_signal_metric_df,)


@app.cell(hide_code=True)
def _(
    agbh_helpers,
    integrated_agbh_df,
    q_max,
    q_min,
    reference_integrated_agbh_df,
):
    kbeta_left_metric_df = agbh_helpers.kbeta_left_shoulder_metric_table(
        integrated_agbh_df,
        reference_integrated_agbh_df,
        q_min=q_min,
        q_max=q_max,
    )
    good_kbeta_left_metric_df = agbh_helpers.kbeta_left_shoulder_metric_table(
        reference_integrated_agbh_df,
        reference_integrated_agbh_df,
        q_min=q_min,
        q_max=q_max,
    )
    return good_kbeta_left_metric_df, kbeta_left_metric_df


@app.cell(hide_code=True)
def _(
    LEFT_OUT_DIR,
    good_kbeta_left_metric_df,
    kbeta_left_metric_df,
    reference_agbh_manifest_df,
    selected_agbh_manifest_df,
):
    selected_agbh_manifest_df.to_csv(
        LEFT_OUT_DIR / "all_batches_selected_agbh.csv",
        index=False,
    )
    reference_agbh_manifest_df.to_csv(
        LEFT_OUT_DIR / "reference_agbh_batch7_latest_april.csv",
        index=False,
    )
    kbeta_left_metric_df.to_csv(
        LEFT_OUT_DIR / "all_batches_kbeta_left_metric.csv",
        index=False,
    )
    good_kbeta_left_metric_df.to_csv(
        LEFT_OUT_DIR / "good_signal_reference_kbeta_left_metric.csv",
        index=False,
    )
    return


@app.cell(hide_code=True)
def _(
    agbh_helpers,
    good_kbeta_left_metric_df,
    kbeta_left_metric_df,
    reference_agbh_manifest_df,
):
    kbeta_left_panel_fig = agbh_helpers.plot_shoulder_metric_by_batch_panels(
        kbeta_left_metric_df,
        good_metric_df=good_kbeta_left_metric_df,
        reference_manifest_df=reference_agbh_manifest_df,
        reference_label="ETALON / REFERENCE = BATCH 7 K-ALPHA (product JSON)",
        metric_column="kbeta_left_mean_net_residual",
        metric_label="left K-beta net residual; K-alpha q-axis fixed",
    )
    kbeta_left_panel_fig
    return (kbeta_left_panel_fig,)


@app.cell
def _(LEFT_OUT_DIR, kbeta_left_panel_fig):
    kbeta_left_panel_fig.savefig(
        LEFT_OUT_DIR / "all_batches_kbeta_left_metric_vertical_panels.png",
        dpi=170,
        bbox_inches="tight",
    )
    return


@app.cell
def _(
    agbh_helpers,
    integrated_agbh_df,
    kbeta_left_metric_df,
    reference_integrated_agbh_df,
):
    kbeta_diagnostic_fig, kbeta_zoom_fig = (
        agbh_helpers.kbeta_left_metric_diagnostic_figures(
            integrated_agbh_df,
            reference_integrated_agbh_df,
            kbeta_left_metric_df,
        )
    )
    kbeta_diagnostic_fig
    return kbeta_diagnostic_fig, kbeta_zoom_fig


@app.cell
def _(kbeta_zoom_fig):
    kbeta_zoom_fig
    return


@app.cell(hide_code=True)
def _(LEFT_OUT_DIR, kbeta_diagnostic_fig, kbeta_zoom_fig):
    kbeta_diagnostic_fig.savefig(
        LEFT_OUT_DIR / "kbeta_left_metric_bad_vs_good_diagnostic.png",
        dpi=180,
        bbox_inches="tight",
    )
    kbeta_zoom_fig.savefig(
        LEFT_OUT_DIR / "kbeta_left_metric_bad_profile_zoom.png",
        dpi=180,
        bbox_inches="tight",
    )
    return


@app.cell(hide_code=True)
def _(
    OUT_DIR,
    agbh_helpers,
    calibration_manifest_df,
    good_signal_metric_df,
    integrated_agbh_df,
    reference_agbh_manifest_df,
    selected_agbh_manifest_df,
    shoulder_metric_df,
):
    output_paths = agbh_helpers.write_outputs(
        out_dir=OUT_DIR,
        manifest_df=calibration_manifest_df,
        agbh_manifest_df=selected_agbh_manifest_df,
        integrated_df=integrated_agbh_df,
        reference_manifest_df=reference_agbh_manifest_df,
        shoulder_metric_df=shoulder_metric_df,
        good_signal_metric_df=good_signal_metric_df,
    )
    run_summary = agbh_helpers.summary_text(
        calibration_manifest_df,
        selected_agbh_manifest_df,
        integrated_agbh_df,
        reference_agbh_manifest_df,
    )
    return output_paths, run_summary


@app.cell(hide_code=True)
def _(mo, output_paths, run_summary):
    mo.md(f"""
    ```text
    {run_summary}

    manifest_csv={output_paths["manifest"]}
    selected_agbh_csv={output_paths["selected_agbh"]}
    integrated_agbh_csv={output_paths["integrated_agbh"]}
    reference_agbh_csv={output_paths["reference_agbh"]}
    shoulder_metric_csv={output_paths["shoulder_metric"]}
    good_signal_metric_csv={output_paths["good_signal_metric"]}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(
    agbh_helpers,
    alpha_peaks,
    beta_peaks,
    integrated_agbh_df,
    normalize_profiles,
    q_max,
    q_min,
    reference_integrated_agbh_df,
):
    profile_overlay_fig = agbh_helpers.plot_profiles_by_batch(
        integrated_agbh_df,
        alpha_peaks=alpha_peaks,
        beta_peaks=beta_peaks,
        q_min=q_min,
        q_max=q_max,
        max_profiles=80,
        normalize=normalize_profiles,
        reference_df=reference_integrated_agbh_df,
    )
    profile_overlay_fig
    return


@app.cell
def _(
    agbh_helpers,
    integrated_agbh_df,
    reference_integrated_agbh_df,
    shoulder_q_max,
    shoulder_q_min,
):
    shoulder_residual_fig = agbh_helpers.plot_shoulder_residuals(
        integrated_agbh_df,
        reference_integrated_agbh_df,
        q_min=2.0,
        q_max=4.0,
        shoulder_min=shoulder_q_min,
        shoulder_max=shoulder_q_max,
    )
    shoulder_residual_fig
    return


@app.cell
def _(agbh_helpers, good_signal_metric_df, shoulder_metric_df):
    shoulder_metric_fig = agbh_helpers.plot_shoulder_metric(
        shoulder_metric_df,
        good_metric_df=good_signal_metric_df,
    )
    shoulder_metric_fig
    return


@app.cell
def _(agbh_helpers, alpha_peaks, beta_peaks, integrated_agbh_df, q_max, q_min):
    profile_heatmap_fig = agbh_helpers.plot_profile_heatmaps(
        integrated_agbh_df,
        alpha_peaks=alpha_peaks,
        beta_peaks=beta_peaks,
        q_min=q_min,
        q_max=q_max,
    )
    profile_heatmap_fig
    return


if __name__ == "__main__":
    app.run()
