import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import importlib
    import joblib
    from pathlib import Path

    import aramis_product_notebook_helpers as helpers
    from xrd_preprocessing import (
        AzimuthalIntegration,
        FaultyPixelDetector,
        H5SessionFilter,
        QRangeValueNormalizer,
        RadialProfileValueFilter,
        SNRFilter,
        SNRTransformer,
        SpecimenValidityFilter,
        calibrant_thickness_h5_filters,
        filter_h5_sessions,
        faulty_pixel_statistics,
        list_h5_measurement_sets,
    )
    from xrd_preprocessing.transformers import (
        ConstantQRangeTransformer,
        DropColumnsTransformer,
        H5ToDataFrameTransformer,
        ProductColumnBuilder,
        ProductStatusGroupFilter,
    )

    helpers = importlib.reload(helpers)

    PRODUCT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = PRODUCT_DIR.parent
    CLINICAL_TRIALS_DIR = (
        Path.home() / "dev" / "eos_play" / "jupyter_notebooks" / "Clinical_trials"
    )
    DATA_DIR = CLINICAL_TRIALS_DIR / "data" / "product-aramis-data"
    DEFAULT_ARCHIVE_PATH = DATA_DIR / "combined_archive.h5"
    DEFAULT_AGBH_CONFIG_PATH = (
        REPO_ROOT / "docs" / "meta" / "aramis_preprocessing_v0_1_config.json"
    )
    DEFAULT_ARAMIS_PREPROCESSING_CONFIG_PATH = (
        REPO_ROOT
        / "config"
        / "preprocessing"
        / "aramis_one_to_many_max_v0_1.yaml"
    )
    DEFAULT_OUTPUT_JOBLIB_PATH = (
        PRODUCT_DIR / "outputs" / "aramis_one_to_many_benign_cancer_dataframe.joblib"
    )
    return (
        AzimuthalIntegration,
        ConstantQRangeTransformer,
        DEFAULT_AGBH_CONFIG_PATH,
        DEFAULT_ARAMIS_PREPROCESSING_CONFIG_PATH,
        DEFAULT_ARCHIVE_PATH,
        DEFAULT_OUTPUT_JOBLIB_PATH,
        DropColumnsTransformer,
        FaultyPixelDetector,
        H5SessionFilter,
        H5ToDataFrameTransformer,
        Path,
        ProductColumnBuilder,
        ProductStatusGroupFilter,
        QRangeValueNormalizer,
        RadialProfileValueFilter,
        SNRFilter,
        SNRTransformer,
        SpecimenValidityFilter,
        calibrant_thickness_h5_filters,
        filter_h5_sessions,
        faulty_pixel_statistics,
        helpers,
        joblib,
        list_h5_measurement_sets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "\n".join(
            [
                "# Aramis DataFrame One-To-Many v0.1",
                "",
                "Research draft preprocessing notebook.",
                "",
                "Goal: build specimen/breast-level BENIGN vs CANCER preprocessing DataFrame for population comparison and save it as joblib.",
                "",
                "H5 branch rule: apply AgBH day, PONI q-range, calibrant-thickness, sample-thickness, and raw BENIGN/CANCER/ATYPICAL/PRE_CANCEROUS status filters before GFRM decode. Product label grouping happens after `H5ToDataFrameTransformer`.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(
    DEFAULT_AGBH_CONFIG_PATH,
    DEFAULT_ARAMIS_PREPROCESSING_CONFIG_PATH,
    DEFAULT_ARCHIVE_PATH,
    Path,
    helpers,
    mo,
):
    _cli_args = mo.cli_args()
    _agbh_default = Path(
        _cli_args.get("agbh-config-path")
        or _cli_args.get("agbh_config_path")
        or DEFAULT_AGBH_CONFIG_PATH
    )
    _aramis_preprocessing_default = Path(
        _cli_args.get("aramis-preprocessing-config-path")
        or _cli_args.get("aramis_preprocessing_config_path")
        or DEFAULT_ARAMIS_PREPROCESSING_CONFIG_PATH
    )
    _preprocessing_config = helpers.read_aramis_preprocessing_config(
        _aramis_preprocessing_default
    )
    _archive_default, _output_joblib_default = helpers.preprocessing_io_paths(
        _aramis_preprocessing_default,
        _preprocessing_config,
    )
    if not _archive_default.exists() and DEFAULT_ARCHIVE_PATH.exists():
        _archive_default = DEFAULT_ARCHIVE_PATH
    _default_settings = {
        "archive_path": str(_archive_default),
        "agbh_config_path": str(_agbh_default),
        "aramis_preprocessing_config_path": str(_aramis_preprocessing_default),
        "output_joblib_path": str(_output_joblib_default),
        "max_sessions": 0,
        "npt": 100,
        "required_q_max_nm_inv": 23.0,
        "snr_threshold_db": 18.0,
        "profile_gate_q_nm_inv": 14.0,
        "profile_gate_min_value": 2.0,
        "min_measurements_per_specimen": 1,
    }
    get_pipeline_settings, set_pipeline_settings = mo.state(_default_settings)
    archive_path_input = mo.ui.text(
        value=str(_archive_default),
        label="combined H5 archive",
    )
    agbh_config_path_input = mo.ui.text(
        value=str(_agbh_default),
        label="AgBH config JSON",
    )
    aramis_preprocessing_config_path_input = mo.ui.text(
        value=str(_aramis_preprocessing_default),
        label="Aramis one-to-many preprocessing YAML",
    )
    output_joblib_path_input = mo.ui.text(
        value=str(_output_joblib_default),
        label="output joblib",
    )
    max_sessions_input = mo.ui.number(
        value=0,
        start=0,
        stop=1000,
        step=1,
        label="max H5 sessions, 0 = all",
    )
    npt_input = mo.ui.number(
        value=100,
        start=100,
        stop=5000,
        step=100,
        label="azimuthal integration npt",
    )
    required_q_max_input = mo.ui.number(
        value=23.0,
        start=1.0,
        stop=40.0,
        step=0.1,
        label="required q max, nm^-1",
    )
    snr_threshold_input = mo.ui.slider(
        value=18.0,
        start=0.0,
        stop=40.0,
        step=0.5,
        label="SNR threshold, dB",
    )
    profile_gate_q_input = mo.ui.number(
        value=14.0,
        start=0.0,
        stop=30.0,
        step=0.1,
        label="profile gate q, nm^-1",
    )
    profile_gate_min_value_input = mo.ui.number(
        value=2.0,
        start=0.0,
        stop=50.0,
        step=0.1,
        label="profile gate min value",
    )
    min_measurements_per_specimen_input = mo.ui.number(
        value=1,
        start=1,
        stop=3,
        step=1,
        label="min measurements per specimen after SNR",
    )

    return (
        agbh_config_path_input,
        aramis_preprocessing_config_path_input,
        archive_path_input,
        get_pipeline_settings,
        max_sessions_input,
        min_measurements_per_specimen_input,
        npt_input,
        output_joblib_path_input,
        profile_gate_min_value_input,
        profile_gate_q_input,
        required_q_max_input,
        set_pipeline_settings,
        snr_threshold_input,
    )


@app.cell(hide_code=True)
def _(
    agbh_config_path_input,
    aramis_preprocessing_config_path_input,
    archive_path_input,
    max_sessions_input,
    min_measurements_per_specimen_input,
    npt_input,
    output_joblib_path_input,
    profile_gate_min_value_input,
    profile_gate_q_input,
    required_q_max_input,
    snr_threshold_input,
):
    def collect_pipeline_settings():
        return {
            "archive_path": archive_path_input.value,
            "agbh_config_path": agbh_config_path_input.value,
            "aramis_preprocessing_config_path": (
                aramis_preprocessing_config_path_input.value
            ),
            "output_joblib_path": output_joblib_path_input.value,
            "max_sessions": int(max_sessions_input.value),
            "npt": int(npt_input.value),
            "required_q_max_nm_inv": float(required_q_max_input.value),
            "snr_threshold_db": float(snr_threshold_input.value),
            "profile_gate_q_nm_inv": float(profile_gate_q_input.value),
            "profile_gate_min_value": float(profile_gate_min_value_input.value),
            "min_measurements_per_specimen": int(
                min_measurements_per_specimen_input.value
            ),
        }

    return (collect_pipeline_settings,)


@app.cell(hide_code=True)
def _(
    agbh_config_path_input,
    aramis_preprocessing_config_path_input,
    archive_path_input,
    collect_pipeline_settings,
    max_sessions_input,
    min_measurements_per_specimen_input,
    mo,
    npt_input,
    output_joblib_path_input,
    profile_gate_min_value_input,
    profile_gate_q_input,
    required_q_max_input,
    set_pipeline_settings,
    snr_threshold_input,
):
    def _validated_settings(_value):
        _ = _value
        set_pipeline_settings(collect_pipeline_settings())

    validate_settings_button = mo.ui.run_button(
        label="Validate settings",
        on_change=_validated_settings,
    )
    controls = mo.vstack(
        [
            archive_path_input,
            agbh_config_path_input,
            aramis_preprocessing_config_path_input,
            output_joblib_path_input,
            max_sessions_input,
            npt_input,
            required_q_max_input,
            snr_threshold_input,
            profile_gate_q_input,
            profile_gate_min_value_input,
            min_measurements_per_specimen_input,
            validate_settings_button,
        ]
    )
    return (
        controls,
    )


@app.cell
def _(controls, mo):
    mo.vstack([controls])
    return


@app.cell(hide_code=True)
def _(
    Path,
    get_pipeline_settings,
    helpers,
):
    _settings = get_pipeline_settings()
    archive_path = Path(_settings["archive_path"])
    agbh_config_path = Path(_settings["agbh_config_path"])
    aramis_preprocessing_config_path = Path(
        _settings["aramis_preprocessing_config_path"]
    )
    output_joblib_path = Path(_settings["output_joblib_path"])
    max_sessions = helpers.max_sessions_from_value(_settings["max_sessions"])
    npt = int(_settings["npt"])
    required_q_max_nm_inv = float(_settings["required_q_max_nm_inv"])
    snr_threshold_db = float(_settings["snr_threshold_db"])
    profile_gate_q_nm_inv = float(_settings["profile_gate_q_nm_inv"])
    profile_gate_min_value = float(_settings["profile_gate_min_value"])
    min_measurements_per_specimen = int(_settings["min_measurements_per_specimen"])
    return (
        agbh_config_path,
        aramis_preprocessing_config_path,
        archive_path,
        max_sessions,
        min_measurements_per_specimen,
        npt,
        output_joblib_path,
        profile_gate_min_value,
        profile_gate_q_nm_inv,
        required_q_max_nm_inv,
        snr_threshold_db,
    )


@app.cell(hide_code=True)
def _(aramis_preprocessing_config_path, helpers):
    aramis_preprocessing_config = helpers.read_aramis_preprocessing_config(
        aramis_preprocessing_config_path
    )
    thickness_settings = helpers.thickness_settings_from_config(
        aramis_preprocessing_config
    )
    return aramis_preprocessing_config, thickness_settings


@app.cell(hide_code=True)
def _(agbh_config_path, helpers):
    agbh_config = helpers.read_agbh_config(agbh_config_path)
    agbh_selection = helpers.agbh_selection(agbh_config)
    accepted_dates = agbh_selection["accepted_dates"]
    agbh_threshold = agbh_selection["threshold"]
    agbh_selected_batches = agbh_selection["selected_batches"]
    product_distance_q_range_policy = agbh_selection["distance_q_range_policy"]
    return (
        accepted_dates,
        agbh_selected_batches,
        agbh_threshold,
        product_distance_q_range_policy,
    )


@app.cell(hide_code=True)
def _(
    H5SessionFilter,
    accepted_dates,
    archive_path,
    calibrant_thickness_h5_filters,
    filter_h5_sessions,
    helpers,
    required_q_max_nm_inv,
    thickness_settings,
):
    h5_filter_plan = helpers.build_branch_h5_filters(
        H5SessionFilter=H5SessionFilter,
        calibrant_thickness_h5_filters=calibrant_thickness_h5_filters,
        filter_h5_sessions=filter_h5_sessions,
        archive_path=archive_path,
        accepted_dates=accepted_dates,
        required_q_max_nm_inv=required_q_max_nm_inv,
        thickness_settings=thickness_settings,
        branch="one_to_many",
    )
    h5_filters = h5_filter_plan["filters"]
    return h5_filter_plan, h5_filters


@app.cell(hide_code=True)
def _(
    archive_path,
    filter_h5_sessions,
    h5_filter_plan,
    helpers,
    list_h5_measurement_sets,
    max_sessions,
):
    h5_stage_frames = helpers.branch_h5_stage_frames(
        archive_path=archive_path,
        filter_h5_sessions=filter_h5_sessions,
        list_h5_measurement_sets=list_h5_measurement_sets,
        filter_plan=h5_filter_plan,
        max_sessions=max_sessions,
    )
    h5_counts_df = helpers.branch_h5_filter_stage_counts(h5_stage_frames)
    h5_metrics_fig = helpers.plot_stage_metrics(h5_counts_df)
    return h5_counts_df, h5_metrics_fig, h5_stage_frames


@app.cell(hide_code=True)
def _(
    accepted_dates,
    agbh_config_path,
    agbh_selected_batches,
    agbh_threshold,
    aramis_preprocessing_config_path,
    archive_path,
    h5_counts_df,
    h5_filter_plan,
    h5_metrics_fig,
    helpers,
    max_sessions,
    mo,
    product_distance_q_range_policy,
    required_q_max_nm_inv,
    thickness_settings,
):
    _q_range_reason = product_distance_q_range_policy.get("reason", "not set")
    mo.vstack(
        [
            mo.md(
                "\n".join(
                    [
                        "## Step 1. H5 filters for one-to-many",
                        "",
                        f"archive: `{archive_path}`",
                        f"AgBH config: `{agbh_config_path}`",
                        f"Aramis preprocessing YAML: `{aramis_preprocessing_config_path}`",
                        f"selected batches: `{agbh_selected_batches}`",
                        f"accepted AgBH dates: `{len(accepted_dates)}`",
                        f"AgBH threshold: `{agbh_threshold}`",
                        f"required q max, nm^-1: `{required_q_max_nm_inv}`",
                        f"max sessions: `{max_sessions}`",
                        "",
                        "Applied before GFRM decode:",
                        "",
                        "```text",
                        "started_at date in accepted AgBH dates",
                        "poni_q_max_nm_inv >= required q max",
                        helpers.thickness_settings_text(thickness_settings),
                        h5_filter_plan["description"],
                        "```",
                        "",
                        "```text",
                        helpers.counts_text(h5_counts_df),
                        "```",
                        "",
                        f"AgBH q-range note: {_q_range_reason}",
                    ]
                )
            ),
            mo.as_html(h5_metrics_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(H5ToDataFrameTransformer, archive_path, h5_filters, max_sessions, thickness_settings):
    h5_reader = H5ToDataFrameTransformer(
        data_preference="gfrm",
        drop_missing_sample_thickness=thickness_settings["require_sample"],
        h5_filters=h5_filters,
        session_category="SAMPLE",
        set_category="SAMPLE",
        max_sessions=max_sessions,
    )
    raw_measurement_df = h5_reader.fit_transform(archive_path)
    calibration_df = h5_reader.calibration_df_
    return calibration_df, h5_reader, raw_measurement_df


@app.cell
def _(ProductColumnBuilder, raw_measurement_df):
    product_column_builder = ProductColumnBuilder()
    decoded_df = product_column_builder.fit_transform(raw_measurement_df)
    return decoded_df, product_column_builder


@app.cell(hide_code=True)
def _(calibration_df, decoded_df, helpers, mo, thickness_settings):
    mo.md(
        "\n".join(
            [
                "## Step 2. H5 reader transformer",
                "",
                "`data_preference='gfrm'` is obligatory.",
                f"calibration rows: `{len(calibration_df)}`",
                f"decoded measurement rows: `{len(decoded_df)}`",
                "`session_category='SAMPLE'` and `set_category='SAMPLE'` mean patient/specimen measurements; calibration records are not loaded into this DataFrame.",
                "Thickness columns and filters are forced by the one-to-many preprocessing YAML.",
                "Level: rows are measurement-level records; clinical label semantics are inherited from `specimenId` / breast side.",
                "",
                "```text",
                helpers.thickness_settings_text(thickness_settings),
                "",
                helpers.calibrant_thickness_summary_text(decoded_df),
                "",
                helpers.stage_summary_text("After GFRM decode", decoded_df),
                "```",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(ProductStatusGroupFilter, decoded_df, helpers):
    binary_label_filter = ProductStatusGroupFilter(
        helpers.LABEL_VALUES,
    )
    binary_df = binary_label_filter.fit_transform(decoded_df)
    binary_label_stats = binary_label_filter.stats_
    return binary_df, binary_label_filter, binary_label_stats


@app.cell(hide_code=True)
def _(binary_df, binary_label_stats, decoded_df, helpers, mo):
    mo.md(
        "\n".join(
            [
                "## Step 3. Product label grouping",
                "",
                "Level: `specimenId` / breast side. All measurement rows from the same specimen inherit the same product label.",
                "`specimenId`-level grouping: `BENIGN -> BENIGN`; `CANCER/ATYPICAL/PRE_CANCEROUS -> CANCER`. `NORMAL` was excluded at H5 level for one-to-many.",
                "",
                f"status groups after filter: `{binary_label_stats.get('after_counts', binary_label_stats.get('counts_pass', {}))}`",
                "",
                "```text",
                helpers.stage_summary_text(
                    "One-to-many BENIGN/CANCER rows",
                    binary_df,
                    before_df=decoded_df,
                ),
                "```",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(FaultyPixelDetector, binary_df, faulty_pixel_statistics):
    faulty_detector = FaultyPixelDetector(
        bright_pixel_min_value=500.0,
        exclude_beam_center_radius=0.04,
    )
    faulty_df = faulty_detector.fit_transform(binary_df)
    faulty_stats = faulty_pixel_statistics(faulty_df)
    return faulty_df, faulty_stats


@app.cell(hide_code=True)
def _(binary_df, faulty_df, faulty_stats, helpers, mo):
    mo.md(
        "\n".join(
            [
                "## Step 4. Measurement-level faulty pixel mask",
                "",
                f"total faulty pixels: `{faulty_stats['total_faulty_pixels']}`",
                "",
                "```text",
                helpers.stage_summary_text(
                    "After faulty-pixel mask creation",
                    faulty_df,
                    before_df=binary_df,
                ),
                "```",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(ConstantQRangeTransformer, faulty_df):
    q_range_transformer = ConstantQRangeTransformer(q_min=2.0, q_max=23.0)
    integration_input_df = q_range_transformer.fit_transform(faulty_df)
    return integration_input_df, q_range_transformer


@app.cell(hide_code=True)
def _(faulty_df, helpers, integration_input_df, mo):
    mo.md(
        "\n".join(
            [
                "## Step 5. Constant q-range preparation",
                "",
                "Level: measurementId. Each row is prepared for q-range-aware azimuthal integration over `2.0..23.0 nm^-1`.",
                "",
                "```text",
                helpers.stage_summary_text(
                    "After constant q-range preparation",
                    integration_input_df,
                    before_df=faulty_df,
                ),
                "```",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(
    AzimuthalIntegration,
    DropColumnsTransformer,
    helpers,
    integration_input_df,
    npt,
    thickness_settings,
):
    helpers.validate_thickness_columns_for_integration(
        integration_input_df,
        thickness_settings,
    )
    integrator = AzimuthalIntegration(
        npt=npt,
        calibration_mode="poni",
        mask_column="faulty_pixel_mask",
        error_model="poisson",
        thickness_adjustment=thickness_settings["enabled"],
        require_thickness_adjustment=thickness_settings["enabled"],
        thickness_reference_column=thickness_settings["calibrant_column"],
        sample_thickness_column=thickness_settings["sample_column"],
    )
    integrated_raw_df = integrator.fit_transform(integration_input_df)
    heavy_column_dropper = DropColumnsTransformer(helpers.HEAVY_DETECTOR_COLUMNS)
    integrated_df = heavy_column_dropper.fit_transform(integrated_raw_df)
    dropped_heavy_columns = heavy_column_dropper.dropped_columns_
    return dropped_heavy_columns, heavy_column_dropper, integrated_df


@app.cell(hide_code=True)
def _(helpers, integrated_df):
    integrated_fig = helpers.plot_profiles(
        integrated_df,
        title="One-to-many after azimuthal integration",
        diagnosis_column="product_status_group",
    )
    return (integrated_fig,)


@app.cell(hide_code=True)
def _(
    dropped_heavy_columns,
    helpers,
    integrated_df,
    integrated_fig,
    integration_input_df,
    mo,
):
    mo.vstack(
        [
            mo.md(
                "\n".join(
                    [
                        "## Step 6. Azimuthal integration",
                        "",
                        "Thickness correction is enabled with specimen thickness and row-level AgBH thickness.",
                        f"Dropped heavy detector columns after integration: `{dropped_heavy_columns}`",
                        "",
                        "```text",
                        helpers.stage_summary_text(
                            "After azimuthal integration",
                            integrated_df,
                            before_df=integration_input_df,
                        ),
                        "```",
                    ]
                )
            ),
            mo.as_html(integrated_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(SNRTransformer, integrated_df):
    snr_transformer = SNRTransformer(snr_method="poisson")
    snr_df = snr_transformer.fit_transform(integrated_df)
    return (snr_df,)


@app.cell(hide_code=True)
def _(helpers, snr_df, snr_threshold_db):
    snr_fig = helpers.plot_snr(snr_df, cutoff_db=snr_threshold_db)
    return (snr_fig,)


@app.cell(hide_code=True)
def _(helpers, integrated_df, mo, snr_df, snr_fig):
    mo.vstack(
        [
            mo.md(
                "\n".join(
                    [
                        "## Step 7. Poisson SNR calculation",
                        "",
                        "```text",
                        helpers.stage_summary_text(
                            "After Poisson SNR calculation",
                            snr_df,
                            before_df=integrated_df,
                        ),
                        "```",
                    ]
                )
            ),
            mo.as_html(snr_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(SNRFilter, snr_df, snr_threshold_db):
    snr_filter = SNRFilter(min_snr_db=snr_threshold_db)
    snr_filtered_df = snr_filter.fit_transform(snr_df)
    snr_filter_stats = snr_filter.stats_
    return snr_filter_stats, snr_filtered_df


@app.cell(hide_code=True)
def _(helpers, mo, snr_df, snr_filter_stats, snr_filtered_df):
    mo.md(
        "\n".join(
            [
                "## Step 8. SNR filter",
                "",
                "Level: measurementId. Rows below the user-selected Poisson SNR threshold are removed.",
                f"SNR rows in/pass/fail: `{snr_filter_stats['rows_in']}/{snr_filter_stats['rows_pass']}/{snr_filter_stats['rows_fail']}`",
                "",
                "```text",
                helpers.stage_summary_text(
                    "After SNR filter",
                    snr_filtered_df,
                    before_df=snr_df,
                ),
                "```",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(
    SpecimenValidityFilter,
    min_measurements_per_specimen,
    snr_filtered_df,
):
    validity_filter = SpecimenValidityFilter(
        min_measurements_per_specimen=min_measurements_per_specimen,
    )
    valid_df = validity_filter.fit_transform(snr_filtered_df)
    validity_stats = validity_filter.stats_
    return valid_df, validity_stats


@app.cell(hide_code=True)
def _(helpers, mo, snr_filtered_df, valid_df, validity_stats):
    mo.md(
        "\n".join(
            [
                "## Step 9. Specimen validity filter",
                "",
                "Level: specimenId. Each retained specimen must have the configured minimum number of valid measurements after SNR filtering.",
                "`patientId` is retained as metadata for later leakage control, not used for this validity rule.",
                f"validity rows in/pass/fail: `{validity_stats['rows_in']}/{validity_stats['rows_pass']}/{validity_stats['rows_fail']}`",
                "",
                "```text",
                helpers.stage_summary_text(
                    "After specimen validity",
                    valid_df,
                    before_df=snr_filtered_df,
                ),
                "```",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(QRangeValueNormalizer, valid_df):
    normalized_df = QRangeValueNormalizer(
        q_min=6.7,
        q_max=7.1,
        statistic="median",
        save_initial_data=False,
    ).fit_transform(valid_df)
    return (normalized_df,)


@app.cell(hide_code=True)
def _(helpers, mo, normalized_df, valid_df):
    mo.md(
        "\n".join(
            [
                "## Step 10. Q 6.7..7.1 normalization",
                "",
                "Level: measurementId. Profiles are normalized by median value in q window `6.7..7.1 nm^-1`.",
                "",
                "```text",
                helpers.stage_summary_text(
                    "After Q 6.7..7.1 normalization",
                    normalized_df,
                    before_df=valid_df,
                ),
                "```",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(
    DropColumnsTransformer,
    RadialProfileValueFilter,
    helpers,
    normalized_df,
    profile_gate_min_value,
    profile_gate_q_nm_inv,
):
    profile_filter = RadialProfileValueFilter(
        q_value_nm_inv=profile_gate_q_nm_inv,
        threshold=profile_gate_min_value,
        op=">",
    )
    final_profile_df = profile_filter.fit_transform(normalized_df)
    final_column_dropper = DropColumnsTransformer(helpers.NON_OUTPUT_PAYLOAD_COLUMNS)
    final_df = final_column_dropper.fit_transform(final_profile_df)
    dropped_final_columns = final_column_dropper.dropped_columns_
    profile_filter_stats = profile_filter.stats_
    return dropped_final_columns, final_column_dropper, final_df, profile_filter_stats


@app.cell(hide_code=True)
def _(final_df, helpers, normalized_df):
    final_fig = helpers.plot_profiles(
        final_df,
        title="One-to-many final normalized DataFrame",
        diagnosis_column="product_status_group",
    )
    final_counts_df = helpers.stage_counts(
        [
            ("h5_reader_transformer", normalized_df),
            ("radial_profile_value_filter", final_df),
        ]
    )
    return final_counts_df, final_fig


@app.cell(hide_code=True)
def _(
    final_df,
    final_fig,
    dropped_final_columns,
    helpers,
    mo,
    normalized_df,
    profile_filter_stats,
):
    mo.vstack(
        [
            mo.md(
                "\n".join(
                    [
                        "## Step 11. Radial-profile signal gate and final payload",
                        "",
                        "Final DataFrame keeps normalized profiles plus all H5/product metadata.",
                        f"Dropped non-output payload columns: `{dropped_final_columns}`",
                        f"profile gate rows in/pass/fail: `{profile_filter_stats['rows_in']}/{profile_filter_stats['rows_pass']}/{profile_filter_stats['rows_fail']}`",
                        "",
                        "```text",
                        helpers.stage_summary_text(
                            "Final one-to-many DataFrame",
                            final_df,
                            before_df=normalized_df,
                        ),
                        "```",
                    ]
                )
            ),
            mo.as_html(final_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    binary_df,
    decoded_df,
    faulty_df,
    final_df,
    helpers,
    integrated_df,
    normalized_df,
    snr_df,
    snr_filtered_df,
    valid_df,
):
    stage_counts_df = helpers.stage_counts(
        [
            ("h5_reader_after_non_na_h5", decoded_df),
            ("binary_product_label_filter", binary_df),
            ("faulty_pixels", faulty_df),
            ("azimuthal_integration", integrated_df),
            ("snr_poisson", snr_df),
            ("snr_filter", snr_filtered_df),
            ("specimen_validity", valid_df),
            ("q_6_7_7_1_normalization", normalized_df),
            ("radial_profile_value_filter", final_df),
        ]
    )
    stage_metrics_fig = helpers.plot_stage_metrics(stage_counts_df)
    return stage_counts_df, stage_metrics_fig


@app.cell(hide_code=True)
def _(helpers, mo, stage_counts_df, stage_metrics_fig):
    mo.vstack(
        [
            mo.md(
                "\n".join(
                    [
                        "## Stage statistics",
                        "",
                        "```text",
                        helpers.counts_text(stage_counts_df),
                        "```",
                    ]
                )
            ),
            mo.as_html(stage_metrics_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(final_df, joblib, output_joblib_path):
    output_joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_df, output_joblib_path)
    saved_joblib_path = output_joblib_path
    return (saved_joblib_path,)


@app.cell(hide_code=True)
def _(final_df, mo, saved_joblib_path):
    mo.md(
        "\n".join(
            [
                "## Output",
                "",
                f"Saved joblib: `{saved_joblib_path}`",
                f"rows: `{len(final_df)}`",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
