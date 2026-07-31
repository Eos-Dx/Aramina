import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import importlib
    import os
    from pathlib import Path

    from experiments.profile_variability import profile_variability

    importlib.reload(profile_variability)
    return Path, os, profile_variability


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Target versus contralateral XRD profile variability

    Research-only analysis of normalized T100 profiles. The primary endpoint
    compares within-breast profile variability for three target positions with
    three contralateral positions in unilateral-biopsy patients. Physical point
    coordinates are unavailable; results describe protocol-conditioned
    variability, not intrinsic tissue heterogeneity.
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo, os):
    _base_dir = Path.cwd()
    _default_input = os.environ.get(
        "ARAMINA_PROFILE_JOBLIB",
        str(
            _base_dir / "examples/outputs/model_input/"
            "aramina_biopsy_patients_model_input_v0_1.joblib"
        ),
    )
    _default_output = os.environ.get(
        "ARAMINA_PROFILE_VARIABILITY_OUTPUT",
        str(_base_dir / "experiments/profile_variability/outputs/primary_3x3"),
    )
    widgets = {
        "input_path": mo.ui.text(
            label="Prepared T100 profile joblib",
            value=_default_input,
            full_width=True,
        ),
        "output_dir": mo.ui.text(
            label="Analysis output folder",
            value=_default_output,
            full_width=True,
        ),
        "minimum_measurements": mo.ui.slider(
            start=2,
            stop=3,
            step=1,
            value=3,
            label="Minimum unique positions per breast",
            show_value=True,
        ),
        "include_bilateral": mo.ui.checkbox(
            value=False,
            label="Include bilateral-biopsy cases (sensitivity analysis only)",
        ),
    }
    mo.vstack(
        [
            mo.md("## Inputs"),
            widgets["input_path"],
            widgets["output_dir"],
            widgets["minimum_measurements"],
            widgets["include_bilateral"],
        ]
    )
    return (widgets,)


@app.cell(hide_code=True)
def _(Path, widgets):
    config = {
        "input_path": Path(str(widgets["input_path"].value).strip()),
        "output_dir": Path(str(widgets["output_dir"].value).strip()),
        "minimum_measurements": int(widgets["minimum_measurements"].value),
        "include_bilateral": bool(widgets["include_bilateral"].value),
    }
    return (config,)


@app.cell(hide_code=True)
def _(config, mo, profile_variability):
    mo.stop(
        not config["input_path"].is_file(),
        mo.callout(
            mo.md(
                f"Prepared profile joblib was not found: `{config['input_path']}`. "
                "Select an existing artifact or set `ARAMINA_PROFILE_JOBLIB`."
            ),
            kind="warn",
        ),
    )
    profile_frame = profile_variability.load_profile_dataframe(config["input_path"])
    variability_analysis = profile_variability.run_variability_analysis(
        profile_frame,
        min_measurements=config["minimum_measurements"],
        include_bilateral_biopsy=config["include_bilateral"],
    )
    profile_variability.save_analysis(
        variability_analysis,
        config["output_dir"],
    )
    return (variability_analysis,)


@app.cell(hide_code=True)
def _(config, mo, variability_analysis):
    _all = variability_analysis.paired_summary.loc[
        variability_analysis.paired_summary["group"].eq("ALL")
    ].iloc[0]
    _contrast = variability_analysis.diagnosis_contrast.iloc[0]
    mo.md(
        f"""
        ## Primary result

        - eligible cases: **{int(_all["cases"])}**
        - minimum unique positions per breast: **{config["minimum_measurements"]}**
        - target / contralateral geometric variability ratio:
          **{_all["geometric_mean_ratio"]:.3f}**
        - bootstrap 95% CI: **{_all["geometric_mean_ratio_bootstrap_95_low"]:.3f}–{_all["geometric_mean_ratio_bootstrap_95_high"]:.3f}**
        - paired log-ratio test: **p={_all["paired_t_log_ratio_p"]:.4g}**
        - fraction with target variability greater than contralateral:
          **{_all["target_more_variable_fraction"]:.1%}**
        - CANCER versus BENIGN log-ratio interaction:
          **{_contrast["mean_difference"]:.3f}**,
          permutation **p={_contrast["diagnosis_label_permutation_p"]:.4g}**

        Output: `{config["output_dir"]}`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, profile_variability, variability_analysis):
    _figure = profile_variability.paired_scatter_figure(variability_analysis.cases)
    mo.vstack([mo.md("## Paired comparison"), mo.as_html(_figure)])
    return


@app.cell(hide_code=True)
def _(mo, profile_variability, variability_analysis):
    _figure = profile_variability.q_variability_figure(
        variability_analysis.q_variability
    )
    mo.vstack([mo.md("## q-dependent variability"), mo.as_html(_figure)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation limits

    - Profiles were already normalized by the frozen preprocessing pipeline; no
      additional smoothing or normalization is applied.
    - Target points are sampled within a suspicious region, whereas contralateral
      points are more widely separated.
    - Only P1/P2/P3 labels are available. Physical distances between points are
      absent, so spatial separation cannot be adjusted statistically.
    - Bilateral-biopsy patients are excluded from the primary comparison.
    - This analysis uses the model-development cohort and is descriptive, not
      independent clinical validation.
    """)
    return


if __name__ == "__main__":
    app.run()
