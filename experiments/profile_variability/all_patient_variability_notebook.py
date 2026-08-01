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

    from experiments.profile_variability import all_patient_variability

    importlib.reload(all_patient_variability)
    return Path, all_patient_variability, os


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # All-patient XRD profile variability

    Research-only analysis of normalized XRD profiles. Historical AgBH/K-beta exclusion and biopsy-patient filtering are disabled during preprocessing. Biopsy and no-biopsy cohorts are analysed separately: target/contralateral for unilateral-biopsy patients and left/right for no-biopsy patients.
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo, os):
    _root = Path.cwd()
    _default_input = os.environ.get(
        "ARAMINA_ALL_PATIENT_PROFILE_JOBLIB",
        str(
            _root / "experiments/profile_variability/local_data/"
            "aramina_all_patients_no_kbeta_profiles.joblib"
        ),
    )
    _default_output = os.environ.get(
        "ARAMINA_ALL_PATIENT_VARIABILITY_OUTPUT",
        str(_root / "experiments/profile_variability/outputs/all_patients_no_kbeta"),
    )
    widgets = {
        "input_path": mo.ui.text(
            label="All-patient normalized profile joblib",
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
    }
    mo.vstack(
        [
            mo.md("## Inputs"),
            widgets["input_path"],
            widgets["output_dir"],
            widgets["minimum_measurements"],
        ]
    )
    return (widgets,)


@app.cell(hide_code=True)
def _(Path, widgets):
    config = {
        "input_path": Path(str(widgets["input_path"].value).strip()),
        "output_dir": Path(str(widgets["output_dir"].value).strip()),
        "minimum_measurements": int(widgets["minimum_measurements"].value),
    }
    return (config,)


@app.cell(hide_code=True)
def _(all_patient_variability, config, mo):
    mo.stop(
        not config["input_path"].is_file(),
        mo.callout(
            mo.md(
                f"All-patient profile joblib was not found: `{config['input_path']}`. "
                "Run the CLI preprocessing route first or set "
                "`ARAMINA_ALL_PATIENT_PROFILE_JOBLIB`."
            ),
            kind="warn",
        ),
    )
    profile_frame = all_patient_variability.load_all_patient_profile_dataframe(
        config["input_path"]
    )
    analysis = all_patient_variability.run_all_patient_variability_analysis(
        profile_frame,
        min_measurements=config["minimum_measurements"],
    )
    all_patient_variability.save_all_patient_analysis(analysis, config["output_dir"])
    return (analysis,)


@app.cell(hide_code=True)
def _(analysis, mo):
    counts = analysis.metadata["cohort_counts"]
    mo.md(
        f"""
        ## Eligible paired cohort

        - post-technical-QC patients: **{analysis.metadata["post_technical_qc_patients"]}**
        - patients with left and right breasts: **{analysis.metadata["patients_with_left_and_right_breasts"]}**
        - patients with the required positions per breast: **{analysis.metadata["eligible_patients"]}**
        - unilateral biopsy BENIGN / CANCER: **{counts.get("BIOPSY_BENIGN", 0)} / {counts.get("BIOPSY_CANCER", 0)}**
        - no-biopsy left/right pairs: **{counts.get("NO_BIOPSY", 0)}**

        Historical K-beta exclusion: **not applied**. This is a descriptive
        variability analysis, not model training or independent validation.
        """
    )
    return


@app.cell(hide_code=True)
def _(all_patient_variability, analysis, mo):
    figure = all_patient_variability.all_patient_variability_figure(analysis.cases)
    mo.vstack([mo.md("## Paired comparisons"), mo.as_html(figure)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Interpretation limit

    Target and contralateral points follow different historical sampling geometries, and physical point coordinates are unavailable. Therefore target/contralateral results are protocol-conditioned. The no-biopsy left/right panel is a descriptive reference cohort, not a clinical comparison or a model-validation result.
    """)
    return


if __name__ == "__main__":
    app.run()
