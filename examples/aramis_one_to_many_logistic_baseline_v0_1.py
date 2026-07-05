import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from aramis.modeling import (
        fit_repeated_one_to_many_logistic,
        load_one_to_many_dataframe,
        summarize_one_to_many_dataframe,
    )

    PRODUCT_DIR = Path(__file__).resolve().parent
    DEFAULT_DATAFRAME_JOBLIB_PATH = (
        PRODUCT_DIR / "outputs" / "aramis_one_to_many_benign_cancer_dataframe.joblib"
    )
    return (
        DEFAULT_DATAFRAME_JOBLIB_PATH,
        Path,
        fit_repeated_one_to_many_logistic,
        load_one_to_many_dataframe,
        np,
        pd,
        plt,
        summarize_one_to_many_dataframe,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "\n".join(
            [
                "# Aramis One-To-Many Logistic Baseline v0.1",
                "",
                "Research draft model notebook.",
                "",
                "Goal: first BENIGN vs CANCER decision-support baseline from the one-to-many preprocessing DataFrame.",
                "",
                "Input is a preprocessed joblib DataFrame. H5 decoding, filtering, faulty-pixel masking, azimuthal integration, SNR filtering, and q-range normalization must already be completed by `aramis_dataframe_one_to_many_v0_1.py`.",
                "",
                "Model test: `LogisticRegression` on the full normalized `radial_profile_data` profile, evaluated over repeated 70/30 patient-safe splits. The same `patientId` is never allowed in train and test for one split.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(DEFAULT_DATAFRAME_JOBLIB_PATH, Path, mo):
    _cli_args = mo.cli_args()
    dataframe_joblib_path = Path(
        _cli_args.get("dataframe-joblib-path")
        or _cli_args.get("dataframe_joblib_path")
        or DEFAULT_DATAFRAME_JOBLIB_PATH
    )
    n_splits = int(_cli_args.get("n-splits") or _cli_args.get("n_splits") or 20)
    test_size = float(_cli_args.get("test-size") or _cli_args.get("test_size") or 0.30)
    random_state = int(
        _cli_args.get("random-state") or _cli_args.get("random_state") or 42
    )
    logreg_c = float(_cli_args.get("logreg-c") or _cli_args.get("logreg_c") or 1.0)
    mo.md(
        "\n".join(
            [
                "## Settings",
                "",
                f"- DataFrame joblib: `{dataframe_joblib_path}`",
                f"- splits: `{n_splits}`",
                "- split rule: `70/30 patient-safe GroupShuffleSplit`",
                f"- test size: `{test_size:.2f}`",
                f"- random state: `{random_state}`",
                f"- LogisticRegression C: `{logreg_c}`",
                "- sample thickness: audit/control metadata only; not used as primary predictor",
            ]
        )
    )
    return dataframe_joblib_path, logreg_c, n_splits, random_state, test_size


@app.cell(hide_code=True)
def _(dataframe_joblib_path, load_one_to_many_dataframe, mo):
    if not dataframe_joblib_path.exists():
        mo.stop(
            True,
            mo.md(
                f"Missing one-to-many joblib: `{dataframe_joblib_path}`. "
                "Run `examples/aramis_dataframe_one_to_many_v0_1.py` first."
            ),
        )
    one_to_many_df = load_one_to_many_dataframe(dataframe_joblib_path)
    one_to_many_df.shape
    return (one_to_many_df,)


@app.cell(hide_code=True)
def _(mo, one_to_many_df, summarize_one_to_many_dataframe):
    dataset_summary_df = summarize_one_to_many_dataframe(one_to_many_df)
    label_counts_df = (
        one_to_many_df["product_status_group"]
        .value_counts()
        .rename_axis("product_status_group")
        .reset_index(name="rows")
    )
    mo.vstack(
        [
            mo.md("## Loaded One-To-Many DataFrame"),
            mo.ui.table(dataset_summary_df, selection=None),
            mo.md("### Label counts"),
            mo.ui.table(label_counts_df, selection=None),
        ]
    )
    return dataset_summary_df, label_counts_df


@app.cell(hide_code=True)
def _(
    fit_repeated_one_to_many_logistic,
    logreg_c,
    n_splits,
    one_to_many_df,
    random_state,
    test_size,
):
    model_result = fit_repeated_one_to_many_logistic(
        one_to_many_df,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        logreg_c=logreg_c,
    )
    split_metrics_df = model_result.split_metrics
    predictions_df = model_result.predictions
    return model_result, predictions_df, split_metrics_df


@app.cell(hide_code=True)
def _(mo, split_metrics_df):
    _summary_df = split_metrics_df[
        [
            "roc_auc",
            "train_rows",
            "test_rows",
            "train_patients",
            "test_patients",
            "test_benign_rows",
            "test_cancer_rows",
        ]
    ].agg(["mean", "std", "min", "max"])
    mo.vstack(
        [
            mo.md("## Repeated 70/30 Patient-Safe Split Metrics"),
            mo.ui.table(split_metrics_df.round(4), selection=None),
            mo.md("### Summary"),
            mo.ui.table(_summary_df.round(4).reset_index(names="metric"), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(model_result, plt):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for _curve in model_result.roc_curves:
        ax.plot(
            _curve["fpr"],
            _curve["tpr"],
            color="#6b7280",
            alpha=0.25,
            linewidth=1.0,
        )
    ax.plot(
        model_result.mean_fpr,
        model_result.mean_tpr,
        color="#d62728",
        linewidth=2.5,
        label=f"mean AUC={model_result.mean_auc:.3f}",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#111827", linestyle="--", linewidth=1.0)
    ax.set_title("One-to-many BENIGN vs CANCER ROC")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig
    return (fig,)


@app.cell(hide_code=True)
def _(mo, model_result, predictions_df, split_metrics_df):
    _auc_mean = split_metrics_df["roc_auc"].mean()
    _auc_std = split_metrics_df["roc_auc"].std(ddof=0)
    _prediction_rows = len(predictions_df)
    mo.md(
        "\n".join(
            [
                "## Current Baseline Readout",
                "",
                f"- ROC AUC across splits: `{_auc_mean:.3f} +/- {_auc_std:.3f}`",
                f"- mean ROC AUC curve: `{model_result.mean_auc:.3f}`",
                f"- held-out prediction rows across all splits: `{_prediction_rows}`",
                "- output meaning: `p_cancer` is a research-draft decision-support risk score, not autonomous diagnosis.",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
