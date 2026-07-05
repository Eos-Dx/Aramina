import marimo

__generated_with = "0.23.9"
app = marimo.App(width="wide")


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
        fit_repeated_one_to_many_product_feature_ablation_comparison,
        load_one_to_many_dataframe,
    )

    PRODUCT_DIR = Path(__file__).resolve().parent
    DEFAULT_STANDARD_JOBLIB_PATH = (
        PRODUCT_DIR
        / "outputs"
        / "real_h5_yaml_validation"
        / "aramis_one_to_many_min_v0_1.joblib"
    )
    DEFAULT_BIOPSY_JOBLIB_PATH = (
        PRODUCT_DIR
        / "outputs"
        / "real_h5_yaml_validation"
        / "aramis_one_to_many_biopsy_min_v0_1.joblib"
    )
    DATASET_COLORS = {
        "standard": "#4c78a8",
        "biopsy_only": "#d62728",
    }
    FEATURE_ROUTE_COLORS = {
        "profile_only": "#4c78a8",
        "thickness_only": "#ff9da7",
        "profile_plus_thickness": "#59a14f",
    }
    return (
        DATASET_COLORS,
        DEFAULT_BIOPSY_JOBLIB_PATH,
        DEFAULT_STANDARD_JOBLIB_PATH,
        FEATURE_ROUTE_COLORS,
        Path,
        fit_repeated_one_to_many_product_feature_ablation_comparison,
        load_one_to_many_dataframe,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "\n".join(
            [
                "# Aramis One-To-Many Model Comparison v0.1",
                "",
                "Research draft model notebook.",
                "",
                "Goal: compare the standard one-to-many BENIGN/CANCER DataFrame with the biopsy-only one-to-many BENIGN/CANCER DataFrame.",
                "",
                "Model routes for each DataFrame: profile only, sample-thickness only, and profile plus sample thickness. This tests whether thickness alone predicts label and whether it improves the profile model.",
                "",
                "Split rule: patient-safe repeated 70/30 split. One patientId is never present in both train and test inside a split.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(DEFAULT_BIOPSY_JOBLIB_PATH, DEFAULT_STANDARD_JOBLIB_PATH, Path, mo):
    _cli_args = mo.cli_args()
    standard_joblib_path = Path(
        _cli_args.get("standard-dataframe-joblib-path")
        or _cli_args.get("standard_dataframe_joblib_path")
        or _cli_args.get("dataframe-joblib-path")
        or _cli_args.get("dataframe_joblib_path")
        or DEFAULT_STANDARD_JOBLIB_PATH
    )
    biopsy_joblib_path = Path(
        _cli_args.get("biopsy-dataframe-joblib-path")
        or _cli_args.get("biopsy_dataframe_joblib_path")
        or DEFAULT_BIOPSY_JOBLIB_PATH
    )
    n_splits = int(_cli_args.get("n-splits") or _cli_args.get("n_splits") or 20)
    test_size = float(_cli_args.get("test-size") or _cli_args.get("test_size") or 0.30)
    random_state = int(
        _cli_args.get("random-state") or _cli_args.get("random_state") or 42
    )
    logreg_c = float(_cli_args.get("logreg-c") or _cli_args.get("logreg_c") or 1.0)
    inner_splits = int(
        _cli_args.get("inner-splits") or _cli_args.get("inner_splits") or 5
    )
    target_sensitivity = float(
        _cli_args.get("target-sensitivity")
        or _cli_args.get("target_sensitivity")
        or 0.95
    )
    aggregation = str(_cli_args.get("aggregation") or "mean")
    feature_routes = {
        "profile_only": {
            "use_profile": True,
            "extra_feature_columns": [],
        },
        "thickness_only": {
            "use_profile": False,
            "extra_feature_columns": ["sample_thickness_mm"],
        },
        "profile_plus_thickness": {
            "use_profile": True,
            "extra_feature_columns": ["sample_thickness_mm"],
        },
    }
    return (
        aggregation,
        biopsy_joblib_path,
        feature_routes,
        inner_splits,
        logreg_c,
        n_splits,
        random_state,
        standard_joblib_path,
        target_sensitivity,
        test_size,
    )


@app.cell(hide_code=True)
def _(
    aggregation,
    biopsy_joblib_path,
    feature_routes,
    inner_splits,
    logreg_c,
    mo,
    n_splits,
    random_state,
    standard_joblib_path,
    target_sensitivity,
    test_size,
):
    mo.md(
        "\n".join(
            [
                "## Settings",
                "",
                f"- standard DataFrame joblib: `{standard_joblib_path}`",
                f"- biopsy-only DataFrame joblib: `{biopsy_joblib_path}`",
                f"- outer splits: `{n_splits}`",
                f"- outer split test size: `{test_size:.2f}`",
                f"- inner OOF splits: `{inner_splits}`",
                f"- random state: `{random_state}`",
                f"- LogisticRegression C: `{logreg_c}`",
                f"- specimen aggregation: `{aggregation}`",
                f"- target sensitivity threshold: `{target_sensitivity:.2f}`",
                f"- feature routes: `{list(feature_routes)}`",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(biopsy_joblib_path, load_one_to_many_dataframe, mo, standard_joblib_path):
    _missing = [
        str(_path)
        for _path in (standard_joblib_path, biopsy_joblib_path)
        if not _path.exists()
    ]
    mo.stop(
        bool(_missing),
        mo.md(
            "Missing one-to-many joblib(s): "
            + ", ".join(f"`{_path}`" for _path in _missing)
            + ". Run `examples/aramis_dataframe_one_to_many_v0_1.py` for standard and biopsy-only outputs first."
        ),
    )
    one_to_many_datasets = {
        "standard": load_one_to_many_dataframe(standard_joblib_path),
        "biopsy_only": load_one_to_many_dataframe(biopsy_joblib_path),
    }
    return (one_to_many_datasets,)


@app.cell(hide_code=True)
def _(
    aggregation,
    feature_routes,
    fit_repeated_one_to_many_product_feature_ablation_comparison,
    inner_splits,
    logreg_c,
    n_splits,
    one_to_many_datasets,
    random_state,
    target_sensitivity,
    test_size,
):
    comparison_result = fit_repeated_one_to_many_product_feature_ablation_comparison(
        one_to_many_datasets,
        feature_routes,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        logreg_c=logreg_c,
        inner_splits=inner_splits,
        target_sensitivity=target_sensitivity,
        aggregation=aggregation,
    )
    dataset_summary_df = comparison_result.dataset_summary
    metric_summary_df = comparison_result.metric_summary
    return comparison_result, dataset_summary_df, metric_summary_df


@app.cell(hide_code=True)
def _(dataset_summary_df, mo):
    _lines = ["## DataFrame Comparison", ""]
    _dataset_rows = dataset_summary_df.drop_duplicates("dataset")
    for _row in _dataset_rows.itertuples(index=False):
        _lines.extend(
            [
                f"### {_row.dataset}",
                f"- measurements: `{int(_row.rows)}`",
                f"- patients: `{int(_row.patients)}`",
                f"- specimens/breasts: `{int(_row.specimens)}`",
                f"- BENIGN rows: `{int(_row.BENIGN_rows)}`",
                f"- CANCER rows: `{int(_row.CANCER_rows)}`",
                f"- biopsy true rows: `{int(_row.biopsy_true_rows)}`",
                "",
            ]
        )
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(DATASET_COLORS, dataset_summary_df, np, plt):
    _metrics = ["rows", "patients", "specimens"]
    _dataset_rows = dataset_summary_df.drop_duplicates("dataset")
    _x = np.arange(len(_metrics))
    _width = 0.36
    _fig, _ax = plt.subplots(figsize=(9.5, 4.8))
    for _idx, _row in enumerate(_dataset_rows.itertuples(index=False)):
        _values = [getattr(_row, _metric) for _metric in _metrics]
        _offset = (_idx - 0.5) * _width
        _bars = _ax.bar(
            _x + _offset,
            _values,
            width=_width,
            label=_row.dataset,
            color=DATASET_COLORS.get(_row.dataset, "#6b7280"),
            alpha=0.86,
        )
        _ax.bar_label(_bars, fmt="%.0f", padding=3, fontsize=9)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(["measurements", "patients", "specimens"])
    _ax.set_ylabel("count")
    _ax.set_title("Input DataFrame size comparison")
    _ax.grid(axis="y", alpha=0.25)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(metric_summary_df, mo):
    _lines = ["## Model Comparison", ""]
    for _row in metric_summary_df.itertuples(index=False):
        _lines.extend(
            [
                f"### {_row.dataset}: {_row.feature_route}",
                f"- ROC AUC: `{_row.roc_auc_mean:.3f} +/- {_row.roc_auc_std:.3f}`",
                f"- PR AUC mean: `{_row.pr_auc_mean:.3f}`",
                f"- target-threshold sensitivity mean: `{_row.sensitivity_target_mean:.3f}`",
                f"- target-threshold specificity mean: `{_row.specificity_target_mean:.3f}`",
                f"- Youden sensitivity mean: `{_row.sensitivity_youden_mean:.3f}`",
                f"- Youden specificity mean: `{_row.specificity_youden_mean:.3f}`",
                "",
            ]
        )
    _thickness_rows = metric_summary_df[
        metric_summary_df["feature_route"] == "thickness_only"
    ]
    _lines.extend(
        [
            "### Thickness-control note",
            "If `thickness_only` performs far above random, sample thickness is label-associated and must be treated as possible confounding.",
            "",
            "Thresholds are selected on train OOF specimen scores and evaluated on held-out test patients.",
        ]
    )
    for _row in _thickness_rows.itertuples(index=False):
        _lines.append(
            f"- {_row.dataset} thickness-only ROC AUC: `{_row.roc_auc_mean:.3f} +/- {_row.roc_auc_std:.3f}`"
        )
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(FEATURE_ROUTE_COLORS, comparison_result, plt):
    _fig, _ax = plt.subplots(figsize=(7.8, 6.0))
    for _name, _result in comparison_result.results.items():
        _dataset, _feature_route = _name.split("__", 1)
        _color = FEATURE_ROUTE_COLORS.get(_feature_route, "#6b7280")
        _linestyle = "-" if _dataset == "biopsy_only" else "--"
        for _curve in _result.roc_curves:
            _ax.plot(
                _curve["fpr"],
                _curve["tpr"],
                color=_color,
                alpha=0.14,
                linewidth=0.9,
                linestyle=_linestyle,
            )
        _ax.plot(
            _result.mean_fpr,
            _result.mean_tpr,
            color=_color,
            linewidth=2.6,
            linestyle=_linestyle,
            label=f"{_name}: mean AUC={_result.mean_auc:.3f}",
        )
    _ax.plot([0.0, 1.0], [0.0, 1.0], color="#111827", linestyle="--", linewidth=1.0)
    _ax.set_title("One-to-many specimen-level ROC comparison")
    _ax.set_xlabel("False positive rate")
    _ax.set_ylabel("True positive rate")
    _ax.set_xlim(0.0, 1.0)
    _ax.set_ylim(0.0, 1.02)
    _ax.grid(alpha=0.25)
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(FEATURE_ROUTE_COLORS, comparison_result, plt):
    _fig, _ax = plt.subplots(figsize=(8.8, 4.8))
    for _name, _result in comparison_result.results.items():
        _dataset, _feature_route = _name.split("__", 1)
        _metrics = _result.split_metrics
        _ax.hist(
            _metrics["roc_auc"],
            bins=10,
            alpha=0.42,
            color=FEATURE_ROUTE_COLORS.get(_feature_route, "#6b7280"),
            label=f"{_name} mean={_metrics['roc_auc'].mean():.3f}",
        )
    _ax.set_title("ROC AUC distribution across patient-safe splits")
    _ax.set_xlabel("ROC AUC")
    _ax.set_ylabel("split count")
    _ax.grid(alpha=0.25)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(FEATURE_ROUTE_COLORS, metric_summary_df, np, plt):
    _metrics = [
        ("roc_auc_mean", "ROC AUC"),
        ("pr_auc_mean", "PR AUC"),
        ("sensitivity_target_mean", "target sens."),
        ("specificity_target_mean", "target spec."),
        ("sensitivity_youden_mean", "Youden sens."),
        ("specificity_youden_mean", "Youden spec."),
    ]
    _x = np.arange(len(_metrics))
    _width = 0.12
    _fig, _ax = plt.subplots(figsize=(13.0, 5.4))
    _n_rows = len(metric_summary_df)
    for _idx, _row in enumerate(metric_summary_df.itertuples(index=False)):
        _values = [getattr(_row, _metric[0]) for _metric in _metrics]
        _offset = (_idx - (_n_rows - 1) / 2) * _width
        _bars = _ax.bar(
            _x + _offset,
            _values,
            width=_width,
            color=FEATURE_ROUTE_COLORS.get(_row.feature_route, "#6b7280"),
            label=f"{_row.dataset}: {_row.feature_route}",
            alpha=0.86,
        )
        _ax.bar_label(_bars, fmt="%.2f", padding=3, fontsize=8)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([_metric[1] for _metric in _metrics], rotation=20, ha="right")
    _ax.set_ylim(0.0, 1.05)
    _ax.set_ylabel("mean score")
    _ax.set_title("Mean metrics across patient-safe splits")
    _ax.grid(axis="y", alpha=0.25)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(comparison_result, metric_summary_df, mo):
    _artifact_lines = [
        "## Generated In-Memory Artifacts",
        "",
        f"- compared datasets: `{', '.join(comparison_result.results)}`",
    ]
    for _name, _result in comparison_result.results.items():
        _artifact_lines.extend(
            [
                f"- `{_name}` split metrics rows: `{len(_result.split_metrics)}`",
                f"- `{_name}` specimen prediction rows: `{len(_result.specimen_predictions)}`",
                f"- `{_name}` measurement prediction rows: `{len(_result.measurement_predictions)}`",
            ]
        )
    _best_auc_row = metric_summary_df.sort_values("roc_auc_mean", ascending=False).iloc[0]
    _artifact_lines.extend(
        [
            "",
            f"Highest mean ROC AUC in this run: `{_best_auc_row.dataset}: {_best_auc_row.feature_route}` (`{_best_auc_row.roc_auc_mean:.3f}`).",
            "",
            "No model joblib is exported in this notebook. This notebook compares research-draft preprocessing cohorts before fixing the final product training pipeline.",
        ]
    )
    mo.md("\n".join(_artifact_lines))
    return


if __name__ == "__main__":
    app.run()
