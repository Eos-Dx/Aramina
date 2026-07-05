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
        fit_repeated_fusion_logistic_models,
        fusion_ablation_feature_sets,
        load_one_to_many_dataframe,
    )

    PRODUCT_DIR = Path(__file__).resolve().parent
    DEFAULT_ONE_TO_MANY_JOBLIB_PATH = (
        PRODUCT_DIR
        / "outputs"
        / "real_h5_yaml_validation"
        / "aramis_one_to_many_biopsy_min_v0_1.joblib"
    )
    DEFAULT_ONE_TO_ONE_JOBLIB_PATH = (
        PRODUCT_DIR
        / "outputs"
        / "real_h5_yaml_validation"
        / "aramis_one_to_one_biopsy_min_v0_1.joblib"
    )
    MODEL_COLORS = {
        "M0_one_to_many_only": "#4c78a8",
        "M1_one_to_many_plus_symmetry": "#59a14f",
        "M2_plus_quality": "#f28e2b",
        "M3_plus_age_bmi": "#e15759",
        "A0_age_only": "#9c755f",
        "A1_bmi_only": "#b07aa1",
        "A2_availability_only": "#bab0ac",
        "F0_symmetry_available_only": "#86bc86",
        "F1_bmi_available_only": "#d4a6c8",
        "F2_replicate_available_only": "#c7c7c7",
        "F3_symmetry_plus_bmi_availability": "#8cd17d",
        "T0_thickness_only": "#ff9da7",
        "S1_cosine_symmetry_block": "#4e79a7",
        "M3a_plus_age_no_bmi": "#76b7b2",
        "M3b_plus_bmi_no_age": "#edc948",
    }
    MODEL_LABELS = {
        "M0_one_to_many_only": "M0: one-to-many",
        "M1_one_to_many_plus_symmetry": "M1: + symmetry",
        "M2_plus_quality": "M2: + quality",
        "M3_plus_age_bmi": "M3: + age/BMI",
        "A0_age_only": "A0: age only",
        "A1_bmi_only": "A1: BMI only",
        "A2_availability_only": "A2: availability only",
        "F0_symmetry_available_only": "F0: symmetry flag only",
        "F1_bmi_available_only": "F1: BMI flag only",
        "F2_replicate_available_only": "F2: replicate flag only",
        "F3_symmetry_plus_bmi_availability": "F3: symmetry+BMI flags",
        "T0_thickness_only": "T0: thickness only control",
        "S1_cosine_symmetry_block": "S1: cosine symmetry block",
        "M3a_plus_age_no_bmi": "M3a: + age, no BMI",
        "M3b_plus_bmi_no_age": "M3b: + BMI, no age",
    }
    return (
        DEFAULT_ONE_TO_MANY_JOBLIB_PATH,
        DEFAULT_ONE_TO_ONE_JOBLIB_PATH,
        MODEL_COLORS,
        MODEL_LABELS,
        Path,
        fit_repeated_fusion_logistic_models,
        fusion_ablation_feature_sets,
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
                "# Aramis Final Experimental Model v0.1",
                "",
                "Research draft notebook.",
                "",
                "Goal: compare first fusion concepts M0-M3 and ablation controls using biopsy-only one-to-many target specimens and paired-breast symmetry features.",
                "",
                "Clinical framing: decision support only. Output is experimental p_cancer behavior and requires radiologist / qualified clinician review.",
                "",
                "Split rule: patient-safe repeated 70/30 split. One patientId is never present in both train and test inside a split.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(DEFAULT_ONE_TO_MANY_JOBLIB_PATH, DEFAULT_ONE_TO_ONE_JOBLIB_PATH, Path, mo):
    _cli_args = mo.cli_args()
    one_to_many_joblib_path = Path(
        _cli_args.get("one-to-many-joblib-path")
        or _cli_args.get("one_to_many_joblib_path")
        or DEFAULT_ONE_TO_MANY_JOBLIB_PATH
    )
    one_to_one_joblib_path = Path(
        _cli_args.get("one-to-one-joblib-path")
        or _cli_args.get("one_to_one_joblib_path")
        or DEFAULT_ONE_TO_ONE_JOBLIB_PATH
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
    return (
        aggregation,
        inner_splits,
        logreg_c,
        n_splits,
        one_to_many_joblib_path,
        one_to_one_joblib_path,
        random_state,
        target_sensitivity,
        test_size,
    )


@app.cell(hide_code=True)
def _(
    aggregation,
    inner_splits,
    logreg_c,
    mo,
    n_splits,
    one_to_many_joblib_path,
    one_to_one_joblib_path,
    random_state,
    target_sensitivity,
    test_size,
):
    mo.md(
        "\n".join(
            [
                "## Settings",
                "",
                f"- one-to-many biopsy DataFrame: `{one_to_many_joblib_path}`",
                f"- one-to-one paired DataFrame: `{one_to_one_joblib_path}`",
                f"- outer splits: `{n_splits}`",
                f"- outer split test size: `{test_size:.2f}`",
                f"- inner one-to-many OOF splits: `{inner_splits}`",
                f"- random state: `{random_state}`",
                f"- LogisticRegression C: `{logreg_c}`",
                f"- specimen aggregation: `{aggregation}`",
                f"- target sensitivity threshold: `{target_sensitivity:.2f}`",
                "- sample thickness: audit/control metadata only; evaluated by `T0_thickness_only`, not used in M0-M3 predictors",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(load_one_to_many_dataframe, mo, one_to_many_joblib_path, one_to_one_joblib_path):
    _missing = [
        str(_path)
        for _path in (one_to_many_joblib_path, one_to_one_joblib_path)
        if not _path.exists()
    ]
    mo.stop(
        bool(_missing),
        mo.md(
            "Missing preprocessing joblib(s): "
            + ", ".join(f"`{_path}`" for _path in _missing)
            + ". Run one-to-many biopsy and one-to-one preprocessing notebooks first."
        ),
    )
    one_to_many_df = load_one_to_many_dataframe(one_to_many_joblib_path)
    one_to_one_df = load_one_to_many_dataframe(one_to_one_joblib_path)
    return one_to_many_df, one_to_one_df


@app.cell(hide_code=True)
def _(mo, one_to_many_df, one_to_one_df):
    mo.md(
        "\n".join(
            [
                "## Input Data",
                "",
                f"- one-to-many rows: `{len(one_to_many_df)}`",
                f"- one-to-many patients: `{one_to_many_df['patientId'].astype(str).nunique()}`",
                f"- one-to-many target specimens: `{one_to_many_df['specimenId'].astype(str).nunique()}`",
                f"- one-to-one rows: `{len(one_to_one_df)}`",
                f"- one-to-one patients: `{one_to_one_df['patientId'].astype(str).nunique()}`",
                f"- one-to-one specimens: `{one_to_one_df['specimenId'].astype(str).nunique()}`",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(
    aggregation,
    fit_repeated_fusion_logistic_models,
    fusion_ablation_feature_sets,
    inner_splits,
    logreg_c,
    n_splits,
    one_to_many_df,
    one_to_one_df,
    random_state,
    target_sensitivity,
    test_size,
):
    fusion_result = fit_repeated_fusion_logistic_models(
        one_to_many_df,
        one_to_one_df,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        logreg_c=logreg_c,
        inner_splits=inner_splits,
        target_sensitivity=target_sensitivity,
        aggregation=aggregation,
        feature_sets=fusion_ablation_feature_sets(),
    )
    feature_table_df = fusion_result.feature_table
    metric_summary_df = fusion_result.metric_summary
    split_metrics_df = fusion_result.split_metrics
    return fusion_result, feature_table_df, metric_summary_df, split_metrics_df


@app.cell(hide_code=True)
def _(feature_table_df, mo):
    _symmetry_available = int(feature_table_df["symmetry_available"].astype(bool).sum())
    _age_available = int(feature_table_df["age_available"].astype(bool).sum())
    _bmi_available = int(feature_table_df["bmi_available"].astype(bool).sum())
    _thickness_available = int(
        feature_table_df["sample_thickness_available"].astype(bool).sum()
    )
    mo.md(
        "\n".join(
            [
                "## Feature Coverage",
                "",
                f"- target specimens: `{len(feature_table_df)}`",
                f"- symmetry available: `{_symmetry_available}`",
                f"- symmetry unavailable: `{len(feature_table_df) - _symmetry_available}`",
                f"- age available: `{_age_available}`",
                f"- BMI available: `{_bmi_available}`",
                f"- sample thickness available: `{_thickness_available}`",
                f"- median sample thickness, mm: `{feature_table_df['sample_thickness_mm_mean'].median():.3f}`",
                f"- median valid target measurements: `{feature_table_df['n_valid_target_measurements'].median():.1f}`",
                f"- median valid contralateral measurements: `{feature_table_df['n_valid_contralateral_measurements'].median():.1f}`",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(feature_table_df, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    _symmetry = feature_table_df[
        feature_table_df["symmetry_available"].astype(bool)
    ]["symmetry_distance_value"].to_numpy(dtype=float)
    _axes[0].hist(_symmetry, bins=24, color="#59a14f", alpha=0.82)
    _axes[0].set_title("Symmetry distance values")
    _axes[0].set_xlabel("between - within cosine distance")
    _axes[0].set_ylabel("specimens")
    _axes[0].grid(alpha=0.25)

    _counts = [
        int(feature_table_df["symmetry_available"].astype(bool).sum()),
        int((~feature_table_df["symmetry_available"].astype(bool)).sum()),
    ]
    _bars = _axes[1].bar(
        np.arange(2),
        _counts,
        color=["#59a14f", "#9ca3af"],
        alpha=0.86,
    )
    _axes[1].bar_label(_bars, fmt="%.0f", padding=3)
    _axes[1].set_xticks(np.arange(2))
    _axes[1].set_xticklabels(["available", "missing"])
    _axes[1].set_title("Symmetry feature availability")
    _axes[1].set_ylabel("specimens")
    _axes[1].grid(axis="y", alpha=0.25)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(MODEL_LABELS, metric_summary_df, mo):
    _lines = ["## M0-M3 And Ablation Comparison", ""]
    for _row in metric_summary_df.itertuples(index=False):
        _label = MODEL_LABELS.get(_row.model_name, _row.model_name)
        _lines.extend(
            [
                f"### {_label}",
                f"- ROC AUC: `{_row.roc_auc_mean:.3f} +/- {_row.roc_auc_std:.3f}`",
                f"- PR AUC mean: `{_row.pr_auc_mean:.3f}`",
                f"- target-threshold sensitivity mean: `{_row.sensitivity_target_mean:.3f}`",
                f"- target-threshold specificity mean: `{_row.specificity_target_mean:.3f}`",
                f"- Youden sensitivity mean: `{_row.sensitivity_youden_mean:.3f}`",
                f"- Youden specificity mean: `{_row.specificity_youden_mean:.3f}`",
                "",
            ]
        )
    _lines.append(
        "Thresholds are selected on train rows and evaluated on held-out test patients."
    )
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(MODEL_COLORS, MODEL_LABELS, fusion_result, plt):
    _fig, _ax = plt.subplots(figsize=(7.8, 6.0))
    for _model_name, _curves in fusion_result.roc_curves.items():
        _color = MODEL_COLORS.get(_model_name, "#6b7280")
        for _curve in _curves:
            _ax.plot(
                _curve["fpr"],
                _curve["tpr"],
                color=_color,
                alpha=0.12,
                linewidth=0.8,
            )
        _mean = fusion_result.mean_roc_curves[_model_name]
        _ax.plot(
            _mean["mean_fpr"],
            _mean["mean_tpr"],
            color=_color,
            linewidth=2.6,
            label=f"{MODEL_LABELS.get(_model_name, _model_name)}: AUC={_mean['mean_auc']:.3f}",
        )
    _ax.plot([0.0, 1.0], [0.0, 1.0], color="#111827", linestyle="--", linewidth=1.0)
    _ax.set_title("Fusion ROC comparison")
    _ax.set_xlabel("False positive rate")
    _ax.set_ylabel("True positive rate")
    _ax.set_xlim(0.0, 1.0)
    _ax.set_ylim(0.0, 1.02)
    _ax.grid(alpha=0.25)
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(MODEL_COLORS, MODEL_LABELS, np, plt, split_metrics_df):
    _fig, _ax = plt.subplots(figsize=(9.2, 4.8))
    for _model_name, _group_df in split_metrics_df.groupby("model_name", sort=False):
        _label = MODEL_LABELS.get(_model_name, _model_name)
        _ax.hist(
            _group_df["roc_auc"],
            bins=np.linspace(0.0, 1.0, 16),
            alpha=0.34,
            color=MODEL_COLORS.get(_model_name, "#6b7280"),
            label=f"{_label} mean={_group_df['roc_auc'].mean():.3f}",
        )
    _ax.set_title("ROC AUC distribution across patient-safe splits")
    _ax.set_xlabel("ROC AUC")
    _ax.set_ylabel("split count")
    _ax.grid(alpha=0.25)
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(MODEL_COLORS, MODEL_LABELS, metric_summary_df, np, plt):
    _metrics = [
        ("roc_auc_mean", "ROC AUC"),
        ("pr_auc_mean", "PR AUC"),
        ("sensitivity_target_mean", "target sens."),
        ("specificity_target_mean", "target spec."),
        ("sensitivity_youden_mean", "Youden sens."),
        ("specificity_youden_mean", "Youden spec."),
    ]
    _x = np.arange(len(_metrics))
    _model_count = max(1, len(metric_summary_df))
    _width = min(0.12, 0.82 / _model_count)
    _fig, _ax = plt.subplots(figsize=(12.0, 5.2))
    for _idx, _row in enumerate(metric_summary_df.itertuples(index=False)):
        _values = [getattr(_row, _metric[0]) for _metric in _metrics]
        _offset = (_idx - ((_model_count - 1) / 2.0)) * _width
        _bars = _ax.bar(
            _x + _offset,
            _values,
            width=_width,
            color=MODEL_COLORS.get(_row.model_name, "#6b7280"),
            label=MODEL_LABELS.get(_row.model_name, _row.model_name),
            alpha=0.86,
        )
        _ax.bar_label(_bars, fmt="%.2f", padding=3, fontsize=7)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([_metric[1] for _metric in _metrics], rotation=20, ha="right")
    _ax.set_ylim(0.0, 1.05)
    _ax.set_ylabel("mean score")
    _ax.set_title("Mean metrics across patient-safe splits")
    _ax.grid(axis="y", alpha=0.25)
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(MODEL_LABELS, fusion_result, metric_summary_df, mo):
    _best = metric_summary_df.sort_values("roc_auc_mean", ascending=False).iloc[0]
    _lines = [
        "## In-Memory Artifacts",
        "",
        f"- one-to-many specimen prediction rows: `{len(fusion_result.one_to_many_result.specimen_predictions)}`",
        f"- fusion prediction rows: `{len(fusion_result.predictions)}`",
        f"- split metric rows: `{len(fusion_result.split_metrics)}`",
        f"- feature table rows: `{len(fusion_result.feature_table)}`",
        "",
        f"Highest mean ROC AUC in this run: `{MODEL_LABELS.get(_best.model_name, _best.model_name)}` (`{_best.roc_auc_mean:.3f}`).",
        "",
        "No model joblib is exported in this notebook. This is a research-draft concept comparison before fixing the product model pipeline.",
    ]
    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
