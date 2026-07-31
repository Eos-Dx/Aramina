"""Output schemas and serialization for recalibrated-joint experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def threshold_score_frame(
    scores: pd.DataFrame,
    *,
    outer_split_id: int | str,
    model_name: str,
    ablation: str,
    threshold_target: float,
    threshold_provenance: str,
    threshold_score_kind: str,
    parameters: dict[str, float],
) -> pd.DataFrame:
    """Preserve exactly the score rows that selected a reported threshold."""
    required = ("patientId", "target_case_id", "label", "p_cancer")
    missing = set(required).difference(scores.columns)
    if missing:
        raise ValueError(f"Threshold scores missing columns: {sorted(missing)}")
    out = scores.copy()
    out["outer_split_id"] = str(outer_split_id)
    if "meta_fold_id" not in out:
        out["meta_fold_id"] = "not_applicable"
    out["model_name"] = model_name
    out["ablation"] = ablation
    out["threshold_target"] = float(threshold_target)
    out["threshold_provenance"] = threshold_provenance
    out["threshold_score_kind"] = threshold_score_kind
    for name in ("lr1_c", "current_lr2_c", "profile_c", "age_c", "symmetry_c"):
        out[name] = parameters.get(name, np.nan)
    return out[
        [
            "outer_split_id", "model_name", "ablation", "meta_fold_id", "patientId",
            "target_case_id", "label", "p_cancer", "threshold_target",
            "threshold_provenance", "threshold_score_kind", "lr1_c", "current_lr2_c",
            "profile_c", "age_c", "symmetry_c",
        ]
    ]


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired outer-fold metrics without calling variation a CI."""
    metric_columns = (
        "roc_auc", "pr_auc", "brier_score", "log_loss", "sensitivity", "specificity",
        "balanced_accuracy", "ppv", "npv", "true_positives", "true_negatives",
        "false_positives", "false_negatives",
    )
    rows: list[dict[str, Any]] = []
    for (model_name, ablation), group in metrics.groupby(["model_name", "ablation"], sort=False):
        row = {"model_name": model_name, "ablation": ablation, "outer_folds": int(len(group))}
        for column in metric_columns:
            row[f"{column}_fold_mean"] = float(group[column].mean())
            row[f"{column}_fold_std"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(metrics: pd.DataFrame, *, reference_model: str) -> pd.DataFrame:
    """Write split-matched deltas versus the explicitly named reference model."""
    reference = metrics.loc[metrics["model_name"].eq(reference_model)].set_index("split_id")
    columns = (
        "roc_auc", "pr_auc", "brier_score", "log_loss", "sensitivity", "specificity",
        "balanced_accuracy",
    )
    rows: list[dict[str, Any]] = []
    for (model_name, ablation), group in metrics.groupby(["model_name", "ablation"], sort=False):
        if model_name == reference_model:
            continue
        paired = group.set_index("split_id").join(reference[list(columns)], rsuffix="_reference")
        for split_id, values in paired.iterrows():
            rows.append(
                {
                    "split_id": int(split_id),
                    "model_name": model_name,
                    "ablation": ablation,
                    **{
                        f"delta_{column}": float(values[column] - values[f"{column}_reference"])
                        for column in columns
                    },
                }
            )
    return pd.DataFrame(rows)


def write_outputs(output, **frames: pd.DataFrame) -> None:
    """Write every declared research output with a stable filename."""
    names = {
        "fold_metrics": "fold_metrics.csv",
        "predictions": "split_predictions.csv",
        "summary": "summary.csv",
        "selection": "regularization_selection.csv",
        "manifest": "fold_manifest.csv",
        "deltas": "paired_fold_deltas.csv",
        "train_all": "train_all_metrics.csv",
        "threshold_scores": "threshold_oof_predictions.csv",
    }
    for key, filename in names.items():
        frames[key].to_csv(output / filename, index=False)


def summary_payload(
    *,
    experiment_name: str,
    metadata: dict[str, str],
    dataframe: pd.DataFrame,
    columns: dict[str, str],
    base: pd.DataFrame,
    controls: dict[str, Any],
    summary: pd.DataFrame,
    train_all: pd.DataFrame,
) -> dict[str, Any]:
    """Build the compact YAML index for all run artifacts."""
    return {
        "experiment": experiment_name,
        "status": "research_only_not_product_compatible",
        "reproducibility": metadata,
        "controls": controls,
        "cohort": {
            "measurements": int(len(dataframe)),
            "patients": int(dataframe[columns["group_column"]].astype(str).nunique()),
            "target_cases": int(len(base)),
            "cancer_target_cases": int((base["label"] == 1).sum()),
            "benign_target_cases": int((base["label"] == 0).sum()),
        },
        "held_out_summary": summary.to_dict(orient="records"),
        "train_all_descriptions": train_all.to_dict(orient="records"),
        "outputs": {
            "fold_metrics": "fold_metrics.csv",
            "split_predictions": "split_predictions.csv",
            "threshold_oof_predictions": "threshold_oof_predictions.csv",
            "regularization_selection": "regularization_selection.csv",
            "fold_manifest": "fold_manifest.csv",
            "paired_fold_deltas": "paired_fold_deltas.csv",
            "train_all_metrics": "train_all_metrics.csv",
        },
    }
