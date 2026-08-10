"""Evaluation summaries, result artifacts, and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

from aramina.model_metrics import binary_metric_values
from aramina.patient_features import TARGET_CASE_ID


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


SUMMARY_METRICS = (
    "roc_auc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "ppv",
    "npv",
    "brier_score",
    "log_loss",
)


def summarize_results(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    train_all_models: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine fold results, repeat-averaged cross-fitted scores, and train-all fit."""
    rows: list[dict[str, Any]] = []
    repeat_averaged_frames: list[pd.DataFrame] = []
    for encoder, group in metrics.groupby("profile_encoder", sort=False):
        prediction_group = predictions[predictions["profile_encoder"] == encoder]
        repeat_averaged = (
            prediction_group.groupby(TARGET_CASE_ID, as_index=False)
            .agg(
                patientId=("patientId", "first"),
                target_side=("target_side", "first"),
                label=("label", "first"),
                p_cancer=("p_cancer", "mean"),
                threshold_target=("threshold_target", "mean"),
            )
            .assign(profile_encoder=encoder)
        )
        repeat_averaged["suggested_class"] = np.where(
            repeat_averaged["p_cancer"]
            >= repeat_averaged["threshold_target"],
            "CANCER",
            "BENIGN",
        )
        repeat_averaged_frames.append(repeat_averaged)
        repeat_averaged_values = binary_metric_values(
            repeat_averaged["label"].to_numpy(dtype=int),
            repeat_averaged["p_cancer"].to_numpy(dtype=float),
            repeat_averaged["threshold_target"].to_numpy(dtype=float),
        )
        row: dict[str, Any] = {
            "profile_encoder": encoder,
            "splits": int(len(group)),
            "repeat_averaged_cross_fitted_target_cases": int(
                len(repeat_averaged)
            ),
        }
        for metric in SUMMARY_METRICS:
            row[f"{metric}_fold_mean"] = float(group[metric].mean())
            row[f"{metric}_fold_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_repeat_averaged_cross_fitted"] = float(
                repeat_averaged_values[metric]
            )
            row[f"{metric}_train_all"] = float(
                train_all_models[encoder]["metrics"][metric]
            )
        row["threshold_train_all"] = float(
            train_all_models[encoder]["thresholds"]["threshold_target"]
        )
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(
        repeat_averaged_frames,
        ignore_index=True,
    )


def build_fold_manifest(
    context: pd.DataFrame,
    split_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    cohort: str,
    folds: int,
    group_column: str,
) -> pd.DataFrame:
    """Record complete target-case membership for every patient-safe fold."""
    rows: list[dict[str, Any]] = []
    for split_id, (train_index, test_index) in enumerate(split_pairs):
        for set_name, indices in (("train", train_index), ("test", test_index)):
            selected = context.iloc[indices]
            for record in selected.itertuples(index=False):
                rows.append(
                    {
                        "cohort": cohort,
                        "split_id": split_id,
                        "repeat_id": split_id // folds,
                        "fold_id": split_id % folds,
                        "set": set_name,
                        "patientId": str(getattr(record, group_column)),
                        TARGET_CASE_ID: str(getattr(record, TARGET_CASE_ID)),
                        "label": int(record.label),
                    }
                )
    manifest = pd.DataFrame(rows)
    set_counts = manifest.groupby(["split_id", "patientId"])["set"].nunique()
    if (set_counts != 1).any():
        raise RuntimeError("Fold manifest contains patient leakage.")
    expected_cases = set(context[TARGET_CASE_ID].astype(str))
    for split_id, group in manifest.groupby("split_id"):
        if set(group[TARGET_CASE_ID]) != expected_cases:
            raise RuntimeError(
                f"Fold manifest split {split_id} does not contain every target case."
            )
        if group[TARGET_CASE_ID].duplicated().any():
            raise RuntimeError(
                f"Fold manifest split {split_id} duplicates target cases."
            )
    return manifest


def paired_fold_deltas(
    metrics: pd.DataFrame,
    *,
    cohort_name: str,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate descriptive paired fold deltas without inferential claims."""
    references = ["raw256"]
    if cohort_name == "common":
        references.insert(0, "raw100")
    fpca_encoders = sorted(
        value
        for value in metrics["profile_encoder"].unique()
        if str(value).startswith("fpca256_")
    )
    delta_rows: list[dict[str, Any]] = []
    for encoder in fpca_encoders:
        candidate = metrics[metrics["profile_encoder"] == encoder].set_index(
            "split_id"
        )
        for reference in references:
            baseline = metrics[
                metrics["profile_encoder"] == reference
            ].set_index("split_id")
            if not candidate.index.equals(baseline.index):
                raise RuntimeError(
                    f"Paired comparison {encoder} vs {reference} has unequal folds."
                )
            for split_id in candidate.index:
                for metric in ("roc_auc", "sensitivity", "specificity"):
                    delta_rows.append(
                        {
                            "cohort": cohort_name,
                            "profile_encoder": encoder,
                            "reference_encoder": reference,
                            "split_id": int(split_id),
                            "repeat_id": int(split_id) // folds,
                            "fold_id": int(split_id) % folds,
                            "metric": metric,
                            "delta": float(
                                candidate.at[split_id, metric]
                                - baseline.at[split_id, metric]
                            ),
                        }
                    )
    deltas = pd.DataFrame(delta_rows)
    summary_rows: list[dict[str, Any]] = []
    for keys, group in deltas.groupby(
        ["cohort", "profile_encoder", "reference_encoder", "metric"],
        sort=False,
    ):
        values = group["delta"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "cohort": keys[0],
                "profile_encoder": keys[1],
                "reference_encoder": keys[2],
                "metric": keys[3],
                "folds": int(values.size),
                "mean_delta": float(np.mean(values)),
                "std_delta": float(np.std(values, ddof=1)),
                "descriptive_quantile_2_5": float(np.quantile(values, 0.025)),
                "descriptive_quantile_97_5": float(np.quantile(values, 0.975)),
                "interval_interpretation": (
                    "descriptive_quantiles_not_inferential_ci_"
                    "because_repeated_folds_overlap"
                ),
            }
        )
    return deltas, pd.DataFrame(summary_rows)


def write_outputs(
    result: dict[str, Any],
    output_folder: Path,
    *,
    config: dict[str, Any],
) -> None:
    """Write the complete readable and executable research footprint."""
    output_folder.mkdir(parents=True, exist_ok=True)
    result["fold_metrics"].to_csv(output_folder / "fold_metrics.csv", index=False)
    result["fold_predictions"].to_csv(
        output_folder / "fold_predictions.csv", index=False
    )
    result["aggregate_summary"].to_csv(
        output_folder / "aggregate_summary.csv", index=False
    )
    result["repeat_averaged_cross_fitted_predictions"].to_csv(
        output_folder / "repeat_averaged_cross_fitted_predictions.csv",
        index=False,
    )
    result["fold_manifest"].to_csv(
        output_folder / "fold_manifest.csv",
        index=False,
    )
    result["paired_fold_deltas"].to_csv(
        output_folder / "paired_fold_deltas.csv",
        index=False,
    )
    result["paired_delta_summary"].to_csv(
        output_folder / "paired_delta_summary.csv",
        index=False,
    )
    result["pca_explained_variance"].to_csv(
        output_folder / "pca_explained_variance.csv", index=False
    )
    result["train_all_basis"].to_csv(
        output_folder / "pca_basis_components.csv", index=False
    )
    joblib.dump(result["fold_pca_basis"], output_folder / "pca_fold_basis.joblib")
    joblib.dump(
        {
            "contract": result["contract"],
            "clinical_stage": result["clinical_stage"],
            "cohort": result["cohort"],
            "dataset": result["dataset"],
            "lineage": result["lineage"],
            "controlled_variables": result["controlled_variables"],
            "models": result["train_all_models"],
        },
        output_folder / "train_all_artifact.joblib",
    )
    readable = {
        key: result[key]
        for key in (
            "contract",
            "clinical_stage",
            "cohort",
            "dataset",
            "lineage",
            "controlled_variables",
            "fpca_definition",
        )
    }
    readable["aggregate_summary"] = result["aggregate_summary"].to_dict(
        orient="records"
    )
    readable["train_all_metrics"] = {
        name: value["metrics"] for name, value in result["train_all_models"].items()
    }
    (output_folder / "aggregate_summary.yaml").write_text(
        yaml.safe_dump(_yaml_safe(readable), sort_keys=False),
        encoding="utf-8",
    )
    (output_folder / "effective_experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _plot_roc(
        result["repeat_averaged_cross_fitted_predictions"],
        output_folder / "roc_comparison.png",
    )
    _plot_convergence(
        result["aggregate_summary"],
        output_folder / "fpca_component_convergence.png",
    )


def record_pca(
    encoder: Any,
    *,
    scope: str,
    split_id: int,
    variance_rows: list[dict[str, Any]],
    basis_store: dict[str, dict[str, Any]] | None = None,
    basis_frames: list[pd.DataFrame] | None = None,
) -> None:
    """Record fitted PCA variance and basis information."""
    pca = encoder.pca_
    if pca is None:
        return
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for index, (variance, cumulative_variance) in enumerate(
        zip(pca.explained_variance_ratio_, cumulative, strict=True),
        start=1,
    ):
        variance_rows.append(
            {
                "profile_encoder": encoder.spec.name,
                "scope": scope,
                "split_id": split_id,
                "component": index,
                "explained_variance_ratio": float(variance),
                "cumulative_explained_variance_ratio": float(cumulative_variance),
            }
        )
    if basis_store is not None:
        basis_store[f"{encoder.spec.name}::split_{split_id}"] = {
            "q_grid": encoder.q_grid_,
            "components": pca.components_,
            "mean_profile": pca.mean_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "training_patient_ids": sorted(encoder.training_patient_ids_),
        }
    if basis_frames is not None:
        frame = pd.DataFrame({"q_nm_inv": encoder.q_grid_})
        frame.insert(0, "profile_encoder", encoder.spec.name)
        for component_index, values in enumerate(pca.components_, start=1):
            frame[f"component_{component_index}"] = values
        frame["mean_profile"] = pca.mean_
        basis_frames.append(frame)


def dataset_summary(
    datasets: dict[int, pd.DataFrame],
    contexts: dict[int, pd.DataFrame],
) -> dict[int, dict[str, int]]:
    """Describe rows, patients, and target cases for each npt input."""
    return {
        npt: {
            "rows": int(len(df)),
            "patients": int(df["patientId"].astype(str).nunique()),
            "target_cases": int(len(contexts[npt])),
            "cancer_target_cases": int((contexts[npt]["label"] == 1).sum()),
            "benign_target_cases": int((contexts[npt]["label"] == 0).sum()),
        }
        for npt, df in datasets.items()
    }


def _plot_roc(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    for encoder, group in predictions.groupby("profile_encoder", sort=False):
        fpr, tpr, _ = roc_curve(group["label"], group["p_cancer"])
        auc = binary_metric_values(
            group["label"].to_numpy(dtype=int),
            group["p_cancer"].to_numpy(dtype=float),
            group["threshold_target"].to_numpy(dtype=float),
        )["roc_auc"]
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{encoder} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_title("Repeat-averaged cross-fitted ROC comparison")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_convergence(summary: pd.DataFrame, output_path: Path) -> None:
    fpca = summary[summary["profile_encoder"].str.startswith("fpca256_")].copy()
    fpca["components"] = fpca["profile_encoder"].str.rsplit("_", n=1).str[-1].astype(int)
    fpca = fpca.sort_values("components")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("roc_auc", "sensitivity", "specificity"),
        ("ROC AUC", "Sensitivity", "Specificity"),
        strict=True,
    ):
        axis.errorbar(
            fpca["components"],
            fpca[f"{metric}_fold_mean"],
            yerr=fpca[f"{metric}_fold_std"],
            marker="o",
            color="#276FBF",
            capsize=3,
            label="FPCA256",
        )
        for baseline, color in (("raw256", "#C44536"), ("raw100", "#5B8E7D")):
            row = summary[summary["profile_encoder"] == baseline]
            if not row.empty:
                axis.axhline(
                    float(row.iloc[0][f"{metric}_fold_mean"]),
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                    label=baseline,
                )
        axis.set(title=title, xlabel="FPCA components", xticks=fpca["components"])
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
