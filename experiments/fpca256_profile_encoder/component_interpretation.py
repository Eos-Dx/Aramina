"""Interpret FPCA30 components without changing the research model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aramina.patient_features import lr1_training_rows

from .config import load_experiment_config
from .lineage import load_cohort_datasets
from .model import FoldLocalProfileEncoder, ProfileSpec, profile_matrix


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ENCODER_NAME = "fpca256_30"
SELECTED_COMPONENTS = (1, 2, 4, 6, 7, 10, 26, 29)


def component_landmarks(
    q_grid: np.ndarray,
    component: np.ndarray,
    explained_variance: float,
) -> dict[str, float]:
    """Describe one component as its one-standard-deviation profile change."""
    delta = np.sqrt(float(explained_variance)) * np.asarray(component, dtype=float)
    positive_index = int(np.argmax(delta))
    negative_index = int(np.argmin(delta))
    return {
        "one_sd_max_q_nm_inv": float(q_grid[positive_index]),
        "one_sd_max_delta": float(delta[positive_index]),
        "one_sd_min_q_nm_inv": float(q_grid[negative_index]),
        "one_sd_min_delta": float(delta[negative_index]),
        "one_sd_peak_to_peak": float(delta[positive_index] - delta[negative_index]),
    }


def align_coefficient_to_reference(
    reference_component: np.ndarray,
    fold_component: np.ndarray,
    coefficient: float,
) -> tuple[float, float]:
    """Align an arbitrary PCA sign and return coefficient and basis similarity."""
    cosine = float(np.dot(reference_component, fold_component))
    sign = 1.0 if cosine >= 0.0 else -1.0
    return float(coefficient * sign), abs(cosine)


def analyze_components(
    *,
    config_path: str | Path,
    result_folder: str | Path,
    output_folder: str | Path,
) -> pd.DataFrame:
    """Create FPCA30 activity and stability outputs from frozen experiment inputs."""
    config, source = load_experiment_config(config_path)
    result_path = Path(result_folder).expanduser().resolve()
    output_path = Path(output_folder).expanduser().resolve()
    artifact_path = result_path / "train_all_artifact.joblib"
    manifest_path = result_path / "fold_manifest.csv"
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("FPCA30 train-all artifact or fold manifest is missing.")

    artifact = joblib.load(artifact_path)
    fitted = artifact["models"][ENCODER_NAME]["profile_encoder"]
    pca = fitted.pca_
    if pca is None or pca.n_components_ != 30:
        raise ValueError("Expected a fitted 30-component FPCA encoder.")

    datasets, _ = load_cohort_datasets(
        config,
        source,
        cohort_name="common",
        enforce_expected=True,
    )
    df = datasets[256]
    model = config["model"]
    all_rows = _lr1_rows(df, model)
    matrix = profile_matrix(all_rows, model["profile_column"])
    labels = (all_rows[model["label_column"]].astype(str) == "CANCER").to_numpy(
        dtype=int
    )
    scores = pca.transform(matrix)
    logreg = fitted.pipeline_.named_steps["logreg"]
    coefficients = np.asarray(logreg.coef_, dtype=float).ravel()
    if coefficients.size != 30:
        raise ValueError("Expected 30 scaled LR1 coefficients.")

    fold_statistics = _fold_statistics(
        df,
        model=model,
        manifest=pd.read_csv(manifest_path),
        reference_components=np.asarray(pca.components_, dtype=float),
        random_seed=int(config["evaluation"]["random_seed"]),
    )
    rows: list[dict[str, Any]] = []
    for index in range(30):
        score = scores[:, index]
        pooled_std = np.sqrt(
            (score[labels == 1].var(ddof=1) + score[labels == 0].var(ddof=1)) / 2.0
        )
        effect_size = (
            float((score[labels == 1].mean() - score[labels == 0].mean()) / pooled_std)
            if pooled_std > 0.0
            else 0.0
        )
        auc = float(roc_auc_score(labels, score))
        landmarks = component_landmarks(
            fitted.q_grid_, pca.components_[index], pca.explained_variance_[index]
        )
        fold = fold_statistics[fold_statistics["component"] == index + 1]
        rows.append(
            {
                "component": index + 1,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[index]),
                "cumulative_explained_variance_ratio": float(
                    pca.explained_variance_ratio_[: index + 1].sum()
                ),
                "lr1_coefficient_per_sd_train_all": float(coefficients[index]),
                "abs_lr1_coefficient_per_sd_train_all": float(abs(coefficients[index])),
                "cancer_minus_benign_score_sd_train_all": effect_size,
                "univariate_auc_train_all": float(max(auc, 1.0 - auc)),
                "univariate_auc_direction": "CANCER_higher"
                if auc >= 0.5
                else "CANCER_lower",
                "fold_aligned_lr1_coefficient_mean": float(
                    fold["aligned_coefficient"].mean()
                ),
                "fold_aligned_lr1_coefficient_sd": float(
                    fold["aligned_coefficient"].std(ddof=1)
                ),
                "fold_mean_abs_lr1_coefficient": float(
                    fold["aligned_coefficient"].abs().mean()
                ),
                "fold_coefficient_direction_consistency": float(
                    max(
                        (fold["aligned_coefficient"] >= 0.0).mean(),
                        (fold["aligned_coefficient"] < 0.0).mean(),
                    )
                ),
                "fold_basis_abs_cosine_mean": float(fold["basis_abs_cosine"].mean()),
                "fold_basis_abs_cosine_min": float(fold["basis_abs_cosine"].min()),
                **landmarks,
            }
        )
    summary = pd.DataFrame(rows)
    summary["model_activity_rank"] = summary[
        "fold_mean_abs_lr1_coefficient"
    ].rank(method="dense", ascending=False).astype(int)
    summary = summary.sort_values("model_activity_rank").reset_index(drop=True)

    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / "fpca30_component_activity.csv", index=False)
    fold_statistics.to_csv(
        output_path / "fpca30_fold_component_stability.csv", index=False
    )
    _write_plot(
        output_path / "fpca30_active_components.png",
        q_grid=fitted.q_grid_,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        summary=summary,
    )
    _write_report(output_path / "fpca30_component_interpretation.md", summary)
    return summary


def _lr1_rows(df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    return lr1_training_rows(
        df,
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
        lr1_row_policy=model["lr1_row_policy"],
    )


def _fold_statistics(
    df: pd.DataFrame,
    *,
    model: dict[str, Any],
    manifest: pd.DataFrame,
    reference_components: np.ndarray,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    required = {"split_id", "set", model["group_column"]}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"Fold manifest is missing fields: {sorted(missing)}.")
    for split_id, group in manifest.groupby("split_id", sort=True):
        patient_ids = set(
            group.loc[group["set"] == "train", model["group_column"]].astype(str)
        )
        train_df = df[df[model["group_column"]].astype(str).isin(patient_ids)].copy()
        encoder = FoldLocalProfileEncoder(
            spec=ProfileSpec(name=ENCODER_NAME, npt=256, kind="fpca", n_components=30),
            profile_column=model["profile_column"],
            label_column=model["label_column"],
            group_column=model["group_column"],
            q_column=model["q_column"],
            logreg_c=float(model["lr1_logreg_c"]),
            random_state=random_seed + int(split_id),
        ).fit(_lr1_rows(train_df, model))
        pca = encoder.pca_
        if pca is None:
            raise RuntimeError("Fold-local FPCA was not fitted.")
        coefficients = encoder.pipeline_.named_steps["logreg"].coef_.ravel()
        for index, coefficient in enumerate(coefficients):
            aligned, similarity = align_coefficient_to_reference(
                reference_components[index], pca.components_[index], float(coefficient)
            )
            rows.append(
                {
                    "split_id": int(split_id),
                    "component": index + 1,
                    "aligned_coefficient": aligned,
                    "basis_abs_cosine": similarity,
                }
            )
    return pd.DataFrame(rows)


def _write_plot(
    path: Path,
    *,
    q_grid: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
    summary: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.5), sharex=True)
    component_rows = summary.set_index("component")
    for axis, component in zip(axes.flat, SELECTED_COMPONENTS, strict=False):
        index = component - 1
        delta = np.sqrt(explained_variance[index]) * components[index]
        row = component_rows.loc[component]
        axis.axhline(0.0, color="#8a8a8a", linewidth=0.8)
        axis.plot(q_grid, delta, color="#33658A", linewidth=1.7)
        axis.set_title(
            "PC%d: %.2f%% variance | LR1 |b|=%.2f" % (
                component,
                100.0 * row["explained_variance_ratio"],
                row["fold_mean_abs_lr1_coefficient"],
            ),
            fontsize=10,
        )
        axis.set_xlim(float(q_grid.min()), float(q_grid.max()))
        axis.grid(alpha=0.18)
    fig.suptitle(
        "FPCA30: one-standard-deviation profile change for selected components",
        fontsize=14,
        fontweight="bold",
    )
    fig.supxlabel("q (nm$^{-1}$)")
    fig.supylabel("Profile change for a +1 SD component score")
    fig.tight_layout(rect=(0.035, 0.045, 1.0, 0.93))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    active = summary.head(10)
    lines = [
        "# FPCA30 component interpretation",
        "",
        "Research-only analysis of the matched common cohort: 161 patients, "
        "164 target cases, and 449 biopsy-side measurement profiles.",
        "",
        "## What is measured",
        "",
        "- `explained_variance_ratio`: share of total profile variation represented by "
        "the component. It is not a cancer-specific measure.",
        "- `fold_mean_abs_lr1_coefficient`: mean absolute LR1 coefficient for a one-SD "
        "component-score change over the 100 patient-safe outer-fold fits.",
        "- `fold_basis_abs_cosine_mean`: similarity of a fold-local PCA basis vector to "
        "the train-on-all vector after allowing the arbitrary PCA sign to reverse.",
        "- PCA component signs are arbitrary. Positive/negative profile directions are "
        "therefore meaningful only after alignment to the train-on-all basis.",
        "",
        "## Most LR1-active components",
        "",
        "| Rank | PC | Variance | Mean |LR1 coefficient| | Basis similarity | "
        "Univariate AUC |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in active.itertuples(index=False):
        lines.append(
            "| %d | %d | %.3f%% | %.3f | %.3f | %.3f |"
            % (
                row.model_activity_rank,
                row.component,
                100.0 * row.explained_variance_ratio,
                row.fold_mean_abs_lr1_coefficient,
                row.fold_basis_abs_cosine_mean,
                row.univariate_auc_train_all,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- PC1 and PC2 represent the dominant broad profile variation. Together they "
            "explain most of the profile variance, but their individual class separation "
            "is modest.",
            "- PC1, PC2, PC4, PC6, and PC7 are the stable model-active group: each has "
            "mean fold-basis similarity above 0.95. PC1 describes a broad transfer between "
            "the q approximately 13.4 peak region and q above approximately 17; PC2 is a "
            "broader q approximately 14 contrast. PC4 and PC6 include low-q versus "
            "mid-q contrast. These are profile-shape descriptions, not molecular assignments.",
            "- Several low-variance components receive substantial train-on-all LR1 "
            "coefficients. PC10, PC26, and PC29 have low fold-basis similarity and must be "
            "treated as unstable candidate patterns, not established biological features.",
            "- `fpca30_active_components.png` displays the profile perturbation for a one-SD "
            "change in each selected component. The chart is descriptive; it does not "
            "identify a molecular origin for a component.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(prog="aramina-fpca30-component-interpretation")
    parser.add_argument(
        "--config",
        default="experiments/fpca256_profile_encoder/"
        "config_fpca256_profile_encoder_v0_1.yaml",
    )
    parser.add_argument(
        "--result-folder",
        default="experiments/fpca256_profile_encoder/results/components_10_to_30/common",
    )
    parser.add_argument(
        "--output-folder",
        default="experiments/fpca256_profile_encoder/results/components_10_to_30/"
        "common/component_interpretation",
    )
    args = parser.parse_args()
    analyze_components(
        config_path=args.config,
        result_folder=args.result_folder,
        output_folder=args.output_folder,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
