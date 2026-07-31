"""Patient-paired variability analysis for normalized breast XRD profiles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    binomtest,
    mannwhitneyu,
    ttest_1samp,
    wasserstein_distance,
    wilcoxon,
)
import yaml


PRIMARY_METRIC = "mean_pairwise_mse_full"
SECONDARY_METRICS = (
    "mean_pairwise_mse_q7_23",
    "mean_pairwise_mse_q7_15",
    "mean_pairwise_mse_q15_23",
    "mean_pairwise_rms_full",
    "mean_pairwise_rms_q7_23",
    "mean_pairwise_rms_q7_15",
    "mean_pairwise_rms_q15_23",
    "median_pairwise_rms_q7_23",
    "rms_about_breast_mean_q7_23",
    "mean_pairwise_cosine_q7_23",
    "mean_pairwise_wasserstein_q7_23",
)
REQUIRED_COLUMNS = {
    "patientId",
    "side",
    "position",
    "product_diagnosis",
    "biopsy",
    "q_range",
    "radial_profile_data",
}
SUPPORTED_LABELS = {"BENIGN", "CANCER"}


@dataclass(frozen=True)
class VariabilityAnalysis:
    """All patient-level and aggregate outputs from one fixed analysis."""

    cases: pd.DataFrame
    paired_summary: pd.DataFrame
    diagnosis_contrast: pd.DataFrame
    secondary_metrics: pd.DataFrame
    q_variability: pd.DataFrame
    metadata: dict[str, Any]


def load_profile_dataframe(path: str | Path) -> pd.DataFrame:
    """Load the standard preprocessing artifact and validate profile columns."""
    source = Path(path).expanduser().resolve()
    value = joblib.load(source)
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, dict) and isinstance(value.get("dataframe"), pd.DataFrame):
        frame = value["dataframe"].copy()
    else:
        raise ValueError(
            "Profile artifact must be a DataFrame or {'dataframe': DataFrame}."
        )
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Profile dataframe is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Profile dataframe is empty.")
    _common_q_grid(frame)
    _profile_matrix(frame)
    artifact_metadata = value.get("metadata", {}) if isinstance(value, dict) else {}
    preprocessing_yaml = (
        str(value.get("preprocessing_config_yaml", ""))
        if isinstance(value, dict)
        else ""
    )
    frame.attrs["artifact_provenance"] = {
        "input_joblib": source.name,
        "input_joblib_sha256": _sha256_file(source),
        "input_h5_sha256": str(artifact_metadata.get("input_h5_sha256", "unknown")),
        "aramina_git_sha": str(artifact_metadata.get("aramina_git_sha", "unknown")),
        "preprocessing_config_yaml_sha256": sha256(
            preprocessing_yaml.encode("utf-8")
        ).hexdigest(),
    }
    return frame


def build_variability_table(
    frame: pd.DataFrame,
    *,
    min_measurements: int = 3,
    include_bilateral_biopsy: bool = False,
) -> pd.DataFrame:
    """Build one paired target/contralateral variability row per target case.

    The primary cohort requires a biopsied target side and a non-biopsied
    contralateral side. Bilateral-biopsy cases can be included only as a
    sensitivity analysis because the opposite side is not a true control side.
    """
    if int(min_measurements) < 2:
        raise ValueError("At least two measurements per breast are required.")
    q_grid = _common_q_grid(frame)
    rows: list[dict[str, Any]] = []
    for patient_id, patient in frame.groupby("patientId", sort=True):
        side_groups = {
            str(side).upper(): side_frame.copy()
            for side, side_frame in patient.groupby("side", sort=True)
        }
        if len(side_groups) != 2:
            continue
        for target_side, target in side_groups.items():
            if not target["biopsy"].astype(bool).any():
                continue
            contralateral_side = next(
                side for side in side_groups if side != target_side
            )
            contralateral = side_groups[contralateral_side]
            contralateral_biopsy = bool(contralateral["biopsy"].astype(bool).any())
            if contralateral_biopsy and not include_bilateral_biopsy:
                continue
            target_collapsed = _collapse_positions(target)
            contralateral_collapsed = _collapse_positions(contralateral)
            if (
                len(target_collapsed) < min_measurements
                or len(contralateral_collapsed) < min_measurements
            ):
                continue
            label = _target_label(target)
            target_metrics = breast_variability(target_collapsed, q_grid=q_grid)
            contralateral_metrics = breast_variability(
                contralateral_collapsed,
                q_grid=q_grid,
            )
            record: dict[str, Any] = {
                "patientId": str(patient_id),
                "target_case_id": f"{patient_id}::{target_side}",
                "target_side": target_side,
                "contralateral_side": contralateral_side,
                "label": label,
                "target_measurements": int(len(target_collapsed)),
                "contralateral_measurements": int(len(contralateral_collapsed)),
                "target_raw_measurements": int(len(target)),
                "contralateral_raw_measurements": int(len(contralateral)),
                "target_positions": ",".join(
                    sorted(target_collapsed["position"].astype(str))
                ),
                "contralateral_positions": ",".join(
                    sorted(contralateral_collapsed["position"].astype(str))
                ),
                "contralateral_biopsy": contralateral_biopsy,
            }
            for metric, target_value in target_metrics.items():
                contralateral_value = contralateral_metrics[metric]
                record[f"target_{metric}"] = target_value
                record[f"contralateral_{metric}"] = contralateral_value
                record[f"delta_{metric}"] = target_value - contralateral_value
                record[f"ratio_{metric}"] = _safe_ratio(
                    target_value,
                    contralateral_value,
                )
                record[f"log_ratio_{metric}"] = np.log(
                    _safe_ratio(target_value, contralateral_value)
                )
            rows.append(record)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No eligible paired target/contralateral cases were found.")
    if not include_bilateral_biopsy and result["patientId"].duplicated().any():
        raise RuntimeError(
            "Primary unilateral-biopsy cohort must contain one row per patient."
        )
    return result.sort_values("target_case_id", kind="stable").reset_index(drop=True)


def breast_variability(frame: pd.DataFrame, *, q_grid: np.ndarray) -> dict[str, float]:
    """Calculate within-breast profile dispersion for one breast."""
    profiles = _profile_matrix(frame)
    if len(profiles) < 2:
        raise ValueError("Within-breast variability requires at least two profiles.")
    masks = {
        "full": np.ones(len(q_grid), dtype=bool),
        "q7_23": (q_grid >= 7.0) & (q_grid <= 23.0),
        "q7_15": (q_grid >= 7.0) & (q_grid < 15.0),
        "q15_23": (q_grid >= 15.0) & (q_grid <= 23.0),
    }
    pair_mse: dict[str, list[float]] = {name: [] for name in masks}
    pair_rms: dict[str, list[float]] = {name: [] for name in masks}
    cosine: list[float] = []
    wasserstein: list[float] = []
    for left_index, right_index in combinations(range(len(profiles)), 2):
        left = profiles[left_index]
        right = profiles[right_index]
        for name, mask in masks.items():
            mse = float(np.mean((left[mask] - right[mask]) ** 2))
            pair_mse[name].append(mse)
            pair_rms[name].append(float(np.sqrt(mse)))
        q_mask = masks["q7_23"]
        left_q = left[q_mask]
        right_q = right[q_mask]
        denominator = float(np.linalg.norm(left_q) * np.linalg.norm(right_q))
        cosine.append(1.0 - float(np.dot(left_q, right_q)) / denominator)
        wasserstein.append(
            float(
                wasserstein_distance(
                    q_grid[q_mask],
                    q_grid[q_mask],
                    u_weights=left_q,
                    v_weights=right_q,
                )
            )
        )
    q_mask = masks["q7_23"]
    centred = profiles[:, q_mask] - profiles[:, q_mask].mean(axis=0)
    return {
        "mean_pairwise_mse_full": float(np.mean(pair_mse["full"])),
        "mean_pairwise_mse_q7_23": float(np.mean(pair_mse["q7_23"])),
        "mean_pairwise_mse_q7_15": float(np.mean(pair_mse["q7_15"])),
        "mean_pairwise_mse_q15_23": float(np.mean(pair_mse["q15_23"])),
        "mean_pairwise_rms_full": float(np.mean(pair_rms["full"])),
        "mean_pairwise_rms_q7_23": float(np.mean(pair_rms["q7_23"])),
        "median_pairwise_rms_q7_23": float(np.median(pair_rms["q7_23"])),
        "mean_pairwise_rms_q7_15": float(np.mean(pair_rms["q7_15"])),
        "mean_pairwise_rms_q15_23": float(np.mean(pair_rms["q15_23"])),
        "mean_pairwise_cosine_q7_23": float(np.mean(cosine)),
        "mean_pairwise_wasserstein_q7_23": float(np.mean(wasserstein)),
        "rms_about_breast_mean_q7_23": float(np.sqrt(np.mean(centred**2))),
    }


def summarize_paired_variability(
    cases: pd.DataFrame,
    *,
    metric: str = PRIMARY_METRIC,
    bootstrap_iterations: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Summarize target-minus-contralateral differences with paired inference."""
    rows = []
    groups = [("ALL", cases)] + [
        (label, cases.loc[cases["label"].eq(label)]) for label in ("BENIGN", "CANCER")
    ]
    for group_name, group in groups:
        target = group[f"target_{metric}"].to_numpy(dtype=float)
        contralateral = group[f"contralateral_{metric}"].to_numpy(dtype=float)
        difference = target - contralateral
        log_ratio = group[f"log_ratio_{metric}"].to_numpy(dtype=float)
        median_low, median_high = bootstrap_interval(
            difference,
            iterations=bootstrap_iterations,
            random_state=random_state,
            statistic="median",
        )
        mean_log_low, mean_log_high = bootstrap_interval(
            log_ratio,
            iterations=bootstrap_iterations,
            random_state=random_state,
            statistic="mean",
        )
        t_test_p = (
            float(ttest_1samp(log_ratio, popmean=0.0).pvalue)
            if len(log_ratio) >= 2
            else float("nan")
        )
        wilcoxon_p = _wilcoxon_zero_p_value(log_ratio)
        positive = int((log_ratio > 0.0).sum())
        nonzero = int((log_ratio != 0.0).sum())
        sign_p = float(binomtest(positive, nonzero, p=0.5).pvalue) if nonzero else 1.0
        rows.append(
            {
                "group": group_name,
                "cases": int(len(group)),
                "median_target": float(np.median(target)),
                "median_contralateral": float(np.median(contralateral)),
                "median_delta": float(np.median(difference)),
                "median_delta_bootstrap_95_low": median_low,
                "median_delta_bootstrap_95_high": median_high,
                "median_ratio": float(
                    np.median(group[f"ratio_{metric}"].to_numpy(dtype=float))
                ),
                "mean_log_ratio": float(np.mean(log_ratio)),
                "geometric_mean_ratio": float(np.exp(np.mean(log_ratio))),
                "geometric_mean_ratio_bootstrap_95_low": float(np.exp(mean_log_low)),
                "geometric_mean_ratio_bootstrap_95_high": float(np.exp(mean_log_high)),
                "target_more_variable_fraction": float(np.mean(difference > 0.0)),
                "paired_t_log_ratio_p": t_test_p,
                "paired_wilcoxon_p": wilcoxon_p,
                "exact_sign_test_p": sign_p,
            }
        )
    return pd.DataFrame(rows)


def summarize_diagnosis_contrast(
    cases: pd.DataFrame,
    *,
    metric: str = PRIMARY_METRIC,
    bootstrap_iterations: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare target/contralateral variability ratios by target diagnosis."""
    column = f"log_ratio_{metric}"
    benign = cases.loc[cases["label"].eq("BENIGN"), column].to_numpy(dtype=float)
    cancer = cases.loc[cases["label"].eq("CANCER"), column].to_numpy(dtype=float)
    if len(benign) == 0 or len(cancer) == 0:
        raise ValueError("Diagnosis contrast requires BENIGN and CANCER cases.")
    rng = np.random.default_rng(random_state)
    sampled_benign = rng.choice(
        benign,
        size=(bootstrap_iterations, len(benign)),
        replace=True,
    )
    sampled_cancer = rng.choice(
        cancer,
        size=(bootstrap_iterations, len(cancer)),
        replace=True,
    )
    mean_difference = sampled_cancer.mean(axis=1) - sampled_benign.mean(axis=1)
    low, high = np.quantile(mean_difference, [0.025, 0.975])
    test = mannwhitneyu(cancer, benign, alternative="two-sided")
    observed = float(np.mean(cancer) - np.mean(benign))
    combined = np.concatenate([cancer, benign])
    permutation_differences = np.empty(bootstrap_iterations, dtype=float)
    for index in range(bootstrap_iterations):
        permuted = rng.permutation(combined)
        permutation_differences[index] = (
            permuted[: len(cancer)].mean() - permuted[len(cancer) :].mean()
        )
    permutation_p = float(
        (np.sum(np.abs(permutation_differences) >= abs(observed)) + 1)
        / (bootstrap_iterations + 1)
    )
    return pd.DataFrame(
        [
            {
                "contrast": "CANCER_minus_BENIGN_log_variability_ratio",
                "cancer_cases": int(len(cancer)),
                "benign_cases": int(len(benign)),
                "median_cancer_log_ratio": float(np.median(cancer)),
                "median_benign_log_ratio": float(np.median(benign)),
                "median_difference": float(np.median(cancer) - np.median(benign)),
                "mean_cancer_log_ratio": float(np.mean(cancer)),
                "mean_benign_log_ratio": float(np.mean(benign)),
                "mean_difference": observed,
                "mean_difference_bootstrap_95_low": float(low),
                "mean_difference_bootstrap_95_high": float(high),
                "diagnosis_label_permutation_p": permutation_p,
                "mann_whitney_u_p": float(test.pvalue),
            }
        ]
    )


def pointwise_q_variability(
    frame: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """Return median within-breast standard deviation as a function of q."""
    q_grid = _common_q_grid(frame)
    rows = []
    for case in cases.itertuples(index=False):
        patient = frame.loc[frame["patientId"].astype(str).eq(str(case.patientId))]
        for role, side in (
            ("target", case.target_side),
            ("contralateral", case.contralateral_side),
        ):
            breast = _collapse_positions(
                patient.loc[patient["side"].astype(str).str.upper().eq(side)]
            )
            profiles = _profile_matrix(breast)
            standard_deviation = profiles.std(axis=0, ddof=1)
            rows.extend(
                {
                    "target_case_id": case.target_case_id,
                    "label": case.label,
                    "role": role,
                    "q_nm_inv": float(q_value),
                    "profile_sd": float(sd_value),
                }
                for q_value, sd_value in zip(
                    q_grid,
                    standard_deviation,
                    strict=True,
                )
            )
    pointwise = pd.DataFrame(rows)
    return (
        pointwise.groupby(["label", "role", "q_nm_inv"], as_index=False)
        .agg(
            median_profile_sd=("profile_sd", "median"),
            mean_profile_sd=("profile_sd", "mean"),
            cases=("target_case_id", "nunique"),
        )
        .sort_values(["label", "role", "q_nm_inv"], kind="stable")
        .reset_index(drop=True)
    )


def summarize_secondary_metrics(cases: pd.DataFrame) -> pd.DataFrame:
    """Audit related variability definitions with multiplicity correction."""
    rows = []
    for metric in SECONDARY_METRICS:
        record: dict[str, Any] = {"metric": metric}
        for group_name, group in (
            ("all", cases),
            ("benign", cases.loc[cases["label"].eq("BENIGN")]),
            ("cancer", cases.loc[cases["label"].eq("CANCER")]),
        ):
            log_ratio = group[f"log_ratio_{metric}"].to_numpy(dtype=float)
            record[f"{group_name}_geometric_mean_ratio"] = float(
                np.exp(np.mean(log_ratio))
            )
            record[f"{group_name}_paired_t_p"] = (
                float(ttest_1samp(log_ratio, popmean=0.0).pvalue)
                if len(log_ratio) >= 2
                else float("nan")
            )
            record[f"{group_name}_wilcoxon_p"] = _wilcoxon_zero_p_value(log_ratio)
        cancer = cases.loc[
            cases["label"].eq("CANCER"),
            f"log_ratio_{metric}",
        ].to_numpy(dtype=float)
        benign = cases.loc[
            cases["label"].eq("BENIGN"),
            f"log_ratio_{metric}",
        ].to_numpy(dtype=float)
        record["diagnosis_mann_whitney_p"] = float(
            mannwhitneyu(cancer, benign, alternative="two-sided").pvalue
        )
        rows.append(record)
    result = pd.DataFrame(rows)
    for column in (
        "all_paired_t_p",
        "benign_paired_t_p",
        "cancer_paired_t_p",
        "diagnosis_mann_whitney_p",
    ):
        result[f"{column}_fdr_bh"] = _benjamini_hochberg(
            result[column].to_numpy(dtype=float)
        )
    return result


def run_variability_analysis(
    frame: pd.DataFrame,
    *,
    min_measurements: int = 3,
    include_bilateral_biopsy: bool = False,
    bootstrap_iterations: int = 10_000,
    random_state: int = 42,
) -> VariabilityAnalysis:
    """Run the fixed primary analysis and return all result tables."""
    cases = build_variability_table(
        frame,
        min_measurements=min_measurements,
        include_bilateral_biopsy=include_bilateral_biopsy,
    )
    paired = summarize_paired_variability(
        cases,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    diagnosis = summarize_diagnosis_contrast(
        cases,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    secondary = summarize_secondary_metrics(cases)
    q_variability = pointwise_q_variability(frame, cases)
    metadata = {
        "primary_metric": PRIMARY_METRIC,
        "min_measurements_per_breast": int(min_measurements),
        "include_bilateral_biopsy": bool(include_bilateral_biopsy),
        "cases": int(len(cases)),
        "patients": int(cases["patientId"].nunique()),
        "cancer_cases": int(cases["label"].eq("CANCER").sum()),
        "benign_cases": int(cases["label"].eq("BENIGN").sum()),
        "q_min_nm_inv": float(_common_q_grid(frame)[0]),
        "q_max_nm_inv": float(_common_q_grid(frame)[-1]),
        "bootstrap_iterations": int(bootstrap_iterations),
        "random_state": int(random_state),
        "analysis_status": "research_descriptive_not_independent_validation",
        "input_provenance": dict(frame.attrs.get("artifact_provenance", {})),
    }
    return VariabilityAnalysis(
        cases=cases,
        paired_summary=paired,
        diagnosis_contrast=diagnosis,
        secondary_metrics=secondary,
        q_variability=q_variability,
        metadata=metadata,
    )


def save_analysis(analysis: VariabilityAnalysis, output_dir: str | Path) -> None:
    """Save aggregate evidence and a local patient-level audit table."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    analysis.cases.to_csv(output / "per_case_variability_local.csv", index=False)
    analysis.paired_summary.to_csv(output / "paired_summary.csv", index=False)
    analysis.diagnosis_contrast.to_csv(output / "diagnosis_contrast.csv", index=False)
    analysis.secondary_metrics.to_csv(output / "secondary_metrics.csv", index=False)
    analysis.q_variability.to_csv(output / "q_variability.csv", index=False)
    (output / "summary.yaml").write_text(
        yaml.safe_dump(analysis.metadata, sort_keys=False),
        encoding="utf-8",
    )


def paired_scatter_figure(
    cases: pd.DataFrame,
    *,
    metric: str = PRIMARY_METRIC,
) -> plt.Figure:
    """Plot paired target and contralateral variability by diagnosis."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    colors = {"BENIGN": "#2f7f8f", "CANCER": "#d1495b"}
    target_column = f"target_{metric}"
    contralateral_column = f"contralateral_{metric}"
    for label, group in cases.groupby("label", sort=True):
        axes[0].scatter(
            group[contralateral_column],
            group[target_column],
            s=30,
            alpha=0.72,
            color=colors[label],
            label=label,
        )
    lower = float(min(cases[target_column].min(), cases[contralateral_column].min()))
    upper = float(max(cases[target_column].max(), cases[contralateral_column].max()))
    axes[0].plot([lower, upper], [lower, upper], color="#555555", linestyle="--")
    axes[0].set_xlabel("Contralateral within-breast variability")
    axes[0].set_ylabel("Target within-breast variability")
    axes[0].legend(frameon=False)

    values = [
        cases.loc[cases["label"].eq(label), f"log_ratio_{metric}"].to_numpy()
        for label in ("BENIGN", "CANCER")
    ]
    box = axes[1].boxplot(
        values,
        tick_labels=["BENIGN", "CANCER"],
        patch_artist=True,
    )
    for patch, color in zip(
        box["boxes"], (colors["BENIGN"], colors["CANCER"]), strict=True
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axes[1].axhline(0.0, color="#555555", linestyle="--")
    axes[1].set_ylabel("log(target variability / contralateral variability)")
    figure.suptitle("Patient-paired XRD profile variability")
    return figure


def q_variability_figure(q_variability: pd.DataFrame) -> plt.Figure:
    """Plot median pointwise profile SD for each role and diagnosis."""
    figure, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.7), sharey=True, constrained_layout=True
    )
    colors = {"target": "#d1495b", "contralateral": "#2f7f8f"}
    for axis, label in zip(axes, ("BENIGN", "CANCER"), strict=True):
        subset = q_variability.loc[q_variability["label"].eq(label)]
        for role, group in subset.groupby("role", sort=True):
            axis.plot(
                group["q_nm_inv"],
                group["median_profile_sd"],
                color=colors[role],
                linewidth=2.0,
                label=role,
            )
        axis.set_title(label)
        axis.set_xlabel("q, nm$^{-1}$")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Median within-breast profile SD")
    return figure


def bootstrap_interval(
    values: np.ndarray,
    *,
    iterations: int,
    random_state: int,
    statistic: str,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for a mean or median."""
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    samples = rng.choice(array, size=(iterations, len(array)), replace=True)
    if statistic == "median":
        estimates = np.median(samples, axis=1)
    elif statistic == "mean":
        estimates = np.mean(samples, axis=1)
    else:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _target_label(target: pd.DataFrame) -> str:
    labels = set(target["product_diagnosis"].dropna().astype(str).str.upper())
    if len(labels) != 1:
        raise ValueError(f"Target breast has ambiguous labels: {sorted(labels)}")
    label = next(iter(labels))
    if label not in SUPPORTED_LABELS:
        raise ValueError(f"Unsupported target diagnosis: {label}")
    return label


def _common_q_grid(frame: pd.DataFrame) -> np.ndarray:
    grids = [np.asarray(value, dtype=float) for value in frame["q_range"]]
    reference = grids[0]
    if reference.ndim != 1 or len(reference) < 2 or not np.isfinite(reference).all():
        raise ValueError("q_range must contain a finite one-dimensional grid.")
    if any(not np.array_equal(reference, grid) for grid in grids[1:]):
        raise ValueError("All normalized profiles must use one common q-grid.")
    return reference


def _profile_matrix(frame: pd.DataFrame) -> np.ndarray:
    profiles = np.vstack(
        frame["radial_profile_data"].map(lambda value: np.asarray(value, dtype=float))
    )
    if profiles.ndim != 2 or not np.isfinite(profiles).all():
        raise ValueError("radial_profile_data must contain finite vectors.")
    return profiles


def _collapse_positions(frame: pd.DataFrame) -> pd.DataFrame:
    """Average technical repeats so each named position contributes once."""
    rows = []
    for position, group in frame.groupby("position", sort=True):
        row = group.iloc[0].copy()
        row["position"] = str(position)
        row["radial_profile_data"] = _profile_matrix(group).mean(axis=0)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _safe_ratio(numerator: float, denominator: float) -> float:
    epsilon = np.finfo(float).eps
    return float((numerator + epsilon) / (denominator + epsilon))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _wilcoxon_zero_p_value(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if np.allclose(array, 0.0):
        return 1.0
    return float(wilcoxon(array, alternative="two-sided").pvalue)
