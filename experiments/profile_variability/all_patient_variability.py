"""All-patient, K-beta-inclusive within-patient XRD variability experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .profile_variability import (
    PRIMARY_METRIC,
    _collapse_positions,
    _common_q_grid,
    _profile_matrix,
    _safe_ratio,
    bootstrap_interval,
    breast_variability,
)


REQUIRED_COLUMNS = {
    "patientId",
    "side",
    "position",
    "biopsy",
    "product_status_group",
    "q_range",
    "radial_profile_data",
}
COHORT_ORDER = (
    "BIOPSY_BENIGN",
    "BIOPSY_CANCER",
    "BIOPSY_UNRESOLVED",
    "NO_BIOPSY",
    "BILATERAL_BIOPSY",
)
COHORT_COLORS = {
    "BIOPSY_BENIGN": "#2f7f8f",
    "BIOPSY_CANCER": "#d1495b",
    "BIOPSY_UNRESOLVED": "#b07d35",
    "NO_BIOPSY": "#6b7280",
    "BILATERAL_BIOPSY": "#8d6a9f",
}


@dataclass(frozen=True)
class AllPatientVariabilityAnalysis:
    """Patient-paired outputs across biopsy and non-biopsy cohorts."""

    cases: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any]


def load_all_patient_profile_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a standalone profile frame or an XRD preprocessing artifact."""
    loaded = joblib.load(path)
    frame = loaded.get("dataframe") if isinstance(loaded, dict) else loaded
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(
            "Profile joblib must contain a DataFrame or an artifact with a "
            "'dataframe' entry."
        )
    return frame


def validate_all_patient_dataframe(frame: pd.DataFrame) -> None:
    """Validate the normalized profiles required by this descriptive analysis."""
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"All-patient profile dataframe is missing columns: {missing}")
    if frame.empty:
        raise ValueError("All-patient profile dataframe is empty.")
    _common_q_grid(frame)
    _profile_matrix(frame)


def build_all_patient_variability_table(
    frame: pd.DataFrame,
    *,
    min_measurements: int = 3,
) -> pd.DataFrame:
    """Create one left/right pair per eligible patient.

    Unilateral-biopsy patients retain target/contralateral orientation. Patients
    with no biopsy are oriented left/right only; left has no clinical target
    meaning. Bilateral-biopsy patients are retained as a descriptive subgroup.
    """
    if int(min_measurements) < 2:
        raise ValueError("At least two measurements per breast are required.")
    validate_all_patient_dataframe(frame)
    q_grid = _common_q_grid(frame)
    rows: list[dict[str, Any]] = []
    for patient_id, patient in frame.groupby("patientId", sort=True):
        sides = _left_right_sides(patient)
        if sides is None:
            continue
        left, right = sides
        left_collapsed = _collapse_positions(left)
        right_collapsed = _collapse_positions(right)
        if (
            len(left_collapsed) < min_measurements
            or len(right_collapsed) < min_measurements
        ):
            continue
        left_biopsy = _has_biopsy(left)
        right_biopsy = _has_biopsy(right)
        cohort, numerator_side, denominator_side = _comparison_definition(
            left_biopsy=left_biopsy,
            right_biopsy=right_biopsy,
            left=left,
            right=right,
        )
        numerator = left_collapsed if numerator_side == "LEFT" else right_collapsed
        denominator = right_collapsed if denominator_side == "RIGHT" else left_collapsed
        numerator_metrics = breast_variability(numerator, q_grid=q_grid)
        denominator_metrics = breast_variability(denominator, q_grid=q_grid)
        record: dict[str, Any] = {
            "patient_id": str(patient_id),
            "cohort": cohort,
            "comparison_orientation": _comparison_orientation(cohort),
            "numerator_side": numerator_side,
            "denominator_side": denominator_side,
            "left_measurements": int(len(left_collapsed)),
            "right_measurements": int(len(right_collapsed)),
            "left_raw_measurements": int(len(left)),
            "right_raw_measurements": int(len(right)),
            "left_biopsy": left_biopsy,
            "right_biopsy": right_biopsy,
        }
        for metric, numerator_value in numerator_metrics.items():
            denominator_value = denominator_metrics[metric]
            record[f"numerator_{metric}"] = numerator_value
            record[f"denominator_{metric}"] = denominator_value
            record[f"left_{metric}"] = (
                numerator_value if numerator_side == "LEFT" else denominator_value
            )
            record[f"right_{metric}"] = (
                numerator_value if numerator_side == "RIGHT" else denominator_value
            )
            record[f"log_ratio_{metric}"] = float(
                np.log(_safe_ratio(numerator_value, denominator_value))
            )
            record[f"geometric_mean_{metric}"] = float(
                np.sqrt(
                    (numerator_value + np.finfo(float).eps)
                    * (denominator_value + np.finfo(float).eps)
                )
            )
        rows.append(record)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(
            "No patients with both left/right breasts and sufficient positions."
        )
    return result.sort_values(["cohort", "patient_id"], kind="stable").reset_index(
        drop=True
    )


def summarize_all_patient_variability(
    cases: pd.DataFrame,
    *,
    metric: str = PRIMARY_METRIC,
    bootstrap_iterations: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Summarize descriptive within-patient variability by cohort."""
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        group = cases.loc[cases["cohort"].eq(cohort)]
        if group.empty:
            continue
        log_ratio = group[f"log_ratio_{metric}"].to_numpy(dtype=float)
        low, high = bootstrap_interval(
            log_ratio,
            iterations=bootstrap_iterations,
            random_state=random_state,
            statistic="mean",
        )
        rows.append(
            {
                "cohort": cohort,
                "patients": int(len(group)),
                "median_numerator_variability": float(
                    np.median(group[f"numerator_{metric}"])
                ),
                "median_denominator_variability": float(
                    np.median(group[f"denominator_{metric}"])
                ),
                "median_per_breast_variability": float(
                    np.median(group[f"geometric_mean_{metric}"])
                ),
                "median_ratio": float(np.median(np.exp(log_ratio))),
                "geometric_mean_ratio": float(np.exp(np.mean(log_ratio))),
                "geometric_mean_ratio_bootstrap_95_low": float(np.exp(low)),
                "geometric_mean_ratio_bootstrap_95_high": float(np.exp(high)),
                "numerator_more_variable_fraction": float(np.mean(log_ratio > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def run_all_patient_variability_analysis(
    frame: pd.DataFrame,
    *,
    min_measurements: int = 3,
    bootstrap_iterations: int = 10_000,
    random_state: int = 42,
) -> AllPatientVariabilityAnalysis:
    """Run the all-patient descriptive analysis from a normalized profile frame."""
    cases = build_all_patient_variability_table(
        frame,
        min_measurements=min_measurements,
    )
    summary = summarize_all_patient_variability(
        cases,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    metadata = {
        "primary_metric": PRIMARY_METRIC,
        "min_measurements_per_breast": int(min_measurements),
        "eligible_patients": int(len(cases)),
        **_eligibility_counts(frame, min_measurements=min_measurements),
        "cohort_counts": {
            str(key): int(value)
            for key, value in cases["cohort"].value_counts().sort_index().items()
        },
        "q_min_nm_inv": float(_common_q_grid(frame)[0]),
        "q_max_nm_inv": float(_common_q_grid(frame)[-1]),
        "kbeta_historical_exclusions_applied": False,
        "biopsy_patient_filter_applied": False,
        "analysis_status": "research_descriptive_not_model_training_or_validation",
    }
    return AllPatientVariabilityAnalysis(
        cases=cases,
        summary=summary,
        metadata=metadata,
    )


def save_all_patient_analysis(
    analysis: AllPatientVariabilityAnalysis,
    output_dir: str | Path,
) -> None:
    """Write cohort-specific evidence and an untracked patient-level table."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    analysis.cases.to_csv(output / "per_patient_variability_local.csv", index=False)
    biopsy_summary = analysis.summary.loc[
        analysis.summary["cohort"].str.startswith("BIOPSY_")
        | analysis.summary["cohort"].eq("BILATERAL_BIOPSY")
    ]
    no_biopsy_summary = analysis.summary.loc[analysis.summary["cohort"].eq("NO_BIOPSY")]
    biopsy_summary.to_csv(output / "biopsy_cohort_summary.csv", index=False)
    no_biopsy_summary.to_csv(output / "no_biopsy_cohort_summary.csv", index=False)
    (output / "summary.yaml").write_text(
        yaml.safe_dump(analysis.metadata, sort_keys=False),
        encoding="utf-8",
    )


def all_patient_variability_figure(
    cases: pd.DataFrame,
    *,
    metric: str = PRIMARY_METRIC,
) -> plt.Figure:
    """Plot independent biopsy and no-biopsy within-cohort analyses."""
    figure = plt.figure(figsize=(13.0, 10.0), constrained_layout=True)
    axes = figure.subplot_mosaic(
        [["biopsy", "no_biopsy"], ["ratios", "ratios"]],
    )
    _scatter_biopsy_cases(axes["biopsy"], cases, metric=metric)
    _scatter_no_biopsy_cases(axes["no_biopsy"], cases, metric=metric)
    _plot_oriented_log_ratios(axes["ratios"], cases, metric=metric)
    figure.suptitle("Within-cohort XRD profile variability")
    return figure


def _left_right_sides(
    patient: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    groups = {
        str(side).strip().upper(): side_frame.copy()
        for side, side_frame in patient.groupby("side", sort=True)
    }
    if set(groups) != {"LEFT", "RIGHT"}:
        return None
    return groups["LEFT"], groups["RIGHT"]


def _eligibility_counts(
    frame: pd.DataFrame,
    *,
    min_measurements: int,
) -> dict[str, int]:
    """Count the technical-frame patients that support a paired comparison."""
    with_left_right = 0
    with_minimum_positions = 0
    for _, patient in frame.groupby("patientId", sort=False):
        sides = _left_right_sides(patient)
        if sides is None:
            continue
        with_left_right += 1
        left, right = sides
        if (
            len(_collapse_positions(left)) >= min_measurements
            and len(_collapse_positions(right)) >= min_measurements
        ):
            with_minimum_positions += 1
    return {
        "post_technical_qc_patients": int(frame["patientId"].nunique()),
        "patients_with_left_and_right_breasts": with_left_right,
        "patients_with_minimum_positions_per_breast": with_minimum_positions,
    }


def _has_biopsy(frame: pd.DataFrame) -> bool:
    values = frame["biopsy"]
    if values.dtype == bool:
        return bool(values.any())
    normalized = values.astype("string").fillna("false").str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("biopsy must contain boolean values.")
    return bool(normalized.isin({"true", "1"}).any())


def _comparison_definition(
    *,
    left_biopsy: bool,
    right_biopsy: bool,
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[str, str, str]:
    if left_biopsy and right_biopsy:
        return "BILATERAL_BIOPSY", "LEFT", "RIGHT"
    if not left_biopsy and not right_biopsy:
        return "NO_BIOPSY", "LEFT", "RIGHT"
    target_side = "LEFT" if left_biopsy else "RIGHT"
    target = left if target_side == "LEFT" else right
    label = _biopsy_target_label(target)
    return f"BIOPSY_{label}", target_side, "RIGHT" if target_side == "LEFT" else "LEFT"


def _biopsy_target_label(target: pd.DataFrame) -> str:
    values = set(
        target["product_status_group"].dropna().astype(str).str.strip().str.upper()
    )
    if values == {"CANCER"}:
        return "CANCER"
    if values == {"BENIGN"}:
        return "BENIGN"
    return "UNRESOLVED"


def _comparison_orientation(cohort: str) -> str:
    return (
        "TARGET_OVER_CONTRALATERAL"
        if cohort.startswith("BIOPSY_")
        else "LEFT_OVER_RIGHT"
    )


def _scatter_biopsy_cases(axis: plt.Axes, cases: pd.DataFrame, *, metric: str) -> None:
    groups = {
        "BIOPSY_BENIGN": "BENIGN",
        "BIOPSY_CANCER": "CANCER",
        "BIOPSY_UNRESOLVED": "Unresolved biopsy label",
    }
    values = cases.loc[cases["cohort"].isin(groups)]
    for cohort, label in groups.items():
        group = values.loc[values["cohort"].eq(cohort)]
        axis.scatter(
            group[f"denominator_{metric}"],
            group[f"numerator_{metric}"],
            color=COHORT_COLORS[cohort],
            alpha=0.72,
            label=label,
        )
    _identity_line(axis, values, metric=metric)
    axis.set_title("Unilateral-biopsy patients")
    axis.set_xlabel("Contralateral within-breast variability")
    axis.set_ylabel("Target within-breast variability")
    axis.legend(frameon=False)


def _scatter_no_biopsy_cases(
    axis: plt.Axes, cases: pd.DataFrame, *, metric: str
) -> None:
    group = cases.loc[cases["cohort"].eq("NO_BIOPSY")]
    axis.scatter(
        group[f"right_{metric}"],
        group[f"left_{metric}"],
        color=COHORT_COLORS["NO_BIOPSY"],
        alpha=0.72,
    )
    _identity_line(axis, group, metric=metric, x_column="right", y_column="left")
    axis.set_title("Non-biopsy patients")
    axis.set_xlabel("Right within-breast variability")
    axis.set_ylabel("Left within-breast variability")


def _plot_oriented_log_ratios(
    axis: plt.Axes, cases: pd.DataFrame, *, metric: str
) -> None:
    available = ["BIOPSY_BENIGN", "BIOPSY_CANCER", "NO_BIOPSY"]
    values = [
        cases.loc[cases["cohort"].eq(cohort), f"log_ratio_{metric}"].to_numpy()
        for cohort in available
    ]
    labels = [
        "BENIGN\ntarget / contralateral",
        "CANCER\ntarget / contralateral",
        "NO BIOPSY\nleft / right",
    ]
    box = axis.boxplot(values, tick_labels=labels, patch_artist=True)
    for patch, cohort in zip(box["boxes"], available, strict=True):
        patch.set_facecolor(COHORT_COLORS[cohort])
        patch.set_alpha(0.55)
    axis.axhline(0.0, color="#555555", linestyle="--")
    axis.set_title("Within-patient variability ratios")
    axis.set_ylabel("log(oriented breast 1 / breast 2 variability)")


def _identity_line(
    axis: plt.Axes,
    cases: pd.DataFrame,
    *,
    metric: str,
    x_column: str = "denominator",
    y_column: str = "numerator",
) -> None:
    if cases.empty:
        return
    x = cases[f"{x_column}_{metric}"]
    y = cases[f"{y_column}_{metric}"]
    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    axis.plot([lower, upper], [lower, upper], color="#555555", linestyle="--")
