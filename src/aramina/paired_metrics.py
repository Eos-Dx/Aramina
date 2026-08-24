"""Paired metric deltas and descriptive patient-cluster intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .model_metrics import binary_metric_values
from .paired_contract import (
    MODEL_NAMES,
    PAIRED_COMPARISONS,
    POOLED_METRIC_NAMES,
    SUMMARY_METRICS,
)
from .patient_features import TARGET_CASE_ID


def assert_shared_evaluation_cases(
    predictions: pd.DataFrame,
    outer_manifest: pd.DataFrame,
    case_manifest: pd.DataFrame,
) -> None:
    """Require every model to score exactly the shared outer test cases."""
    all_cases = set(case_manifest[TARGET_CASE_ID])
    for split_id, manifest in outer_manifest.groupby("split_id"):
        expected = set(
            manifest.loc[manifest["role"].eq("outer_test"), TARGET_CASE_ID]
        )
        if not expected.issubset(all_cases):
            raise RuntimeError("Outer test manifest contains unknown target cases.")
        split_predictions = predictions.loc[predictions["split_id"].eq(split_id)]
        for model_name in MODEL_NAMES:
            model_predictions = split_predictions.loc[
                split_predictions["model_name"].eq(model_name)
            ]
            if set(model_predictions[TARGET_CASE_ID]) != expected:
                raise RuntimeError(
                    f"{model_name} split {split_id} does not match shared test cases."
                )
            if model_predictions[TARGET_CASE_ID].duplicated().any():
                raise RuntimeError(
                    f"{model_name} split {split_id} duplicates target cases."
                )


def paired_fold_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate explicit encoder, architecture, and total paired effects."""
    rows: list[dict[str, Any]] = []
    for comparison, candidate_name, reference_name in PAIRED_COMPARISONS:
        candidate = metrics.loc[metrics["model_name"].eq(candidate_name)].set_index(
            "split_id"
        )
        reference = metrics.loc[metrics["model_name"].eq(reference_name)].set_index(
            "split_id"
        )
        if not candidate.index.equals(reference.index):
            raise RuntimeError(
                f"{comparison} candidate and reference have unequal fold indexes."
            )
        for split_id in candidate.index:
            rows.append(
                {
                    "comparison": comparison,
                    "split_id": int(split_id),
                    "candidate_model": candidate_name,
                    "reference_model": reference_name,
                    **{
                        f"delta_{metric}": float(
                            candidate.at[split_id, metric]
                            - reference.at[split_id, metric]
                        )
                        for metric in SUMMARY_METRICS
                    },
                }
            )
    return pd.DataFrame(rows)


def paired_delta_summary(
    deltas: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    random_state: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    """Summarize fold deltas and paired patient-cluster bootstrap intervals."""
    rows: list[dict[str, Any]] = []
    for comparison_index, comparison_spec in enumerate(PAIRED_COMPARISONS):
        comparison, candidate_name, reference_name = comparison_spec
        comparison_deltas = deltas.loc[deltas["comparison"].eq(comparison)]
        paired_cases = _paired_repeat_averaged_cases(
            predictions,
            candidate_name=candidate_name,
            reference_name=reference_name,
        )
        rng = np.random.default_rng(random_state + comparison_index)
        bootstrap = _paired_patient_bootstrap_deltas(
            paired_cases,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )
        point = _paired_case_metric_deltas(paired_cases)
        for metric in SUMMARY_METRICS:
            values = comparison_deltas[f"delta_{metric}"].to_numpy(dtype=float)
            pooled_name = POOLED_METRIC_NAMES[metric]
            sampled = bootstrap[pooled_name]
            rows.append(
                {
                    "comparison": comparison,
                    "candidate_model": candidate_name,
                    "reference_model": reference_name,
                    "metric": metric,
                    "folds": int(len(values)),
                    "mean_fold_delta": float(np.mean(values)),
                    "std_fold_delta": float(np.std(values, ddof=0)),
                    "fold_delta_q2_5": float(np.quantile(values, 0.025)),
                    "fold_delta_q97_5": float(np.quantile(values, 0.975)),
                    "pooled_repeat_averaged_delta": point[pooled_name],
                    "paired_patient_bootstrap_ci_low": (
                        float(np.quantile(sampled, 0.025))
                        if sampled
                        else float("nan")
                    ),
                    "paired_patient_bootstrap_ci_high": (
                        float(np.quantile(sampled, 0.975))
                        if sampled
                        else float("nan")
                    ),
                    "valid_bootstrap_samples": int(len(sampled)),
                    "interval_scope": (
                        "descriptive_patient_cluster_bootstrap_of_"
                        "repeat_averaged_oof_predictions"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_repeat_averaged_cases(
    predictions: pd.DataFrame,
    *,
    candidate_name: str,
    reference_name: str,
) -> pd.DataFrame:
    averaged = (
        predictions.loc[
            predictions["model_name"].isin([reference_name, candidate_name])
        ]
        .groupby(["model_name", TARGET_CASE_ID], as_index=False)
        .agg(
            patientId=("patientId", "first"),
            label=("label", "first"),
            p_cancer=("p_cancer", "mean"),
            threshold_target=("threshold_target", "mean"),
        )
    )
    reference = averaged.loc[averaged["model_name"].eq(reference_name)].drop(
        columns="model_name"
    )
    candidate = averaged.loc[averaged["model_name"].eq(candidate_name)].drop(
        columns="model_name"
    )
    paired = candidate.merge(
        reference,
        on=[TARGET_CASE_ID, "patientId", "label"],
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(reference) or len(paired) != len(candidate):
        raise RuntimeError(
            f"{candidate_name} and {reference_name} cases are not exactly paired."
        )
    return paired


def _paired_case_metric_deltas(cases: pd.DataFrame) -> dict[str, float]:
    y = cases["label"].to_numpy(dtype=int)
    candidate = binary_metric_values(
        y,
        cases["p_cancer_candidate"].to_numpy(dtype=float),
        cases["threshold_target_candidate"].to_numpy(dtype=float),
    )
    reference = binary_metric_values(
        y,
        cases["p_cancer_reference"].to_numpy(dtype=float),
        cases["threshold_target_reference"].to_numpy(dtype=float),
    )
    return {name: float(candidate[name] - reference[name]) for name in candidate}


def _paired_patient_bootstrap_deltas(
    cases: pd.DataFrame,
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, list[float]]:
    sampled = {name: [] for name in POOLED_METRIC_NAMES.values()}
    patient_ids = cases["patientId"].drop_duplicates().to_numpy()
    for _ in range(max(0, int(bootstrap_samples))):
        chosen = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sample = pd.concat(
            [cases.loc[cases["patientId"].eq(patient_id)] for patient_id in chosen],
            ignore_index=True,
        )
        if sample["label"].nunique() != 2:
            continue
        values = _paired_case_metric_deltas(sample)
        for name in sampled:
            if np.isfinite(values[name]):
                sampled[name].append(values[name])
    return sampled
