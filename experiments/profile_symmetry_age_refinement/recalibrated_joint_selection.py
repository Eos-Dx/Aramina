"""Nested scoring and ablation-specific regularization selection utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from aramina.m2q_model import GatedSymmetryLogistic
from aramina.model_metrics import binary_metric_values

from recalibrated_joint_data import FullChainMetaPair
from recalibrated_joint_model import RecalibratedJointAdditiveClassifier


ABLATIONS = {
    "calibrated_profile": {"use_age": False, "use_symmetry": False},
    "profile_age": {"use_age": True, "use_symmetry": False},
    "profile_symmetry": {"use_age": False, "use_symmetry": True},
    "profile_age_symmetry": {"use_age": True, "use_symmetry": True},
}
# Two orders below and one order above the original product-scale grid. Boundary
# selections remain an explicit result rather than an implicit optimum claim.
DEFAULT_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def score_joint_pairs(
    pairs: Sequence[FullChainMetaPair],
    *,
    profile_c: float,
    age_c: float,
    symmetry_c: float,
    use_age: bool,
    use_symmetry: bool,
    random_state: int,
) -> pd.DataFrame:
    """Score cached full-chain validation folds without refitting LR1."""
    rows: list[pd.DataFrame] = []
    for pair in pairs:
        model = RecalibratedJointAdditiveClassifier(
            profile_c=profile_c,
            age_c=age_c,
            symmetry_c=symmetry_c,
            use_age=use_age,
            use_symmetry=use_symmetry,
            random_state=random_state + pair.fold_id,
        ).fit(
            pair.meta_train_features,
            pair.meta_train_features["label"].to_numpy(dtype=int),
        )
        validation = pair.meta_validation_features
        out = validation[["target_case_id", "patientId", "label", "label_name"]].copy()
        out["meta_fold_id"] = int(pair.fold_id)
        out["p_cancer"] = model.predict_proba(validation)[:, 1]
        rows.append(out)
    scored = pd.concat(rows, ignore_index=True)
    if scored["target_case_id"].duplicated().any():
        raise RuntimeError("Full-chain meta validation must score each case exactly once.")
    return scored.sort_values("target_case_id", kind="stable").reset_index(drop=True)


def score_current_pairs(
    pairs: Sequence[FullChainMetaPair],
    *,
    logreg_c: float,
    random_state: int,
) -> pd.DataFrame:
    """Score current LR2 architecture on the same cached full-chain folds."""
    rows: list[pd.DataFrame] = []
    for pair in pairs:
        model = GatedSymmetryLogistic(
            logreg_c=logreg_c,
            random_state=random_state + pair.fold_id,
        ).fit(
            pair.meta_train_features,
            pair.meta_train_features["label"].to_numpy(dtype=int),
        )
        validation = pair.meta_validation_features
        out = validation[["target_case_id", "patientId", "label", "label_name"]].copy()
        out["meta_fold_id"] = int(pair.fold_id)
        out["p_cancer"] = model.predict_proba(validation)[:, 1]
        rows.append(out)
    scored = pd.concat(rows, ignore_index=True)
    if scored["target_case_id"].duplicated().any():
        raise RuntimeError("Current architecture OOF must score each case exactly once.")
    return scored.sort_values("target_case_id", kind="stable").reset_index(drop=True)


def select_ablation_regularization(
    pairs: Sequence[FullChainMetaPair],
    *,
    ablation: str,
    candidate_c: Sequence[float] = DEFAULT_C_GRID,
    random_state: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Coordinate-select penalties entirely within one ablation architecture."""
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation!r}")
    grid = _validated_grid(candidate_c)
    enabled = ABLATIONS[ablation]
    selected = {"profile_c": 0.1, "age_c": 0.1, "symmetry_c": 0.1}
    steps = ["profile_c"]
    if enabled["use_age"]:
        steps.append("age_c")
    if enabled["use_symmetry"]:
        steps.append("symmetry_c")
    score_cache: dict[tuple[float, float, float], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for step_index, key in enumerate(steps):
        candidates: list[tuple[dict[str, float], pd.DataFrame]] = []
        for value in grid:
            trial = {**selected, key: value}
            cache_key = (trial["profile_c"], trial["age_c"], trial["symmetry_c"])
            scored = score_cache.get(cache_key)
            if scored is None:
                scored = score_joint_pairs(
                    pairs,
                    **trial,
                    **enabled,
                    random_state=random_state + step_index * 1_000,
                )
                score_cache[cache_key] = scored
            metrics = _probability_metrics(scored)
            candidates.append((trial, scored))
            rows.append(
                {
                    "ablation": ablation,
                    "selection_step": key,
                    "candidate_c": value,
                    **trial,
                    **metrics,
                    "selected": False,
                    "selected_at_grid_boundary": False,
                }
            )
        ranked = sorted(
            candidates,
            key=lambda item: _ranking_key(_probability_metrics(item[1]), item[0][key]),
        )
        selected = ranked[0][0]
        for row in rows:
            if row["ablation"] == ablation and row["selection_step"] == key:
                row["selected"] = row["candidate_c"] == selected[key]
                row["selected_at_grid_boundary"] = bool(
                    row["selected"] and selected[key] in {grid[0], grid[-1]}
                )
    selected_score = score_cache.get(
        (selected["profile_c"], selected["age_c"], selected["symmetry_c"])
    )
    if selected_score is None:
        selected_score = score_joint_pairs(
            pairs,
            **selected,
            **enabled,
            random_state=random_state + 9_999,
        )
    return selected, pd.DataFrame(rows), selected_score


def fit_joint_model(
    features: pd.DataFrame,
    *,
    parameters: dict[str, float],
    ablation: str,
    random_state: int,
) -> RecalibratedJointAdditiveClassifier:
    """Fit selected joint model on LR1-OOF rows for deployed-chain scoring."""
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation!r}")
    return RecalibratedJointAdditiveClassifier(
        **parameters,
        **ABLATIONS[ablation],
        random_state=random_state,
    ).fit(features, features["label"].to_numpy(dtype=int))


def _probability_metrics(scored: pd.DataFrame) -> dict[str, float]:
    y = scored["label"].to_numpy(dtype=int)
    score = scored["p_cancer"].to_numpy(dtype=float)
    return binary_metric_values(y, score, np.full(len(score), 0.5))


def _ranking_key(metrics: dict[str, float], candidate_c: float) -> tuple[float, float, float, float]:
    return (
        float(metrics["log_loss"]),
        float(metrics["brier_score"]),
        -float(metrics["roc_auc"]),
        float(candidate_c),
    )


def _validated_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(sorted({float(value) for value in values}))
    if len(grid) < 3 or any(not np.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError("candidate_c must contain at least three positive finite values.")
    return grid
