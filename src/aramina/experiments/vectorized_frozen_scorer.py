"""Vectorized scorer for the immutable Aramina 0.2.15 target-breast model.

The scorer keeps the product architecture unchanged.  LR1 is evaluated over the
complete ``draw x measurement`` profile cube in one call, target-side LR1
logits are aggregated per draw, Core4 symmetry is calculated with the product
implementation, and LR2 is evaluated for all draw/target rows in one call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..patient_features import (
    TARGET_CASE_ID,
    normalize_side,
)
from ..symmetry_features import SK_FEATURE_CONTRACT_V0_2
from ..target_breast_model import SK_CORE4_FEATURE_COLUMNS


FROZEN_MODEL_NAME = "aramina_target_breast_risk"
FROZEN_MODEL_VERSION = "0.2.15-beta"
FROZEN_THRESHOLD = 0.24041049078429919
PROFILE_BINS = 100


class VectorizedFrozenScorerError(ValueError):
    """Raised when cube inputs cannot be scored by the frozen contract."""


@dataclass(frozen=True)
class FrozenScoreCube:
    """Frozen-model scores ordered as ``draw x target``."""

    p_cancer: np.ndarray
    target_case_ids: tuple[str, ...]
    threshold: float
    target_measurements: np.ndarray
    contralateral_measurements: np.ndarray
    symmetry_available: np.ndarray

    def as_dataframe(self) -> pd.DataFrame:
        """Return one auditable record per draw and target breast."""
        draw_count, target_count = self.p_cancer.shape
        return pd.DataFrame(
            {
                "draw_index": np.repeat(np.arange(draw_count), target_count),
                TARGET_CASE_ID: np.tile(self.target_case_ids, draw_count),
                "p_cancer": self.p_cancer.reshape(-1),
                "threshold": self.threshold,
                "target_measurements": self.target_measurements.reshape(-1),
                "contralateral_measurements": self.contralateral_measurements.reshape(
                    -1
                ),
                "symmetry_available": self.symmetry_available.reshape(-1),
            }
        )


def score_frozen_aramina_0_2_15_cube(
    profile_cube: np.ndarray,
    *,
    patient_manifest: pd.DataFrame,
    q_grid: np.ndarray,
    target_manifest: pd.DataFrame,
    model_artifact: Mapping[str, Any],
    model_name: str = FROZEN_MODEL_NAME,
) -> FrozenScoreCube:
    """Score normalized profile draws with the frozen 0.2.15 architecture.

    ``profile_cube`` is ordered ``(draw, measurement, 100 profile bins)``.
    ``patient_manifest`` has one row per measurement and must retain product
    patient, side, specimen, and age fields.  ``q_grid`` is either one common
    100-bin grid or one grid per measurement.  ``target_manifest`` must expose
    ``patient_id`` and ``target_side``; every row denotes one target breast.
    """
    cube = _validate_profile_cube(profile_cube)
    manifest, q_values = _validated_measurement_inputs(
        patient_manifest,
        q_grid,
        measurement_count=cube.shape[1],
        model_artifact=model_artifact,
    )
    targets = _validated_targets(target_manifest, manifest)
    model_info = _frozen_model_info(model_artifact, model_name=model_name)
    lr1_scores = _vectorized_lr1_scores(cube, model_info["lr1_model"])
    draw_count = cube.shape[0]
    target_count = len(targets)
    feature_blocks: list[pd.DataFrame] = []
    target_counts: list[int] = []
    contralateral_counts: list[int] = []
    symmetry_flags: list[np.ndarray] = []
    for target in targets.itertuples(index=False):
        block, target_count_value, contralateral_count_value, available = (
            _vectorized_feature_block(
                cube,
                lr1_scores,
                manifest,
                q_values,
                patient_id=str(target.patient_id),
                target_side=str(target.target_side),
                model_info=model_info,
            )
        )
        feature_blocks.append(block)
        target_counts.append(target_count_value)
        contralateral_counts.append(contralateral_count_value)
        symmetry_flags.append(available)
    feature_table = pd.concat(feature_blocks, ignore_index=True)
    probabilities = model_info["final_model"].predict_proba(feature_table)[:, 1]
    p_cancer = probabilities.reshape(target_count, draw_count).T
    target_measurements = np.broadcast_to(
        np.asarray(target_counts, dtype=int), (draw_count, target_count)
    ).copy()
    contralateral_measurements = np.broadcast_to(
        np.asarray(contralateral_counts, dtype=int), (draw_count, target_count)
    ).copy()
    symmetry_available = np.column_stack(symmetry_flags).astype(int, copy=False)
    return FrozenScoreCube(
        p_cancer=p_cancer,
        target_case_ids=tuple(target.target_case_id for target in targets.itertuples()),
        threshold=FROZEN_THRESHOLD,
        target_measurements=target_measurements,
        contralateral_measurements=contralateral_measurements,
        symmetry_available=symmetry_available,
    )


def _validate_profile_cube(profile_cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(profile_cube, dtype=float)
    if cube.ndim != 3 or cube.shape[2] != PROFILE_BINS:
        raise VectorizedFrozenScorerError(
            "profile_cube must have shape (draw, measurement, 100)."
        )
    if cube.shape[0] == 0 or cube.shape[1] == 0 or not np.isfinite(cube).all():
        raise VectorizedFrozenScorerError(
            "profile_cube must contain finite profiles for at least one draw and measurement."
        )
    return cube


def _validated_measurement_inputs(
    patient_manifest: pd.DataFrame,
    q_grid: np.ndarray,
    *,
    measurement_count: int,
    model_artifact: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(patient_manifest) != measurement_count:
        raise VectorizedFrozenScorerError(
            "patient_manifest row count must equal profile_cube measurements."
        )
    columns = model_artifact.get("model_columns", {})
    required = {
        str(columns.get("group_column", "patientId")),
        str(columns.get("specimen_column", "specimenId")),
        str(columns.get("side_column", "side")),
    }
    missing = sorted(required.difference(patient_manifest.columns))
    if missing:
        raise VectorizedFrozenScorerError(
            f"patient_manifest is missing product columns: {missing}."
        )
    manifest = patient_manifest.reset_index(drop=True).copy(deep=True)
    q_values = np.asarray(q_grid, dtype=float)
    if q_values.ndim == 1:
        if q_values.shape[0] != PROFILE_BINS:
            raise VectorizedFrozenScorerError("q_grid must contain 100 bins.")
        q_values = np.broadcast_to(q_values, (measurement_count, PROFILE_BINS)).copy()
    if q_values.shape != (measurement_count, PROFILE_BINS):
        raise VectorizedFrozenScorerError(
            "q_grid must have shape (100,) or (measurement, 100)."
        )
    if not np.isfinite(q_values).all():
        raise VectorizedFrozenScorerError("q_grid contains non-finite values.")
    return manifest, q_values


def _validated_targets(
    target_manifest: pd.DataFrame,
    patient_manifest: pd.DataFrame,
) -> pd.DataFrame:
    required = {"patient_id", "target_side"}
    missing = sorted(required.difference(target_manifest.columns))
    if missing:
        raise VectorizedFrozenScorerError(
            f"target_manifest is missing columns: {missing}."
        )
    targets = target_manifest.loc[:, ["patient_id", "target_side"]].copy()
    if targets.empty:
        raise VectorizedFrozenScorerError("target_manifest cannot be empty.")
    targets["patient_id"] = targets["patient_id"].astype(str)
    targets["target_side"] = targets["target_side"].map(normalize_side)
    if targets["target_side"].isna().any():
        raise VectorizedFrozenScorerError("target_manifest contains an invalid target_side.")
    targets["target_case_id"] = (
        targets["patient_id"] + "::" + targets["target_side"]
    )
    if targets["target_case_id"].duplicated().any():
        raise VectorizedFrozenScorerError("target_manifest contains duplicate target breasts.")
    patient_ids = set(patient_manifest["patientId"].astype(str))
    unknown = sorted(set(targets["patient_id"]).difference(patient_ids))
    if unknown:
        raise VectorizedFrozenScorerError(
            f"target_manifest refers to absent patients: {unknown}."
        )
    return targets.reset_index(drop=True)


def _frozen_model_info(
    model_artifact: Mapping[str, Any],
    *,
    model_name: str,
) -> Mapping[str, Any]:
    identity = model_artifact.get("model_identity", {})
    if not isinstance(identity, Mapping) or str(identity.get("version")) != FROZEN_MODEL_VERSION:
        raise VectorizedFrozenScorerError(
            f"Expected immutable {FROZEN_MODEL_VERSION} artifact."
        )
    models = model_artifact.get("models")
    if not isinstance(models, Mapping) or model_name not in models:
        raise VectorizedFrozenScorerError(f"Model artifact has no {model_name!r} route.")
    model_info = models[model_name]
    if not isinstance(model_info, Mapping):
        raise VectorizedFrozenScorerError("Frozen model route is malformed.")
    required = {"lr1_model", "final_model", "thresholds"}
    if not required.issubset(model_info):
        raise VectorizedFrozenScorerError("Frozen model route misses required estimators.")
    threshold = float(model_info["thresholds"].get("threshold_target", np.nan))
    if not np.isclose(threshold, FROZEN_THRESHOLD, rtol=0.0, atol=0.0):
        raise VectorizedFrozenScorerError(
            "Frozen artifact threshold does not match 0.2.15 contract."
        )
    return model_info


def _vectorized_lr1_scores(profile_cube: np.ndarray, lr1_model: Any) -> np.ndarray:
    draws, measurements, bins = profile_cube.shape
    scores = lr1_model.predict_proba(profile_cube.reshape(draws * measurements, bins))[
        :, 1
    ]
    return scores.reshape(draws, measurements)


def _vectorized_feature_block(
    profile_cube: np.ndarray,
    lr1_scores: np.ndarray,
    manifest: pd.DataFrame,
    q_values: np.ndarray,
    *,
    patient_id: str,
    target_side: str,
    model_info: Mapping[str, Any],
) -> tuple[pd.DataFrame, int, int, np.ndarray]:
    columns = model_info.get("model_columns", {})
    group_column = str(columns.get("group_column", "patientId"))
    side_column = str(columns.get("side_column", "side"))
    age_column = str(columns.get("age_column", "age"))
    patient_mask = manifest[group_column].astype(str).eq(patient_id).to_numpy()
    patient_frame = manifest.loc[patient_mask]
    if patient_frame.empty:
        raise VectorizedFrozenScorerError(f"Patient is absent: {patient_id!r}.")
    side_norms = patient_frame[side_column].map(normalize_side)
    available_sides = sorted(side_norms.dropna().unique())
    if target_side not in available_sides:
        raise VectorizedFrozenScorerError(
            f"Target side {target_side!r} is absent for patient {patient_id!r}."
        )
    contralateral = next((side for side in available_sides if side != target_side), None)
    target_mask_local = side_norms.eq(target_side).to_numpy()
    contralateral_mask_local = side_norms.eq(contralateral).to_numpy()
    patient_scores = lr1_scores[:, patient_mask]
    target_scores = patient_scores[:, target_mask_local]
    if target_scores.shape[1] == 0:
        raise VectorizedFrozenScorerError(
            f"No target measurements for {patient_id!r}/{target_side!r}."
        )
    clipped = np.clip(target_scores, 1e-6, 1.0 - 1e-6)
    profile_logit_average = 1.0 / (
        1.0
        + np.exp(-np.mean(np.log(clipped / (1.0 - clipped)), axis=1))
    )
    patient_profiles = profile_cube[:, patient_mask, :]
    patient_q = q_values[patient_mask]
    core4, symmetry_available = _vectorized_core4(
        patient_profiles,
        patient_q,
        target_mask=target_mask_local,
        contralateral_mask=contralateral_mask_local,
        feature_contract=str(model_info.get("symmetry_feature_contract", "")),
    )
    age_values = pd.to_numeric(patient_frame[age_column], errors="coerce")
    age_available = int(age_values.notna().any())
    age = float(age_values.median()) if age_available else 0.0
    block = pd.DataFrame(
        {
            "profile_p_cancer_logit_average": profile_logit_average,
            "age": age,
            "age_available": age_available,
            "symmetry_available": symmetry_available.astype(int),
            **core4,
        }
    )
    return (
        block,
        int(target_mask_local.sum()),
        int(contralateral_mask_local.sum()),
        symmetry_available,
    )


def _vectorized_core4(
    profiles: np.ndarray,
    q_values: np.ndarray,
    *,
    target_mask: np.ndarray,
    contralateral_mask: np.ndarray,
    feature_contract: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    draws = profiles.shape[0]
    if feature_contract != SK_FEATURE_CONTRACT_V0_2:
        raise VectorizedFrozenScorerError(
            "Vectorized scorer requires the frozen v0.2 symmetry contract."
        )
    if int(target_mask.sum()) < 2 or int(contralateral_mask.sum()) < 2:
        return (
            {column: np.zeros(draws, dtype=float) for column in SK_CORE4_FEATURE_COLUMNS},
            np.zeros(draws, dtype=bool),
        )
    reference_q = q_values[0]
    if not np.allclose(q_values, reference_q, rtol=0.0, atol=1e-12):
        raise VectorizedFrozenScorerError(
            "Vectorized symmetry requires one common q grid per patient."
        )
    smoothed = savgol_filter(profiles, window_length=11, polyorder=3, axis=-1)
    target = smoothed[:, target_mask, :]
    contralateral = smoothed[:, contralateral_mask, :]
    target_mean = np.mean(target, axis=1)
    contralateral_mean = np.mean(contralateral, axis=1)
    target_std = np.std(target, axis=1, ddof=1)
    contralateral_std = np.std(contralateral, axis=1, ddof=1)
    mask1 = (reference_q >= 6.7) & (reference_q <= 15.0)
    mask2 = (reference_q >= 15.0) & (reference_q <= 23.0)
    full = (reference_q >= 2.0) & (reference_q <= 23.0)
    core4 = {
        "sk_wasserstein_distance_full_q2": _vectorized_wasserstein(
            reference_q[full], target_mean[:, full], contralateral_mean[:, full]
        ),
        "sk_weightedrms1": _vectorized_weighted_rms(
            target_mean[:, mask1],
            contralateral_mean[:, mask1],
            target_std[:, mask1],
            contralateral_std[:, mask1],
        ),
        "sk_weightedrms2": _vectorized_weighted_rms(
            target_mean[:, mask2],
            contralateral_mean[:, mask2],
            target_std[:, mask2],
            contralateral_std[:, mask2],
        ),
        "sk_mean_peak_value_abs_delta": _vectorized_mean_peak_delta(
            profiles,
            reference_q,
            target_mask=target_mask,
            contralateral_mask=contralateral_mask,
        ),
    }
    available = np.logical_and.reduce(
        [np.isfinite(values) for values in core4.values()]
    )
    return (
        {
            column: np.where(available, values, 0.0)
            for column, values in core4.items()
        },
        available,
    )


def _vectorized_weighted_rms(
    target_mean: np.ndarray,
    contralateral_mean: np.ndarray,
    target_std: np.ndarray,
    contralateral_std: np.ndarray,
) -> np.ndarray:
    difference = target_mean - contralateral_mean
    variance = target_std**2 + contralateral_std**2
    floor = np.percentile(variance, 5.0, axis=1)
    weight = 1.0 / np.maximum(variance, floor[:, None] + 1e-12)
    return np.sqrt(
        np.sum(weight * difference**2, axis=1) / np.sum(weight, axis=1)
    )


def _vectorized_wasserstein(
    q: np.ndarray,
    target: np.ndarray,
    contralateral: np.ndarray,
) -> np.ndarray:
    target_positive = np.clip(target, 0.0, None)
    contralateral_positive = np.clip(contralateral, 0.0, None)
    target_sum = np.sum(target_positive, axis=1)
    contralateral_sum = np.sum(contralateral_positive, axis=1)
    valid = (target_sum > 1e-12) & (contralateral_sum > 1e-12)
    output = np.full(target.shape[0], np.nan, dtype=float)
    target_pdf = target_positive[valid] / target_sum[valid, None]
    contralateral_pdf = (
        contralateral_positive[valid] / contralateral_sum[valid, None]
    )
    output[valid] = np.sum(
        np.abs(
            np.cumsum(target_pdf, axis=1)[:, :-1]
            - np.cumsum(contralateral_pdf, axis=1)[:, :-1]
        )
        * np.diff(q),
        axis=1,
    )
    return output


def _vectorized_mean_peak_delta(
    profiles: np.ndarray,
    q: np.ndarray,
    *,
    target_mask: np.ndarray,
    contralateral_mask: np.ndarray,
) -> np.ndarray:
    peak_mask = (q >= 13.0) & (q <= 14.8)
    target_peak = np.max(profiles[:, target_mask][:, :, peak_mask], axis=2)
    contralateral_peak = np.max(
        profiles[:, contralateral_mask][:, :, peak_mask], axis=2
    )
    return np.abs(np.mean(target_peak, axis=1) - np.mean(contralateral_peak, axis=1))
