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

from ..patient_features import (
    TARGET_CASE_ID,
    display_side,
    has_numeric,
    normalize_side,
    numeric_median,
)
from ..symmetry_features import (
    SK_FEATURE_CONTRACT_V0_1,
    target_contralateral_symmetry_features,
)


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

    feature_rows: list[dict[str, Any]] = []
    row_index: list[tuple[int, int]] = []
    for draw_index, draw_profiles in enumerate(cube):
        draw_frame = manifest.copy(deep=False)
        draw_frame["radial_profile_data"] = list(draw_profiles)
        draw_frame["q_range"] = list(q_values)
        for target_index, target in enumerate(targets.itertuples(index=False)):
            feature_rows.append(
                _feature_row_for_draw_target(
                    draw_frame,
                    lr1_scores[draw_index],
                    patient_id=str(target.patient_id),
                    target_side=str(target.target_side),
                    model_info=model_info,
                )
            )
            row_index.append((draw_index, target_index))

    feature_table = pd.DataFrame(feature_rows)
    probabilities = model_info["final_model"].predict_proba(feature_table)[:, 1]
    draw_count = cube.shape[0]
    target_count = len(targets)
    p_cancer = np.empty((draw_count, target_count), dtype=float)
    target_measurements = np.empty((draw_count, target_count), dtype=int)
    contralateral_measurements = np.empty((draw_count, target_count), dtype=int)
    symmetry_available = np.empty((draw_count, target_count), dtype=int)
    for value, (draw_index, target_index), feature_row in zip(
        probabilities, row_index, feature_rows, strict=True
    ):
        p_cancer[draw_index, target_index] = float(value)
        target_measurements[draw_index, target_index] = int(
            feature_row["target_measurements"]
        )
        contralateral_measurements[draw_index, target_index] = int(
            feature_row["contralateral_measurements"]
        )
        symmetry_available[draw_index, target_index] = int(
            feature_row["symmetry_available"]
        )
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


def _feature_row_for_draw_target(
    draw_frame: pd.DataFrame,
    lr1_scores: np.ndarray,
    *,
    patient_id: str,
    target_side: str,
    model_info: Mapping[str, Any],
) -> dict[str, Any]:
    columns = model_info.get("model_columns", {})
    group_column = str(columns.get("group_column", "patientId"))
    specimen_column = str(columns.get("specimen_column", "specimenId"))
    side_column = str(columns.get("side_column", "side"))
    age_column = str(columns.get("age_column", "age"))
    patient_mask = draw_frame[group_column].astype(str).eq(patient_id).to_numpy()
    patient_frame = draw_frame.loc[patient_mask]
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
    target_scores = lr1_scores[patient_mask][target_mask_local]
    if target_scores.size == 0:
        raise VectorizedFrozenScorerError(
            f"No target measurements for {patient_id!r}/{target_side!r}."
        )
    clipped = np.clip(target_scores, 1e-6, 1.0 - 1e-6)
    profile_logit_average = float(
        1.0 / (1.0 + np.exp(-np.mean(np.log(clipped / (1.0 - clipped)))))
    )
    symmetry = target_contralateral_symmetry_features(
        patient_frame,
        profile_column="radial_profile_data",
        q_column="q_range",
        side_column=side_column,
        target_side_norm=target_side,
        contralateral_side_norm=contralateral,
        feature_contract=str(
            model_info.get("symmetry_feature_contract", SK_FEATURE_CONTRACT_V0_1)
        ),
    )
    return {
        TARGET_CASE_ID: f"{patient_id}::{target_side}",
        "patientId": patient_id,
        "target_side": display_side(target_side),
        "contralateral_side": display_side(contralateral),
        "specimens": int(patient_frame[specimen_column].astype(str).nunique()),
        "measurements": int(len(patient_frame)),
        "age": numeric_median(patient_frame, age_column, default=0.0),
        "age_available": int(has_numeric(patient_frame, age_column)),
        "profile_p_cancer_probability_mean": float(np.mean(target_scores)),
        "profile_p_cancer_logit_average": profile_logit_average,
        "profile_p_cancer_n_measurements": int(target_scores.size),
        **symmetry,
    }
