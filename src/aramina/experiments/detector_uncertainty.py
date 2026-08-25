"""Detector-level uncertainty and polar-cake helpers for Aramina experiments."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xrd_preprocessing import (
    perform_azimuthal_integration,
    perform_polar_cake_integration,
    sample_detector_centered_poisson,
)


RAW_FRAME_COLUMN = "measurement_data"
MASK_COLUMN = "faulty_pixel_mask"
CALIBRATION_SESSION_COLUMN = "calibration_session_uid"
MASKED_PIXEL_POLICY = (
    "zero_masked_pixels_then_centered_poisson_on_positive_component"
)


def iter_detector_poisson_profiles(
    patient_frame: pd.DataFrame,
    *,
    draws: int,
    random_state: int,
    normalization_q_range: tuple[float, float],
) -> Iterator[pd.DataFrame]:
    """Yield product-shaped profiles after seeded detector Poisson draws."""
    _require_detector_columns(patient_frame)
    observed = np.stack(
        [
            _centered_poisson_observation(row[RAW_FRAME_COLUMN], row[MASK_COLUMN])
            for _, row in patient_frame.iterrows()
        ]
    )
    session_ids = patient_frame[CALIBRATION_SESSION_COLUMN].astype(str).tolist()
    rng = np.random.default_rng(random_state)

    for _ in range(draws):
        sampled = sample_detector_centered_poisson(
            observed,
            1,
            measurement_axis=0,
            measurement_session_ids=session_ids,
            random_state=rng,
        ).values[0]
        out = patient_frame.copy(deep=True)
        q_values: list[np.ndarray] = []
        raw_profiles: list[np.ndarray] = []
        normalized_profiles: list[np.ndarray] = []
        for row_index, (_, row) in enumerate(patient_frame.iterrows()):
            sampled_row = row.copy()
            sampled_row[RAW_FRAME_COLUMN] = sampled[row_index]
            q, intensity, _sigma, _distance = perform_azimuthal_integration(
                sampled_row,
                column=RAW_FRAME_COLUMN,
                npt=100,
                mask_column=MASK_COLUMN,
                calibration_mode="poni",
                error_model="poisson",
                thickness_adjustment=True,
                require_thickness_adjustment=True,
                thickness_reference_column="calibrant_thickness_mm",
                sample_thickness_column="sample_thickness_mm",
            )
            q = np.asarray(q, dtype=float)
            intensity = np.asarray(intensity, dtype=float)
            q_values.append(q)
            raw_profiles.append(intensity)
            normalized_profiles.append(
                normalize_profile(
                    q,
                    intensity,
                    q_range=normalization_q_range,
                )
            )
        out["q_range"] = q_values
        out["radial_profile_data_raw"] = raw_profiles
        out["radial_profile_data"] = normalized_profiles
        yield out


def _centered_poisson_observation(frame: Any, faulty_pixels: Any) -> np.ndarray:
    """Prepare baseline-corrected detector values for centered perturbation."""
    observed = np.asarray(frame, dtype=float).copy()
    if observed.ndim != 2:
        raise ValueError("Detector observation must be a two-dimensional frame.")
    mask = _pixel_mask(faulty_pixels, observed.shape)
    invalid = ~np.isfinite(observed)
    if np.any(invalid & ~mask):
        raise ValueError(
            "Every non-finite detector pixel must be present in faulty_pixel_mask "
            "before centered Poisson perturbation."
        )
    observed[mask] = 0.0
    if not np.isfinite(observed).all():
        raise ValueError("Prepared detector observations must be finite.")
    return observed


def _pixel_mask(faulty_pixels: Any, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(faulty_pixels)
    if values.shape == shape:
        return values.astype(bool)
    if values.size == 0:
        return np.zeros(shape, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            "faulty_pixel_mask must be a 2D mask or an Nx2 coordinate array."
        )
    coordinates = np.asarray(values, dtype=int)
    rows, columns = coordinates[:, 0], coordinates[:, 1]
    if (
        np.any(rows < 0)
        or np.any(columns < 0)
        or np.any(rows >= shape[0])
        or np.any(columns >= shape[1])
    ):
        raise ValueError("faulty_pixel_mask contains an out-of-bounds coordinate.")
    mask = np.zeros(shape, dtype=bool)
    mask[rows, columns] = True
    return mask


def normalize_profile(
    q: np.ndarray,
    profile: np.ndarray,
    *,
    q_range: tuple[float, float],
) -> np.ndarray:
    """Apply the frozen median normalization to one integrated profile."""
    q_values = np.asarray(q, dtype=float).ravel()
    values = np.asarray(profile, dtype=float).ravel()
    if q_values.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Integrated profile is not finite on the expected q grid.")
    band = (q_values >= q_range[0]) & (q_values <= q_range[1])
    if int(np.sum(band)) < 1:
        raise ValueError("Normalization q range has no integrated profile bins.")
    scale = float(np.median(values[band]))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Integrated profile has a non-positive normalization value.")
    return values / scale


def write_polar_cake_artifacts(
    dataframe: pd.DataFrame,
    *,
    target_cases: Sequence[tuple[str, str]],
    output_folder: Path,
    n_q: int,
    n_chi: int,
    parity_max_relative_rmse: float,
) -> pd.DataFrame:
    """Persist baseline cakes and one-dimensional parity records."""
    _require_detector_columns(dataframe)
    output_folder.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for patient_id, target_side in target_cases:
        patient = dataframe[dataframe["patientId"].astype(str) == patient_id]
        for row_number, (_, row) in enumerate(patient.iterrows()):
            cake = perform_polar_cake_integration(
                row,
                column=RAW_FRAME_COLUMN,
                npt=n_q,
                npt_azimuthal=n_chi,
                mask_column=MASK_COLUMN,
                calibration_mode="poni",
                error_model="poisson",
                thickness_adjustment=True,
                require_thickness_adjustment=True,
                thickness_reference_column="calibrant_thickness_mm",
                sample_thickness_column="sample_thickness_mm",
            )
            q_1d, intensity_1d, _sigma_1d, _distance = perform_azimuthal_integration(
                row,
                column=RAW_FRAME_COLUMN,
                npt=n_q,
                mask_column=MASK_COLUMN,
                calibration_mode="poni",
                error_model="poisson",
                thickness_adjustment=True,
                require_thickness_adjustment=True,
                thickness_reference_column="calibrant_thickness_mm",
                sample_thickness_column="sample_thickness_mm",
            )
            radial_from_cake = _weighted_angular_mean(cake.intensity, cake.count)
            relative_rmse = _relative_rmse(radial_from_cake, intensity_1d)
            measurement_id = _measurement_id(row, row_number)
            artifact_name = f"{_safe(patient_id)}_{measurement_id}.npz"
            artifact_path = output_folder / artifact_name
            np.savez_compressed(
                artifact_path,
                intensity=cake.intensity,
                q=cake.q,
                azimuth=cake.azimuth,
                sigma=cake.sigma,
                sum_variance=cake.sum_variance,
                count=cake.count,
                radial_reference_q=np.asarray(q_1d),
                radial_reference_intensity=np.asarray(intensity_1d),
                radial_from_cake=radial_from_cake,
            )
            records.append(
                {
                    "patient_id": patient_id,
                    "target_side": target_side,
                    "measured_side": str(row["side"]),
                    "specimen_id": str(row["specimenId"]),
                    "position": str(row.get("position", "")),
                    "calibration_session_uid": str(row[CALIBRATION_SESSION_COLUMN]),
                    "n_q": n_q,
                    "n_chi": n_chi,
                    "relative_rmse": relative_rmse,
                    "parity_max_relative_rmse": parity_max_relative_rmse,
                    "parity_pass": bool(relative_rmse <= parity_max_relative_rmse),
                    "artifact": str(artifact_path.relative_to(output_folder.parent)),
                }
            )
    return pd.DataFrame(records)


def _weighted_angular_mean(
    intensity: np.ndarray,
    count: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(intensity, dtype=float)
    if count is None:
        return np.nanmean(values, axis=0)
    weights = np.asarray(count, dtype=float)
    weighted = np.sum(values * weights, axis=0)
    denominator = np.sum(weights, axis=0)
    return np.divide(
        weighted,
        denominator,
        out=np.full(weighted.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _relative_rmse(observed: np.ndarray, reference: np.ndarray) -> float:
    observed_values = np.asarray(observed, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    valid = np.isfinite(observed_values) & np.isfinite(reference_values)
    if not np.any(valid):
        return float("inf")
    scale = float(np.sqrt(np.mean(reference_values[valid] ** 2)))
    if scale <= 1e-12:
        return float("inf")
    return float(
        np.sqrt(np.mean((observed_values[valid] - reference_values[valid]) ** 2))
        / scale
    )


def _measurement_id(row: pd.Series, row_number: int) -> str:
    parts = (
        str(row.get("specimenId", "specimen")),
        str(row.get("position", row_number)),
        str(row.get("started_at", row_number)),
    )
    return _safe("_".join(parts))


def _safe(value: str) -> str:
    clean = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return clean.strip("_") or "measurement"


def _require_detector_columns(dataframe: pd.DataFrame) -> None:
    required = {
        "patientId",
        "specimenId",
        "side",
        "ponifile",
        "sample_thickness_mm",
        "calibrant_thickness_mm",
        RAW_FRAME_COLUMN,
        MASK_COLUMN,
        CALIBRATION_SESSION_COLUMN,
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Detector experiment requires columns: {missing}.")
