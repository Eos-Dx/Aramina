from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from xrd_preprocessing import load_preprocessing_config


EXPECTED_PROFILE_LENGTH = 100
SYNTHETIC_PONI = """# Synthetic PONI for Aramis tests
poni_version: 2.1
Detector: Detector
Detector_config: {"pixel1": 0.0001, "pixel2": 0.0001, "max_shape": [32, 32], "orientation": 3}
Distance: 0.005
Poni1: 0.0016
Poni2: 0.0016
Rot1: 0
Rot2: 0
Rot3: 0
Wavelength: 1e-10
"""
ONE_TO_MANY_OUTPUT_COLUMNS = {
    "age",
    "azimuthal_mask_pixels",
    "azimuthal_mask_source",
    "azimuthal_mode",
    "azimuthal_npt",
    "azimuthal_npt_azimuthal",
    "biopsy",
    "birads",
    "breast_density",
    "calculated_distance",
    "calibrant_thickness_mm",
    "faulty_pixel_mask",
    "id",
    "interpolation_q_range",
    "meas_name",
    "measurementDate",
    "measurement_data_source",
    "noise_std",
    "patientId",
    "patient_product_diagnosis",
    "patient_specimen_valid",
    "patient_specimen_validity_reason",
    "patient_valid_specimen_count",
    "poni_q_max_nm_inv",
    "position",
    "product_diagnosis",
    "product_status_group",
    "q_range",
    "radial_profile_data_raw",
    "radial_profile_data",
    "radial_profile_sigma",
    "sample_biopsy",
    "sample_biopsy_type",
    "sample_height_in",
    "sample_thickness_mm",
    "sample_weight_lb",
    "set_name",
    "side",
    "snr_db",
    "snr_linear",
    "snr_method_used",
    "specimen_measurement_count",
    "specimenId",
    "specimen_status",
    "started_at",
    "thickness_adjusted_distance_m",
    "thickness_adjustment_applied",
    "thickness_adjustment_reliable",
    "thickness_adjustment_warning",
    "thickness_reference_mm",
    "thickness_reference_source",
}
ONE_TO_ONE_OUTPUT_COLUMNS = ONE_TO_MANY_OUTPUT_COLUMNS | {
    "one_to_one_pair_type",
    "patient_valid_specimen_count",
}
PAYLOAD_COLUMNS = {
    "measurement_data",
    "raw_data",
    "processed_data",
    "detector_measurements",
    "gfrm_data",
}


def load_synthetic_config(branch: str = "one_to_one") -> dict:
    config_file = {
        "one_to_one": "aramis_one_to_one_max_v0_1.yaml",
        "one_to_one_biopsy": "aramis_one_to_one_biopsy_max_v0_1.yaml",
        "one_to_many": "aramis_one_to_many_max_v0_1.yaml",
        "one_to_many_benign_cancer": "aramis_one_to_many_max_v0_1.yaml",
        "one_to_many_benign_cancer_biopsy": "aramis_one_to_many_biopsy_max_v0_1.yaml",
    }[branch]
    config_path = Path(__file__).parents[1] / "config" / "preprocessing" / config_file
    config = load_preprocessing_config(config_path)
    config["raw_data"]["source"] = "npy"
    config["raw_data"]["allowed_sources"] = ["gfrm", "npy"]
    config["raw_data"]["h5_dataset_candidates"]["npy"] = ["raw/data"]
    config["pipeline"]["steps"][0] = {
        "name": "h5_blob_to_df",
        "transformer": "H5BlobDataFrameTransformer",
        "params": {
            "source": {"$ref": "raw_data.source"},
            "dataset_candidates": {
                "$ref": "raw_data.h5_dataset_candidates.npy",
            },
        },
    }
    config["snr"]["min_snr_db"] = -100.0
    config["integration"]["q_range_nm_inv"] = [2.0, 23.0]
    config["normalization"]["q_range_nm_inv"] = [6.7, 7.1]
    config["profile_gate"]["q_nm_inv"] = 14.0
    config["profile_gate"]["min_value"] = -1_000_000.0
    return config


def write_known_synthetic_h5(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        _write_measurement(
            h5,
            "p1_left",
            patient_id="P1",
            specimen_id="P1_LEFT",
            side="Left",
            status="BENIGN",
            seed=1,
            biopsy=True,
        )
        _write_measurement(
            h5,
            "p1_right",
            patient_id="P1",
            specimen_id="P1_RIGHT",
            side="Right",
            status="CANCER",
            seed=2,
        )
        _write_measurement(
            h5,
            "p2_left_only",
            patient_id="P2",
            specimen_id="P2_LEFT",
            side="Left",
            status="BENIGN",
            seed=3,
        )
        _write_measurement(
            h5,
            "p3_left_normal",
            patient_id="P3",
            specimen_id="P3_LEFT",
            side="Left",
            status="NORMAL",
            seed=4,
        )
        _write_measurement(
            h5,
            "p3_right_atypical",
            patient_id="P3",
            specimen_id="P3_RIGHT",
            side="Right",
            status="ATYPICAL",
            seed=5,
        )
        _write_measurement(
            h5,
            "p4_left_missing_thickness",
            patient_id="P4",
            specimen_id="P4_LEFT",
            side="Left",
            status="BENIGN",
            seed=6,
            sample_thickness_mm=None,
        )
        _write_measurement(
            h5,
            "p4_right_orphan_after_thickness",
            patient_id="P4",
            specimen_id="P4_RIGHT",
            side="Right",
            status="CANCER",
            seed=7,
        )
        _write_measurement(
            h5,
            "p5_bad_calibrant",
            patient_id="P5",
            specimen_id="P5_LEFT",
            side="Left",
            status="CANCER",
            seed=8,
            calibrant_thickness_mm=50.0,
        )


def assert_common_output_contract(frame) -> None:
    assert PAYLOAD_COLUMNS.isdisjoint(set(frame.columns))
    assert bool(frame["thickness_adjustment_applied"].all())
    assert bool(frame["thickness_adjustment_reliable"].all())
    assert set(frame["thickness_reference_source"]) == {"calibrant_thickness_mm"}
    assert set(frame["thickness_reference_mm"]) == {10.0}
    assert set(frame["calibrant_thickness_mm"]) == {10.0}
    assert frame["sample_thickness_mm"].notna().all()
    assert set(frame["measurement_data_source"]).issubset(
        {"npy:raw/data", "npy:processed/data"}
    )
    assert all(len(values) == EXPECTED_PROFILE_LENGTH for values in frame["q_range"])
    assert all(
        len(values) == EXPECTED_PROFILE_LENGTH for values in frame["radial_profile_data"]
    )
    for q_values, profile_values in zip(
        frame["q_range"],
        frame["radial_profile_data"],
        strict=True,
    ):
        q = np.asarray(q_values, dtype=float)
        profile = np.asarray(profile_values, dtype=float)
        band = (q >= 6.7) & (q <= 7.1)
        np.testing.assert_allclose(np.median(profile[band]), 1.0)


def _write_measurement(
    h5: h5py.File,
    name: str,
    *,
    patient_id: str,
    specimen_id: str,
    side: str,
    status: str,
    seed: int,
    sample_thickness_mm: float | None = 10.0,
    calibrant_thickness_mm: float = 10.0,
    biopsy: bool | None = None,
) -> None:
    biopsy_flag = status not in {"NORMAL", "BENIGN"} if biopsy is None else bool(biopsy)
    group = h5.require_group(f"measurements/{name}")
    raw = group.require_group("raw")
    processed = group.require_group("processed")
    raw.create_dataset("data", data=_image(seed, offset=200.0))
    processed.create_dataset("data", data=_image(seed, offset=180.0))
    group.attrs.update(
        {
            "patientId": patient_id,
            "ponifile": SYNTHETIC_PONI,
            "specimenId": specimen_id,
            "side": side,
            "position": "P1",
            "started_at": "2026-05-01 10:00:00",
            "specimen_status": status,
            "biopsy": biopsy_flag,
            "sample_biopsy": biopsy_flag,
            "sample_biopsy_type": "Post-biopsy" if status == "CANCER" else "Pre-biopsy",
            "age": 55.0 + seed,
            "sample_height_in": 64.0,
            "sample_weight_lb": 160.0,
            "breast_density": "C",
            "birads": "BI-RADS 4 Suspicious for Malignancy/High Risk",
            "calibrant_thickness_mm": calibrant_thickness_mm,
            "poni_q_max_nm_inv": 25.0,
        }
    )
    if sample_thickness_mm is not None:
        group.attrs["sample_thickness_mm"] = sample_thickness_mm


def _image(seed: int, *, offset: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((32, 32), offset + seed)
    return base + rng.normal(0.0, 2.0, size=base.shape)
