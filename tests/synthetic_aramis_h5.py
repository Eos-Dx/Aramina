from __future__ import annotations

import json
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
PAYLOAD_COLUMNS = {
    "measurement_data",
    "raw_data",
    "processed_data",
    "detector_measurements",
    "gfrm_data",
}


def load_synthetic_config(cohort: str = "biopsy_patients") -> dict:
    config_file = {
        "biopsy_patients": "config_preprocessing_biopsy_patients_v0_1.yaml",
        "prediction_patient": "config_preprocessing_prediction_patient_v0_1.yaml",
    }[cohort]
    config_path = Path(__file__).parents[1] / "config" / "preprocessing" / config_file
    config = load_preprocessing_config(config_path)
    config.pop("aramis_preprocessing", None)
    config["raw_data"]["source"] = "npy"
    config["raw_data"]["allowed_sources"] = ["gfrm", "npy"]
    config["raw_data"]["h5_dataset_candidates"]["npy"] = ["raw/data"]
    blob_reader = {
        "name": "h5_blob_to_df",
        "transformer": "H5BlobDataFrameTransformer",
        "params": {
            "source": {"$ref": "raw_data.source"},
            "dataset_candidates": {
                "$ref": "raw_data.h5_dataset_candidates.npy",
            },
        },
    }
    h5_steps = {
        "H5PoniGeometryCalculatorTransformer",
        "H5SessionSelectorTransformer",
        "H5ToDataFrameTransformer",
    }
    config["pipeline"]["steps"] = [
        blob_reader,
        *(
            step
            for step in config["pipeline"]["steps"]
            if step["transformer"] not in h5_steps
        ),
    ]
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


def write_v0_3_one_patient_h5(
    path: Path,
    *,
    patient_id: str,
    left_status: str,
    right_status: str,
    target_side: str,
    seed: int,
    measurements_per_breast: int = 3,
) -> None:
    """Write one v0.3-style H5 session with one patient and both breasts."""
    with h5py.File(path, "w") as h5:
        h5.attrs.update(
            {
                "NX_class": "NXroot",
                "format": "xrd-session",
                "schema_version": "0.3",
                "container_type": "session",
                "session_uid": f"session-{patient_id}",
                "container_id": f"container-{patient_id}",
                "created_at": "2026-07-01 10:00:00",
                "producer_software": "Aramis synthetic test",
                "producer_version": "0.1",
            }
        )
        session = h5.require_group("session")
        session.attrs.update(
            {
                "NX_class": "NXentry",
                "session_pk": int(seed),
                "session_uid": f"session-{patient_id}",
                "category": "SAMPLE",
                "status": "COMPLETED",
                "operator_username": "test",
                "started_at": "2026-07-01 10:00:00",
                "calibrant_thickness_mm": 10.0,
            }
        )
        _write_v0_3_sample(session, patient_id)
        _write_v0_3_instrument(session)
        sets = session.require_group("sets")
        set_index = 1
        for side, status in (("Left", left_status), ("Right", right_status)):
            for position_index in range(1, measurements_per_breast + 1):
                _write_v0_3_set(
                    sets,
                    set_index=set_index,
                    patient_id=patient_id,
                    side=side,
                    status=status,
                    target_side=target_side,
                    position=f"P{position_index}",
                    seed=seed + set_index,
                )
                set_index += 1


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


def _write_v0_3_sample(session: h5py.Group, patient_id: str) -> None:
    sample = session.require_group("sample")
    sample.attrs["NX_class"] = "NXsample"
    _write_text_dataset(sample, "name", f"{patient_id}_study")
    _write_text_dataset(sample, "patient_name", patient_id)
    _write_text_dataset(sample, "sample_type", "breast_tissue")


def _write_v0_3_instrument(session: h5py.Group) -> None:
    instrument = session.require_group("instrument")
    instrument.attrs.update(
        {
            "NX_class": "NXinstrument",
            "machine_serial": "HUMAN1-SYNTH",
            "machine_type": "EosDx",
            "machine_location": "test",
            "source_type": "Cu",
        }
    )
    _write_scalar_dataset(instrument, "wavelength", 1.0, units="angstrom")
    _write_scalar_dataset(instrument, "beam_energy", 12.398, units="keV")
    detector_sets = instrument.require_group("detector_sets")
    detector_sets.attrs["NX_class"] = "NXcollection"
    detector_set = detector_sets.require_group("ds_1_SYNTH")
    detector_set.attrs.update(
        {
            "NX_class": "NXcollection",
            "detector_set_id": 1,
            "detector_set_hardware_id": "SYNTH",
            "primary_detector_id": 1,
        }
    )
    _write_json_dataset(
        detector_set,
        "layout",
        {"detectors": [{"detector_id": 1, "x_mm": 0.0, "y_mm": 0.0}]},
    )
    detectors = detector_set.require_group("detectors")
    detector = detectors.require_group("det_1_SYNTH")
    detector.attrs.update(
        {
            "NX_class": "NXdetector",
            "detector_id": 1,
            "detector_hardware_id": "SYNTH",
            "manufacturer": "Synthetic",
            "model": "Synthetic32",
            "material": "Si",
        }
    )
    _write_scalar_dataset(detector, "x_pixel_size", 100.0, units="um")
    _write_scalar_dataset(detector, "y_pixel_size", 100.0, units="um")
    _write_scalar_dataset(detector, "x_pixel_count", 32, units="pixel")
    _write_scalar_dataset(detector, "y_pixel_count", 32, units="pixel")
    _write_scalar_dataset(detector, "sensor_thickness", 500.0, units="um")


def _write_v0_3_set(
    sets: h5py.Group,
    *,
    set_index: int,
    patient_id: str,
    side: str,
    status: str,
    target_side: str,
    position: str,
    seed: int,
) -> None:
    name = f"set_{set_index:03d}_sample_main"
    specimen_id = f"{patient_id}_{side}"
    group = sets.require_group(name)
    group.attrs.update(
        {
            "NX_class": "NXcollection",
            "set_pk": set_index,
            "set_uid": f"{patient_id}-{side}-{position}",
            "workflow_id": "synthetic-workflow",
            "batch_id": "synthetic-batch",
            "detector_set_id": 1,
            "status": "COMPLETED",
            "is_approved": True,
            "measurement_type_name": "sample_main",
            "measurement_type_category": "SAMPLE",
            "workflow_key": "aramis_predict_synthetic_v0_3",
            "sample_name": specimen_id,
            "created_at": "2026-07-01 10:00:00",
            "patientId": patient_id,
            "specimenId": specimen_id,
            "side": side,
            "target_side": target_side,
            "position": position,
            "started_at": "2026-07-01 10:00:00",
            "specimen_status": status,
            "biopsy": side.lower() == target_side.lower(),
            "sample_biopsy": side.lower() == target_side.lower(),
            "sample_biopsy_type": "Post-biopsy" if status == "CANCER" else "Pre-biopsy",
            "age": 50.0 + seed,
            "breast_density": "C",
            "birads": "BI-RADS 4 Suspicious for Malignancy/High Risk",
            "calibrant_thickness_mm": 10.0,
            "poni_q_max_nm_inv": 25.0,
        }
    )
    acquisition = group.require_group("acquisition")
    _write_scalar_dataset(acquisition, "distance", 5.0, units="mm")
    _write_scalar_dataset(acquisition, "voltage", 40.0, units="kV")
    _write_scalar_dataset(acquisition, "current", 30.0, units="uA")
    _write_scalar_dataset(acquisition, "exposure_time", 60.0, units="s")
    _write_scalar_dataset(acquisition, "sample_thickness", 10.0, units="mm")
    _write_json_dataset(
        group,
        "metadata",
        {
            "patientId": patient_id,
            "specimenId": specimen_id,
            "side": side,
            "position": position,
            "specimen_status": status,
            "target_side": target_side,
            "calibrant_thickness_mm": 10.0,
        },
    )
    _write_v0_3_data_group(group, "raw", _image(seed, offset=260.0))
    _write_v0_3_data_group(group, "processed", _image(seed, offset=240.0))
    measurements = group.require_group("measurements")
    detector = measurements.require_group("det_1_SYNTH")
    detector.attrs.update(
        {
            "NX_class": "NXdetector",
            "measurement_pk": set_index,
            "measurement_uid": f"measurement-{patient_id}-{side}-{position}",
            "detector_id": 1,
            "file_path": f"/synthetic/{patient_id}_{side}_{position}.gfrm",
            "metadata_file_path": f"/synthetic/{patient_id}_{side}_{position}.dsc",
            "signal": "data",
        }
    )
    detector.create_dataset("data", data=_image(seed, offset=260.0), compression="gzip")
    detector.create_dataset("mask", data=np.zeros((32, 32), dtype=np.uint8))
    detector.create_dataset("detector_meta", data=np.bytes_("<synthetic dsc>"))
    artifacts = group.require_group("artifacts")
    _write_text_dataset(artifacts, "poni", SYNTHETIC_PONI)


def _write_v0_3_data_group(parent: h5py.Group, name: str, data: np.ndarray) -> None:
    group = parent.require_group(name)
    group.attrs.update({"NX_class": "NXdata", "signal": "data"})
    group.create_dataset("data", data=data, compression="gzip")


def _write_scalar_dataset(
    group: h5py.Group,
    name: str,
    value: float | int,
    *,
    units: str,
) -> None:
    dataset = group.create_dataset(name, data=value)
    dataset.attrs["units"] = units


def _write_json_dataset(group: h5py.Group, name: str, value: dict) -> None:
    _write_text_dataset(group, name, json.dumps(value))


def _write_text_dataset(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=value, dtype=h5py.string_dtype(encoding="utf-8"))


def _image(seed: int, *, offset: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((32, 32), offset + seed)
    return base + rng.normal(0.0, 2.0, size=base.shape)
