from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from aramina.attenuation_experiment import (
    ATTENUATION_SYMMETRY_COLUMNS,
    AttenuationExperimentUnavailable,
    audit_archive_transmission_metadata,
    evaluate_paired_attenuation_contribution,
    extract_three_point_attenuation_features,
    write_archive_audit_artifacts,
)


def test_extracts_validated_three_point_and_bilateral_symmetry_features():
    result = extract_three_point_attenuation_features(_validated_measurements())

    assert result.status == "available"
    left = result.features.query("patientId == 'P01' and side == 'LEFT'").iloc[0]
    assert left["attenuation_p1"] == pytest.approx(0.2)
    assert left["attenuation_mean"] == pytest.approx(0.3)
    assert left["attenuation_std"] == pytest.approx(np.std([0.2, 0.3, 0.4]))
    assert left["attenuation_delta_p1"] == pytest.approx(-0.3)
    assert left["attenuation_mean_abs_delta"] == pytest.approx(0.3)
    assert left["attenuation_symmetry_available"] == 1
    assert left["attenuation_evaluation_eligible"] == 1
    assert result.coverage["complete_three_point"].eq(1).all()


def test_marks_missing_point_and_does_not_silently_select_duplicate_measurement():
    measurements = _validated_measurements()
    measurements = measurements.loc[
        ~(
            measurements["patientId"].eq("P01")
            & measurements["side"].eq("Left")
            & measurements["position"].eq("P3")
        )
    ].copy()
    measurements = pd.concat([measurements, measurements.iloc[[0]]], ignore_index=True)

    result = extract_three_point_attenuation_features(measurements)

    coverage = result.coverage.query("patientId == 'P01' and side == 'LEFT'").iloc[0]
    assert result.status == "unavailable"
    assert coverage["complete_three_point"] == 0
    assert "p1_row_count_2" in coverage["availability_reason"]
    assert "p3_row_count_0" in coverage["availability_reason"]
    assert set(result.features["side"]) == {"RIGHT"}
    assert result.features["attenuation_evaluation_eligible"].eq(0).all()


def test_rejects_categorical_density_and_requires_validated_provenance():
    measurements = _validated_measurements()

    with pytest.raises(AttenuationExperimentUnavailable, match="breast_density"):
        extract_three_point_attenuation_features(
            measurements,
            value_column="breast_density",
        )

    measurements["attenuation_provenance_status"] = "unvalidated"
    result = extract_three_point_attenuation_features(measurements)
    assert result.status == "unavailable"
    assert result.coverage["validated_rows"].eq(0).all()


def test_archive_audit_does_not_promote_transmission_metadata_to_attenuation(tmp_path: Path):
    archive_path = tmp_path / "archive.h5"
    _write_transmission_archive(archive_path)

    audit = audit_archive_transmission_metadata(archive_path)

    assert audit.status == "unavailable"
    assert len(audit.inventory) == 3
    assert audit.inventory["paired_main_adjacent"].eq(1).all()
    assert audit.inventory["transmission_pct"].eq(10.0).all()
    assert audit.inventory["correction_factor"].eq(1.5964).all()
    assert audit.inventory["attenuation_input_usable"].eq(0).all()
    assert audit.coverage["raw_three_point_pairing"].tolist() == [1]
    assert audit.coverage["validated_three_point_eligible"].tolist() == [0]
    assert "reference identifier" in audit.unavailable_reason
    paths = write_archive_audit_artifacts(
        audit,
        archive_path=archive_path,
        output_dir=tmp_path / "audit",
    )
    assert set(paths) == {"inventory", "coverage", "status"}
    assert all(path.is_file() for path in paths.values())
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    assert status["status"] == "unavailable"
    assert status["canonical_labelled_raw_three_point_breast_sessions"] == 1


def test_archive_audit_keeps_noncanonical_status_unlabelled(tmp_path: Path):
    archive_path = tmp_path / "archive.h5"
    _write_transmission_archive(archive_path, specimen_status="NORMAL")

    audit = audit_archive_transmission_metadata(archive_path)

    assert audit.inventory["product_label"].isna().all()
    assert audit.coverage["product_label"].isna().all()


def test_paired_evaluation_uses_identical_cases_and_patient_safe_folds():
    cases = _evaluation_cases()

    result = evaluate_paired_attenuation_contribution(
        cases,
        baseline_feature_columns=("baseline_profile", "age"),
        n_splits=2,
        n_repeats=2,
        random_state=7,
    )

    assert len(result.split_metrics) == 8
    assert len(result.paired_deltas) == 4
    assert set(result.split_metrics["model_name"]) == {
        "baseline",
        "baseline_plus_attenuation",
    }
    assert set(result.split_metrics) >= {
        "roc_auc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "ppv",
        "npv",
        "threshold",
        "false_negatives",
    }
    assert set(result.paired_deltas) >= {
        "delta_roc_auc",
        "delta_sensitivity",
        "delta_false_negatives",
    }
    for _, split in result.predictions.groupby("split_id"):
        cases_by_model = split.groupby("model_name")["target_case_id"].agg(set)
        assert len(cases_by_model) == 2
        assert cases_by_model.iloc[0] == cases_by_model.iloc[1]
        for patient_id, patient_rows in split.groupby("patientId"):
            assert len(patient_rows["model_name"].unique()) == 2
            assert patient_rows["target_case_id"].nunique() == 2
    assert result.coverage.loc[0, "excluded_cases"] == 0


def test_paired_evaluation_fails_when_no_complete_validated_cases():
    cases = _evaluation_cases()
    cases["attenuation_evaluation_eligible"] = 0

    with pytest.raises(AttenuationExperimentUnavailable, match="no complete validated"):
        evaluate_paired_attenuation_contribution(
            cases,
            baseline_feature_columns=("baseline_profile", "age"),
            n_splits=2,
        )


def _validated_measurements() -> pd.DataFrame:
    records = []
    values = {
        ("P01", "Left"): [0.2, 0.3, 0.4],
        ("P01", "Right"): [0.5, 0.6, 0.7],
    }
    for (patient_id, side), point_values in values.items():
        for position, value in zip(("P1", "P2", "P3"), point_values, strict=True):
            records.append(
                {
                    "patientId": patient_id,
                    "side": side,
                    "position": position,
                    "attenuation_value": value,
                    "attenuation_provenance_status": "measured_validated",
                    "attenuation_formula_id": "validated_formula_v1",
                    "attenuation_reference_id": f"reference-{patient_id}-{position}",
                    "attenuation_units": "1",
                    "breast_density": "C",
                }
            )
    return pd.DataFrame(records)


def _write_transmission_archive(path: Path, *, specimen_status: str = "BENIGN") -> None:
    with h5py.File(path, "w") as archive:
        calibration = archive.create_group("calib_001")
        session = calibration.create_group("sample_01_Nova_400_Left")
        sample = session.create_group("sample")
        sample.attrs.update(
            {"side": "Left", "specimen_status": specimen_status, "biopsy": True}
        )
        sample.create_dataset("patient_name", data="Nova_400")
        sets = session.create_group("sets")
        for index, position in enumerate(("P1", "P2", "P3"), start=1):
            transmission = sets.create_group(f"set_{index * 2 - 1:03d}_sample_transmission")
            transmission.attrs.update(
                {
                    "set_pk": index * 2 - 1,
                    "measurement_type_name": "Sample Transmission",
                    "position": position,
                    "transmission_pct": 10.0,
                    "correction_factor": 1.5964,
                }
            )
            transmission.create_dataset("metadata", data=json.dumps({}))
            processing = transmission.create_group("processing")
            processing.create_dataset("config", data=json.dumps({}))
            main = sets.create_group(f"set_{index * 2:03d}_sample_main")
            main.attrs.update(
                {
                    "set_pk": index * 2,
                    "measurement_type_name": "Sample Main",
                    "position": position,
                }
            )


def _evaluation_cases() -> pd.DataFrame:
    records = []
    for patient_index in range(8):
        label = int(patient_index >= 4)
        for side, sign in (("LEFT", -1.0), ("RIGHT", 1.0)):
            base = patient_index / 10.0
            record = {
                "target_case_id": f"P{patient_index:02d}::{side}",
                "patientId": f"P{patient_index:02d}",
                "label": label,
                "baseline_profile": base,
                "age": 45.0 + patient_index,
                "attenuation_evaluation_eligible": 1,
            }
            for point_index, point in enumerate(("p1", "p2", "p3"), start=1):
                value = base + sign * point_index / 100.0
                record[f"attenuation_{point}"] = value
                record[f"attenuation_delta_{point}"] = sign * point_index / 50.0
                record[f"attenuation_abs_delta_{point}"] = point_index / 50.0
            record["attenuation_mean"] = np.mean(
                [record["attenuation_p1"], record["attenuation_p2"], record["attenuation_p3"]]
            )
            record["attenuation_std"] = np.std(
                [record["attenuation_p1"], record["attenuation_p2"], record["attenuation_p3"]]
            )
            record["attenuation_range"] = record["attenuation_p3"] - record["attenuation_p1"]
            record["attenuation_mean_delta"] = np.mean(
                [record["attenuation_delta_p1"], record["attenuation_delta_p2"], record["attenuation_delta_p3"]]
            )
            record["attenuation_mean_abs_delta"] = np.mean(
                [record["attenuation_abs_delta_p1"], record["attenuation_abs_delta_p2"], record["attenuation_abs_delta_p3"]]
            )
            record["attenuation_rms_delta"] = np.sqrt(
                np.mean(
                    np.square(
                        [
                            record["attenuation_delta_p1"],
                            record["attenuation_delta_p2"],
                            record["attenuation_delta_p3"],
                        ]
                    )
                )
            )
            records.append(record)
    assert set(ATTENUATION_SYMMETRY_COLUMNS).issubset(records[0])
    return pd.DataFrame(records)
