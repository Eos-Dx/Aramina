"""Focused unit tests for T130 paired held-out comparison helpers."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments/profile_symmetry_age_refinement"
MODULE_PATH = EXPERIMENT_DIR / "t130_holdout_comparison.py"


def _module():
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    spec = importlib.util.spec_from_file_location("t130_holdout_comparison_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest() -> pd.DataFrame:
    rows = []
    for index in range(22):
        patient = f"P{index:02d}" if index < 17 else f"P{index - 17:02d}"
        side = "LEFT" if index % 2 else "RIGHT"
        rows.append({
            "patient_id": patient,
            "target_side": side,
            "target_case_id": f"{patient}::{side}",
            "label": 1 if index < 11 else 0,
        })
    return pd.DataFrame(rows)


def test_assert_t130_composition_and_overlap_guard():
    module = _module()
    manifest = _manifest()
    module.assert_t130_composition(manifest)
    train = pd.DataFrame({"patientId": [f"T{index:03d}" for index in range(164)]})
    module.assert_no_patient_overlap(train, manifest)
    overlapping_train = train.copy()
    overlapping_train.loc[0, "patientId"] = "P00"
    with pytest.raises(ValueError, match="T100/T130 patient overlap"):
        module.assert_no_patient_overlap(overlapping_train, manifest)


def test_wilson_intervals_and_paired_disagreement_are_exact_counts():
    module = _module()
    low, high = module.wilson_interval(8, 11)
    assert low == pytest.approx(0.434, abs=0.002)
    assert high == pytest.approx(0.903, abs=0.002)
    predictions = pd.DataFrame({
        "target_case_id": ["A", "B", "A", "B"],
        "procedure": ["left", "left", "right", "right"],
        "label": [1, 0, 1, 0],
        "prediction": [1, 1, 1, 0],
    })
    result = module.paired_disagreement(predictions)
    all_cases = result.loc[result.reference_group.eq("all")].iloc[0]
    assert all_cases[["same_decision", "different_decision", "both_correct", "left_only_correct", "right_only_correct", "both_incorrect"]].to_dict() == {
        "same_decision": 1,
        "different_decision": 1,
        "both_correct": 1,
        "left_only_correct": 0,
        "right_only_correct": 1,
        "both_incorrect": 0,
    }


def test_oof_seed_matches_locked_train_all_experiment():
    module = _module()
    assert module.TRAIN_ALL_RANDOM_STATE == module.RANDOM_STATE + 9_000_000
    assert module.TRAIN_ALL_LR1_OOF_RANDOM_STATE == module.TRAIN_ALL_RANDOM_STATE + 10


def test_shared_test_features_reject_any_model_input_drift():
    module = _module()
    columns = [
        "target_case_id",
        "profile_p_cancer_logit_average",
        "age",
        "age_available",
        "symmetry_available",
        *module.product_symmetry_columns(),
    ]
    rows = pd.DataFrame(
        [
            ["P1::LEFT", 0.4, 50.0, 1, 1, 0.1, 0.2, 0.3, 0.4],
            ["P2::RIGHT", 0.6, 60.0, 1, 0, 0.0, 0.0, 0.0, 0.0],
        ],
        columns=columns,
    )
    module.assert_shared_test_features(rows, rows.copy())
    changed = rows.copy()
    changed.loc[0, "profile_p_cancer_logit_average"] = np.nextafter(0.4, 1.0)
    module.assert_shared_test_features(rows, changed)
    changed.loc[0, "profile_p_cancer_logit_average"] = 0.41
    with pytest.raises(ValueError, match="feature rows differ"):
        module.assert_shared_test_features(rows, changed)


def test_one_patient_h5_contains_target_and_contralateral_sessions(tmp_path):
    _module()
    from t130_full_patient_preprocessing import extract_one_patient_h5

    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        archive = handle.create_group("calibration")
        archive.attrs["operator_id"] = "operator"
        for session_name, side in (("left_session", "Left"), ("right_session", "Right")):
            session = archive.create_group(session_name)
            sample = session.create_group("sample")
            sample.create_dataset("name", data=f"specimen-{side}".encode())
            sample.attrs["side"] = side
            sample.attrs["age"] = 50
            sample.attrs["biopsy"] = side == "Left"
            sample.attrs["specimen_status"] = "BENIGN"
            sets = session.create_group("sets")
            measurement = sets.create_group("measurement")
            measurement.attrs["position"] = "P1_L"
            measurement.create_dataset("raw_file", data=np.arange(4))

    output = extract_one_patient_h5(
        source,
        {
            "archive_group": "calibration",
            "patient_id": "P001",
            "target_side": "left",
            "left_session": "left_session",
            "right_session": "right_session",
        },
        target_side="left",
        output_path=tmp_path / "patient.h5",
    )
    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema_version"] == "0.3"
        assert set(handle["session/sets"]) == {
            "measurement",
            "contralateral_measurement",
        }
        assert handle["session/sets/measurement"].attrs["patientId"] == "P001"
        assert handle["session/sets/measurement"].attrs["side"] == "Left"
        assert handle["session/sets/contralateral_measurement"].attrs["side"] == "Right"
        assert handle["session/sets/measurement"].attrs["position"] == "P1"


def test_frozen_recalibrated_artifacts_load_and_score_from_repository_root():
    repository = EXPERIMENT_DIR.parents[1]
    model_dir = EXPERIMENT_DIR / "evidence/t130_holdout_20260731/models"
    code = """
import joblib
import pandas as pd
from pathlib import Path
from aramina.m2q_model import SK_CORE4_FEATURE_COLUMNS

for path in sorted(Path('experiments/profile_symmetry_age_refinement/evidence/t130_holdout_20260731/models').glob('*.joblib')):
    artifact = joblib.load(path)
    row = {
        'profile_p_cancer_logit_average': 0.5,
        'age': 50.0,
        'age_available': 1,
        'symmetry_available': 0,
        **{column: 0.0 for column in SK_CORE4_FEATURE_COLUMNS},
    }
    score = artifact['final_model'].predict_proba(pd.DataFrame([row]))
    assert score.shape == (1, 2)
    assert artifact['lr1_model'] is not None
"""
    assert len(list(model_dir.glob("*.joblib"))) == 2
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repository / "src")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        env=env,
        check=True,
    )
