from __future__ import annotations

import pandas as pd
import pytest

from aramina.training_evaluation import (
    _split_assignment_frame,
    _validate_patient_split_assignments,
)


def test_split_assignment_frame_rejects_patient_overlap():
    with pytest.raises(RuntimeError, match="Patient leakage"):
        _split_assignment_frame(
            split_id=0,
            n_splits=5,
            train_patients={"P00", "P01"},
            test_patients={"P01", "P02"},
        )


def test_split_assignment_validation_rejects_invalid_held_out_frequency():
    assignments = pd.concat(
        [
            _split_assignment_frame(
                split_id=0,
                n_splits=2,
                train_patients={"P01"},
                test_patients={"P00"},
            ),
            _split_assignment_frame(
                split_id=1,
                n_splits=2,
                train_patients={"P00"},
                test_patients={"P01"},
            ),
        ],
        ignore_index=True,
    )
    assignments.loc[assignments["split_id"] == 1, "partition"] = ["test", "train"]

    with pytest.raises(RuntimeError, match="held out exactly once per repeat"):
        _validate_patient_split_assignments(
            assignments,
            expected_patients={"P00", "P01"},
            n_splits=2,
            n_repeats=1,
        )
