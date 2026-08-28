from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from aramina.experiments.measurement_uncertainty import TargetRequest, _score_patient_frame
from aramina.experiments.vectorized_frozen_scorer import (
    FROZEN_THRESHOLD,
    score_frozen_aramina_0_2_15_cube,
)


ROOT = __import__("pathlib").Path(__file__).parents[1]
MODEL_PATH = (
    ROOT
    / "models/aramina_target_breast_risk_0_2_15-beta_43b2865632ea/model.joblib"
)
MODEL_NAME = "aramina_target_breast_risk"


@pytest.fixture(scope="module")
def model_artifact():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def measurement_inputs():
    q_grid = np.linspace(6.7, 23.0, 100)
    manifest = pd.DataFrame(
        {
            "patientId": ["P1"] * 6 + ["P2"] * 3,
            "specimenId": [
                "P1-L-1",
                "P1-L-2",
                "P1-L-3",
                "P1-R-1",
                "P1-R-2",
                "P1-R-3",
                "P2-L-1",
                "P2-L-2",
                "P2-L-3",
            ],
            "side": ["Left"] * 3 + ["Right"] * 3 + ["Left"] * 3,
            "age": [49] * 6 + [63] * 3,
        }
    )
    base = np.linspace(0.82, 1.18, 100)
    profiles = np.vstack(
        [
            base + 0.010 * np.sin(q_grid * index)
            for index in range(1, len(manifest) + 1)
        ]
    )
    targets = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2"],
            "target_side": ["left", "right", "left"],
        }
    )
    return manifest, q_grid, profiles, targets


def _original_score(
    profiles: np.ndarray,
    *,
    manifest: pd.DataFrame,
    q_grid: np.ndarray,
    model_info: dict,
    patient_id: str,
    target_side: str,
) -> float:
    frame = manifest.copy(deep=True)
    frame["q_range"] = [q_grid.copy() for _ in range(len(frame))]
    frame["radial_profile_data"] = [row.copy() for row in profiles]
    return _score_patient_frame(
        frame,
        model_info=model_info,
        model_name=MODEL_NAME,
        target=TargetRequest(patient_id=patient_id, target_side=target_side),
        columns={
            "profile_column": "radial_profile_data",
            "group_column": "patientId",
            "specimen_column": "specimenId",
            "side_column": "side",
            "q_column": "q_range",
            "age_column": "age",
        },
    )["p_cancer"]


def test_vectorized_scorer_matches_deterministic_frozen_route(
    model_artifact, measurement_inputs
):
    manifest, q_grid, profiles, targets = measurement_inputs
    result = score_frozen_aramina_0_2_15_cube(
        profiles[None, :, :],
        patient_manifest=manifest,
        q_grid=q_grid,
        target_manifest=targets,
        model_artifact=model_artifact,
    )
    model_info = model_artifact["models"][MODEL_NAME]
    expected = np.array(
        [
            _original_score(
                profiles,
                manifest=manifest,
                q_grid=q_grid,
                model_info=model_info,
                patient_id=row.patient_id,
                target_side=row.target_side,
            )
            for row in targets.itertuples(index=False)
        ]
    )

    np.testing.assert_allclose(result.p_cancer[0], expected, rtol=0.0, atol=1e-14)
    assert result.threshold == FROZEN_THRESHOLD
    assert result.target_case_ids == ("P1::LEFT", "P1::RIGHT", "P2::LEFT")
    assert result.symmetry_available[0].tolist() == [1, 1, 0]
    assert result.contralateral_measurements[0].tolist() == [3, 3, 0]


def test_vectorized_scorer_matches_random_draws(model_artifact, measurement_inputs):
    manifest, q_grid, profiles, targets = measurement_inputs
    rng = np.random.default_rng(813)
    cube = np.stack(
        [profiles + rng.normal(0.0, 0.006, size=profiles.shape) for _ in range(5)]
    )
    result = score_frozen_aramina_0_2_15_cube(
        cube,
        patient_manifest=manifest,
        q_grid=np.broadcast_to(q_grid, profiles.shape),
        target_manifest=targets,
        model_artifact=model_artifact,
    )
    model_info = model_artifact["models"][MODEL_NAME]
    expected = np.array(
        [
            [
                _original_score(
                    draw_profiles,
                    manifest=manifest,
                    q_grid=q_grid,
                    model_info=model_info,
                    patient_id=row.patient_id,
                    target_side=row.target_side,
                )
                for row in targets.itertuples(index=False)
            ]
            for draw_profiles in cube
        ]
    )

    np.testing.assert_allclose(result.p_cancer, expected, rtol=0.0, atol=1e-14)
