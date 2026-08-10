from __future__ import annotations

from pathlib import Path

import joblib


ROOT = Path(__file__).parents[1]
RETRAINED_CANDIDATE_MODEL = (
    ROOT
    / "models"
    / "aramina_target_breast_risk_0_2_13-beta_f5e4a04cad11"
    / "model.joblib"
)


def test_retrained_candidate_records_pinned_xrd_release_lineage():
    artifact = joblib.load(RETRAINED_CANDIDATE_MODEL)

    assert artifact["model_identity"]["version"] == "0.2.13-beta"
    assert artifact["reproducibility"]["source_code"]["xrd_preprocessing"] == {
        "git_commit": "88dcaa277c5a0d4be2ab637bc5827a14bd106bea",
        "requested_revision": "88dcaa277c5a0d4be2ab637bc5827a14bd106bea",
        "url": "https://github.com/Eos-Dx/XRD-preprocessing.git",
        "version": "0.1.9b0",
    }
