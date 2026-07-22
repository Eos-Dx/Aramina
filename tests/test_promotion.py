from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import joblib
import pytest
import yaml

from aramis.promotion import promote_model_run


def _completed_run(root: Path) -> Path:
    run = root / "run"
    run.mkdir()
    artifact = {
        "kind": "aramis_training_artifact",
        "model_identity": {
            "name": "aramis_target_breast_risk",
            "version": "0.2.7-beta",
        },
        "feature_schema": {"final_model": {"feature_columns": ["age"]}},
        "evaluation": {"requested": True},
    }
    joblib.dump(artifact, run / "model.joblib")
    model_sha256 = sha256((run / "model.joblib").read_bytes()).hexdigest()
    (run / "model_description.yaml").write_text(
        yaml.safe_dump(
            {
                "output_type": "aramis_model_description",
                "model": {
                    "name": "aramis_target_breast_risk",
                    "version": "0.2.7-beta",
                    "artifact_sha256": model_sha256,
                },
                "feature_schema": artifact["feature_schema"],
            }
        ),
        encoding="utf-8",
    )
    for filename in (
        "preprocessing_config.yaml",
        "prediction_preprocessing_config.yaml",
        "training_config.yaml",
        "preprocess_and_train_config.yaml",
    ):
        (run / filename).write_text("contract: test\n", encoding="utf-8")
    (run / "evaluation.yaml").write_text(
        yaml.safe_dump(
            {
                "output_type": "aramis_evaluation_artifact",
                "model": {
                    "name": "aramis_target_breast_risk",
                    "version": "0.2.7-beta",
                    "artifact_sha256": model_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "evaluation_metrics.csv").write_text("fold_id\n0\n", encoding="utf-8")
    (run / "evaluation_predictions.csv").write_text(
        "target_case_id\ncase-1\n", encoding="utf-8"
    )
    return run


def test_promote_copies_a_reviewed_run_without_mutating_source(tmp_path: Path):
    run = _completed_run(tmp_path)
    before = (run / "model.joblib").read_bytes()

    result = promote_model_run(run, models_root=tmp_path / "models")

    destination = Path(result["model_folder"])
    assert destination.name == result["model_id"]
    assert (destination / "model.joblib").read_bytes() == before
    assert (destination / "evaluation.yaml").is_file()
    assert (run / "model.joblib").read_bytes() == before

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        promote_model_run(run, models_root=tmp_path / "models")


def test_promote_rejects_incomplete_source_run(tmp_path: Path):
    run = _completed_run(tmp_path)
    (run / "training_config.yaml").unlink()

    with pytest.raises(ValueError, match="missing files"):
        promote_model_run(run, models_root=tmp_path / "models")
