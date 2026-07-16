from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from aramis import __main__ as cli
from aramis import workflows
from aramis.training_config import (
    PRODUCT_EVALUATION,
    resolved_recipe_path,
    validate_training_config,
)


def test_cli_commands_delegate_to_product_entrypoints(monkeypatch, capsys, tmp_path: Path):
    config = tmp_path / "config.yaml"
    frame = pd.DataFrame({"value": [1, 2]})
    monkeypatch.setattr(cli, "run_preprocessing_from_config", lambda _: frame)
    monkeypatch.setattr(
        cli,
        "run_training_from_config",
        lambda _: {
            "kind": "aramis_training_artifact",
            "run_folder": str(tmp_path / "training"),
            "model_id": "model-id",
            "model_path": str(tmp_path / "model.joblib"),
        },
    )
    monkeypatch.setattr(
        cli,
        "run_preprocess_train_from_config",
        lambda _: {
            "preprocessing_dataframe": frame,
            "training_artifact": {"kind": "aramis_training_artifact"},
            "run_folder": str(tmp_path / "workflow"),
        },
    )
    monkeypatch.setattr(
        cli,
        "run_prediction_from_config",
        lambda _: {
            "external_report": {
                "patient_id": "P01",
                "target_side": "left",
                "suggested_class": "BENIGN",
                "reliability": "high",
            },
            "internal_report": {"model": {"name": "M2Q"}},
        },
    )

    assert cli.main(["preprocess", "--config", str(config)]) == 0
    assert "rows=2" in capsys.readouterr().out
    assert cli.main(["train", "--config", str(config)]) == 0
    assert "model_id=model-id" in capsys.readouterr().out
    assert cli.main(["preprocess-train", "--config", str(config)]) == 0
    assert "preprocess_columns=1" in capsys.readouterr().out
    assert cli.main(["predict", "--config", str(config)]) == 0
    assert "suggested_class=BENIGN" in capsys.readouterr().out


def test_cli_train_requires_config_when_not_listing_recipes(capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.main(["train"])
    assert "--config is required" in capsys.readouterr().err


def test_cli_verbose_preprocess_forwards_progress_flag(monkeypatch, tmp_path: Path):
    config = tmp_path / "config.yaml"
    received: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_preprocessing_from_config",
        lambda _, **kwargs: received.update(kwargs) or pd.DataFrame(),
    )

    assert cli.main(["preprocess", "--config", str(config), "--verbose"]) == 0
    assert received == {"verbose": True}


def test_workflow_passes_preprocessing_dataframe_directly_to_training(
    monkeypatch,
    tmp_path: Path,
):
    config_path = tmp_path / "preprocess_train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "contract": workflows.PREPROCESS_TRAIN_CONTRACT,
                "preprocess_train": {
                    "name": "product",
                    "created_by": "test",
                    "output_folder": str(tmp_path / "runs"),
                },
                "preprocessing_config_path": str(tmp_path / "preprocess.yaml"),
                "training_config_path": str(tmp_path / "train.yaml"),
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame({"patientId": ["P01"]})
    preprocessing_artifact = {
        "dataframe": frame,
        "preprocessing_config_yaml": "pipeline: {}\n",
        "metadata": {"input_h5_sha256": "abc"},
    }
    received: dict[str, object] = {}
    monkeypatch.setattr(
        workflows,
        "run_preprocessing_artifact_from_config",
        lambda config, output_joblib_path: preprocessing_artifact,
    )

    def train_stub(config, **kwargs):
        received.update(kwargs)
        return {"kind": "aramis_training_artifact"}

    monkeypatch.setattr(workflows, "run_training_from_config", train_stub)

    result = workflows.run_preprocess_train_from_config(config_path)

    assert received["dataframe"] is frame
    assert received["preprocessing_artifact"] is preprocessing_artifact
    assert Path(received["dataframe_joblib_path"]).name == "dataframe.joblib"
    assert received["preprocess_train_config_yaml"] == config_path.read_text(encoding="utf-8")
    assert result["preprocessing_dataframe"] is frame
    summary = json.loads(
        (Path(result["run_folder"]) / "preprocessing" / "cohort_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["rows"] == 1
    assert summary["patients"] == 1
    assert summary["input_h5_sha256"] == "abc"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "Missing preprocess-train fields"),
        ({"contract": "wrong", "preprocess_train": {}}, "Missing preprocess-train fields"),
    ],
)
def test_workflow_contract_rejects_incomplete_config(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
):
    config_path = tmp_path / "preprocess_train.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        workflows.run_preprocess_train_from_config(config_path)


def test_training_config_rejects_unknown_and_resolves_packaged_path(tmp_path: Path):
    config = {
        "contract": "aramis_training_config_v0_1",
        "training": {
            "name": "test",
            "version": "0.1-beta",
            "created_by": "test",
            "clinical_stage": "research draft",
            "intended_use": "test",
        },
        "run": {"evaluation": True, "train_on_all": False},
        "input": {"dataframe_joblib_path": "input.joblib"},
        "output": {"folder": "out"},
        "model": {"recipe": "m2q_gated_target_case_v0_1"},
        "evaluation": {**PRODUCT_EVALUATION, "unexpected": True},
    }
    with pytest.raises(ValueError, match="Unknown evaluation fields"):
        validate_training_config(config, tmp_path / "train.yaml")

    registry = tmp_path / "config" / "model_recipes.yaml"
    registry.parent.mkdir()
    assert resolved_recipe_path("missing.yaml", registry) == registry.parents[2] / "missing.yaml"
