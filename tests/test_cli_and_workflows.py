from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from aramis import __main__ as cli
from aramis import workflows
from aramis.training_config import (
    PRODUCT_MODEL_NAME,
    PRODUCT_EVALUATION,
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
                "risk_probability": 0.12345,
                "target_class_risk_level": "low",
                "reliability": "high",
            },
            "internal_report": {"model": {"name": PRODUCT_MODEL_NAME}},
        },
    )

    assert cli.main(["preprocess", "--config", str(config)]) == 0
    assert "rows=2" in capsys.readouterr().out
    assert cli.main(["train", "--config", str(config)]) == 0
    assert "model_id=model-id" in capsys.readouterr().out
    assert cli.main(["preprocess-train", "--config", str(config)]) == 0
    assert "preprocess_columns=1" in capsys.readouterr().out
    assert cli.main(["predict", "--config", str(config)]) == 0
    assert "risk_probability=0.12345" in capsys.readouterr().out


def test_cli_train_requires_config_when_not_listing_models(capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.main(["train"])
    assert "--config is required" in capsys.readouterr().err


def test_cli_promote_delegates_to_product_promotion(monkeypatch, capsys, tmp_path: Path):
    promoted = {
        "model_id": "model-id",
        "artifact_sha256": "sha256",
        "model_folder": tmp_path / "models" / "model-id",
    }
    monkeypatch.setattr(cli, "promote_model_run", lambda *_args, **_kwargs: promoted)

    assert cli.main(["promote", "--run-folder", str(tmp_path / "run")]) == 0
    output = capsys.readouterr().out
    assert "model_id=model-id" in output
    assert "artifact_sha256=sha256" in output


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
                "preprocessing_and_training": {
                    "name": "product",
                    "run_author": "test",
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


def test_workflow_resolves_root_relative_paths_from_external_config_tree(
    monkeypatch,
    tmp_path: Path,
):
    config_path = tmp_path / "Aramis" / "config" / "preprocess_train" / "product.yaml"
    config_path.parent.mkdir(parents=True)
    (config_path.parents[2] / "pyproject.toml").touch()
    config_path.write_text(
        yaml.safe_dump(
            {
                "contract": workflows.PREPROCESS_TRAIN_CONTRACT,
                "preprocessing_and_training": {
                    "name": "product",
                    "run_author": "test",
                    "output_folder": "./examples/outputs/preprocess_train",
                },
                "preprocessing_config_path": "./config/preprocessing/preprocess.yaml",
                "training_config_path": "./config/training/train.yaml",
            }
        ),
        encoding="utf-8",
    )
    received: dict[str, object] = {}
    frame = pd.DataFrame({"patientId": ["P01"]})
    monkeypatch.setattr(
        workflows,
        "run_preprocessing_artifact_from_config",
        lambda config, output_joblib_path: received.update(preprocess=config)
        or {"dataframe": frame, "metadata": {}},
    )
    monkeypatch.setattr(
        workflows,
        "run_training_from_config",
        lambda config, **kwargs: received.update(training=config) or {},
    )

    result = workflows.run_preprocess_train_from_config(config_path)

    project_root = config_path.parents[2]
    assert received["preprocess"] == project_root / "config/preprocessing/preprocess.yaml"
    assert received["training"] == project_root / "config/training/train.yaml"
    assert result["run_folder"].is_relative_to(project_root / "examples/outputs")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "Missing preprocessing-and-training fields"),
        ({"contract": "wrong", "preprocessing_and_training": {}}, "Missing preprocessing-and-training fields"),
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


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("run_author", 42, "preprocessing_and_training.run_author must be a string"),
        ("output_folder", " ", "preprocessing_and_training.output_folder must not be empty"),
        ("preprocessing_config_path", "", "preprocessing_config_path must not be empty"),
    ],
)
def test_workflow_contract_rejects_invalid_string_values(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
):
    payload = {
        "contract": workflows.PREPROCESS_TRAIN_CONTRACT,
        "preprocessing_and_training": {
            "name": "product",
            "run_author": "test",
            "output_folder": "outputs",
        },
        "preprocessing_config_path": "config/preprocessing/input.yaml",
        "training_config_path": "config/training/train.yaml",
    }
    target = (
        payload["preprocessing_and_training"]
        if field in payload["preprocessing_and_training"]
        else payload
    )
    target[field] = value
    config_path = tmp_path / "preprocess_train.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=error):
        workflows._load_preprocess_train_config(config_path)


def test_training_config_rejects_unknown_and_resolves_packaged_path(tmp_path: Path):
    config = {
        "contract": "aramis_training_config_v0_3",
        "model": {
            "name": "test",
            "version": "0.1-beta",
            "model_author": "test",
            "clinical_stage": "research draft",
            "intended_use": "test",
        },
        "run": {"evaluation": True, "train_on_all": False},
        "input": {"dataframe_joblib_path": "input.joblib"},
        "output": {"folder": "out"},
        "evaluation": {**PRODUCT_EVALUATION, "unexpected": True},
    }
    config["model"]["name"] = PRODUCT_MODEL_NAME
    with pytest.raises(ValueError, match="Unknown evaluation fields"):
        validate_training_config(config, tmp_path / "train.yaml")
