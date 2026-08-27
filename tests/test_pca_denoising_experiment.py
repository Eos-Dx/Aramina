from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from aramina.experiments.pca_denoising import (
    _validate_experiment_config,
    evaluate_pca_denoising_methods,
    run_pca_denoising_experiment,
    score_pca_denoising_models_on_held_out,
)


def _frame() -> pd.DataFrame:
    q = np.linspace(2.0, 23.0, 30)
    rows = []
    for patient_index in range(20):
        cancer = patient_index % 2 == 0
        for side in ("Left", "Right"):
            biopsy = side == "Left"
            label = "CANCER" if cancer and biopsy else "BENIGN"
            for measurement_index in range(2):
                signal = np.sin(q / 3.0) + 0.01 * measurement_index
                if label == "CANCER":
                    signal = signal + np.exp(-0.5 * ((q - 8.0) / 0.5) ** 2)
                rows.append(
                    {
                        "patientId": f"P{patient_index:02d}",
                        "specimenId": f"P{patient_index:02d}_{side}",
                        "measurementId": (
                            f"P{patient_index:02d}_{side}_{measurement_index}"
                        ),
                        "side": side,
                        "product_status_group": label,
                        "radial_profile_data": signal.copy(),
                        "q_range": q.copy(),
                        "age": 40 + patient_index,
                        "biopsy": biopsy,
                    }
                )
    return pd.DataFrame(rows)


def test_denoising_comparison_reuses_patient_safe_splits(tmp_path: Path):
    dataframe = _frame()
    result = evaluate_pca_denoising_methods(
        dataframe,
        methods=[
            {"name": "raw", "type": "raw", "params": {}},
            {
                "name": "smooth",
                "type": "smoothed_pca",
                "params": {"n_components": 5, "smoothing": 20.0},
            },
            {
                "name": "sparse",
                "type": "sparse_pca",
                "params": {
                    "n_components": 5,
                    "alpha": 0.2,
                    "max_iter": 100,
                    "tol": 1.0e-5,
                },
            },
        ],
        evaluation={
            "folds": 2,
            "repeats": 1,
            "random_seed": 7,
            "bootstrap_samples": 0,
        },
        train_on_all=True,
        artifact_folder=tmp_path / "models",
    )

    assignments = result["split_assignments"]
    for _split_id, split in assignments.groupby("split_id"):
        train = set(split.loc[split["partition"] == "train", "patientId"])
        test = set(split.loc[split["partition"] == "test", "patientId"])
        assert not train.intersection(test)
    assert set(result["summary"]["model_name"]) == {"raw", "smooth", "sparse"}
    assert set(result["train_on_all_metrics"]["model_name"]) == {
        "raw",
        "smooth",
        "sparse",
    }
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 3
    np.testing.assert_array_equal(
        dataframe["radial_profile_data"].iloc[0],
        _frame()["radial_profile_data"].iloc[0],
    )

    held_out_path = tmp_path / "held_out.joblib"
    joblib.dump(
        {
            "dataframe": dataframe,
            "case_manifest": [
                {
                    "patient_id": f"P{patient_index:02d}",
                    "target_side": "left",
                    "reference_label": "CANCER"
                    if patient_index % 2 == 0
                    else "BENIGN",
                }
                for patient_index in range(20)
            ],
        },
        held_out_path,
    )
    held_out = score_pca_denoising_models_on_held_out(
        held_out_path,
        methods=[
            {"name": "raw", "type": "raw", "params": {}},
            {
                "name": "smooth",
                "type": "smoothed_pca",
                "params": {"n_components": 5, "smoothing": 20.0},
            },
            {
                "name": "sparse",
                "type": "sparse_pca",
                "params": {
                    "n_components": 5,
                    "alpha": 0.2,
                    "max_iter": 100,
                    "tol": 1.0e-5,
                },
            },
        ],
        artifact_folder=tmp_path / "models",
    )
    assert len(held_out["held_out_predictions"]) == 60
    assert set(held_out["held_out_metrics"]["model_name"]) == {
        "raw",
        "smooth",
        "sparse",
    }


def test_denoising_comparison_uses_identical_method_split_ids():
    result = evaluate_pca_denoising_methods(
        _frame(),
        methods=[
            {"name": "raw", "type": "raw", "params": {}},
            {
                "name": "smooth",
                "type": "smoothed_pca",
                "params": {"n_components": 5, "smoothing": 5.0},
            },
        ],
        evaluation={
            "folds": 2,
            "repeats": 2,
            "random_seed": 11,
            "bootstrap_samples": 0,
        },
        train_on_all=False,
    )

    split_sets = {
        name: set(group["split_id"])
        for name, group in result["split_metrics"].groupby("model_name")
    }
    assert split_sets["raw"] == split_sets["smooth"] == {0, 1, 2, 3}


def test_experiment_config_requires_final_fit_and_safe_method_names():
    config = {
        "contract": "aramina_pca_denoising_experiment_v0_1",
        "model": {
            "name": "aramina_target_breast_risk",
            "version": "0.2.14-beta",
        },
        "run": {"train_on_all": False},
        "input": {
            "dataframe_joblib_path": "input.joblib",
            "held_out_artifact_path": "held_out.joblib",
            "dataset_id": "dataset",
            "dataset_fingerprint": "fingerprint",
            "source_h5_sha256": "sha256",
        },
        "output": {"folder": "outputs"},
        "evaluation": {
            "method": "repeated_stratified_kfold",
            "folds": 2,
            "repeats": 1,
            "random_seed": 42,
            "bootstrap_samples": 0,
        },
        "methods": [{"name": "../unsafe", "type": "raw", "params": {}}],
    }

    with pytest.raises(ValueError, match="train_on_all must be true"):
        _validate_experiment_config(config)
    config["run"]["train_on_all"] = True
    with pytest.raises(ValueError, match="lowercase letters"):
        _validate_experiment_config(config)


def test_experiment_yaml_writes_complete_run(tmp_path: Path):
    dataframe = _frame()
    input_path = tmp_path / "input.joblib"
    held_out_path = tmp_path / "held_out.joblib"
    joblib.dump(dataframe, input_path)
    joblib.dump(
        {
            "dataframe": dataframe,
            "case_manifest": [
                {
                    "patient_id": f"P{patient_index:02d}",
                    "target_side": "left",
                    "reference_label": "CANCER"
                    if patient_index % 2 == 0
                    else "BENIGN",
                }
                for patient_index in range(20)
            ],
        },
        held_out_path,
    )
    config = {
        "contract": "aramina_pca_denoising_experiment_v0_1",
        "model": {
            "name": "aramina_target_breast_risk",
            "version": "0.2.14-beta",
        },
        "run": {"train_on_all": True},
        "input": {
            "dataframe_joblib_path": str(input_path),
            "held_out_artifact_path": str(held_out_path),
            "dataset_id": "synthetic",
            "dataset_fingerprint": "synthetic-fingerprint",
            "source_h5_sha256": "synthetic-sha256",
        },
        "output": {"folder": str(tmp_path / "runs")},
        "evaluation": {
            "method": "repeated_stratified_kfold",
            "folds": 2,
            "repeats": 1,
            "random_seed": 42,
            "bootstrap_samples": 0,
        },
        "methods": [
            {"name": "raw", "type": "raw", "params": {}},
            {
                "name": "smooth",
                "type": "smoothed_pca",
                "params": {"n_components": 5, "smoothing": 20.0},
            },
        ],
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_pca_denoising_experiment(config_path)

    run_folder = result["run_folder"]
    assert (run_folder / "summary.csv").is_file()
    assert (run_folder / "held_out_metrics.csv").is_file()
    assert (run_folder / "patient_split_assignments.csv").is_file()
    assert (run_folder / "run_manifest.json").is_file()
    assert result["manifest"]["dataset_fingerprint"] == "synthetic-fingerprint"
