from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from aramina.experiments import polar_harmonic_ablation as ablation
from aramina.experiments.polar_basis_compression import load_config as load_core_config


def test_validation_requires_explicit_target_sensitivity(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)
    del config["evaluation"]["target_sensitivity"]
    path = tmp_path / "missing_target_sensitivity.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(
        ablation.PolarHarmonicAblationError,
        match="target_sensitivity",
    ):
        ablation.load_config(path)


def test_child_mapping_is_deterministic_and_isolates_core_output(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)
    config_path = tmp_path / "parent.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    parent = tmp_path / "parent_run"

    child = ablation._child_core_config(config, config_path, parent, n_chi=36)

    assert child["contract"] == "aramina_polar_basis_compression_v0_1"
    assert child["polar_cakes"]["n_chi"] == 36
    assert child["polar_cakes"]["cache_folder"] == str(tmp_path / "cache")
    assert child["output"]["folder"] == str(parent / "children" / "n_chi_36")
    assert child["data_version"]["pointer_path"] == "input.h5.dvc"
    assert child["evaluation"] == {
        "method": "repeated_stratified_patient_kfold",
        "folds": 2,
        "repeats": 1,
        "seed": 42,
        "target_sensitivity": 0.95,
        "threshold_policy": "training_fold_target_sensitivity",
    }
    child_path = tmp_path / "child.yaml"
    child_path.write_text(yaml.safe_dump(child), encoding="utf-8")
    assert load_core_config(child_path)["contract"] == "aramina_polar_basis_compression_v0_1"


def test_parent_run_combines_complete_children_and_writes_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _valid_config(tmp_path)
    config["polar_representation"]["n_chi_values"] = [12, 36]
    path = tmp_path / "parent.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fake_core(child_config_path: Path, *, verbose: bool) -> dict[str, object]:
        del verbose
        child = yaml.safe_load(child_config_path.read_text(encoding="utf-8"))
        root = Path(child["output"]["folder"])
        run = root / "polar_basis_compression_fake"
        run.mkdir(parents=True)
        _write_child_tables(run, n_chi=int(child["polar_cakes"]["n_chi"]))
        return {
            "run_folder": run,
            "mlflow": {"run_id": f"child-{child['polar_cakes']['n_chi']}"},
        }

    def fake_statistics(predictions: pd.DataFrame, fold_metrics: pd.DataFrame, **_: object) -> dict[str, pd.DataFrame]:
        assert set(predictions["n_chi"]) == {12, 36}
        assert set(fold_metrics["n_chi"]) == {12, 36}
        assert set(predictions["coefficients_per_channel"]) == {12}
        assert set(fold_metrics["coefficients_per_channel"]) == {12}
        frame = pd.DataFrame(
            {
                "n_chi": [36],
                "contrast": ["A0+A2 minus A0"],
                "metric": ["sensitivity"],
                "delta_mean": [0.1],
                "ci_low": [0.01],
                "ci_high": [0.2],
            }
        )
        return {
            "fingerprints": pd.DataFrame({"value": ["ok"]}),
            "paired_split_deltas": pd.DataFrame({"value": ["ok"]}),
            "bootstrap_confidence_intervals": frame.copy(),
            "holm_correction": pd.DataFrame({"value": ["ok"]}),
            "paired_contrasts": frame,
            "chi_resolution_per_split": pd.DataFrame({"value": ["ok"]}),
            "chi_resolution_summary": pd.DataFrame({"value": ["ok"]}),
            "direction_consistency": pd.DataFrame({"value": ["ok"]}),
        }

    monkeypatch.setattr(ablation, "run_polar_basis_compression_from_config", fake_core)
    monkeypatch.setattr(ablation, "analyze_polar_harmonic_runs", fake_statistics)
    monkeypatch.setattr(
        ablation,
        "_log_parent_mlflow",
        lambda **_: {"enabled": True, "run_id": "parent", "status": "FINISHED"},
    )

    result = ablation.run_polar_harmonic_ablation_from_config(path)
    run = result["run_folder"]
    assert result["child_runs"] == 2
    assert (run / "combined_predictions.csv").is_file()
    assert (run / "statistics_paired_contrasts.csv").is_file()
    assert (run / "child_configs" / "n_chi_12.yaml").is_file()
    assert (run / "run_manifest.json").is_file()
    assert json.loads((run / "run_manifest.json").read_text())["status"] == "complete"


def test_parent_fails_closed_on_partial_child_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _valid_config(tmp_path)
    config["polar_representation"]["n_chi_values"] = [36]
    path = tmp_path / "parent.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fake_core(child_config_path: Path, *, verbose: bool) -> dict[str, object]:
        del verbose
        child = yaml.safe_load(child_config_path.read_text(encoding="utf-8"))
        run = Path(child["output"]["folder"]) / "partial"
        run.mkdir(parents=True)
        return {"run_folder": run}

    monkeypatch.setattr(ablation, "run_polar_basis_compression_from_config", fake_core)

    with pytest.raises(ablation.PolarHarmonicAblationError, match="partial"):
        ablation.run_polar_harmonic_ablation_from_config(path)


def _valid_config(tmp_path: Path) -> dict[str, object]:
    return {
        "contract": "aramina_polar_harmonic_ablation_v0_1",
        "experiment": {
            "name": "polar_harmonic_test",
            "model_name": "aramina_target_breast_risk",
            "model_version": "0.2.14-beta",
            "purpose": "test",
        },
        "input": {
            "input_h5_path": "input.h5",
            "model_joblib_path": "model.joblib",
        },
        "data_version": {
            "contract": "aramina_dvc_input_v0_1",
            "system": "dvc",
            "dataset_id": "data",
            "dvc_version": "3.67.1",
            "pointer_path": "input.h5.dvc",
        },
        "cohort": {
            "selection": "all_accepted_target_measurements_and_cases",
            "accepted_target_measurements": 4,
            "target_cases": 4,
            "patient_grouping": "patient_safe",
        },
        "polar_representation": {
            "n_q": 256,
            "n_chi_values": [12, 18, 36, 72],
            "radial_q_range": [2.0, 23.0],
            "azimuthal_range": [-180.0, 180.0],
            "normalization_q_range": [6.7, 7.1],
            "harmonic_q_range": [2.1, 12.2],
            "missing_sector_policy": "weighted_fit_with_zero_weight_for_missing_sectors",
            "cache_folder": "cache",
            "force_rebuild": False,
        },
        "harmonic_ablation": {
            "mode_sets": {"A0": [0], "A0_A2": [0, 2], "A0_A2_A4": [0, 2, 4]},
            "encoder": "cubic_bspline",
            "coefficients_per_channel": [8, 12, 16],
            "primary": {"n_chi": 36, "coefficients_per_channel": 12},
        },
        "evaluation": {
            "method": "repeated_stratified_patient_kfold",
            "folds": 2,
            "repeats": 1,
            "seed": 42,
            "inner_oof_lr1_to_lr2": True,
            "threshold_policy": "training_fold_target_sensitivity",
            "target_sensitivity": 0.95,
            "compare_on_identical_measurement_and_fold_manifests": True,
            "metrics": [
                "sensitivity",
                "specificity",
                "roc_auc",
                "balanced_accuracy",
                "ppv",
                "npv",
                "confusion_matrix",
            ],
            "confidence_interval": "paired_patient_cluster_bootstrap_95",
        },
        "controls": {
            "radial_baseline": "raw100",
            "qc_modes": [1, 3],
            "confounder_fields": [
                "age",
                "thickness",
                "session",
                "date",
                "target_side",
                "measurement_count",
                "snr",
            ],
            "permutation_control": "not_executed_partial_arc_limitation",
            "session_stress_test": "unavailable_sparse_or_high_cardinality_session_labels",
        },
        "mlflow": {
            "enabled": True,
            "tracking_uri": "sqlite:///mlflow.db",
            "experiment_name": "polar_harmonic_test",
        },
        "output": {
            "folder": "runs",
            "product_artifact_changes": False,
            "report_changes": False,
        },
    }


def _write_child_tables(run: Path, *, n_chi: int) -> None:
    cases = pd.DataFrame(
        {
            "target_case_id": ["c1", "c2", "c3", "c4"],
            "patientId": ["p1", "p2", "p3", "p4"],
            "label": [0, 0, 1, 1],
        }
    )
    cases.to_csv(run / "cohort_manifest.csv", index=False)
    pd.DataFrame(
        {
            "measurement_key": ["m1", "m2", "m3", "m4"],
            "dataset_sha256": ["data"] * 4,
            "patient_id": ["p1", "p2", "p3", "p4"],
            "target_case_id": ["c1", "c2", "c3", "c4"],
            "label": [0, 0, 1, 1],
            "n_chi": [n_chi] * 4,
            "calibration_session_uid": ["s"] * 4,
            "poni_sha256": ["p"] * 4,
        }
    ).to_csv(run / "polar_cake_manifest.csv", index=False)
    manifest = pd.concat(
        [
            cases.assign(split_id=0, partition="test"),
            cases.assign(split_id=1, partition="test"),
        ],
        ignore_index=True,
    )
    manifest.to_csv(run / "fold_manifest.csv", index=False)
    predictions = []
    metrics = []
    for mode_set in ("A0", "A0+A2", "A0+A2+A4"):
        for coefficient in (8, 12, 16):
            for split_id in (0, 1):
                scores = [0.1, 0.2, 0.8, 0.9]
                predictions.extend(
                    {
                        "mode_set": mode_set,
                        "n_chi": n_chi,
                        "coefficients_per_channel": coefficient,
                        "budget": coefficient,
                        "split_id": split_id,
                        "target_case_id": row.target_case_id,
                        "patientId": row.patientId,
                        "label": row.label,
                        "p_cancer": scores[index],
                        "threshold": 0.5,
                    }
                    for index, row in enumerate(cases.itertuples(index=False))
                )
                metrics.append(
                    {
                        "mode_set": mode_set,
                        "n_chi": n_chi,
                        "coefficients_per_channel": coefficient,
                        "budget": coefficient,
                        "split_id": split_id,
                        "sensitivity": 1.0,
                        "specificity": 1.0,
                        "roc_auc": 1.0,
                        "balanced_accuracy": 1.0,
                        "ppv": 1.0,
                        "npv": 1.0,
                    }
                )
    pd.DataFrame(predictions).to_csv(run / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(run / "fold_metrics.csv", index=False)
    raw_predictions = pd.DataFrame(predictions).query("mode_set == 'A0' and coefficients_per_channel == 8")
    raw_predictions.to_csv(run / "raw100_predictions.csv", index=False)
    pd.DataFrame(metrics).query("mode_set == 'A0' and coefficients_per_channel == 8").to_csv(
        run / "raw100_fold_metrics.csv", index=False
    )
    (run / "lineage.json").write_text("{}\n", encoding="utf-8")
    (run / "run_manifest.json").write_text("{}\n", encoding="utf-8")
