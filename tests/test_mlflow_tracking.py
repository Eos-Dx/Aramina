from __future__ import annotations

import json
import sys
import types

import joblib
import pandas as pd

from aramis import build_run_name, dataset_fingerprint, log_product_run


def test_dataset_fingerprint_changes_when_values_change():
    left = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    right = pd.DataFrame({"a": [1, 3], "b": ["x", "y"]})

    assert dataset_fingerprint(left) != dataset_fingerprint(right)


def test_dataset_fingerprint_is_stable_for_same_frame():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    assert dataset_fingerprint(df) == dataset_fingerprint(df.copy())


def test_build_run_name_contains_product_prefix():
    assert build_run_name("Aramis").startswith("aramis_draft_")


def test_log_product_run_dry_run_writes_required_artifacts(tmp_path):
    df = pd.DataFrame({"patientId": ["P1"], "p_cancer": [0.2]})
    model = {"kind": "dummy"}

    result = log_product_run(
        tracking_uri="file:///tmp/mlruns",
        experiment_name="Aramis",
        run_name="unit",
        product_name="Aramis",
        preprocessing_config={"raw_data": {"source": "npy"}},
        product_filter_rules={"cohort": "biopsy_patients"},
        dataset_df=df,
        model=model,
        metrics={"roc_auc": 0.5},
        params={"pipeline_version": "test"},
        artifacts_dir=tmp_path,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["mlflow_run_id"] is None
    assert result["dataset_rows"] == 1
    assert (tmp_path / "preprocessing_config.json").exists()
    assert (tmp_path / "product_filter_rules.json").exists()
    assert (tmp_path / "feature_schema.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "params.json").exists()
    assert joblib.load(tmp_path / "model.joblib") == model
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"roc_auc": 0.5}


def test_log_product_run_non_dry_run_uses_mlflow_api(monkeypatch, tmp_path):
    calls: list[tuple[str, object]] = []

    class _Run:
        info = types.SimpleNamespace(run_id="run-1")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_mlflow = types.SimpleNamespace(
        sklearn=types.SimpleNamespace(
            log_model=lambda model, artifact_path: calls.append(
                ("log_model", artifact_path)
            )
        ),
        set_tracking_uri=lambda uri: calls.append(("tracking_uri", uri)),
        set_experiment=lambda name: calls.append(("experiment", name)),
        start_run=lambda run_name: _Run(),
        set_tag=lambda key, value: calls.append(("tag", key)),
        log_param=lambda key, value: calls.append(("param", key)),
        log_metric=lambda key, value: calls.append(("metric", key)),
        log_artifact=lambda path: calls.append(("artifact", path)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setenv("ARAMIS_LOG_MLFLOW_MODEL", "1")

    result = log_product_run(
        tracking_uri="file:///tmp/mlruns",
        experiment_name="Aramis",
        run_name="unit",
        product_name="Aramis",
        preprocessing_config={},
        product_filter_rules={},
        dataset_df=pd.DataFrame({"a": [1]}),
        model={"kind": "dummy"},
        metrics={"roc_auc": 0.5},
        params={"n_splits": 2, "complex": {"ignored": True}},
        artifacts_dir=tmp_path,
        dry_run=False,
    )

    assert result["mlflow_run_id"] == "run-1"
    assert ("tracking_uri", "file:///tmp/mlruns") in calls
    assert ("experiment", "Aramis") in calls
    assert ("metric", "roc_auc") in calls
    assert ("param", "n_splits") in calls
    assert ("log_model", "sklearn_model") in calls
    assert sum(name == "artifact" for name, _ in calls) == 7
