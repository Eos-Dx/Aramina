from __future__ import annotations

from pathlib import Path

import pytest

from aramina.mlflow_tracking import MlflowRun, MlflowTrackingError

mlflow = pytest.importorskip("mlflow", minversion="3.0")
MlflowClient = mlflow.MlflowClient


def _tracking_uri(tmp_path: Path) -> str:
    store = tmp_path / "mlruns"
    store.mkdir()
    return store.as_uri()


def _artifact_paths(client: MlflowClient, run_id: str, path: str = "") -> set[str]:
    paths: set[str] = set()
    for artifact in client.list_artifacts(run_id, path or None):
        if artifact.is_dir:
            paths.update(_artifact_paths(client, run_id, artifact.path))
        else:
            paths.add(artifact.path)
    return paths


def test_finished_run_logs_flattened_values_metrics_and_required_artifacts(
    tmp_path: Path,
):
    tracking_uri = _tracking_uri(tmp_path)
    artifacts = tmp_path / "training-output"
    (artifacts / "metadata").mkdir(parents=True)
    (artifacts / "model.joblib").write_bytes(b"model")
    (artifacts / "metadata" / "metrics.json").write_text(
        '{"roc_auc": 0.71}', encoding="utf-8"
    )

    with MlflowRun(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name="aramina-product",
        run_name="product-training",
        params={"profile": {"components": 30}, "evaluation": {"enabled": True}},
        tags={"product": "aramina", "lineage": {"stage": "research-draft"}},
    ) as run:
        assert run.run_id is not None
        assert run.status == "RUNNING"
        run.log_params(
            {"dataset": {"fingerprint": "sha256:abc", "target_cases": 171}}
        )
        run.set_tags(
            {"lineage": {"model_git_sha": "deadbeef", "complete": True}}
        )
        run.log_metrics(
            {"roc_auc": 0.71, "threshold": {"sensitivity": 0.95}}, step=2
        )
        run.log_artifact_directory(
            artifacts,
            required_files=("model.joblib", "metadata/metrics.json"),
            artifact_path="training",
        )

    assert run.status == "FINISHED"
    client = MlflowClient(tracking_uri=tracking_uri)
    stored = client.get_run(run.run_id)
    assert stored.info.status == "FINISHED"
    assert stored.data.params == {
        "evaluation.enabled": "true",
        "profile.components": "30",
        "dataset.fingerprint": "sha256:abc",
        "dataset.target_cases": "171",
    }
    assert stored.data.tags["product"] == "aramina"
    assert stored.data.tags["lineage.stage"] == "research-draft"
    assert stored.data.tags["lineage.model_git_sha"] == "deadbeef"
    assert stored.data.tags["lineage.complete"] == "true"
    assert stored.data.tags["mlflow.runName"] == "product-training"
    assert stored.data.metrics == {
        "roc_auc": pytest.approx(0.71),
        "threshold.sensitivity": pytest.approx(0.95),
    }
    assert _artifact_paths(client, run.run_id) == {
        "training/metadata/metrics.json",
        "training/model.joblib",
    }


def test_body_failure_records_failed_run_status(tmp_path: Path):
    tracking_uri = _tracking_uri(tmp_path)

    with pytest.raises(RuntimeError, match="training failed"):
        with MlflowRun(
            enabled=True,
            tracking_uri=tracking_uri,
            experiment_name="aramina-product",
        ) as run:
            raise RuntimeError("training failed")

    assert run.status == "FAILED"
    stored = MlflowClient(tracking_uri=tracking_uri).get_run(run.run_id)
    assert stored.info.status == "FAILED"


def test_enabled_logging_failure_is_closed_and_records_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    tracking_uri = _tracking_uri(tmp_path)

    with pytest.raises(MlflowTrackingError, match="metric logging failed"):
        with MlflowRun(
            enabled=True,
            tracking_uri=tracking_uri,
            experiment_name="aramina-product",
        ) as run:
            def fail_log_metric(*args, **kwargs):
                del args, kwargs
                raise OSError("tracking backend unavailable")

            monkeypatch.setattr(run._client, "log_metric", fail_log_metric)
            run.log_metrics({"roc_auc": 0.7})

    assert run.status == "FAILED"
    stored = MlflowClient(tracking_uri=tracking_uri).get_run(run.run_id)
    assert stored.info.status == "FAILED"


@pytest.mark.parametrize(
    ("method_name", "client_method", "payload", "error"),
    [
        (
            "log_params",
            "log_param",
            {"dataset": {"fingerprint": "sha256:abc"}},
            "parameter logging failed",
        ),
        (
            "set_tags",
            "set_tag",
            {"lineage": {"model_git_sha": "deadbeef"}},
            "tag logging failed",
        ),
    ],
)
def test_post_start_metadata_logging_failures_close_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    client_method: str,
    payload: dict[str, object],
    error: str,
):
    tracking_uri = _tracking_uri(tmp_path)

    with pytest.raises(MlflowTrackingError, match=error):
        with MlflowRun(
            enabled=True,
            tracking_uri=tracking_uri,
            experiment_name="aramina-product",
        ) as run:
            def fail_metadata_logging(*args, **kwargs):
                del args, kwargs
                raise OSError("tracking backend unavailable")

            monkeypatch.setattr(run._client, client_method, fail_metadata_logging)
            getattr(run, method_name)(payload)

    assert run.status == "FAILED"
    stored = MlflowClient(tracking_uri=tracking_uri).get_run(run.run_id)
    assert stored.info.status == "FAILED"
