from __future__ import annotations

import pytest

from aramina import runtime_identity, workflows
from aramina.training_artifacts import _distribution_provenance


def test_embedded_source_commits_are_used(monkeypatch):
    monkeypatch.setenv("ARAMINA_GIT_SHA", "a" * 40)
    monkeypatch.setenv("XRD_PREPROCESSING_GIT_SHA", "b" * 40)

    assert runtime_identity.aramina_git_sha() == "a" * 40
    assert runtime_identity.xrd_preprocessing_git_sha() == "b" * 40


def test_embedded_source_commit_requires_full_sha(monkeypatch):
    monkeypatch.setenv("ARAMINA_GIT_SHA", "abc123")

    with pytest.raises(ValueError, match="full 40-character Git SHA"):
        runtime_identity.aramina_git_sha()


def test_mlflow_provenance_fails_before_product_run(monkeypatch):
    monkeypatch.setattr(workflows, "aramina_git_sha", lambda: "unavailable")
    monkeypatch.setattr(
        workflows,
        "xrd_preprocessing_git_sha",
        lambda: "b" * 40,
    )

    with pytest.raises(ValueError, match="Aramina"):
        workflows._require_product_source_provenance({"enabled": True})


def test_xrd_source_override_keeps_one_consistent_provenance(monkeypatch):
    git_sha = "c" * 40
    monkeypatch.setenv("XRD_PREPROCESSING_GIT_SHA", git_sha)
    monkeypatch.setenv("XRD_PREPROCESSING_VERSION", "0.1.10b0")
    monkeypatch.setenv(
        "XRD_PREPROCESSING_URL",
        "https://github.com/Eos-Dx/XRD-preprocessing.git",
    )

    assert _distribution_provenance("xrd-preprocessing") == {
        "version": "0.1.10b0",
        "git_commit": git_sha,
        "url": "https://github.com/Eos-Dx/XRD-preprocessing.git",
        "requested_revision": git_sha,
    }
