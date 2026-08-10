"""Tests for the cross-platform model-artifact comparison utility."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _comparison_module():
    source = Path(__file__).parents[1] / "scripts" / "compare_model_artifacts.py"
    spec = spec_from_file_location("compare_model_artifacts", source)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_float_round_off_is_reproducible():
    module = _comparison_module()

    assert module._values_match(
        {"threshold": 0.24451397236244887, "summary": [{"brier": 0.24761569970072841}]},
        {"threshold": 0.2445139723624505, "summary": [{"brier": 0.2476156997006145}]},
    )


def test_non_numeric_metadata_still_requires_exact_match():
    module = _comparison_module()

    assert not module._values_match({"model": "candidate-a"}, {"model": "candidate-b"})


def test_archived_created_by_identity_matches_current_model_author():
    module = _comparison_module()

    assert module._canonical_model_identity(
        {"name": "aramina", "version": "0.2.7-beta", "created_by": "Sergey"}
    ) == {
        "name": "aramina",
        "version": "0.2.7-beta",
        "model_author": "Sergey",
    }
