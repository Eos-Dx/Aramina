#!/usr/bin/env python3
"""Compare two Aramis final-fit model artifacts for reproducible training."""

from __future__ import annotations

import argparse
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from aramis.training_config import PRODUCT_MODEL_NAME


def main() -> int:
    """Compare executable model values and immutable provenance fields."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    reference = joblib.load(args.reference)
    candidate = joblib.load(args.candidate)
    differences = _differences(reference, candidate)
    if differences:
        print("MODEL COMPARISON: FAILED")
        print("\n".join(f"- {message}" for message in differences))
        return 1
    print("MODEL COMPARISON: PASSED")
    print("Executable parameters, threshold, H5 checksum, model identity, and evaluation match.")
    return 0


def _differences(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    differences: list[str] = []
    for label, artifact in (("reference", reference), ("candidate", candidate)):
        if "reproducibility" not in artifact:
            differences.append(
                f"{label} artifact has no reproducibility record; retrain it with Aramis 0.2.7-beta or later."
            )
    if differences:
        return differences
    _compare_value(
        differences,
        "model_identity",
        _canonical_model_identity(reference["model_identity"]),
        _canonical_model_identity(candidate["model_identity"]),
    )
    _compare_value(
        differences,
        "source_h5.sha256",
        reference["reproducibility"]["source_h5"]["sha256"],
        candidate["reproducibility"]["source_h5"]["sha256"],
    )
    # A reference artifact may be trained from its preserved preprocessing
    # joblib while the Docker bundle deliberately reruns raw-H5 preprocessing.
    # These modes are different provenance routes to the same accepted cohort,
    # so executable parameters and evaluation, rather than the route label,
    # establish reproducibility here.
    _compare_value(
        differences,
        "evaluation.protocol",
        reference["evaluation"]["protocol"],
        candidate["evaluation"]["protocol"],
    )
    _compare_value(
        differences,
        "evaluation.summary",
        reference["evaluation"]["summary"],
        candidate["evaluation"]["summary"],
    )
    # Bundle configs are external by design. Their workflow output directory and
    # input-H5 path differ from the in-repository training reference, despite
    # resolving to the same H5 checksum and transformer settings. The full YAML
    # and its checksums remain in each artifact for traceability; model equality
    # is established through the executable parameters and evaluation instead.
    _compare_model(
        differences,
        reference["models"][PRODUCT_MODEL_NAME],
        candidate["models"][PRODUCT_MODEL_NAME],
    )
    return differences


def _compare_model(
    differences: list[str], reference: dict[str, Any], candidate: dict[str, Any]
) -> None:
    _compare_value(differences, "feature_columns", reference["feature_columns"], candidate["feature_columns"])
    _compare_value(differences, "thresholds", reference["thresholds"], candidate["thresholds"])

    for step_name in ("scaler", "logreg"):
        reference_step = reference["lr1_model"].named_steps[step_name]
        candidate_step = candidate["lr1_model"].named_steps[step_name]
        for attribute in ("mean_", "scale_", "coef_", "intercept_"):
            if hasattr(reference_step, attribute):
                _compare_array(
                    differences,
                    f"lr1.{step_name}.{attribute}",
                    getattr(reference_step, attribute),
                    getattr(candidate_step, attribute),
                )

    for attribute in (
        "base_fill_values_",
        "symmetry_means_",
        "symmetry_scales_",
    ):
        _compare_array(
            differences,
            f"lr2.{attribute}",
            getattr(reference["final_model"], attribute),
            getattr(candidate["final_model"], attribute),
        )
    _compare_array(
        differences,
        "lr2.logreg_.coef_",
        reference["final_model"].logreg_.coef_,
        candidate["final_model"].logreg_.coef_,
    )
    _compare_array(
        differences,
        "lr2.logreg_.intercept_",
        reference["final_model"].logreg_.intercept_,
        candidate["final_model"].logreg_.intercept_,
    )


def _canonical_model_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Normalize an archived author-key spelling before reproducibility comparison."""
    canonical = dict(identity)
    if "created_by" in canonical and "model_author" not in canonical:
        canonical["model_author"] = canonical.pop("created_by")
    return canonical


def _compare_value(
    differences: list[str], name: str, reference: Any, candidate: Any
) -> None:
    if not _values_match(reference, candidate):
        differences.append(f"{name} differs: {reference!r} != {candidate!r}")


def _values_match(reference: Any, candidate: Any) -> bool:
    """Compare nested artifact metadata, allowing harmless float round-off."""
    if isinstance(reference, bool) or isinstance(candidate, bool):
        return reference is candidate
    if isinstance(reference, Real) and isinstance(candidate, Real):
        return bool(np.isclose(reference, candidate, rtol=1e-12, atol=1e-12))
    if isinstance(reference, dict) and isinstance(candidate, dict):
        return (
            reference.keys() == candidate.keys()
            and all(_values_match(reference[key], candidate[key]) for key in reference)
        )
    if isinstance(reference, (list, tuple)) and isinstance(candidate, (list, tuple)):
        return len(reference) == len(candidate) and all(
            _values_match(left, right) for left, right in zip(reference, candidate)
        )
    return reference == candidate


def _compare_array(
    differences: list[str], name: str, reference: Any, candidate: Any
) -> None:
    if not np.allclose(reference, candidate, rtol=1e-12, atol=1e-12, equal_nan=True):
        differences.append(f"{name} differs")


if __name__ == "__main__":
    raise SystemExit(main())
