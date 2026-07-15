#!/usr/bin/env python3
"""Compare two Aramis final-fit model artifacts for reproducible training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np


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
    print("Executable parameters, threshold, H5 checksum, recipe, and YAML checksums match.")
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
        "model_identity.recipe",
        reference["model_identity"]["recipe"],
        candidate["model_identity"]["recipe"],
    )
    _compare_value(
        differences,
        "source_h5.sha256",
        reference["reproducibility"]["source_h5"]["sha256"],
        candidate["reproducibility"]["source_h5"]["sha256"],
    )
    _compare_value(
        differences,
        "reproduction_mode",
        reference["reproducibility"]["reproduction_mode"],
        candidate["reproducibility"]["reproduction_mode"],
    )
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
    _compare_value(
        differences,
        "reproducibility.checksums",
        reference["reproducibility"]["checksums"],
        candidate["reproducibility"]["checksums"],
    )
    _compare_model(differences, reference["models"]["M2Q"], candidate["models"]["M2Q"])
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


def _compare_value(
    differences: list[str], name: str, reference: Any, candidate: Any
) -> None:
    if reference != candidate:
        differences.append(f"{name} differs: {reference!r} != {candidate!r}")


def _compare_array(
    differences: list[str], name: str, reference: Any, candidate: Any
) -> None:
    if not np.allclose(reference, candidate, rtol=1e-12, atol=1e-12, equal_nan=True):
        differences.append(f"{name} differs")


if __name__ == "__main__":
    raise SystemExit(main())
