"""Fixed feature schema and warnings for the selected product model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .target_breast_model import SK_CORE4_FEATURE_COLUMNS


def target_breast_model_input_columns() -> list[str]:
    """Return fixed final-model input columns in report order."""
    return [
        "profile_p_cancer_logit_average",
        "age",
        "age_available",
        *SK_CORE4_FEATURE_COLUMNS,
        "symmetry_available",
    ]


def target_breast_feature_schema() -> dict[str, Any]:
    """Return the serialised feature contract carried by the model artifact."""
    return {
        "final_model": {
            "feature_columns": target_breast_model_input_columns(),
            "learned_feature_columns": [
                "profile_p_cancer_logit_average",
                "age",
                "age_available",
                *SK_CORE4_FEATURE_COLUMNS,
            ],
            "symmetry_gate": "symmetry_available",
            "symmetry_policy": (
                "single_model_gated_optional_refinement_requires_2_valid_measurements_"
                "per_breast_and_finite_core4"
            ),
            "reliability_fields": [
                "profile_p_cancer_n_measurements",
                "target_measurements",
                "contralateral_measurements",
                "symmetry_available",
            ],
            "unit": "target_breast_case",
            "label": "BENIGN vs CANCER decision-support class",
        }
    }


def target_breast_warnings(feature_table: pd.DataFrame) -> list[str]:
    """Create research-draft and measurement-sufficiency warnings."""
    warnings = [
        "Research-draft decision support only; requires radiologist review.",
        "Not for autonomous diagnosis.",
        "This model includes age; its contribution must be reviewed separately.",
        "Measurement sufficiency is reported separately; reliability fields are not model predictors.",
    ]
    unavailable = int((feature_table["symmetry_available"] == 0).sum())
    if unavailable:
        warnings.append(
            f"{unavailable} target-breast cases have unavailable paired-breast symmetry features."
        )
    low_target = int((feature_table["target_measurements"] < 2).sum())
    if low_target:
        warnings.append(
            f"{low_target} target-breast cases have fewer than 2 valid target-breast measurements."
        )
    low_contralateral = int((feature_table["contralateral_measurements"] < 2).sum())
    if low_contralateral:
        warnings.append(
            f"{low_contralateral} target-breast cases have fewer than 2 valid contralateral-breast measurements."
        )
    return warnings
