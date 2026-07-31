"""Research-only staged Profile -> Symmetry -> Age classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from aramina.m2q_model import SK_CORE4_FEATURE_COLUMNS


PROFILE_PROBABILITY_COLUMN = "profile_p_cancer_logit_average"
PROBABILITY_EPSILON = 1e-6


def _clip_probability(values: Any) -> np.ndarray:
    probability = np.asarray(values, dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("Profile probabilities must be finite.")
    return np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = _clip_probability(probability)
    return np.log(probability / (1.0 - probability))


def _expit(logit: np.ndarray) -> np.ndarray:
    values = np.asarray(logit, dtype=float)
    positive = values >= 0.0
    out = np.empty_like(values)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    out[~positive] = negative_exp / (1.0 + negative_exp)
    return out


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    classes, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("Staged correction fitting requires BENIGN and CANCER.")
    class_weight = {
        int(label): float(len(labels) / (len(classes) * count))
        for label, count in zip(classes, counts, strict=True)
    }
    return np.asarray([class_weight[int(label)] for label in labels], dtype=float)


class _OffsetLogisticCorrection:
    """Fit an L2-regularized additive correction to a fixed input logit."""

    def __init__(self, *, c: float) -> None:
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError("Correction C must be a positive finite value.")
        self.c = float(c)

    def fit(
        self,
        design: np.ndarray,
        y: np.ndarray,
        *,
        offset: np.ndarray,
    ) -> "_OffsetLogisticCorrection":
        matrix = np.asarray(design, dtype=float)
        labels = np.asarray(y, dtype=int)
        base_logit = np.asarray(offset, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Correction design must be a two-dimensional matrix.")
        if len(matrix) != len(labels) or len(base_logit) != len(labels):
            raise ValueError("Correction design, labels, and offset must align.")
        if not np.isfinite(matrix).all() or not np.isfinite(base_logit).all():
            raise ValueError("Correction design and offset must be finite.")
        sample_weight = _balanced_sample_weight(labels)
        weight_total = float(sample_weight.sum())

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            linear = base_logit + matrix @ coefficients
            loss = np.logaddexp(0.0, linear) - labels * linear
            probability = _expit(linear)
            penalty = 0.5 * float(np.dot(coefficients[1:], coefficients[1:]))
            value = float(np.dot(sample_weight, loss) / weight_total)
            value += penalty / (self.c * weight_total)
            gradient = matrix.T @ (sample_weight * (probability - labels))
            gradient = np.asarray(gradient / weight_total, dtype=float)
            gradient[1:] += coefficients[1:] / (self.c * weight_total)
            return value, gradient

        result = minimize(
            objective,
            np.zeros(matrix.shape[1], dtype=float),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 5_000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(
                "Offset logistic correction did not converge: "
                f"{result.message}"
            )
        self.coef_ = np.asarray(result.x, dtype=float)
        self.optimization_result_ = result
        return self

    def correction(self, design: np.ndarray) -> np.ndarray:
        matrix = np.asarray(design, dtype=float)
        if not hasattr(self, "coef_"):
            raise RuntimeError("Correction model has not been fitted.")
        if matrix.ndim != 2 or matrix.shape[1] != len(self.coef_):
            raise ValueError("Correction design does not match fitted coefficients.")
        return np.asarray(matrix @ self.coef_, dtype=float)


class StagedProfileSymmetryAgeClassifier(BaseEstimator):
    """Sequential logit refinements with exact identity for missing blocks.

    The profile probability is the fixed baseline. Symmetry is fitted first as
    an additive logit correction gated by ``symmetry_available``. Age is fitted
    second as an additive correction to the symmetry-refined logit. Its compact
    interaction term allows the age contribution to depend on incoming risk.

    This class is research-only and is not part of the product prediction
    contract.
    """

    def __init__(
        self,
        *,
        symmetry_c: float = 0.3,
        age_c: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self.symmetry_c = symmetry_c
        self.age_c = age_c
        self.random_state = random_state

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
    ) -> "StagedProfileSymmetryAgeClassifier":
        """Fit symmetry and age corrections using the profile score as offset."""
        _ = self.random_state  # Optimization is deterministic.
        self._require_columns(x)
        labels = np.asarray(y, dtype=int)
        if len(x) != len(labels):
            raise ValueError("Feature rows and labels must align.")

        profile_logit = self._profile_logit(x)
        symmetry_design = self._fit_symmetry_design(x)
        self.symmetry_model_ = _OffsetLogisticCorrection(c=self.symmetry_c).fit(
            symmetry_design,
            labels,
            offset=profile_logit,
        )
        symmetry_logit = profile_logit + self.symmetry_model_.correction(
            symmetry_design
        )

        age_design = self._fit_age_design(x, symmetry_logit)
        self.age_model_ = _OffsetLogisticCorrection(c=self.age_c).fit(
            age_design,
            labels,
            offset=symmetry_logit,
        )
        self.symmetry_feature_names_ = [
            "symmetry_block_intercept",
            *(f"gated_{column}" for column in SK_CORE4_FEATURE_COLUMNS),
        ]
        self.age_feature_names_ = [
            "age_block_intercept",
            "gated_age",
            "gated_age_x_incoming_logit",
        ]
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return BENIGN/CANCER probabilities after both refinements."""
        probability = self.predict_stage_probabilities(x)[
            "final_p_cancer"
        ].to_numpy(dtype=float)
        return np.column_stack([1.0 - probability, probability])

    def predict_stage_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return profile, symmetry-refined, and final probabilities."""
        profile_logit, symmetry_logit, final_logit = self._stage_logits(x)
        return pd.DataFrame(
            {
                "profile_p_cancer": _expit(profile_logit),
                "after_symmetry_p_cancer": _expit(symmetry_logit),
                "final_p_cancer": _expit(final_logit),
            },
            index=x.index,
        )

    def stage_logit_corrections(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return additive evidence contributed by each optional block."""
        self._check_fitted()
        profile_logit = self._profile_logit(x)
        symmetry_correction = self.symmetry_model_.correction(
            self._symmetry_design(x)
        )
        symmetry_logit = profile_logit + symmetry_correction
        age_correction = self.age_model_.correction(
            self._age_design(x, symmetry_logit)
        )
        return pd.DataFrame(
            {
                "symmetry_logit_correction": symmetry_correction,
                "age_logit_correction": age_correction,
            },
            index=x.index,
        )

    def _stage_logits(
        self,
        x: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_fitted()
        profile_logit = self._profile_logit(x)
        symmetry_logit = profile_logit + self.symmetry_model_.correction(
            self._symmetry_design(x)
        )
        final_logit = symmetry_logit + self.age_model_.correction(
            self._age_design(x, symmetry_logit)
        )
        return profile_logit, symmetry_logit, final_logit

    def _fit_symmetry_design(self, x: pd.DataFrame) -> np.ndarray:
        available = x["symmetry_available"].astype(bool).to_numpy()
        symmetry = x.loc[:, SK_CORE4_FEATURE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        paired = symmetry.loc[available]
        if paired.empty:
            raise ValueError("At least one symmetry-available training row is required.")
        self.symmetry_fill_values_ = paired.median().fillna(0.0)
        filled = symmetry.fillna(self.symmetry_fill_values_)
        self.symmetry_scaler_ = StandardScaler().fit(
            filled.loc[available].to_numpy(dtype=float)
        )
        return self._symmetry_design(x)

    def _symmetry_design(self, x: pd.DataFrame) -> np.ndarray:
        available = x["symmetry_available"].astype(bool).to_numpy(dtype=float)
        symmetry = (
            x.loc[:, SK_CORE4_FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(self.symmetry_fill_values_)
        )
        scaled = self.symmetry_scaler_.transform(
            symmetry.to_numpy(dtype=float)
        )
        scaled[available == 0.0, :] = 0.0
        return np.column_stack([available, scaled * available[:, None]])

    def _fit_age_design(
        self,
        x: pd.DataFrame,
        incoming_logit: np.ndarray,
    ) -> np.ndarray:
        available = x["age_available"].astype(bool).to_numpy()
        age = pd.to_numeric(x["age"], errors="coerce")
        observed_age = age.loc[available]
        if observed_age.empty:
            raise ValueError("At least one age-available training row is required.")
        self.age_fill_value_ = float(observed_age.median())
        filled_age = age.fillna(self.age_fill_value_).to_numpy(dtype=float)
        self.age_scaler_ = StandardScaler().fit(filled_age[available, None])
        incoming = np.asarray(incoming_logit, dtype=float)
        self.incoming_logit_scaler_ = StandardScaler().fit(
            incoming[available, None]
        )
        return self._age_design(x, incoming)

    def _age_design(
        self,
        x: pd.DataFrame,
        incoming_logit: np.ndarray,
    ) -> np.ndarray:
        available = x["age_available"].astype(bool).to_numpy(dtype=float)
        age = (
            pd.to_numeric(x["age"], errors="coerce")
            .fillna(self.age_fill_value_)
            .to_numpy(dtype=float)
        )
        age_scaled = self.age_scaler_.transform(age[:, None])[:, 0]
        incoming = np.asarray(incoming_logit, dtype=float)
        incoming_scaled = self.incoming_logit_scaler_.transform(
            incoming[:, None]
        )[:, 0]
        return np.column_stack(
            [
                available,
                available * age_scaled,
                available * age_scaled * incoming_scaled,
            ]
        )

    @staticmethod
    def _profile_logit(x: pd.DataFrame) -> np.ndarray:
        probability = pd.to_numeric(
            x[PROFILE_PROBABILITY_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)
        return _logit(probability)

    @staticmethod
    def _require_columns(x: pd.DataFrame) -> None:
        required = {
            PROFILE_PROBABILITY_COLUMN,
            "symmetry_available",
            "age",
            "age_available",
            *SK_CORE4_FEATURE_COLUMNS,
        }
        missing = sorted(required.difference(x.columns))
        if missing:
            raise ValueError(f"Missing staged model feature columns: {missing}")

    def _check_fitted(self) -> None:
        if not hasattr(self, "age_model_") or not hasattr(self, "symmetry_model_"):
            raise RuntimeError("Staged classifier has not been fitted.")
