"""Research-only additive recalibration model ported from experiment2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from .target_breast_model import SK_CORE4_FEATURE_COLUMNS


PROFILE_PROBABILITY_COLUMN = "profile_p_cancer_logit_average"
PROBABILITY_EPSILON = 1e-6
DELTA_LOWER_BOUND = -1.0 + 1e-6


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
    negative = np.exp(values[~positive])
    out[~positive] = negative / (1.0 + negative)
    return out


class RecalibratedJointAdditiveClassifier(BaseEstimator):
    """Recalibrate LR1 logit with gated age and SK Core4 contributions.

    This is the full profile + age + symmetry architecture from experiment2.
    Availability flags gate optional blocks and are not learned predictors.
    The profile delta is regularized toward the identity slope and bounded so
    the total profile-logit slope remains positive.
    """

    def __init__(
        self,
        *,
        profile_c: float = 0.001,
        age_c: float = 0.3,
        symmetry_c: float = 0.001,
        random_state: int = 42,
    ) -> None:
        self.profile_c = profile_c
        self.age_c = age_c
        self.symmetry_c = symmetry_c
        self.random_state = random_state

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
    ) -> RecalibratedJointAdditiveClassifier:
        """Fit the full additive model on patient-safe LR1 OOF features."""
        self._validate_hyperparameters()
        self._require_columns(x)
        labels = np.asarray(y, dtype=int)
        if len(x) != len(labels):
            raise ValueError("Feature rows and labels must align.")
        if np.unique(labels).size != 2:
            raise ValueError("Additive recalibration requires BENIGN and CANCER.")
        self._fit_scalers(x)
        design, penalties = self._design_and_penalties(x)
        profile_logit = self._profile_logit(x)
        sample_count = float(len(labels))

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            linear = profile_logit + design @ coefficients
            loss = np.logaddexp(0.0, linear) - labels * linear
            probability = _expit(linear)
            value = float(loss.mean())
            value += (
                0.5 * float(np.dot(penalties, coefficients**2)) / sample_count
            )
            gradient = design.T @ (probability - labels) / sample_count
            gradient = np.asarray(gradient, dtype=float)
            gradient += penalties * coefficients / sample_count
            return value, gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1], dtype=float),
            method="L-BFGS-B",
            jac=True,
            bounds=[(None, None), (DELTA_LOWER_BOUND, None)]
            + [(None, None)] * (design.shape[1] - 2),
            options={"maxiter": 5_000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(
                f"Additive recalibration did not converge: {result.message}"
            )
        self.coef_ = np.asarray(result.x, dtype=float)
        self.optimization_result_ = result
        self.classes_ = np.array([0, 1], dtype=int)
        self.feature_names_ = [
            "intercept",
            "profile_logit_delta",
            "gated_age",
            *(f"gated_{column}" for column in SK_CORE4_FEATURE_COLUMNS),
        ]
        self.penalty_by_feature_ = dict(
            zip(self.feature_names_, penalties.tolist(), strict=True)
        )
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return BENIGN/CANCER probabilities."""
        probability = _expit(self.decision_function(x))
        return np.column_stack([1.0 - probability, probability])

    def decision_function(self, x: pd.DataFrame) -> np.ndarray:
        """Return final additive logits."""
        self._check_fitted()
        self._require_columns(x)
        design, _ = self._design_and_penalties(x)
        return np.asarray(self._profile_logit(x) + design @ self.coef_, dtype=float)

    @property
    def recalibration_parameters_(self) -> dict[str, float]:
        """Return intercept, profile delta, and total profile slope."""
        self._check_fitted()
        return {
            "intercept": float(self.coef_[0]),
            "profile_logit_slope": float(1.0 + self.coef_[1]),
            "profile_logit_delta": float(self.coef_[1]),
        }

    def _fit_scalers(self, x: pd.DataFrame) -> None:
        age_available = self._age_gate(x)
        age = pd.to_numeric(x["age"], errors="coerce")
        observed_age = age.loc[age_available]
        if age_available.any():
            if observed_age.isna().any():
                raise ValueError("age_available requires a finite age value.")
            self.age_fill_value_ = float(observed_age.median())
            self.age_scaler_ = StandardScaler().fit(
                observed_age.to_numpy(dtype=float)[:, None]
            )
        else:
            self.age_fill_value_ = 0.0
            self.age_scaler_ = StandardScaler().fit(
                np.zeros((1, 1), dtype=float)
            )

        symmetry_available = self._symmetry_gate(x)
        symmetry = x.loc[:, SK_CORE4_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        observed_symmetry = symmetry.loc[symmetry_available]
        if symmetry_available.any():
            if not np.isfinite(observed_symmetry.to_numpy(dtype=float)).all():
                raise ValueError(
                    "symmetry_available requires finite SK Core4 values."
                )
            self.symmetry_scaler_ = StandardScaler().fit(
                observed_symmetry.to_numpy(dtype=float)
            )
        else:
            self.symmetry_scaler_ = StandardScaler().fit(
                np.zeros((1, len(SK_CORE4_FEATURE_COLUMNS)), dtype=float)
            )

    def _design_and_penalties(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        profile_logit = self._profile_logit(x)
        age_gate = self._age_gate(x).astype(float)
        age = pd.to_numeric(x["age"], errors="coerce")
        if age.loc[age_gate.astype(bool)].isna().any():
            raise ValueError("age_available requires a finite age value.")
        age_scaled = self.age_scaler_.transform(
            age.fillna(self.age_fill_value_).to_numpy(dtype=float)[:, None]
        )[:, 0]

        symmetry_gate = self._symmetry_gate(x).astype(float)
        symmetry = x.loc[:, SK_CORE4_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        active_symmetry = symmetry.loc[symmetry_gate.astype(bool)]
        if not np.isfinite(active_symmetry.to_numpy(dtype=float)).all():
            raise ValueError("symmetry_available requires finite SK Core4 values.")
        symmetry_scaled = self.symmetry_scaler_.transform(
            symmetry.fillna(0.0).to_numpy(dtype=float)
        )
        design = np.column_stack(
            [
                np.ones(len(x), dtype=float),
                profile_logit,
                age_gate * age_scaled,
                *(symmetry_scaled * symmetry_gate[:, None]).T,
            ]
        )
        penalties = np.asarray(
            [
                0.0,
                1.0 / float(self.profile_c),
                1.0 / float(self.age_c),
                *(
                    [1.0 / float(self.symmetry_c)]
                    * len(SK_CORE4_FEATURE_COLUMNS)
                ),
            ],
            dtype=float,
        )
        return design, penalties

    @staticmethod
    def _age_gate(x: pd.DataFrame) -> np.ndarray:
        return x["age_available"].astype(bool).to_numpy()

    @staticmethod
    def _symmetry_gate(x: pd.DataFrame) -> np.ndarray:
        return x["symmetry_available"].astype(bool).to_numpy()

    @staticmethod
    def _profile_logit(x: pd.DataFrame) -> np.ndarray:
        probability = pd.to_numeric(
            x[PROFILE_PROBABILITY_COLUMN], errors="coerce"
        ).to_numpy(dtype=float)
        return _logit(probability)

    @staticmethod
    def _require_columns(x: pd.DataFrame) -> None:
        required = {
            PROFILE_PROBABILITY_COLUMN,
            "age",
            "age_available",
            "symmetry_available",
            *SK_CORE4_FEATURE_COLUMNS,
        }
        missing = sorted(required.difference(x.columns))
        if missing:
            raise ValueError(f"Missing additive model feature columns: {missing}")

    def _validate_hyperparameters(self) -> None:
        for name, value in {
            "profile_c": self.profile_c,
            "age_c": self.age_c,
            "symmetry_c": self.symmetry_c,
        }.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite value.")

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Additive recalibration has not been fitted.")
