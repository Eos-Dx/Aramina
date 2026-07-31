"""Research-only recalibrated joint additive meta-model.

The product model is deliberately not imported or modified here.  This module
implements a compact stacked logistic model whose input is an out-of-fold LR1
profile probability.  Optional age and symmetry blocks are multiplicative
gates: their availability flags are never columns in the learned design.
"""

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
DELTA_LOWER_BOUND_EPSILON = 1e-6
DELTA_LOWER_BOUND = -1.0 + DELTA_LOWER_BOUND_EPSILON


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
    """Jointly calibrate LR1 evidence and optional age/SK contributions.

    The fitted logit is:

    ``z0 + alpha + delta * z0 + age_gate * g_age(age)
    + symmetry_gate * g_symmetry(SK Core4)``, where ``z0`` is the LR1 logit.

    Availability flags are used solely as multiplicative gates.  They are not
    learned predictors and therefore cannot independently encode risk.  The
    intercept is unpenalized. ``delta`` is L2-penalized toward zero, so the
    regularization reference is the LR1 identity slope of one, not a slope of
    zero. Age and symmetry blocks have independent L2 strengths. The primary
    fit uses ordinary, unweighted logistic likelihood.
    """

    def __init__(
        self,
        *,
        profile_c: float = 0.1,
        age_c: float = 0.1,
        symmetry_c: float = 0.1,
        use_age: bool = True,
        use_symmetry: bool = True,
        random_state: int = 42,
    ) -> None:
        self.profile_c = profile_c
        self.age_c = age_c
        self.symmetry_c = symmetry_c
        self.use_age = use_age
        self.use_symmetry = use_symmetry
        self.random_state = random_state

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
    ) -> "RecalibratedJointAdditiveClassifier":
        """Fit all enabled blocks jointly on cross-fitted LR1 feature rows."""
        self._validate_hyperparameters()
        self._require_columns(x)
        labels = np.asarray(y, dtype=int)
        if len(x) != len(labels):
            raise ValueError("Feature rows and labels must align.")
        self._fit_scalers(x)
        design, penalties = self._design_and_penalties(x)
        profile_logit = self._profile_logit(x)
        if np.unique(labels).size != 2:
            raise ValueError("Joint refinement fitting requires BENIGN and CANCER.")
        sample_count = float(len(labels))

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            linear = profile_logit + design @ coefficients
            loss = np.logaddexp(0.0, linear) - labels * linear
            probability = _expit(linear)
            value = float(loss.mean())
            value += 0.5 * float(np.dot(penalties, coefficients**2)) / sample_count
            gradient = design.T @ (probability - labels)
            gradient = np.asarray(gradient / sample_count, dtype=float)
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
                "Recalibrated joint logistic fitting did not converge: "
                f"{result.message}"
            )
        self.coef_ = np.asarray(result.x, dtype=float)
        self.optimization_result_ = result
        self.classes_ = np.array([0, 1], dtype=int)
        self.feature_names_ = self._feature_names()
        self.penalty_by_feature_ = dict(
            zip(self.feature_names_, penalties.tolist(), strict=True)
        )
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return BENIGN/CANCER probabilities for the enabled architecture."""
        probability = _expit(self.decision_function(x))
        return np.column_stack([1.0 - probability, probability])

    def decision_function(self, x: pd.DataFrame) -> np.ndarray:
        """Return final model logits."""
        self._check_fitted()
        design, _ = self._design_and_penalties(x)
        return np.asarray(self._profile_logit(x) + design @ self.coef_, dtype=float)

    def prediction_components(self, x: pd.DataFrame) -> pd.DataFrame:
        """Expose profile, age, and symmetry logit contributions for audit."""
        self._check_fitted()
        design, _ = self._design_and_penalties(x)
        index = self._feature_index()
        base_profile = self._profile_logit(x)
        profile = base_profile + design[:, index["profile_logit_delta"]] * self.coef_[
            index["profile_logit_delta"]
        ]
        age_columns = [index[name] for name in self._age_feature_names() if name in index]
        symmetry_columns = [
            index[name] for name in self._symmetry_feature_names() if name in index
        ]
        age = design[:, age_columns] @ self.coef_[age_columns] if age_columns else np.zeros(len(x))
        symmetry = (
            design[:, symmetry_columns] @ self.coef_[symmetry_columns]
            if symmetry_columns
            else np.zeros(len(x))
        )
        intercept = np.full(len(x), self.coef_[index["intercept"]], dtype=float)
        return pd.DataFrame(
            {
                "profile_logit_contribution": profile,
                "age_logit_contribution": age,
                "symmetry_logit_contribution": symmetry,
                "intercept_logit_contribution": intercept,
                "final_logit": intercept + profile + age + symmetry,
            },
            index=x.index,
        )

    @property
    def recalibration_parameters_(self) -> dict[str, float]:
        """Return ``alpha`` and ``1 + delta`` in the raw LR1-logit formula."""
        self._check_fitted()
        index = self._feature_index()
        return {
            "intercept": float(self.coef_[index["intercept"]]),
            "profile_logit_slope": float(1.0 + self.coef_[index["profile_logit_delta"]]),
            "profile_logit_delta": float(self.coef_[index["profile_logit_delta"]]),
        }

    def _fit_scalers(self, x: pd.DataFrame) -> None:
        age_available = self._age_gate(x)
        age = pd.to_numeric(x["age"], errors="coerce")
        if self.use_age and age_available.any():
            observed = age.loc[age_available]
            if observed.isna().any():
                raise ValueError("age_available requires a finite age value.")
            self.age_fill_value_ = float(observed.median())
            self.age_scaler_ = StandardScaler().fit(observed.to_numpy(dtype=float)[:, None])
        else:
            self.age_fill_value_ = 0.0
            self.age_scaler_ = StandardScaler().fit(np.zeros((1, 1), dtype=float))

        symmetry_available = self._symmetry_gate(x)
        symmetry = x.loc[:, SK_CORE4_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if self.use_symmetry and symmetry_available.any():
            observed = symmetry.loc[symmetry_available]
            if not np.isfinite(observed.to_numpy(dtype=float)).all():
                raise ValueError(
                    "symmetry_available requires finite SK Core4 values."
                )
            self.symmetry_scaler_ = StandardScaler().fit(observed.to_numpy(dtype=float))
        else:
            self.symmetry_scaler_ = StandardScaler().fit(
                np.zeros((1, len(SK_CORE4_FEATURE_COLUMNS)), dtype=float)
            )

    def _design_and_penalties(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        profile_logit = self._profile_logit(x)
        columns = [np.ones(len(x), dtype=float), profile_logit]
        penalties = [0.0, 1.0 / float(self.profile_c)]

        if self.use_age:
            age_gate = self._age_gate(x).astype(float)
            age = pd.to_numeric(x["age"], errors="coerce")
            if age.loc[age_gate.astype(bool)].isna().any():
                raise ValueError("age_available requires a finite age value.")
            filled = age.fillna(self.age_fill_value_).to_numpy(dtype=float)
            age_scaled = self.age_scaler_.transform(filled[:, None])[:, 0]
            columns.append(age_gate * age_scaled)
            penalties.append(1.0 / float(self.age_c))

        if self.use_symmetry:
            symmetry_gate = self._symmetry_gate(x).astype(float)
            symmetry = x.loc[:, SK_CORE4_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
            active = symmetry.loc[symmetry_gate.astype(bool)]
            if not np.isfinite(active.to_numpy(dtype=float)).all():
                raise ValueError("symmetry_available requires finite SK Core4 values.")
            filled = symmetry.fillna(0.0).to_numpy(dtype=float)
            scaled = self.symmetry_scaler_.transform(filled)
            columns.extend((scaled * symmetry_gate[:, None]).T)
            penalties.extend([1.0 / float(self.symmetry_c)] * len(SK_CORE4_FEATURE_COLUMNS))

        return np.column_stack(columns), np.asarray(penalties, dtype=float)

    def _feature_names(self) -> list[str]:
        return [
            "intercept",
            "profile_logit_delta",
            *self._age_feature_names(),
            *self._symmetry_feature_names(),
        ]

    def _age_feature_names(self) -> list[str]:
        return ["gated_age"] if self.use_age else []

    def _symmetry_feature_names(self) -> list[str]:
        if not self.use_symmetry:
            return []
        return [f"gated_{column}" for column in SK_CORE4_FEATURE_COLUMNS]

    def _feature_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.feature_names_)}

    @staticmethod
    def _age_gate(x: pd.DataFrame) -> np.ndarray:
        return x["age_available"].astype(bool).to_numpy()

    @staticmethod
    def _symmetry_gate(x: pd.DataFrame) -> np.ndarray:
        return x["symmetry_available"].astype(bool).to_numpy()

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
            raise ValueError(f"Missing joint model feature columns: {missing}")

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
            raise RuntimeError("Recalibrated joint classifier has not been fitted.")

    @staticmethod
    def _profile_logit(x: pd.DataFrame) -> np.ndarray:
        probability = pd.to_numeric(
            x[PROFILE_PROBABILITY_COLUMN], errors="coerce"
        ).to_numpy(dtype=float)
        return _logit(probability)
