"""Executable M2Q model components."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SK_CORE4_FEATURE_COLUMNS = (
    "sk_wasserstein_distance_full_q2",
    "sk_weightedrms1",
    "sk_weightedrms2",
    "sk_mean_peak_value_abs_delta",
)


def build_profile_logistic(*, logreg_c: float, random_state: int) -> Pipeline:
    """Build the standard-scaled LR1 radial-profile classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(logreg_c),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=int(random_state),
                    solver="lbfgs",
                ),
            ),
        ]
    )


class GatedSymmetryLogistic(BaseEstimator):
    """LR2 with profile and age plus an optional neutral SK refinement.

    Profile evidence and age are always evaluated. SK Core4 terms are scaled
    from paired training cases and set to zero when no contralateral breast is
    available. The availability flag is a gate, never a learned risk feature.
    """

    def __init__(self, *, logreg_c: float = 0.1, random_state: int = 42) -> None:
        self.logreg_c = logreg_c
        self.random_state = random_state

    @property
    def base_feature_columns_(self) -> list[str]:
        """Return the profile and age fields evaluated for every target breast."""
        return ["profile_p_cancer_logit_average", "age", "age_available"]

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "GatedSymmetryLogistic":
        """Fit LR2 and paired-case symmetry scaling."""
        base = x.loc[:, self.base_feature_columns_].apply(
            pd.to_numeric, errors="coerce"
        )
        self.base_fill_values_ = base.median().fillna(0.0)
        self.base_scaler_ = StandardScaler().fit(
            base.fillna(self.base_fill_values_).to_numpy(dtype=float)
        )

        paired = x["symmetry_available"].astype(bool).to_numpy()
        symmetry = x.loc[:, SK_CORE4_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        paired_values = symmetry.loc[paired]
        self.symmetry_means_ = paired_values.mean().fillna(0.0)
        self.symmetry_scales_ = paired_values.std(ddof=0).replace(0.0, 1.0).fillna(1.0)

        self.logreg_ = LogisticRegression(
            C=float(self.logreg_c),
            class_weight="balanced",
            max_iter=5000,
            random_state=int(self.random_state),
            solver="lbfgs",
        ).fit(self._matrix(x), np.asarray(y, dtype=int))
        self.feature_names_ = [
            *self.base_feature_columns_,
            *(f"gated_{column}" for column in SK_CORE4_FEATURE_COLUMNS),
        ]
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return BENIGN/CANCER probabilities for target-breast feature rows."""
        return self.logreg_.predict_proba(self._matrix(x))

    def _matrix(self, x: pd.DataFrame) -> np.ndarray:
        base = x.loc[:, self.base_feature_columns_].apply(
            pd.to_numeric, errors="coerce"
        )
        base_scaled = self.base_scaler_.transform(
            base.fillna(self.base_fill_values_).to_numpy(dtype=float)
        )
        paired = x["symmetry_available"].astype(bool).to_numpy()
        symmetry = (
            x.loc[:, SK_CORE4_FEATURE_COLUMNS]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .fillna(self.symmetry_means_)
        )
        symmetry_scaled = (
            symmetry.to_numpy(dtype=float) - self.symmetry_means_.to_numpy(dtype=float)
        ) / self.symmetry_scales_.to_numpy(dtype=float)
        symmetry_scaled[~paired, :] = 0.0
        return np.hstack([base_scaled, symmetry_scaled])
