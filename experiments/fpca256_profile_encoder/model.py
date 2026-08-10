"""Fold-local FPCA profile encoding under the current Aramina architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aramina.m2q_model import GatedSymmetryLogistic, SK_CORE4_FEATURE_COLUMNS
from aramina.model_metrics import binary_metric_values
from aramina.model_utils import compute_binary_thresholds, profile_matrix
from aramina.patient_features import (
    TARGET_CASE_ID,
    empty_lr1_scores,
    lr1_training_rows,
    patient_feature_table,
    row_labels,
    score_lr1_rows,
)


PROFILE_SCORE_COLUMNS = (
    "profile_p_cancer_probability_mean",
    "profile_p_cancer_logit_average",
    "profile_p_cancer_n_measurements",
)


@dataclass(frozen=True)
class ProfileSpec:
    """One controlled LR1 representation."""

    name: str
    npt: int
    kind: Literal["raw", "fpca"]
    n_components: int | None = None


class FoldLocalProfileEncoder:
    """Raw or discrete-FPCA LR1 fitted exclusively on supplied patients."""

    def __init__(
        self,
        *,
        spec: ProfileSpec,
        profile_column: str,
        label_column: str,
        group_column: str,
        q_column: str,
        logreg_c: float,
        random_state: int,
    ) -> None:
        self.spec = spec
        self.profile_column = profile_column
        self.label_column = label_column
        self.group_column = group_column
        self.q_column = q_column
        self.logreg_c = logreg_c
        self.random_state = random_state

    def fit(self, rows: pd.DataFrame) -> "FoldLocalProfileEncoder":
        """Fit PCA and LR1 on one training partition only."""
        validate_profile_grid(
            rows,
            profile_column=self.profile_column,
            q_column=self.q_column,
            expected_npt=self.spec.npt,
        )
        matrix = profile_matrix(rows, self.profile_column)
        labels = row_labels(rows, self.label_column)
        steps: list[tuple[str, Any]] = []
        if self.spec.kind == "fpca":
            n_components = int(self.spec.n_components or 0)
            if n_components >= min(matrix.shape):
                raise ValueError(
                    f"FPCA components {n_components} must be smaller than "
                    f"min(profile matrix shape)={min(matrix.shape)}."
                )
            steps.append(
                (
                    "fpca",
                    PCA(
                        n_components=n_components,
                        svd_solver="full",
                    ),
                )
            )
        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=float(self.logreg_c),
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=int(self.random_state),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.pipeline_ = Pipeline(steps).fit(matrix, labels)
        self.training_patient_ids_ = frozenset(
            rows[self.group_column].astype(str).unique()
        )
        self.q_grid_ = np.asarray(rows[self.q_column].iloc[0], dtype=float)
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        """Expose product-compatible probability prediction."""
        return self.pipeline_.predict_proba(matrix)

    @property
    def pca_(self) -> PCA | None:
        """Return fitted discrete-FPCA transform when present."""
        return self.pipeline_.named_steps.get("fpca")


def validate_profile_grid(
    df: pd.DataFrame,
    *,
    profile_column: str,
    q_column: str,
    expected_npt: int,
) -> np.ndarray:
    """Require finite profiles on one shared, uniformly spaced q grid."""
    if df.empty:
        raise ValueError("Profile dataset must not be empty.")
    matrix = profile_matrix(df, profile_column)
    if matrix.shape[1] != expected_npt:
        raise ValueError(
            f"Expected {expected_npt}-bin profiles; received {matrix.shape[1]}."
        )
    grids = [np.asarray(value, dtype=float).ravel() for value in df[q_column]]
    reference = grids[0]
    if reference.size != expected_npt or not np.isfinite(reference).all():
        raise ValueError("q grid must be finite and match expected npt.")
    for grid in grids[1:]:
        if grid.shape != reference.shape or not np.allclose(
            grid,
            reference,
            rtol=1e-9,
            atol=1e-10,
        ):
            raise ValueError("FPCA requires one shared q grid across measurements.")
    spacing = np.diff(reference)
    if not np.all(spacing > 0.0) or not np.allclose(
        spacing,
        spacing[0],
        rtol=1e-6,
        atol=1e-10,
    ):
        raise ValueError("FPCA requires a uniformly spaced increasing q grid.")
    return reference


def build_dataset_context(df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    """Compute npt-specific Core4 context once, before LR1 scores are attached."""
    scores = empty_lr1_scores(
        df,
        group_column=model["group_column"],
        side_column=model["side_column"],
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
    )
    context = patient_feature_table(
        df,
        scores,
        profile_column=model["profile_column"],
        label_column=model["label_column"],
        group_column=model["group_column"],
        specimen_column=model["specimen_column"],
        side_column=model["side_column"],
        q_column=model["q_column"],
        age_column=model["age_column"],
        biopsy_column=model["biopsy_column"],
    )
    return context.drop(columns=list(PROFILE_SCORE_COLUMNS))


def fit_split(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_context: pd.DataFrame,
    test_context: pd.DataFrame,
    spec: ProfileSpec,
    model: dict[str, Any],
    target_sensitivity: float,
    random_state: int,
) -> dict[str, Any]:
    """Fit one outer split using current product LR1/LR2 and threshold policies."""
    encoder = _fit_encoder(train_df, spec=spec, model=model, random_state=random_state)
    test_patients = frozenset(test_df[model["group_column"]].astype(str).unique())
    if encoder.training_patient_ids_.intersection(test_patients):
        raise RuntimeError("Patient leakage detected in fold-local profile encoder.")

    train_features = attach_profile_scores(
        encoder,
        train_df,
        train_context,
        model=model,
        require_two_classes=True,
    )
    test_features = attach_profile_scores(
        encoder,
        test_df,
        test_context,
        model=model,
        require_two_classes=False,
    )
    final_model = GatedSymmetryLogistic(
        logreg_c=float(model["lr2_logreg_c"]),
        random_state=random_state,
    ).fit(train_features, train_features["label"].to_numpy(dtype=int))
    train_score = final_model.predict_proba(train_features)[:, 1]
    thresholds = compute_binary_thresholds(
        train_features["label"].to_numpy(dtype=int),
        train_score,
        target_sensitivity=target_sensitivity,
    )
    test_score = final_model.predict_proba(test_features)[:, 1]
    return {
        "encoder": encoder,
        "final_model": final_model,
        "thresholds": thresholds,
        "features": test_features,
        "score": test_score,
    }


def fit_train_all(
    *,
    df: pd.DataFrame,
    context: pd.DataFrame,
    spec: ProfileSpec,
    model: dict[str, Any],
    target_sensitivity: float,
    random_state: int,
) -> dict[str, Any]:
    """Fit one research model and threshold on all available target cases."""
    encoder = _fit_encoder(df, spec=spec, model=model, random_state=random_state)
    features = attach_profile_scores(
        encoder,
        df,
        context,
        model=model,
        require_two_classes=True,
    )
    final_model = GatedSymmetryLogistic(
        logreg_c=float(model["lr2_logreg_c"]),
        random_state=random_state,
    ).fit(features, features["label"].to_numpy(dtype=int))
    score = final_model.predict_proba(features)[:, 1]
    thresholds = compute_binary_thresholds(
        features["label"].to_numpy(dtype=int),
        score,
        target_sensitivity=target_sensitivity,
    )
    threshold = float(thresholds["threshold_target"])
    return {
        "profile_encoder": encoder,
        "final_model": final_model,
        "thresholds": thresholds,
        "metrics": metric_row(
            features["label"].to_numpy(dtype=int),
            score,
            threshold=threshold,
        ),
        "feature_schema": {
            "base": [
                "profile_p_cancer_logit_average",
                "age",
                "age_available",
            ],
            "symmetry": list(SK_CORE4_FEATURE_COLUMNS),
            "symmetry_available": "neutral gate; not a learned feature",
        },
    }


def attach_profile_scores(
    encoder: FoldLocalProfileEncoder,
    df: pd.DataFrame,
    context: pd.DataFrame,
    *,
    model: dict[str, Any],
    require_two_classes: bool,
) -> pd.DataFrame:
    """Score biopsy-only LR1 rows and merge with precomputed patient context."""
    rows = lr1_training_rows(
        df,
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
        lr1_row_policy=model["lr1_row_policy"],
        require_two_classes=require_two_classes,
    )
    scores = score_lr1_rows(
        encoder,
        rows,
        full_df=df,
        profile_column=model["profile_column"],
        group_column=model["group_column"],
        side_column=model["side_column"],
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
    )
    out = context.merge(scores, on=TARGET_CASE_ID, how="inner")
    if len(out) != len(context):
        raise RuntimeError("LR1 scoring did not produce one row per target case.")
    if require_two_classes and out["label"].nunique() != 2:
        raise ValueError("Training features require BENIGN and CANCER cases.")
    return out.reset_index(drop=True)


def metric_row(
    y: np.ndarray,
    score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Return complete thresholded metrics for one split."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    thresholds = np.full(len(y), float(threshold), dtype=float)
    values = binary_metric_values(y, score, thresholds)
    pred = (score >= thresholds).astype(int)
    return {
        **values,
        "threshold_target": float(threshold),
        "true_positives": int(((pred == 1) & (y == 1)).sum()),
        "true_negatives": int(((pred == 0) & (y == 0)).sum()),
        "false_positives": int(((pred == 1) & (y == 0)).sum()),
        "false_negatives": int(((pred == 0) & (y == 1)).sum()),
    }


def _fit_encoder(
    df: pd.DataFrame,
    *,
    spec: ProfileSpec,
    model: dict[str, Any],
    random_state: int,
) -> FoldLocalProfileEncoder:
    rows = lr1_training_rows(
        df,
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
        lr1_row_policy=model["lr1_row_policy"],
    )
    return FoldLocalProfileEncoder(
        spec=spec,
        profile_column=model["profile_column"],
        label_column=model["label_column"],
        group_column=model["group_column"],
        q_column=model["q_column"],
        logreg_c=float(model["lr1_logreg_c"]),
        random_state=random_state,
    ).fit(rows)
