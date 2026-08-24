"""Fold-local model fitting and nested manifests for paired evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .additive_recalibration import RecalibratedJointAdditiveClassifier
from .model_metrics import binary_metric_values
from .model_utils import compute_binary_thresholds, profile_matrix
from .paired_cohort import patient_subset
from .paired_contract import (
    ADDITIVE_MODEL,
    ADDITIVE_REGULARIZATION,
    FPCA30_MODEL,
)
from .patient_features import (
    TARGET_CASE_ID,
    lr1_training_rows,
    row_labels,
    score_lr1_rows,
)
from .target_breast_model import GatedSymmetryLogistic
from .training_evaluation import _patient_split_pairs


@dataclass(frozen=True)
class ProfileSpec:
    """One fixed LR1 profile representation."""

    name: str
    npt: int
    kind: Literal["raw", "fpca"]
    n_components: int | None = None


@dataclass(frozen=True)
class FeaturePair:
    """Train and validation features from one train-fitted LR1."""

    train: pd.DataFrame
    validation: pd.DataFrame
    encoder: Pipeline


@dataclass(frozen=True)
class MetaPair:
    """Nested additive meta-train and meta-validation features."""

    fold_id: int
    train: pd.DataFrame
    validation: pd.DataFrame


def fit_feature_pair(
    frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    train_ids: set[str],
    validation_ids: set[str],
    spec: ProfileSpec,
    model: dict[str, Any],
    random_state: int,
) -> FeaturePair:
    """Fit one LR1 representation and score train and validation cases."""
    if train_ids.intersection(validation_ids):
        raise RuntimeError("Patient leakage detected before LR1 fitting.")
    train_frame = patient_subset(frame, train_ids)
    validation_frame = patient_subset(frame, validation_ids)
    train_context = patient_subset(context, train_ids)
    validation_context = patient_subset(context, validation_ids)
    encoder = _fit_profile_encoder(
        train_frame,
        spec=spec,
        model=model,
        random_state=random_state,
    )
    fitted_ids = set(train_frame["patientId"].astype(str))
    if fitted_ids.intersection(validation_ids):
        raise RuntimeError("Validation patient entered fold-local LR1 fitting.")
    return FeaturePair(
        train=_attach_profile_scores(
            encoder,
            train_frame,
            train_context,
            model=model,
            require_two_classes=True,
        ),
        validation=_attach_profile_scores(
            encoder,
            validation_frame,
            validation_context,
            model=model,
            require_two_classes=False,
        ),
        encoder=encoder,
    )


def evaluate_product_comparator(
    frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    train_ids: set[str],
    test_ids: set[str],
    spec: ProfileSpec,
    model: dict[str, Any],
    split_id: int,
    n_splits: int,
    random_state: int,
    target_sensitivity: float,
) -> dict[str, Any]:
    """Evaluate one exact same-data LR1-to-product-LR2 path."""
    pair = fit_feature_pair(
        frame,
        context,
        train_ids=train_ids,
        validation_ids=test_ids,
        spec=spec,
        model=model,
        random_state=random_state,
    )
    final_model = GatedSymmetryLogistic(
        logreg_c=float(model["lr2_logreg_c"]),
        random_state=random_state,
    ).fit(pair.train, pair.train["label"].to_numpy(dtype=int))
    train_score = final_model.predict_proba(pair.train)[:, 1]
    test_score = final_model.predict_proba(pair.validation)[:, 1]
    thresholds = compute_binary_thresholds(
        pair.train["label"].to_numpy(dtype=int),
        train_score,
        target_sensitivity=target_sensitivity,
    )
    result = _metric_prediction_result(
        model_name=spec.name,
        split_id=split_id,
        n_splits=n_splits,
        train_features=pair.train,
        test_features=pair.validation,
        test_score=test_score,
        thresholds=thresholds,
        model_fit_provenance="outer_train_same_data_lr1_to_lr2",
        threshold_provenance="outer_train_fitted_lr1_fitted_lr2_scores",
    )
    result["threshold_scores"] = _threshold_score_frame(
        pair.train,
        train_score,
        model_name=spec.name,
        split_id=split_id,
        threshold=float(thresholds["threshold_target"]),
        provenance="outer_train_fitted_lr1_fitted_lr2_scores",
    )
    result["test_features"] = pair.validation
    return result


def evaluate_additive_comparator(
    frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    train_ids: set[str],
    test_ids: set[str],
    test_features: pd.DataFrame,
    model: dict[str, Any],
    split_id: int,
    n_splits: int,
    inner_lr1_splits: int,
    meta_splits: int,
    random_state: int,
    target_sensitivity: float,
) -> dict[str, Any]:
    """Evaluate full additive model from strictly nested FPCA30/LR1 inputs."""
    train_frame = patient_subset(frame, train_ids)
    train_context = patient_subset(context, train_ids)
    outer_oof, outer_oof_manifest = _cross_fitted_profile_features(
        train_frame,
        train_context,
        outer_split_id=split_id,
        parent_fold_id=-1,
        level="additive_outer_lr1",
        n_splits=inner_lr1_splits,
        random_state=random_state,
        model=model,
    )
    meta_pairs, meta_manifest = _full_chain_meta_pairs(
        train_frame,
        train_context,
        outer_split_id=split_id,
        meta_splits=meta_splits,
        inner_lr1_splits=inner_lr1_splits,
        random_state=random_state + 1_000,
        model=model,
    )
    threshold_scores = _score_additive_meta_pairs(
        meta_pairs,
        random_state=random_state + 2_000,
    )
    thresholds = compute_binary_thresholds(
        threshold_scores["label"].to_numpy(dtype=int),
        threshold_scores["p_cancer"].to_numpy(dtype=float),
        target_sensitivity=target_sensitivity,
    )
    additive = RecalibratedJointAdditiveClassifier(
        **ADDITIVE_REGULARIZATION,
        random_state=random_state + 3_000,
    ).fit(outer_oof, outer_oof["label"].to_numpy(dtype=int))
    if set(test_features["patientId"].astype(str)) != test_ids:
        raise RuntimeError("Additive test features do not match shared outer test fold.")
    test_score = additive.predict_proba(test_features)[:, 1]
    fit_provenance = (
        "outer_train_patient_safe_fpca30_lr1_oof_additive_meta_fit"
    )
    threshold_provenance = (
        "outer_train_nested_full_chain_fpca30_lr1_oof_additive_meta_oof_scores"
    )
    result = _metric_prediction_result(
        model_name=ADDITIVE_MODEL,
        split_id=split_id,
        n_splits=n_splits,
        train_features=outer_oof,
        test_features=test_features,
        test_score=test_score,
        thresholds=thresholds,
        model_fit_provenance=fit_provenance,
        threshold_provenance=threshold_provenance,
    )
    result["threshold_scores"] = _threshold_score_frame(
        threshold_scores,
        threshold_scores["p_cancer"].to_numpy(dtype=float),
        model_name=ADDITIVE_MODEL,
        split_id=split_id,
        threshold=float(thresholds["threshold_target"]),
        provenance=threshold_provenance,
    )
    result["nested_manifest"] = pd.concat(
        [outer_oof_manifest, meta_manifest], ignore_index=True
    )
    return result


def strict_split_pairs(
    context: pd.DataFrame,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    description: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic patient-safe splits without reducing fold count."""
    patient_labels = context.groupby("patientId")["label"].max()
    class_counts = patient_labels.value_counts()
    if len(class_counts) != 2:
        raise ValueError(f"{description} splitting requires two patient classes.")
    if int(n_splits) > int(class_counts.min()):
        raise ValueError(
            f"{description} n_splits={n_splits} exceeds smallest patient class "
            f"count={int(class_counts.min())}."
        )
    return _patient_split_pairs(
        mode="stratified_kfold",
        base_features=context.reset_index(drop=True),
        y_patients=context["label"].to_numpy(dtype=int),
        n_splits=int(n_splits),
        n_repeats=int(n_repeats),
        random_state=int(random_state),
    )


def outer_fold_manifest(
    context: pd.DataFrame,
    split_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_splits: int,
) -> pd.DataFrame:
    """Persist shared outer target-case roles for all models."""
    rows: list[pd.DataFrame] = []
    for split_id, (train_index, test_index) in enumerate(split_pairs):
        for role, index in (("outer_train", train_index), ("outer_test", test_index)):
            part = context.iloc[index][
                [TARGET_CASE_ID, "patientId", "label", "label_name"]
            ].copy()
            part.insert(0, "split_id", int(split_id))
            part.insert(1, "repeat_id", int(split_id) // int(n_splits))
            part.insert(2, "level", "outer")
            part.insert(3, "parent_fold_id", -1)
            part.insert(4, "fold_id", int(split_id) % int(n_splits))
            part.insert(5, "role", role)
            rows.append(part)
    manifest = pd.concat(rows, ignore_index=True)
    for split_id, group in manifest.groupby("split_id"):
        if group[TARGET_CASE_ID].duplicated().any():
            raise RuntimeError(f"Outer manifest split {split_id} duplicates cases.")
        if set(group[TARGET_CASE_ID]) != set(context[TARGET_CASE_ID]):
            raise RuntimeError(f"Outer manifest split {split_id} omits cases.")
        if group.groupby("patientId")["role"].nunique().max() != 1:
            raise RuntimeError(f"Outer manifest split {split_id} leaks patients.")
    return manifest


def patient_ids(context: pd.DataFrame, index: np.ndarray) -> set[str]:
    """Return patient IDs represented by case indexes."""
    return set(context.iloc[index]["patientId"].astype(str))


def _fit_profile_encoder(
    frame: pd.DataFrame,
    *,
    spec: ProfileSpec,
    model: dict[str, Any],
    random_state: int,
) -> Pipeline:
    rows = lr1_training_rows(
        frame,
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
        lr1_row_policy=model["lr1_row_policy"],
    )
    matrix = profile_matrix(rows, model["profile_column"])
    if matrix.shape[1] != spec.npt:
        raise ValueError(
            f"{spec.name} requires {spec.npt}-bin profiles; "
            f"received {matrix.shape[1]}."
        )
    steps: list[tuple[str, Any]] = []
    if spec.kind == "fpca":
        components = int(spec.n_components or 0)
        if components > min(matrix.shape):
            raise ValueError(
                f"FPCA components {components} exceed training matrix "
                f"limit {min(matrix.shape)}."
            )
        steps.append(("fpca", PCA(n_components=components, svd_solver="full")))
    steps.extend(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(model["lr1_logreg_c"]),
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=int(random_state),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return Pipeline(steps).fit(matrix, row_labels(rows, model["label_column"]))


def _attach_profile_scores(
    encoder: Pipeline,
    frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    model: dict[str, Any],
    require_two_classes: bool,
) -> pd.DataFrame:
    rows = lr1_training_rows(
        frame,
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
        lr1_row_policy=model["lr1_row_policy"],
        require_two_classes=require_two_classes,
    )
    scores = score_lr1_rows(
        encoder,
        rows,
        full_df=frame,
        profile_column=model["profile_column"],
        group_column=model["group_column"],
        side_column=model["side_column"],
        label_column=model["label_column"],
        biopsy_column=model["biopsy_column"],
    )
    out = context.merge(scores, on=TARGET_CASE_ID, how="inner", validate="one_to_one")
    if len(out) != len(context):
        raise RuntimeError("LR1 scoring silently dropped target cases.")
    if require_two_classes and out["label"].nunique() != 2:
        raise ValueError("Training target cases require BENIGN and CANCER.")
    return out.sort_values(TARGET_CASE_ID, kind="stable").reset_index(drop=True)


def _cross_fitted_profile_features(
    frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    outer_split_id: int,
    parent_fold_id: int,
    level: str,
    n_splits: int,
    random_state: int,
    model: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = strict_split_pairs(
        context,
        n_splits=n_splits,
        n_repeats=1,
        random_state=random_state,
        description=level,
    )
    features: list[pd.DataFrame] = []
    manifests: list[pd.DataFrame] = []
    spec = ProfileSpec(FPCA30_MODEL, 256, "fpca", 30)
    for fold_id, (train_index, validation_index) in enumerate(pairs):
        train_ids = patient_ids(context, train_index)
        validation_ids = patient_ids(context, validation_index)
        pair = fit_feature_pair(
            frame,
            context,
            train_ids=train_ids,
            validation_ids=validation_ids,
            spec=spec,
            model=model,
            random_state=random_state + fold_id,
        )
        part = pair.validation.copy()
        part["lr1_crossfit_fold"] = int(fold_id)
        features.append(part)
        manifests.append(
            _nested_manifest_rows(
                context,
                outer_split_id=outer_split_id,
                level=level,
                parent_fold_id=parent_fold_id,
                fold_id=fold_id,
                train_ids=train_ids,
                validation_ids=validation_ids,
            )
        )
    out = pd.concat(features, ignore_index=True).sort_values(
        TARGET_CASE_ID, kind="stable"
    )
    if out[TARGET_CASE_ID].duplicated().any() or set(out[TARGET_CASE_ID]) != set(
        context[TARGET_CASE_ID]
    ):
        raise RuntimeError("LR1 OOF construction must score each target case once.")
    return out.reset_index(drop=True), pd.concat(manifests, ignore_index=True)


def _full_chain_meta_pairs(
    frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    outer_split_id: int,
    meta_splits: int,
    inner_lr1_splits: int,
    random_state: int,
    model: dict[str, Any],
) -> tuple[list[MetaPair], pd.DataFrame]:
    split_pairs = strict_split_pairs(
        context,
        n_splits=meta_splits,
        n_repeats=1,
        random_state=random_state,
        description="additive_meta",
    )
    pairs: list[MetaPair] = []
    manifests: list[pd.DataFrame] = []
    fpca_spec = ProfileSpec(FPCA30_MODEL, 256, "fpca", 30)
    for meta_fold, (train_index, validation_index) in enumerate(split_pairs):
        train_ids = patient_ids(context, train_index)
        validation_ids = patient_ids(context, validation_index)
        meta_train_frame = patient_subset(frame, train_ids)
        meta_train_context = patient_subset(context, train_ids)
        meta_train, lr1_manifest = _cross_fitted_profile_features(
            meta_train_frame,
            meta_train_context,
            outer_split_id=outer_split_id,
            parent_fold_id=meta_fold,
            level="additive_meta_lr1",
            n_splits=inner_lr1_splits,
            random_state=random_state + meta_fold * 1_000,
            model=model,
        )
        validation_pair = fit_feature_pair(
            frame,
            context,
            train_ids=train_ids,
            validation_ids=validation_ids,
            spec=fpca_spec,
            model=model,
            random_state=random_state + meta_fold * 1_000 + 999,
        )
        pairs.append(MetaPair(meta_fold, meta_train, validation_pair.validation))
        manifests.extend(
            [
                _nested_manifest_rows(
                    context,
                    outer_split_id=outer_split_id,
                    level="additive_meta",
                    parent_fold_id=-1,
                    fold_id=meta_fold,
                    train_ids=train_ids,
                    validation_ids=validation_ids,
                ),
                lr1_manifest,
            ]
        )
    return pairs, pd.concat(manifests, ignore_index=True)


def _score_additive_meta_pairs(
    pairs: list[MetaPair],
    *,
    random_state: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for pair in pairs:
        model = RecalibratedJointAdditiveClassifier(
            **ADDITIVE_REGULARIZATION,
            random_state=random_state + pair.fold_id,
        ).fit(pair.train, pair.train["label"].to_numpy(dtype=int))
        out = pair.validation[
            [TARGET_CASE_ID, "patientId", "label", "label_name"]
        ].copy()
        out["meta_fold_id"] = int(pair.fold_id)
        out["p_cancer"] = model.predict_proba(pair.validation)[:, 1]
        rows.append(out)
    scores = pd.concat(rows, ignore_index=True).sort_values(
        TARGET_CASE_ID, kind="stable"
    )
    if scores[TARGET_CASE_ID].duplicated().any():
        raise RuntimeError("Additive meta OOF scoring duplicated target cases.")
    return scores.reset_index(drop=True)


def _metric_prediction_result(
    *,
    model_name: str,
    split_id: int,
    n_splits: int,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    test_score: np.ndarray,
    thresholds: dict[str, Any],
    model_fit_provenance: str,
    threshold_provenance: str,
) -> dict[str, Any]:
    y = test_features["label"].to_numpy(dtype=int)
    decision_threshold = float(thresholds["threshold_target"])
    decision_thresholds = np.full(len(y), decision_threshold, dtype=float)
    values = binary_metric_values(y, test_score, decision_thresholds)
    prediction = np.asarray(test_score >= decision_threshold, dtype=int)
    metric_row = {
        "model_name": model_name,
        "split_id": int(split_id),
        "repeat_id": int(split_id) // int(n_splits),
        "fold_id": int(split_id) % int(n_splits),
        "evaluation_mode": "paired_repeated_stratified_patient_safe_kfold",
        "model_fit_provenance": model_fit_provenance,
        "threshold_provenance": threshold_provenance,
        "roc_auc": values["roc_auc"],
        "pr_auc": values["pr_auc"],
        "brier_score": values["brier_score"],
        "log_loss": values["log_loss"],
        "calibration_intercept": values["calibration_intercept"],
        "calibration_slope": values["calibration_slope"],
        "sensitivity_target": values["sensitivity"],
        "specificity_target": values["specificity"],
        "balanced_accuracy_target": values["balanced_accuracy"],
        "ppv_target": values["ppv"],
        "npv_target": values["npv"],
        "tp_target": int(((y == 1) & (prediction == 1)).sum()),
        "tn_target": int(((y == 0) & (prediction == 0)).sum()),
        "fp_target": int(((y == 0) & (prediction == 1)).sum()),
        "fn_target": int(((y == 1) & (prediction == 0)).sum()),
        "train_patients": int(train_features["patientId"].nunique()),
        "test_patients": int(test_features["patientId"].nunique()),
        "train_target_cases": int(len(train_features)),
        "test_target_cases": int(len(test_features)),
        "train_cancer_target_cases": int((train_features["label"] == 1).sum()),
        "test_cancer_target_cases": int((test_features["label"] == 1).sum()),
        **thresholds,
    }
    predictions = test_features[
        [TARGET_CASE_ID, "patientId", "label", "label_name", "target_side"]
    ].copy()
    predictions.insert(0, "model_name", model_name)
    predictions.insert(1, "split_id", int(split_id))
    predictions.insert(2, "repeat_id", int(split_id) // int(n_splits))
    predictions.insert(3, "fold_id", int(split_id) % int(n_splits))
    predictions["evaluation_mode"] = metric_row["evaluation_mode"]
    predictions["model_fit_provenance"] = model_fit_provenance
    predictions["threshold_provenance"] = threshold_provenance
    predictions["p_cancer"] = np.asarray(test_score, dtype=float)
    predictions["threshold_target"] = decision_threshold
    predictions["y_pred_target"] = prediction
    return {"metrics": metric_row, "predictions": predictions}


def _threshold_score_frame(
    features: pd.DataFrame,
    scores: np.ndarray,
    *,
    model_name: str,
    split_id: int,
    threshold: float,
    provenance: str,
) -> pd.DataFrame:
    out = features[[TARGET_CASE_ID, "patientId", "label", "label_name"]].copy()
    out["meta_fold_id"] = (
        features["meta_fold_id"].to_numpy() if "meta_fold_id" in features else -1
    )
    out.insert(0, "model_name", model_name)
    out.insert(1, "split_id", int(split_id))
    out["p_cancer"] = np.asarray(scores, dtype=float)
    out["threshold_target"] = float(threshold)
    out["threshold_provenance"] = provenance
    return out


def _nested_manifest_rows(
    context: pd.DataFrame,
    *,
    outer_split_id: int,
    level: str,
    parent_fold_id: int,
    fold_id: int,
    train_ids: set[str],
    validation_ids: set[str],
) -> pd.DataFrame:
    if train_ids.intersection(validation_ids):
        raise RuntimeError(f"Patient leakage in {level} fold {fold_id}.")
    rows: list[pd.DataFrame] = []
    for role, selected_ids in (
        (f"{level}_train", train_ids),
        (f"{level}_validation", validation_ids),
    ):
        part = patient_subset(context, selected_ids)[
            [TARGET_CASE_ID, "patientId", "label", "label_name"]
        ].copy()
        part.insert(0, "split_id", int(outer_split_id))
        part.insert(1, "repeat_id", -1)
        part.insert(2, "level", level)
        part.insert(3, "parent_fold_id", int(parent_fold_id))
        part.insert(4, "fold_id", int(fold_id))
        part.insert(5, "role", role)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)
