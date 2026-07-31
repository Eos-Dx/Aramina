"""Nested patient-safe evaluation of recalibrated joint-additive research models.

This experiment is not product code. It does not update model artifacts,
contracts, preprocessing, or prediction behavior.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aramina.m2q_model import GatedSymmetryLogistic
from aramina.model_metrics import binary_metric_values, final_fit_training_metrics
from aramina.model_utils import compute_binary_thresholds
from aramina.training_model import _fit_patient_model_input

from recalibrated_joint_data import (
    base_features,
    fit_feature_pair,
    full_chain_meta_pairs,
    input_metadata,
    load_input_dataframe,
    lr1_cross_fitted_features,
    model_columns,
    patient_ids,
    raw_patient_subset,
    split_pairs,
)
from recalibrated_joint_selection import (
    ABLATIONS,
    DEFAULT_C_GRID,
    fit_joint_model,
    score_current_pairs,
    select_ablation_regularization,
)
from recalibrated_joint_output import (
    paired_deltas,
    summarize,
    summary_payload,
    threshold_score_frame,
    write_outputs,
)


EXPERIMENT_NAME = "recalibrated_joint_profile_age_symmetry"
CURRENT_EXACT_NAME = "current_product_exact_legacy"
CURRENT_OOF_NAME = "current_architecture_oof_retrained"
JOINT_MODEL_NAME = "recalibrated_joint_additive"
DEFAULT_OUTER_SPLITS = 5
DEFAULT_OUTER_REPEATS = 10
DEFAULT_INNER_LR1_SPLITS = 5
DEFAULT_META_SPLITS = 4
DEFAULT_RANDOM_STATE = 42
TARGET_SENSITIVITY = 0.95
DEFAULT_LR1_C = 0.1
DEFAULT_CURRENT_LR2_C = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-joblib", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--outer-splits", type=int, default=DEFAULT_OUTER_SPLITS)
    parser.add_argument("--outer-repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    parser.add_argument("--inner-lr1-splits", type=int, default=DEFAULT_INNER_LR1_SPLITS)
    parser.add_argument("--meta-splits", type=int, default=DEFAULT_META_SPLITS)
    parser.add_argument("--candidate-c", nargs="+", type=float, default=DEFAULT_C_GRID)
    parser.add_argument("--lr1-c", type=float, default=DEFAULT_LR1_C)
    parser.add_argument("--current-lr2-c", type=float, default=DEFAULT_CURRENT_LR2_C)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def run_experiment(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    outer_splits: int = DEFAULT_OUTER_SPLITS,
    outer_repeats: int = DEFAULT_OUTER_REPEATS,
    inner_lr1_splits: int = DEFAULT_INNER_LR1_SPLITS,
    meta_splits: int = DEFAULT_META_SPLITS,
    candidate_c: Sequence[float] = DEFAULT_C_GRID,
    lr1_c: float = DEFAULT_LR1_C,
    current_lr2_c: float = DEFAULT_CURRENT_LR2_C,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Run strict nested outer evaluation and descriptive train-all fits."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    columns = model_columns()
    base = base_features(dataframe, columns)
    metadata = _metadata(input_path, repository)
    outer_pairs = split_pairs(
        base,
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=random_state,
    )
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_selection: list[pd.DataFrame] = []
    all_manifests: list[pd.DataFrame] = []
    all_threshold_scores: list[pd.DataFrame] = []
    for split_id, (train_index, test_index) in enumerate(outer_pairs):
        result = _evaluate_outer_split(
            dataframe=dataframe,
            columns=columns,
            base=base,
            train_index=train_index,
            test_index=test_index,
            split_id=split_id,
            candidate_c=candidate_c,
            lr1_c=lr1_c,
            current_lr2_c=current_lr2_c,
            inner_lr1_splits=inner_lr1_splits,
            meta_splits=meta_splits,
            random_state=random_state + split_id * 10_000,
        )
        all_metrics.extend(result["metrics"])
        all_predictions.extend(result["predictions"])
        all_selection.append(result["selection"])
        all_manifests.append(result["manifest"])
        all_threshold_scores.extend(result["threshold_scores"])
    fold_metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    selection = pd.concat(all_selection, ignore_index=True)
    manifest = pd.concat(all_manifests, ignore_index=True)
    summary = summarize(fold_metrics)
    deltas = paired_deltas(fold_metrics, reference_model=CURRENT_EXACT_NAME)
    train_all, train_selection, train_manifest, train_threshold_scores = _train_all(
        dataframe,
        columns,
        candidate_c=candidate_c,
        lr1_c=lr1_c,
        current_lr2_c=current_lr2_c,
        inner_lr1_splits=inner_lr1_splits,
        meta_splits=meta_splits,
        random_state=random_state + 9_000_000,
    )
    selection = pd.concat([selection, train_selection], ignore_index=True)
    manifest = pd.concat([manifest, train_manifest], ignore_index=True)
    threshold_scores = pd.concat(
        [*all_threshold_scores, train_threshold_scores], ignore_index=True
    )
    write_outputs(
        output,
        fold_metrics=fold_metrics,
        predictions=predictions,
        summary=summary,
        selection=selection,
        manifest=manifest,
        deltas=deltas,
        train_all=train_all,
        threshold_scores=threshold_scores,
    )
    payload = summary_payload(
        experiment_name=EXPERIMENT_NAME,
        metadata=metadata,
        dataframe=dataframe,
        columns=columns,
        base=base,
        controls={
            "outer_evaluation": "repeated_stratified_patient_safe_kfold",
            "outer_splits": int(outer_splits),
            "outer_repeats": int(outer_repeats),
            "inner_lr1_crossfit": "strictly_nested_within_each_meta_train_fold",
            "inner_lr1_splits": int(inner_lr1_splits),
            "meta_splits": int(meta_splits),
            "random_state": int(random_state),
            "lr1_c": float(lr1_c),
            "current_lr2_c": float(current_lr2_c),
            "candidate_c": [float(value) for value in candidate_c],
            "target_sensitivity": TARGET_SENSITIVITY,
            "primary_fit": "ordinary_unweighted_logistic_likelihood",
            "availability_flags": "gates_only_not_learned_predictors",
            "fold_std_interpretation": "descriptive_fold_variability_not_confidence_interval",
            "patient_cluster_confidence_interval": "pending_not_implemented",
        },
        summary=summary,
        train_all=train_all,
    )
    (output / "summary.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return payload


def _evaluate_outer_split(**kwargs: Any) -> dict[str, Any]:
    frame: pd.DataFrame = kwargs["dataframe"]
    columns: dict[str, str] = kwargs["columns"]
    base: pd.DataFrame = kwargs["base"]
    split_id = int(kwargs["split_id"])
    random_state = int(kwargs["random_state"])
    train_ids = patient_ids(base, kwargs["train_index"])
    test_ids = patient_ids(base, kwargs["test_index"])
    if train_ids.intersection(test_ids):
        raise RuntimeError("Patient leakage in outer split.")
    train_raw = raw_patient_subset(frame, columns, train_ids)
    test_raw = raw_patient_subset(frame, columns, test_ids)
    exact_train, exact_test = fit_feature_pair(
        train_raw,
        test_raw,
        columns,
        lr1_c=kwargs["lr1_c"],
        random_state=random_state,
    )
    outer_oof = lr1_cross_fitted_features(
        train_raw,
        columns,
        lr1_c=kwargs["lr1_c"],
        n_splits=kwargs["inner_lr1_splits"],
        random_state=random_state + 100,
        context={"outer_split_id": split_id, "meta_fold_id": "outer_train_full"},
    )
    meta_pairs = full_chain_meta_pairs(
        train_raw,
        columns,
        lr1_c=kwargs["lr1_c"],
        meta_splits=kwargs["meta_splits"],
        random_state=random_state + 1_000,
        outer_split_id=split_id,
        inner_lr1_splits=kwargs["inner_lr1_splits"],
    )
    if set(outer_oof.features["patientId"].astype(str)).intersection(test_ids):
        raise RuntimeError("Outer test patient entered outer-train OOF features.")
    manifest = _outer_manifest(split_id, train_ids, test_ids)
    manifest = pd.concat(
        [manifest, outer_oof.manifest, *(pair.manifest for pair in meta_pairs)],
        ignore_index=True,
    )
    exact_metrics, exact_predictions, exact_threshold_scores = _exact_current_comparator(
        split_id=split_id,
        train_features=exact_train,
        test_features=exact_test,
        lr1_c=kwargs["lr1_c"],
        current_lr2_c=kwargs["current_lr2_c"],
        random_state=random_state + 2_000,
    )
    oof_metrics, oof_predictions, oof_threshold_scores = _oof_current_comparator(
        split_id=split_id,
        oof_features=outer_oof.features,
        test_features=exact_test,
        meta_pairs=meta_pairs,
        lr1_c=kwargs["lr1_c"],
        current_lr2_c=kwargs["current_lr2_c"],
        random_state=random_state + 3_000,
    )
    metrics = [exact_metrics, oof_metrics]
    predictions = [exact_predictions, oof_predictions]
    threshold_score_frames = [exact_threshold_scores, oof_threshold_scores]
    selection_rows: list[pd.DataFrame] = []
    for offset, ablation in enumerate(ABLATIONS):
        parameters, selection, threshold_scores = select_ablation_regularization(
            meta_pairs,
            ablation=ablation,
            candidate_c=kwargs["candidate_c"],
            random_state=random_state + 4_000 + offset * 1_000,
        )
        selection["outer_split_id"] = split_id
        selection["selection_scope"] = "outer_train_nested_full_chain"
        selection_rows.append(selection)
        model = fit_joint_model(
            outer_oof.features,
            parameters=parameters,
            ablation=ablation,
            random_state=random_state + 8_000 + offset,
        )
        test_score = model.predict_proba(exact_test)[:, 1]
        row, prediction, threshold_score_frame = _oof_metric_and_prediction(
            model_name=JOINT_MODEL_NAME,
            ablation=ablation,
            split_id=split_id,
            threshold_scores=threshold_scores,
            test_features=exact_test,
            test_score=test_score,
            parameters={"lr1_c": float(kwargs["lr1_c"]), **parameters},
            threshold_provenance="outer_train_full_chain_lr1_oof_meta_oof_scores",
            threshold_score_kind="nested_full_chain_oof_scores",
        )
        metrics.append(row)
        predictions.append(prediction)
        threshold_score_frames.append(threshold_score_frame)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "selection": pd.concat(selection_rows, ignore_index=True),
        "manifest": manifest,
        "threshold_scores": threshold_score_frames,
    }


def _exact_current_comparator(
    **kwargs: Any,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train = kwargs["train_features"]
    test = kwargs["test_features"]
    model = GatedSymmetryLogistic(
        logreg_c=kwargs["current_lr2_c"], random_state=kwargs["random_state"]
    ).fit(train, train["label"].to_numpy(dtype=int))
    threshold_scores = _score_frame(train, model.predict_proba(train)[:, 1])
    test_score = model.predict_proba(test)[:, 1]
    return _oof_metric_and_prediction(
        model_name=CURRENT_EXACT_NAME,
        ablation="current_product_exact_legacy",
        split_id=kwargs["split_id"],
        threshold_scores=threshold_scores,
        test_features=test,
        test_score=test_score,
        parameters={
            "lr1_c": float(kwargs["lr1_c"]),
            "current_lr2_c": float(kwargs["current_lr2_c"]),
        },
        threshold_provenance="legacy_outer_train_fitted_lr1_fitted_lr2_scores",
        threshold_score_kind="legacy_fitted_outer_train_scores",
    )


def _oof_current_comparator(
    **kwargs: Any,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    oof = kwargs["oof_features"]
    model = GatedSymmetryLogistic(
        logreg_c=kwargs["current_lr2_c"], random_state=kwargs["random_state"]
    ).fit(oof, oof["label"].to_numpy(dtype=int))
    threshold_scores = score_current_pairs(
        kwargs["meta_pairs"],
        logreg_c=kwargs["current_lr2_c"],
        random_state=kwargs["random_state"] + 100,
    )
    test_score = model.predict_proba(kwargs["test_features"])[:, 1]
    return _oof_metric_and_prediction(
        model_name=CURRENT_OOF_NAME,
        ablation="current_architecture_oof_retrained",
        split_id=kwargs["split_id"],
        threshold_scores=threshold_scores,
        test_features=kwargs["test_features"],
        test_score=test_score,
        parameters={
            "lr1_c": float(kwargs["lr1_c"]),
            "current_lr2_c": float(kwargs["current_lr2_c"]),
        },
        threshold_provenance="outer_train_full_chain_lr1_oof_current_lr2_oof_scores",
        threshold_score_kind="nested_full_chain_oof_scores",
    )


def _oof_metric_and_prediction(
    **kwargs: Any,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    threshold_scores: pd.DataFrame = kwargs["threshold_scores"]
    y_train = threshold_scores["label"].to_numpy(dtype=int)
    train_score = threshold_scores["p_cancer"].to_numpy(dtype=float)
    threshold = compute_binary_thresholds(
        y_train, train_score, target_sensitivity=TARGET_SENSITIVITY
    )
    test = kwargs["test_features"]
    test_score = np.asarray(kwargs["test_score"], dtype=float)
    values = binary_metric_values(
        test["label"].to_numpy(dtype=int),
        test_score,
        np.full(len(test_score), threshold["threshold_target"]),
    )
    confusion = _confusion_counts(
        test["label"].to_numpy(dtype=int), test_score, float(threshold["threshold_target"])
    )
    row = {
        "model_name": kwargs["model_name"],
        "ablation": kwargs["ablation"],
        "split_id": int(kwargs["split_id"]),
        "threshold_provenance": kwargs["threshold_provenance"],
        "threshold_sample_count": int(len(threshold_scores)),
        "threshold_cancer_cases": int((y_train == 1).sum()),
        "threshold_benign_cases": int((y_train == 0).sum()),
        "test_patients": int(test["patientId"].nunique()),
        "test_target_cases": int(len(test)),
        "test_cancer_cases": int((test["label"] == 1).sum()),
        "test_benign_cases": int((test["label"] == 0).sum()),
        **kwargs["parameters"],
        **threshold,
        **values,
        **confusion,
    }
    prediction = test[["target_case_id", "patientId", "label", "label_name"]].copy()
    prediction["model_name"] = kwargs["model_name"]
    prediction["ablation"] = kwargs["ablation"]
    prediction["split_id"] = int(kwargs["split_id"])
    prediction["p_cancer"] = test_score
    prediction["threshold_target"] = float(threshold["threshold_target"])
    prediction["threshold_provenance"] = kwargs["threshold_provenance"]
    prediction["y_pred_target"] = (test_score >= float(threshold["threshold_target"])).astype(int)
    threshold_frame = threshold_score_frame(
        threshold_scores,
        outer_split_id=kwargs["split_id"],
        model_name=kwargs["model_name"],
        ablation=kwargs["ablation"],
        threshold_target=float(threshold["threshold_target"]),
        threshold_provenance=kwargs["threshold_provenance"],
        threshold_score_kind=kwargs["threshold_score_kind"],
        parameters=kwargs["parameters"],
    )
    return row, prediction, threshold_frame


def _train_all(
    frame: pd.DataFrame,
    columns: dict[str, str],
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_features, _ = _fit_patient_model_input(
        frame,
        profile_column=columns["profile_column"],
        label_column=columns["label_column"],
        group_column=columns["group_column"],
        specimen_column=columns["specimen_column"],
        side_column=columns["side_column"],
        q_column=columns["q_column"],
        age_column=columns["age_column"],
        biopsy_column=columns["biopsy_column"],
        lr1_row_policy=columns["lr1_row_policy"],
        lr1_logreg_c=kwargs["lr1_c"],
        random_state=kwargs["random_state"],
    )
    exact = GatedSymmetryLogistic(
        logreg_c=kwargs["current_lr2_c"], random_state=kwargs["random_state"]
    ).fit(full_features, full_features["label"].to_numpy(dtype=int))
    exact_score = exact.predict_proba(full_features)[:, 1]
    exact_threshold = compute_binary_thresholds(
        full_features["label"].to_numpy(dtype=int),
        exact_score,
        target_sensitivity=TARGET_SENSITIVITY,
    )
    rows = [
        {
            "model_name": CURRENT_EXACT_NAME,
            "ablation": "current_product_exact_legacy",
            "description_status": "training_cohort_current_product_exact_not_independent",
            "threshold_provenance": "training_cohort_fitted_lr1_fitted_lr2_scores",
            "threshold_sample_count": int(len(full_features)),
            "threshold_cancer_cases": int((full_features["label"] == 1).sum()),
            "threshold_benign_cases": int((full_features["label"] == 0).sum()),
            "lr1_c": float(kwargs["lr1_c"]),
            "current_lr2_c": float(kwargs["current_lr2_c"]),
            **final_fit_training_metrics(
                full_features["label"].to_numpy(dtype=int),
                exact_score,
                threshold=float(exact_threshold["threshold_target"]),
            ),
            **exact_threshold,
        }
    ]
    threshold_score_frames = [
        threshold_score_frame(
            _score_frame(full_features, exact_score),
            outer_split_id="train_all",
            model_name=CURRENT_EXACT_NAME,
            ablation="current_product_exact_legacy",
            threshold_target=float(exact_threshold["threshold_target"]),
            threshold_provenance="training_cohort_fitted_lr1_fitted_lr2_scores",
            threshold_score_kind="training_cohort_fitted_scores",
            parameters={
                "lr1_c": float(kwargs["lr1_c"]),
                "current_lr2_c": float(kwargs["current_lr2_c"]),
            },
        )
    ]
    all_oof = lr1_cross_fitted_features(
        frame,
        columns,
        lr1_c=kwargs["lr1_c"],
        n_splits=kwargs["inner_lr1_splits"],
        random_state=kwargs["random_state"] + 10,
        context={"outer_split_id": "train_all", "meta_fold_id": "full_train"},
    )
    meta_pairs = full_chain_meta_pairs(
        frame,
        columns,
        lr1_c=kwargs["lr1_c"],
        meta_splits=kwargs["meta_splits"],
        random_state=kwargs["random_state"] + 20,
        outer_split_id="train_all",
        inner_lr1_splits=kwargs["inner_lr1_splits"],
    )
    selection_rows: list[pd.DataFrame] = []
    for offset, ablation in enumerate(ABLATIONS):
        parameters, selection, threshold_scores = select_ablation_regularization(
            meta_pairs,
            ablation=ablation,
            candidate_c=kwargs["candidate_c"],
            random_state=kwargs["random_state"] + 100 + offset * 1_000,
        )
        selection["outer_split_id"] = "train_all"
        selection["selection_scope"] = "train_all_nested_full_chain"
        selection_rows.append(selection)
        model = fit_joint_model(
            all_oof.features,
            parameters=parameters,
            ablation=ablation,
            random_state=kwargs["random_state"] + 5_000 + offset,
        )
        score = model.predict_proba(full_features)[:, 1]
        y_oof = threshold_scores["label"].to_numpy(dtype=int)
        threshold = compute_binary_thresholds(
            y_oof,
            threshold_scores["p_cancer"].to_numpy(dtype=float),
            target_sensitivity=TARGET_SENSITIVITY,
        )
        rows.append(
            {
                "model_name": JOINT_MODEL_NAME,
                "ablation": ablation,
                "description_status": "training_cohort_deployed_chain_not_independent",
                "threshold_provenance": "training_cohort_full_chain_lr1_oof_meta_oof_scores",
                "threshold_sample_count": int(len(threshold_scores)),
                "threshold_cancer_cases": int((y_oof == 1).sum()),
                "threshold_benign_cases": int((y_oof == 0).sum()),
                "lr1_c": float(kwargs["lr1_c"]),
                **parameters,
                **final_fit_training_metrics(
                    full_features["label"].to_numpy(dtype=int),
                    score,
                    threshold=float(threshold["threshold_target"]),
                ),
                **threshold,
            }
        )
        threshold_score_frames.append(
            threshold_score_frame(
                threshold_scores,
                outer_split_id="train_all",
                model_name=JOINT_MODEL_NAME,
                ablation=ablation,
                threshold_target=float(threshold["threshold_target"]),
                threshold_provenance="training_cohort_full_chain_lr1_oof_meta_oof_scores",
                threshold_score_kind="training_cohort_nested_full_chain_oof_scores",
                parameters={"lr1_c": float(kwargs["lr1_c"]), **parameters},
            )
        )
    manifest = pd.concat(
        [all_oof.manifest, *(pair.manifest for pair in meta_pairs)], ignore_index=True
    )
    return (
        pd.DataFrame(rows),
        pd.concat(selection_rows, ignore_index=True),
        manifest,
        pd.concat(threshold_score_frames, ignore_index=True),
    )


def _metadata(input_path: str | Path | None, repository: Path) -> dict[str, str]:
    if input_path is None:
        from recalibrated_joint_data import _git_sha

        return {
            "input_joblib_path": "not_provided",
            "input_joblib_sha256": "not_provided",
            "git_sha": _git_sha(repository),
            "git_worktree_dirty": "not_provided",
        }
    return input_metadata(input_path, repository=repository)


def _outer_manifest(split_id: int, train_ids: set[str], test_ids: set[str]) -> pd.DataFrame:
    rows = [
        {"outer_split_id": split_id, "meta_fold_id": "not_applicable", "level": "outer", "fold_id": split_id, "role": "outer_train", "patient_id": patient}
        for patient in sorted(train_ids)
    ] + [
        {"outer_split_id": split_id, "meta_fold_id": "not_applicable", "level": "outer", "fold_id": split_id, "role": "outer_test", "patient_id": patient}
        for patient in sorted(test_ids)
    ]
    return pd.DataFrame(rows)


def _score_frame(features: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    out = features[["target_case_id", "patientId", "label", "label_name"]].copy()
    out["p_cancer"] = np.asarray(score, dtype=float)
    return out


def _confusion_counts(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, int]:
    prediction = np.asarray(score >= threshold, dtype=int)
    return {
        "true_positives": int(((y == 1) & (prediction == 1)).sum()),
        "true_negatives": int(((y == 0) & (prediction == 0)).sum()),
        "false_positives": int(((y == 0) & (prediction == 1)).sum()),
        "false_negatives": int(((y == 1) & (prediction == 0)).sum()),
    }


def main() -> None:
    args = parse_args()
    payload = run_experiment(
        load_input_dataframe(args.input_joblib),
        args.output_dir,
        input_path=args.input_joblib,
        outer_splits=args.outer_splits,
        outer_repeats=args.outer_repeats,
        inner_lr1_splits=args.inner_lr1_splits,
        meta_splits=args.meta_splits,
        candidate_c=args.candidate_c,
        lr1_c=args.lr1_c,
        current_lr2_c=args.current_lr2_c,
        random_state=args.random_state,
    )
    print(yaml.safe_dump(payload["held_out_summary"], sort_keys=False))


if __name__ == "__main__":
    main()
