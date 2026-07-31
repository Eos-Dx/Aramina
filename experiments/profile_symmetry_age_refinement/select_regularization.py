"""Patient-safe sequential regularization selection for the staged model.

This research runner selects one frozen regularization value for each stage
before a single train-all staged model description is written. It does not
modify the Aramina product model or its model artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aramina.model_metrics import final_fit_training_metrics
from aramina.model_utils import compute_binary_thresholds
from aramina.training_model import _fit_patient_model_input

from run_experiment import (
    DEFAULT_FOLDS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_REPEATS,
    STAGED_MODEL_NAME,
    TARGET_SENSITIVITY,
    _base_feature_table,
    _fit_feature_tables,
    _load_staged_classifier_class,
    _metric_record,
    _model_columns,
    _staged_scores,
    load_input_dataframe,
    run_experiment,
)
from staged_model import PROFILE_PROBABILITY_COLUMN


EXPERIMENT_NAME = "profile_symmetry_age_regularization_selection"
DEFAULT_CANDIDATE_C = (0.03, 0.1, 0.3, 1.0)
SELECTION_METRIC = "mean_held_out_log_loss"


def parse_args() -> argparse.Namespace:
    """Parse the small, reproducible selection runner interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-joblib", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _candidate_c(values: Sequence[float]) -> tuple[float, ...]:
    """Validate a concise ordered C grid."""
    candidate_c = tuple(sorted({float(value) for value in values}))
    if not candidate_c or any(not np.isfinite(value) or value <= 0.0 for value in candidate_c):
        raise ValueError("candidate_c must contain positive finite values.")
    return candidate_c


def _split_features(
    dataframe: pd.DataFrame,
    *,
    columns: dict[str, str],
    base_features: pd.DataFrame,
    split_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    lr1_c: float,
    random_state: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Fit LR1 only on each training fold and return case-level fold tables."""
    rows: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for split_id, (train_index, test_index) in enumerate(split_pairs):
        train_patients = set(base_features.iloc[train_index]["patientId"].astype(str))
        test_patients = set(base_features.iloc[test_index]["patientId"].astype(str))
        if train_patients.intersection(test_patients):
            raise RuntimeError("Patient leakage detected before model fitting.")
        train_df = dataframe[
            dataframe[columns["group_column"]].astype(str).isin(train_patients)
        ].copy()
        test_df = dataframe[
            dataframe[columns["group_column"]].astype(str).isin(test_patients)
        ].copy()
        train_features, test_features = _fit_feature_tables(
            train_df,
            test_df,
            columns,
            lr1_c=lr1_c,
            random_state=random_state + split_id,
        )
        rows.append((split_id, train_features, test_features))
    return rows


def _metric_rows(
    feature_rows: Sequence[tuple[int, pd.DataFrame, pd.DataFrame]],
    *,
    stage: str,
    candidate_c: float,
    symmetry_c: float | None,
    age_c: float | None,
    random_state: int,
) -> pd.DataFrame:
    """Score one candidate C at the requested stage on fixed patient folds."""
    records: list[dict[str, Any]] = []
    staged_class = _load_staged_classifier_class()
    for split_id, train_features, test_features in feature_rows:
        if stage == "profile_p_cancer":
            train_score = train_features[PROFILE_PROBABILITY_COLUMN].to_numpy(dtype=float)
            test_score = test_features[PROFILE_PROBABILITY_COLUMN].to_numpy(dtype=float)
        else:
            if symmetry_c is None or age_c is None:
                raise ValueError("Correction-stage scoring requires symmetry_c and age_c.")
            model = staged_class(
                symmetry_c=symmetry_c,
                age_c=age_c,
                random_state=random_state + split_id,
            ).fit(train_features, train_features["label"].to_numpy(dtype=int))
            train_score = _staged_scores(model, train_features)[stage].to_numpy(dtype=float)
            test_score = _staged_scores(model, test_features)[stage].to_numpy(dtype=float)
        record, _ = _metric_record(
            model_name=STAGED_MODEL_NAME,
            stage=stage,
            split_id=split_id,
            train_features=train_features,
            test_features=test_features,
            train_score=train_score,
            test_score=test_score,
            target_sensitivity=TARGET_SENSITIVITY,
        )
        record["candidate_c"] = float(candidate_c)
        records.append(record)
    return pd.DataFrame(records)


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise candidates using a predeclared probability-first rule."""
    summary = (
        rows.groupby(["stage", "candidate_c"], as_index=False)
        .agg(
            splits=("split_id", "size"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_score_mean=("brier_score", "mean"),
            brier_score_std=("brier_score", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
        )
        .sort_values(
            [
                "stage",
                "log_loss_mean",
                "brier_score_mean",
                "roc_auc_mean",
                "specificity_mean",
                "candidate_c",
            ],
            ascending=[True, True, True, False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return summary


def _selected_c(summary: pd.DataFrame, stage: str) -> float:
    """Return the top candidate under the documented ordering."""
    stage_rows = summary.loc[summary["stage"].eq(stage)]
    if stage_rows.empty:
        raise ValueError(f"No regularization summary for stage {stage!r}.")
    return float(stage_rows.iloc[0]["candidate_c"])


def _train_all_metrics(
    dataframe: pd.DataFrame,
    *,
    columns: dict[str, str],
    lr1_c: float,
    symmetry_c: float,
    age_c: float,
    random_state: int,
) -> dict[str, Any]:
    """Fit the selected staged model on all accepted cases and describe it."""
    features, _ = _fit_patient_model_input(
        dataframe,
        profile_column=columns["profile_column"],
        label_column=columns["label_column"],
        group_column=columns["group_column"],
        specimen_column=columns["specimen_column"],
        side_column=columns["side_column"],
        q_column=columns["q_column"],
        age_column=columns["age_column"],
        biopsy_column=columns["biopsy_column"],
        lr1_row_policy=columns["lr1_row_policy"],
        lr1_logreg_c=lr1_c,
        random_state=random_state,
    )
    y = features["label"].to_numpy(dtype=int)
    staged_class = _load_staged_classifier_class()
    model = staged_class(
        symmetry_c=symmetry_c,
        age_c=age_c,
        random_state=random_state,
    ).fit(features, y)
    score = _staged_scores(model, features)["final_p_cancer"].to_numpy(dtype=float)
    thresholds = compute_binary_thresholds(
        y,
        score,
        target_sensitivity=TARGET_SENSITIVITY,
    )
    return {
        "evaluation_status": "in_sample_not_independent",
        "target_cases": int(len(features)),
        "cancer_target_cases": int((y == 1).sum()),
        "benign_target_cases": int((y == 0).sum()),
        "selected_regularization": {
            "lr1_c": float(lr1_c),
            "symmetry_c": float(symmetry_c),
            "age_c": float(age_c),
        },
        **final_fit_training_metrics(
            y,
            score,
            threshold=float(thresholds["threshold_target"]),
        ),
        **thresholds,
    }


def run_regularization_selection(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    *,
    candidate_c: Sequence[float] = DEFAULT_CANDIDATE_C,
    n_splits: int = DEFAULT_FOLDS,
    n_repeats: int = DEFAULT_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Select C values sequentially, then describe the selected train-all fit."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_c = _candidate_c(candidate_c)
    columns = _model_columns()
    base_features = _base_feature_table(dataframe, columns)
    from aramina.training_evaluation import _patient_split_pairs

    split_pairs = _patient_split_pairs(
        mode="stratified_kfold",
        base_features=base_features,
        y_patients=base_features["label"].to_numpy(dtype=int),
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    profile_rows: list[pd.DataFrame] = []
    profile_cache: dict[float, list[tuple[int, pd.DataFrame, pd.DataFrame]]] = {}
    for lr1_c in candidate_c:
        print(f"Selecting LR1 C={lr1_c}", flush=True)
        feature_rows = _split_features(
            dataframe,
            columns=columns,
            base_features=base_features,
            split_pairs=split_pairs,
            lr1_c=lr1_c,
            random_state=random_state,
        )
        profile_cache[lr1_c] = feature_rows
        profile_rows.append(
            _metric_rows(
                feature_rows,
                stage="profile_p_cancer",
                candidate_c=lr1_c,
                symmetry_c=None,
                age_c=None,
                random_state=random_state,
            )
        )
    profile_metrics = pd.concat(profile_rows, ignore_index=True)
    profile_summary = _summary(profile_metrics)
    selected_lr1_c = _selected_c(profile_summary, "profile_p_cancer")
    selected_features = profile_cache[selected_lr1_c]

    symmetry_rows: list[pd.DataFrame] = []
    for symmetry_c in candidate_c:
        print(f"Selecting symmetry C={symmetry_c}", flush=True)
        symmetry_rows.append(
            _metric_rows(
                selected_features,
                stage="after_symmetry_p_cancer",
                candidate_c=symmetry_c,
                symmetry_c=symmetry_c,
                age_c=candidate_c[0],
                random_state=random_state,
            )
        )
    symmetry_metrics = pd.concat(symmetry_rows, ignore_index=True)
    symmetry_summary = _summary(symmetry_metrics)
    selected_symmetry_c = _selected_c(symmetry_summary, "after_symmetry_p_cancer")

    age_rows: list[pd.DataFrame] = []
    for age_c in candidate_c:
        print(f"Selecting age C={age_c}", flush=True)
        age_rows.append(
            _metric_rows(
                selected_features,
                stage="final_p_cancer",
                candidate_c=age_c,
                symmetry_c=selected_symmetry_c,
                age_c=age_c,
                random_state=random_state,
            )
        )
    age_metrics = pd.concat(age_rows, ignore_index=True)
    age_summary = _summary(age_metrics)
    selected_age_c = _selected_c(age_summary, "final_p_cancer")

    selection_metrics = pd.concat(
        [profile_metrics, symmetry_metrics, age_metrics],
        ignore_index=True,
    )
    selection_summary = pd.concat(
        [profile_summary, symmetry_summary, age_summary],
        ignore_index=True,
    )
    selected = {
        "lr1_c": selected_lr1_c,
        "symmetry_c": selected_symmetry_c,
        "age_c": selected_age_c,
    }
    train_all = _train_all_metrics(
        dataframe,
        columns=columns,
        random_state=random_state,
        **selected,
    )
    print("Running selected configuration across patient-safe folds", flush=True)
    selected_run = run_experiment(
        dataframe,
        output / "selected_configuration",
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        lr1_c=selected_lr1_c,
        symmetry_c=selected_symmetry_c,
        age_c=selected_age_c,
    )
    selection_metrics.to_csv(output / "regularization_selection_metrics.csv", index=False)
    selection_summary.to_csv(output / "regularization_selection_summary.csv", index=False)
    (output / "selected_train_all_metrics.yaml").write_text(
        yaml.safe_dump(train_all, sort_keys=False),
        encoding="utf-8",
    )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "status": "research_only_not_product_compatible",
        "selection_protocol": {
            "evaluation": "repeated_stratified_5fold_x20_patient_safe",
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "candidate_c": list(candidate_c),
            "selection_metric": SELECTION_METRIC,
            "tie_breakers": [
                "lower_brier_score",
                "higher_roc_auc",
                "higher_specificity_at_train_selected_threshold",
                "smaller_c",
            ],
            "selection_order": ["lr1", "symmetry", "age"],
            "threshold_selection": "train_fold_only",
        },
        "selected_regularization": selected,
        "train_all_metrics": train_all,
        "selected_configuration_reused_fold_summary": selected_run["held_out_summary"],
        "limitations": [
            "Selection folds determine the selected C values.",
            "The selected-configuration fold summary is not an independent validation estimate.",
            "Use a nested selection protocol or independent cohort before product selection.",
        ],
        "outputs": {
            "selection_metrics": "regularization_selection_metrics.csv",
            "selection_summary": "regularization_selection_summary.csv",
            "selected_train_all_metrics": "selected_train_all_metrics.yaml",
            "selected_configuration": "selected_configuration/summary.yaml",
        },
    }
    (output / "regularization_selection.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = run_regularization_selection(
        load_input_dataframe(args.input_joblib),
        args.output_dir,
    )
    print(yaml.safe_dump(payload["selected_regularization"], sort_keys=False))


if __name__ == "__main__":
    main()
