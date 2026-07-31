"""Research-only patient-safe comparison of current and staged classifiers.

The staged implementation is intentionally owned by ``staged_model.py``.  This
runner owns only experiment orchestration, validation, and serialised outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from aramina.m2q_model import GatedSymmetryLogistic
from aramina.model_metrics import binary_metric_values, final_fit_training_metrics
from aramina.model_utils import compute_binary_thresholds
from aramina.patient_features import empty_lr1_scores, patient_feature_table
from aramina.training_config import PRODUCT_MODEL_NAME, resolve_model_definition
from aramina.training_evaluation import _fit_split_feature_tables, _patient_split_pairs
from aramina.training_model import _fit_patient_model_input


EXPERIMENT_NAME = "profile_symmetry_age_refinement"
CURRENT_MODEL_NAME = "current_gated_symmetry_logistic"
STAGED_MODEL_NAME = "staged_profile_symmetry_age"
STAGED_SCORES = (
    "profile_p_cancer",
    "after_symmetry_p_cancer",
    "final_p_cancer",
)
DEFAULT_FOLDS = 5
DEFAULT_REPEATS = 20
DEFAULT_RANDOM_STATE = 42
TARGET_SENSITIVITY = 0.95
DEFAULT_LR1_C = 0.1
DEFAULT_CURRENT_LR2_C = 0.3
DEFAULT_SYMMETRY_C = 0.3
DEFAULT_AGE_C = 0.3


def parse_args() -> argparse.Namespace:
    """Parse the deliberately small research runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-joblib", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_input_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a preprocessing artifact or a DataFrame supplied for research."""
    value = joblib.load(Path(path))
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict) and isinstance(value.get("dataframe"), pd.DataFrame):
        return value["dataframe"].copy()
    raise ValueError(
        "--input-joblib must contain a pandas DataFrame or a preprocessing artifact "
        "with a 'dataframe' field."
    )


def _load_staged_classifier_class() -> type[Any]:
    """Load the experiment-owned staged classifier without touching product code."""
    source = Path(__file__).with_name("staged_model.py")
    if not source.exists():
        raise FileNotFoundError(
            "Missing staged research model: "
            f"{source}. Add StagedProfileSymmetryAgeClassifier before running."
        )
    spec = importlib.util.spec_from_file_location(
        "profile_symmetry_age_refinement_staged_model", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load staged research model: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.StagedProfileSymmetryAgeClassifier
    except AttributeError as exc:
        raise ImportError(
            "staged_model.py must define StagedProfileSymmetryAgeClassifier."
        ) from exc


def _model_columns() -> dict[str, str]:
    definition = resolve_model_definition(PRODUCT_MODEL_NAME)["model"]
    return {
        name: str(definition[name])
        for name in (
            "profile_column",
            "label_column",
            "group_column",
            "specimen_column",
            "side_column",
            "q_column",
            "age_column",
            "biopsy_column",
            "lr1_row_policy",
        )
    }


def _base_feature_table(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Build neutral-score cases solely to define patient-safe folds."""
    return patient_feature_table(
        df,
        empty_lr1_scores(
            df,
            group_column=columns["group_column"],
            side_column=columns["side_column"],
            label_column=columns["label_column"],
            biopsy_column=columns["biopsy_column"],
        ),
        profile_column=columns["profile_column"],
        label_column=columns["label_column"],
        group_column=columns["group_column"],
        specimen_column=columns["specimen_column"],
        side_column=columns["side_column"],
        q_column=columns["q_column"],
        age_column=columns["age_column"],
        biopsy_column=columns["biopsy_column"],
    )


def _fit_feature_tables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: dict[str, str],
    *,
    lr1_c: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _fit_split_feature_tables(
        train_df,
        test_df,
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


def _confusion_counts(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, int]:
    prediction = np.asarray(score >= threshold, dtype=int)
    return {
        "tp": int(((y == 1) & (prediction == 1)).sum()),
        "tn": int(((y == 0) & (prediction == 0)).sum()),
        "fp": int(((y == 0) & (prediction == 1)).sum()),
        "fn": int(((y == 1) & (prediction == 0)).sum()),
    }


def _metric_record(
    *,
    model_name: str,
    stage: str,
    split_id: int,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_score: np.ndarray,
    test_score: np.ndarray,
    target_sensitivity: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = train_features["label"].to_numpy(dtype=int)
    y_test = test_features["label"].to_numpy(dtype=int)
    thresholds = compute_binary_thresholds(
        y_train,
        train_score,
        target_sensitivity=target_sensitivity,
    )
    threshold = float(thresholds["threshold_target"])
    values = binary_metric_values(
        y_test,
        test_score,
        np.full(len(test_score), threshold, dtype=float),
    )
    return (
        {
            "model_name": model_name,
            "stage": stage,
            "split_id": int(split_id),
            "train_patients": int(train_features["patientId"].nunique()),
            "test_patients": int(test_features["patientId"].nunique()),
            "train_target_cases": int(len(train_features)),
            "test_target_cases": int(len(test_features)),
            "target_sensitivity": float(target_sensitivity),
            **thresholds,
            **values,
            **_confusion_counts(y_test, test_score, threshold),
        },
        thresholds,
    )


def _prediction_rows(
    *,
    model_name: str,
    stage: str,
    split_id: int,
    test_features: pd.DataFrame,
    score: np.ndarray,
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    out = test_features[["target_case_id", "patientId", "label", "label_name"]].copy()
    out["model_name"] = model_name
    out["stage"] = stage
    out["split_id"] = int(split_id)
    out["p_cancer"] = np.asarray(score, dtype=float)
    out["threshold_target"] = float(thresholds["threshold_target"])
    out["y_pred_target"] = (out["p_cancer"] >= out["threshold_target"]).astype(int)
    return out


def _staged_scores(model: Any, x: pd.DataFrame) -> pd.DataFrame:
    values = model.predict_stage_probabilities(x)
    missing = [column for column in STAGED_SCORES if column not in values.columns]
    if missing:
        raise ValueError(f"Staged model is missing stage probability columns: {missing}")
    out = values.loc[:, STAGED_SCORES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("Staged model returned non-finite stage probabilities.")
    if ((out < 0.0) | (out > 1.0)).any().any():
        raise ValueError("Staged model probabilities must be within [0, 1].")
    return out


def _stage_corrections(model: Any, x: pd.DataFrame) -> pd.DataFrame:
    """Validate and preserve staged correction evidence for split inspection."""
    values = model.stage_logit_corrections(x)
    if not isinstance(values, pd.DataFrame):
        raise TypeError("stage_logit_corrections(X) must return a pandas DataFrame.")
    if len(values) != len(x):
        raise ValueError("stage_logit_corrections(X) must return one row per input.")
    out = values.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("Staged model returned non-finite logit corrections.")
    return out.reset_index(drop=True).add_prefix("stage_")


def _run_split(
    *,
    split_id: int,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    current_lr2_c: float,
    symmetry_c: float,
    age_c: float,
    random_state: int,
    target_sensitivity: float,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    y_train = train_features["label"].to_numpy(dtype=int)
    current = GatedSymmetryLogistic(
        logreg_c=current_lr2_c,
        random_state=random_state,
    ).fit(
        train_features, y_train
    )
    staged_class = _load_staged_classifier_class()
    staged = staged_class(
        symmetry_c=symmetry_c,
        age_c=age_c,
        random_state=random_state,
    ).fit(train_features, y_train)

    scores: list[tuple[str, str, np.ndarray, np.ndarray]] = [
        (
            CURRENT_MODEL_NAME,
            "final",
            current.predict_proba(train_features)[:, 1],
            current.predict_proba(test_features)[:, 1],
        )
    ]
    train_staged = _staged_scores(staged, train_features)
    test_staged = _staged_scores(staged, test_features)
    test_corrections = _stage_corrections(staged, test_features)
    scores.extend(
        (
            STAGED_MODEL_NAME,
            stage,
            train_staged[stage].to_numpy(dtype=float),
            test_staged[stage].to_numpy(dtype=float),
        )
        for stage in STAGED_SCORES
    )

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for model_name, stage, train_score, test_score in scores:
        metric, thresholds = _metric_record(
            model_name=model_name,
            stage=stage,
            split_id=split_id,
            train_features=train_features,
            test_features=test_features,
            train_score=train_score,
            test_score=test_score,
            target_sensitivity=target_sensitivity,
        )
        metrics.append(metric)
        prediction = _prediction_rows(
            model_name=model_name,
            stage=stage,
            split_id=split_id,
            test_features=test_features,
            score=test_score,
            thresholds=thresholds,
        )
        if model_name == STAGED_MODEL_NAME:
            prediction = pd.concat(
                [prediction.reset_index(drop=True), test_corrections], axis=1
            )
        predictions.append(prediction)
    return metrics, predictions


def _summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "ppv",
        "npv",
        "tp",
        "tn",
        "fp",
        "fn",
    )
    rows: list[dict[str, Any]] = []
    for (model_name, stage), group in metrics.groupby(["model_name", "stage"], sort=False):
        row = {
            "model_name": model_name,
            "stage": stage,
            "splits": int(len(group)),
            "threshold_target_mean": float(group["threshold_target"].mean()),
            "threshold_target_std": float(group["threshold_target"].std(ddof=0)),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_train_all(
    df: pd.DataFrame,
    columns: dict[str, str],
    *,
    lr1_c: float,
    current_lr2_c: float,
    symmetry_c: float,
    age_c: float,
    random_state: int,
    target_sensitivity: float,
) -> pd.DataFrame:
    features, _ = _fit_patient_model_input(
        df,
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
    current = GatedSymmetryLogistic(
        logreg_c=current_lr2_c,
        random_state=random_state,
    ).fit(
        features, y
    )
    staged_class = _load_staged_classifier_class()
    staged = staged_class(
        symmetry_c=symmetry_c,
        age_c=age_c,
        random_state=random_state,
    ).fit(features, y)
    values: list[tuple[str, str, np.ndarray]] = [
        (CURRENT_MODEL_NAME, "final", current.predict_proba(features)[:, 1])
    ]
    staged_values = _staged_scores(staged, features)
    values.extend(
        (STAGED_MODEL_NAME, stage, staged_values[stage].to_numpy(dtype=float))
        for stage in STAGED_SCORES
    )
    rows: list[dict[str, Any]] = []
    for model_name, stage, score in values:
        thresholds = compute_binary_thresholds(
            y, score, target_sensitivity=target_sensitivity
        )
        rows.append(
            {
                "model_name": model_name,
                "stage": stage,
                **final_fit_training_metrics(
                    y,
                    score,
                    threshold=float(thresholds["threshold_target"]),
                ),
                **thresholds,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    *,
    n_splits: int = DEFAULT_FOLDS,
    n_repeats: int = DEFAULT_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
    target_sensitivity: float = TARGET_SENSITIVITY,
    lr1_c: float = DEFAULT_LR1_C,
    current_lr2_c: float = DEFAULT_CURRENT_LR2_C,
    symmetry_c: float = DEFAULT_SYMMETRY_C,
    age_c: float = DEFAULT_AGE_C,
) -> dict[str, Any]:
    """Compare current and staged research models on identical patient splits."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    columns = _model_columns()
    base_features = _base_feature_table(dataframe, columns)
    split_pairs = _patient_split_pairs(
        mode="stratified_kfold",
        base_features=base_features,
        y_patients=base_features["label"].to_numpy(dtype=int),
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
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
        metrics, predictions = _run_split(
            split_id=split_id,
            train_features=train_features,
            test_features=test_features,
            current_lr2_c=current_lr2_c,
            symmetry_c=symmetry_c,
            age_c=age_c,
            random_state=random_state + split_id,
            target_sensitivity=target_sensitivity,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    fold_metrics = pd.DataFrame(all_metrics)
    split_predictions = pd.concat(all_predictions, ignore_index=True)
    summary = _summary_table(fold_metrics)
    train_all = _fit_train_all(
        dataframe,
        columns,
        lr1_c=lr1_c,
        current_lr2_c=current_lr2_c,
        symmetry_c=symmetry_c,
        age_c=age_c,
        random_state=random_state,
        target_sensitivity=target_sensitivity,
    )
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    split_predictions.to_csv(output / "split_predictions.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    train_all.to_csv(output / "train_all_metrics.csv", index=False)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "status": "research_only_not_product_compatible",
        "architecture": "profile_to_optional_symmetry_to_optional_age",
        "controls": {
            "evaluation": "repeated_stratified_5fold_x20_patient_safe",
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "random_state": int(random_state),
            "target_sensitivity": float(target_sensitivity),
            "lr1_logreg_c": float(lr1_c),
            "current_lr2_c": float(current_lr2_c),
            "symmetry_c": float(symmetry_c),
            "age_c": float(age_c),
            "threshold_selection": "train_fold_only",
        },
        "cohort": {
            "measurements": int(len(dataframe)),
            "patients": int(dataframe[columns["group_column"]].astype(str).nunique()),
            "target_cases": int(len(base_features)),
            "cancer_target_cases": int((base_features["label"] == 1).sum()),
            "benign_target_cases": int((base_features["label"] == 0).sum()),
        },
        "held_out_summary": summary.to_dict(orient="records"),
        "train_all_metrics": train_all.to_dict(orient="records"),
        "outputs": {
            "fold_metrics": "fold_metrics.csv",
            "split_predictions": "split_predictions.csv",
            "summary": "summary.csv",
            "train_all_metrics": "train_all_metrics.csv",
        },
    }
    (output / "summary.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = run_experiment(load_input_dataframe(args.input_joblib), args.output_dir)
    print(yaml.safe_dump(payload["held_out_summary"], sort_keys=False))


if __name__ == "__main__":
    main()
