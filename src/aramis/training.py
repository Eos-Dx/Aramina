"""Training entrypoints for Aramis research-draft models."""

from __future__ import annotations

import subprocess
import tomllib
from collections.abc import Sequence
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xrd_preprocessing import (
    load_preprocessing_artifact,
    load_preprocessing_config,
    load_preprocessing_dataframe,
)

from .modeling import (
    LABEL_MAP,
    compute_binary_thresholds,
    profile_matrix,
)


class PatientModelInputBuilder(BaseEstimator):
    """Build the patient-level table used by Aramis M0/M1/M2 models.

    The input is a preprocessed measurement-level DataFrame. The builder first
    selects LR1 training rows according to the product policy, trains the profile
    LogisticRegression, scores measurements, and aggregates those scores to one
    patient-level `p_cancer` value by averaging LR1 evidence in logit space. It
    then adds patient label, symmetry features, age, and audit counters.
    """

    def __init__(
        self,
        *,
        profile_column: str = "radial_profile_data",
        label_column: str = "product_status_group",
        group_column: str = "patientId",
        specimen_column: str = "specimenId",
        side_column: str = "side",
        q_column: str = "q_range",
        age_column: str = "age",
        biopsy_column: str = "biopsy",
        lr1_row_policy: str = "all_rows",
        lr1_logreg_c: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.profile_column = profile_column
        self.label_column = label_column
        self.group_column = group_column
        self.specimen_column = specimen_column
        self.side_column = side_column
        self.q_column = q_column
        self.age_column = age_column
        self.biopsy_column = biopsy_column
        self.lr1_row_policy = lr1_row_policy
        self.lr1_logreg_c = lr1_logreg_c
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: Any = None) -> "PatientModelInputBuilder":
        """Fit LR1 on profile rows selected by `lr1_row_policy`."""
        self.lr1_rows_ = _lr1_training_rows(
            x,
            label_column=self.label_column,
            biopsy_column=self.biopsy_column,
            lr1_row_policy=self.lr1_row_policy,
        )
        self.lr1_model_ = _profile_logistic(
            logreg_c=self.lr1_logreg_c,
            random_state=self.random_state,
        )
        self.lr1_model_.fit(
            profile_matrix(self.lr1_rows_, self.profile_column),
            _row_labels(self.lr1_rows_, self.label_column),
        )
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return one row per patient with LR1, symmetry, age, and label fields."""
        scored_lr1 = _score_lr1_rows(
            self.lr1_model_,
            self.lr1_rows_,
            full_df=x,
            profile_column=self.profile_column,
            group_column=self.group_column,
            side_column=self.side_column,
            label_column=self.label_column,
            biopsy_column=self.biopsy_column,
        )
        return _patient_feature_table(
            x,
            scored_lr1,
            profile_column=self.profile_column,
            label_column=self.label_column,
            group_column=self.group_column,
            specimen_column=self.specimen_column,
            side_column=self.side_column,
            q_column=self.q_column,
            age_column=self.age_column,
            biopsy_column=self.biopsy_column,
        )

    def fit_transform(self, x: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit LR1 and return the patient-level feature table."""
        return self.fit(x, y).transform(x)


class PatientModelSetTrainer(BaseEstimator):
    """Train selected Aramis patient models.

    The trainer consumes the patient-level feature table produced by
    `PatientModelInputBuilder` and the LR1 measurement rows retained by that
    builder. `M0` uses the patient-level LR1 score directly. `M1` fits a scalar
    LogisticRegression on LR1 plus the SK symmetry block. `M0Q` tests profile
    plus reliability only. `M1Q` adds explicit reliability counters to M1. `M2`
    adds age to M1, and `M2Q` adds age to M1Q.
    """

    def __init__(
        self,
        *,
        selected_models: Sequence[str] = ("M0", "M0Q", "M1", "M1Q", "M2", "M2Q"),
        profile_column: str = "radial_profile_data",
        label_column: str = "product_status_group",
        lr1_logreg_c: float = 1.0,
        lr2_logreg_c: float = 1.0,
        random_state: int = 42,
        target_sensitivity: float = 0.95,
    ) -> None:
        self.selected_models = list(selected_models)
        self.profile_column = profile_column
        self.label_column = label_column
        self.lr1_logreg_c = lr1_logreg_c
        self.lr2_logreg_c = lr2_logreg_c
        self.random_state = random_state
        self.target_sensitivity = target_sensitivity

    def fit(
        self,
        feature_table: pd.DataFrame,
        lr1_rows: pd.DataFrame,
    ) -> "PatientModelSetTrainer":
        """Fit final patient models and store them in `models_`."""
        self.models_ = _fit_patient_model_set(
            feature_table,
            lr1_rows,
            profile_column=self.profile_column,
            label_column=self.label_column,
            lr1_logreg_c=self.lr1_logreg_c,
            lr2_logreg_c=self.lr2_logreg_c,
            random_state=self.random_state,
            target_sensitivity=self.target_sensitivity,
            selected_models=self.selected_models,
        )
        return self


class PatientModelSetEvaluator(BaseEstimator):
    """Evaluate selected patient models under the requested validation mode.

    Supported modes are `all_on_all`, `loovm`, `stratified_kfold`, and repeated
    stratified patient-level 70/30 splits. All split modes operate at patient
    level, so measurements from one patient cannot appear in both train and test.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        selected_models: Sequence[str],
        profile_column: str = "radial_profile_data",
        label_column: str = "product_status_group",
        group_column: str = "patientId",
        specimen_column: str = "specimenId",
        side_column: str = "side",
        q_column: str = "q_range",
        age_column: str = "age",
        biopsy_column: str = "biopsy",
        lr1_row_policy: str = "all_rows",
        lr1_logreg_c: float = 1.0,
        lr2_logreg_c: float = 1.0,
        random_state: int = 42,
        target_sensitivity: float = 0.95,
    ) -> None:
        self.config = config
        self.selected_models = list(selected_models)
        self.profile_column = profile_column
        self.label_column = label_column
        self.group_column = group_column
        self.specimen_column = specimen_column
        self.side_column = side_column
        self.q_column = q_column
        self.age_column = age_column
        self.biopsy_column = biopsy_column
        self.lr1_row_policy = lr1_row_policy
        self.lr1_logreg_c = lr1_logreg_c
        self.lr2_logreg_c = lr2_logreg_c
        self.random_state = random_state
        self.target_sensitivity = target_sensitivity

    def fit(self, x: pd.DataFrame, y: Any = None) -> "PatientModelSetEvaluator":
        """Run evaluation and store `split_metrics_` and `split_predictions_`."""
        self.split_metrics_, self.split_predictions_ = _evaluate_patient_model_set(
            x,
            config=self.config,
            profile_column=self.profile_column,
            label_column=self.label_column,
            group_column=self.group_column,
            specimen_column=self.specimen_column,
            side_column=self.side_column,
            q_column=self.q_column,
            age_column=self.age_column,
            biopsy_column=self.biopsy_column,
            lr1_row_policy=self.lr1_row_policy,
            lr1_logreg_c=self.lr1_logreg_c,
            lr2_logreg_c=self.lr2_logreg_c,
            random_state=self.random_state,
            target_sensitivity=self.target_sensitivity,
            selected_models=self.selected_models,
        )
        return self


class AramisPatientTrainingPipeline(BaseEstimator):
    """Complete sklearn-compatible training estimator for Aramis patient models.

    This estimator is the product-level training unit. It receives a
    measurement-level preprocessing DataFrame, builds the patient feature table,
    fits selected M0/M0Q/M1/M1Q/M2/M2Q models, evaluates them, and exposes the
    final traceable model artifact as `artifact_`.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        config_text: str,
        input_dataframe_joblib_path: str | Path,
        preprocessing_artifact: dict[str, Any],
        prediction_preprocessing: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.config_text = config_text
        self.input_dataframe_joblib_path = input_dataframe_joblib_path
        self.preprocessing_artifact = preprocessing_artifact
        self.prediction_preprocessing = prediction_preprocessing

    def fit(self, x: pd.DataFrame, y: Any = None) -> "AramisPatientTrainingPipeline":
        """Fit the full patient-level training route and build `artifact_`."""
        _ = y
        model_config = self.config.get("model", {})
        evaluation_config = self.config.get("evaluation", {})
        profile_column = str(model_config.get("profile_column", "radial_profile_data"))
        label_column = str(model_config.get("label_column", "product_status_group"))
        group_column = str(model_config.get("group_column", "patientId"))
        specimen_column = str(model_config.get("specimen_column", "specimenId"))
        side_column = str(model_config.get("side_column", "side"))
        q_column = str(model_config.get("q_column", "q_range"))
        age_column = str(model_config.get("age_column", "age"))
        biopsy_column = str(model_config.get("biopsy_column", "biopsy"))
        lr1_row_policy = str(model_config.get("lr1_row_policy", "all_rows"))
        selected_models = _selected_patient_models(model_config)
        default_logreg_c = float(model_config.get("logreg_c", 1.0))
        lr1_logreg_c = float(model_config.get("lr1_logreg_c", default_logreg_c))
        lr2_logreg_c = float(model_config.get("lr2_logreg_c", default_logreg_c))
        require_paired_breasts = bool(model_config.get("require_paired_breasts", False))
        random_state = int(evaluation_config.get("random_state", 42))
        target_sensitivity = float(evaluation_config.get("target_sensitivity", 0.95))

        self.training_dataframe_, self.paired_breast_audit_ = _paired_breast_training_dataframe(
            x,
            group_column=group_column,
            side_column=side_column,
            required=require_paired_breasts,
        )
        self.input_builder_ = PatientModelInputBuilder(
            profile_column=profile_column,
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            side_column=side_column,
            q_column=q_column,
            age_column=age_column,
            biopsy_column=biopsy_column,
            lr1_row_policy=lr1_row_policy,
            lr1_logreg_c=lr1_logreg_c,
            random_state=random_state,
        )
        self.feature_table_ = self.input_builder_.fit_transform(self.training_dataframe_)
        self.model_trainer_ = PatientModelSetTrainer(
            selected_models=selected_models,
            profile_column=profile_column,
            label_column=label_column,
            lr1_logreg_c=lr1_logreg_c,
            lr2_logreg_c=lr2_logreg_c,
            random_state=random_state,
            target_sensitivity=target_sensitivity,
        )
        self.model_trainer_.fit(self.feature_table_, self.input_builder_.lr1_rows_)
        self.evaluator_ = PatientModelSetEvaluator(
            config=self.config,
            selected_models=selected_models,
            profile_column=profile_column,
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            side_column=side_column,
            q_column=q_column,
            age_column=age_column,
            biopsy_column=biopsy_column,
            lr1_row_policy=lr1_row_policy,
            lr1_logreg_c=lr1_logreg_c,
            lr2_logreg_c=lr2_logreg_c,
            random_state=random_state,
            target_sensitivity=target_sensitivity,
        )
        self.evaluator_.fit(self.training_dataframe_)
        self.artifact_ = _patient_training_artifact(
            df=self.training_dataframe_,
            config=self.config,
            config_text=self.config_text,
            input_dataframe_joblib_path=self.input_dataframe_joblib_path,
            preprocessing_artifact=self.preprocessing_artifact,
            prediction_preprocessing=self.prediction_preprocessing,
            selected_models=selected_models,
            models=self.model_trainer_.models_,
            feature_table=self.feature_table_,
            lr1_rows=self.input_builder_.lr1_rows_,
            split_metrics=self.evaluator_.split_metrics_,
            split_predictions=self.evaluator_.split_predictions_,
        )
        self.artifact_["paired_breast_eligibility"] = self.paired_breast_audit_
        return self


def build_patient_training_pipeline(
    *,
    config: dict[str, Any],
    config_text: str,
    input_dataframe_joblib_path: str | Path,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None = None,
) -> Pipeline:
    """Return one sklearn Pipeline object for patient-level Aramis training."""
    return Pipeline(
        [
            (
                "patient_training",
                AramisPatientTrainingPipeline(
                    config=config,
                    config_text=config_text,
                    input_dataframe_joblib_path=input_dataframe_joblib_path,
                    preprocessing_artifact=preprocessing_artifact,
                    prediction_preprocessing=prediction_preprocessing,
                ),
            )
        ]
    )


def run_training_from_config(
    config_path: str | Path,
    *,
    dataframe: pd.DataFrame | None = None,
    preprocessing_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Aramis training using paths and parameters stored in YAML."""
    config_path = Path(config_path)
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    _validate_training_config(config, config_path)

    branch = config["training"]["branch"]
    if branch != "one_to_many":
        raise ValueError(f"Unsupported training branch: {branch}")

    input_path = _config_path(config, config_path, "input_dataframe_joblib_path")
    output_path = _config_path(config, config_path, "output_model_joblib_path")
    output_json_path = _optional_config_path(config, config_path, "output_json_path")
    output_yaml_path = _optional_config_path(config, config_path, "output_yaml_path")
    prediction_preprocessing_config_path = _optional_config_path(
        config,
        config_path,
        "prediction_preprocessing_config_path",
    )
    prediction_preprocessing = _prediction_preprocessing_payload(
        prediction_preprocessing_config_path
    )
    df = dataframe if dataframe is not None else load_preprocessing_dataframe(input_path)
    if preprocessing_artifact is None:
        preprocessing_artifact = load_preprocessing_artifact(input_path)

    model_type = str(config.get("model", {}).get("type", "patient_m0_m1_m2_logistic_set"))
    if model_type != "patient_m0_m1_m2_logistic_set":
        raise ValueError(f"Unsupported training model.type: {model_type!r}")
    artifact = train_patient_m0_m1_m2_model_artifact(
        df,
        config=config,
        config_text=config_text,
        input_dataframe_joblib_path=input_path,
        preprocessing_artifact=preprocessing_artifact,
        prediction_preprocessing=prediction_preprocessing,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    if output_json_path is not None:
        _write_json_summary(artifact, output_json_path)
    if output_yaml_path is not None:
        _write_yaml_description(artifact, output_yaml_path)
    return artifact


def train_patient_m0_m1_m2_model_artifact(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    config_text: str,
    input_dataframe_joblib_path: str | Path,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one artifact containing M0, M1, and M2 patient-level models."""
    pipeline = build_patient_training_pipeline(
        config=config,
        config_text=config_text,
        input_dataframe_joblib_path=input_dataframe_joblib_path,
        preprocessing_artifact=preprocessing_artifact,
        prediction_preprocessing=prediction_preprocessing,
    )
    pipeline.fit(df)
    return pipeline.named_steps["patient_training"].artifact_


def _patient_training_artifact(
    *,
    df: pd.DataFrame,
    config: dict[str, Any],
    config_text: str,
    input_dataframe_joblib_path: str | Path,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None,
    selected_models: Sequence[str],
    models: dict[str, Any],
    feature_table: pd.DataFrame,
    lr1_rows: pd.DataFrame,
    split_metrics: pd.DataFrame,
    split_predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Build the traceable joblib payload for patient-level model training."""
    metric_summary = _summarize_patient_model_metrics(split_metrics)
    dataset_summary = _patient_dataset_summary(df, feature_table, lr1_rows)
    model_descriptions = {
        name: description
        for name, description in _patient_model_descriptions().items()
        if name in selected_models
    }

    return {
        "kind": "aramis_training_artifact",
        "version": "0.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "patient_m0_m1_m2_logistic_set",
        "models": models,
        "model_descriptions": model_descriptions,
        "feature_schema": _patient_model_feature_schema(selected_models),
        "warnings": _patient_model_warnings(config, selected_models, feature_table),
        "training_config": config,
        "training_config_yaml": config_text,
        "training_config_text": config_text,
        "training_config_sha256": sha256(config_text.encode("utf-8")).hexdigest(),
        **_preprocessing_lineage_fields(
            preprocessing_artifact,
            prediction_preprocessing,
        ),
        "input_dataframe_joblib_sha256": _file_sha256(input_dataframe_joblib_path),
        "dataset_summary": dataset_summary,
        "feature_table": feature_table,
        "metric_summary": metric_summary,
        "split_metrics": split_metrics,
        "split_predictions": split_predictions,
        "metadata": {
            "branch": "one_to_many",
            "aramis_version": _aramis_version(),
            "aramis_git_sha": _aramis_git_sha(),
        },
    }


def _fit_patient_model_input(
    df: pd.DataFrame,
    *,
    profile_column: str,
    label_column: str,
    group_column: str,
    specimen_column: str,
    side_column: str,
    q_column: str,
    age_column: str,
    biopsy_column: str,
    lr1_row_policy: str,
    lr1_logreg_c: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lr1_rows = _lr1_training_rows(
        df,
        label_column=label_column,
        biopsy_column=biopsy_column,
        lr1_row_policy=lr1_row_policy,
    )
    lr1_model = _profile_logistic(logreg_c=lr1_logreg_c, random_state=random_state)
    lr1_model.fit(profile_matrix(lr1_rows, profile_column), _row_labels(lr1_rows, label_column))
    scored_lr1 = _score_lr1_rows(
        lr1_model,
        lr1_rows,
        full_df=df,
        profile_column=profile_column,
        group_column=group_column,
        side_column=side_column,
        label_column=label_column,
        biopsy_column=biopsy_column,
    )
    feature_table = _patient_feature_table(
        df,
        scored_lr1,
        profile_column=profile_column,
        label_column=label_column,
        group_column=group_column,
        specimen_column=specimen_column,
        side_column=side_column,
        q_column=q_column,
        age_column=age_column,
        biopsy_column=biopsy_column,
    )
    return feature_table, lr1_rows


def _fit_patient_model_set(
    feature_table: pd.DataFrame,
    lr1_rows: pd.DataFrame,
    *,
    profile_column: str,
    label_column: str,
    lr1_logreg_c: float,
    lr2_logreg_c: float,
    random_state: int,
    target_sensitivity: float,
    selected_models: Sequence[str],
) -> dict[str, Any]:
    lr1_model = _profile_logistic(logreg_c=lr1_logreg_c, random_state=random_state)
    lr1_model.fit(profile_matrix(lr1_rows, profile_column), _row_labels(lr1_rows, label_column))
    y = feature_table["label"].to_numpy(dtype=int)
    models = {}
    if "M0" in selected_models:
        models["M0"] = {
            "name": "M0_profile_only",
            "lr1_model": lr1_model,
            "final_model": None,
            "feature_columns": ["profile_p_cancer_logit_average"],
            "thresholds": compute_binary_thresholds(
                y,
                feature_table["profile_p_cancer_logit_average"].to_numpy(dtype=float),
                target_sensitivity=target_sensitivity,
            ),
        }
    for model_name, columns in _patient_model_feature_columns().items():
        if model_name not in selected_models:
            continue
        final_model = _scalar_logistic(logreg_c=lr2_logreg_c, random_state=random_state)
        final_model.fit(feature_table[columns], y)
        score = final_model.predict_proba(feature_table[columns])[:, 1]
        models[model_name] = {
            "name": _patient_model_descriptions()[model_name]["name"],
            "lr1_model": lr1_model,
            "final_model": final_model,
            "feature_columns": columns,
            "thresholds": compute_binary_thresholds(
                y,
                score,
                target_sensitivity=target_sensitivity,
            ),
        }
    return models


def _evaluate_patient_model_set(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    profile_column: str,
    label_column: str,
    group_column: str,
    specimen_column: str,
    side_column: str,
    q_column: str,
    age_column: str,
    biopsy_column: str,
    lr1_row_policy: str,
    lr1_logreg_c: float,
    lr2_logreg_c: float,
    random_state: int,
    target_sensitivity: float,
    selected_models: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_config = config.get("evaluation", {})
    mode = _evaluation_mode(evaluation_config)
    n_splits = int(evaluation_config.get("n_splits", 20))
    n_repeats = int(evaluation_config.get("n_repeats", 1))
    test_size = float(evaluation_config.get("test_size", 0.30))
    base_features = _patient_feature_table(
        df,
        _empty_lr1_scores(df, group_column),
        profile_column=profile_column,
        label_column=label_column,
        group_column=group_column,
        specimen_column=specimen_column,
        side_column=side_column,
        q_column=q_column,
        age_column=age_column,
        biopsy_column=biopsy_column,
    )
    if mode == "all_on_all":
        return _evaluate_patient_all_on_all(
            df,
            profile_column=profile_column,
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            side_column=side_column,
            q_column=q_column,
            age_column=age_column,
            biopsy_column=biopsy_column,
            lr1_row_policy=lr1_row_policy,
            lr1_logreg_c=lr1_logreg_c,
            lr2_logreg_c=lr2_logreg_c,
            random_state=random_state,
            target_sensitivity=target_sensitivity,
            selected_models=selected_models,
        )
    metrics = []
    predictions = []
    y_patients = base_features["label"].to_numpy(dtype=int)
    split_pairs = _patient_split_pairs(
        mode=mode,
        base_features=base_features,
        y_patients=y_patients,
        n_splits=n_splits,
        n_repeats=n_repeats,
        test_size=test_size,
        random_state=random_state,
    )
    for split_id, (train_idx, test_idx) in enumerate(split_pairs):
        train_patients = set(base_features.iloc[train_idx]["patientId"].astype(str))
        test_patients = set(base_features.iloc[test_idx]["patientId"].astype(str))
        train_df = df[df[group_column].astype(str).isin(train_patients)].copy()
        test_df = df[df[group_column].astype(str).isin(test_patients)].copy()
        if set(train_df[group_column].astype(str)).intersection(
            set(test_df[group_column].astype(str))
        ):
            raise RuntimeError("Patient leakage detected in training split.")
        train_features, train_lr1_rows = _fit_patient_model_input(
            train_df,
            profile_column=profile_column,
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            side_column=side_column,
            q_column=q_column,
            age_column=age_column,
            biopsy_column=biopsy_column,
            lr1_row_policy=lr1_row_policy,
            lr1_logreg_c=lr1_logreg_c,
            random_state=random_state + split_id,
        )
        lr1_model = _profile_logistic(
            logreg_c=lr1_logreg_c,
            random_state=random_state + split_id,
        )
        lr1_model.fit(
            profile_matrix(train_lr1_rows, profile_column),
            _row_labels(train_lr1_rows, label_column),
        )
        test_lr1_rows = _lr1_training_rows(
            test_df,
            label_column=label_column,
            biopsy_column=biopsy_column,
            lr1_row_policy=lr1_row_policy,
            require_two_classes=False,
        )
        test_features = _patient_feature_table(
            test_df,
            _score_lr1_rows(
                lr1_model,
                test_lr1_rows,
                full_df=test_df,
                profile_column=profile_column,
                group_column=group_column,
                side_column=side_column,
                label_column=label_column,
                biopsy_column=biopsy_column,
            ),
            profile_column=profile_column,
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            side_column=side_column,
            q_column=q_column,
            age_column=age_column,
            biopsy_column=biopsy_column,
            require_two_classes=False,
        )
        for model_name in selected_models:
            train_score, test_score = _split_model_scores(
                model_name,
                train_features,
                test_features,
                lr2_logreg_c=lr2_logreg_c,
                random_state=random_state + split_id,
            )
            thresholds = compute_binary_thresholds(
                train_features["label"].to_numpy(dtype=int),
                train_score,
                target_sensitivity=target_sensitivity,
            )
            if mode != "loovm":
                metrics.append(
                    _patient_metric_row(
                        model_name,
                        split_id,
                        train_features,
                        test_features,
                        test_score,
                        thresholds,
                        evaluation_mode=mode,
                    )
                )
            predictions.append(
                _patient_prediction_frame(
                    model_name,
                    split_id,
                    test_features,
                    test_score,
                    thresholds,
                    evaluation_mode=mode,
                )
            )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    if mode == "loovm":
        return (
            _pooled_patient_metrics(prediction_frame, evaluation_mode=mode),
            prediction_frame,
        )
    return (
        pd.DataFrame(metrics),
        prediction_frame,
    )


def _split_model_scores(
    model_name: str,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    lr2_logreg_c: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "M0":
        return (
            train_features["profile_p_cancer_logit_average"].to_numpy(dtype=float),
            test_features["profile_p_cancer_logit_average"].to_numpy(dtype=float),
        )
    columns = _patient_model_feature_columns()[model_name]
    model = _scalar_logistic(logreg_c=lr2_logreg_c, random_state=random_state)
    model.fit(train_features[columns], train_features["label"].to_numpy(dtype=int))
    return (
        model.predict_proba(train_features[columns])[:, 1],
        model.predict_proba(test_features[columns])[:, 1],
    )


def _selected_patient_models(model_config: dict[str, Any]) -> list[str]:
    selected = model_config.get(
        "selected_models",
        ["M1Q"],
    )
    if isinstance(selected, str):
        selected = [selected]
    out = [str(model_name).upper() for model_name in selected]
    supported = {"M0", "M0Q", "M1", "M1Q", "M2", "M2Q"}
    unknown = [model_name for model_name in out if model_name not in supported]
    if unknown:
        raise ValueError(f"Unsupported patient models: {unknown}")
    return out


def _evaluation_mode(evaluation_config: dict[str, Any]) -> str:
    mode = str(evaluation_config.get("mode", "repeated_stratified_shuffle")).lower()
    aliases = {
        "70_30": "repeated_stratified_shuffle",
        "repeated_70_30": "repeated_stratified_shuffle",
        "patient_70_30": "repeated_stratified_shuffle",
        "loo": "loovm",
        "loocv": "loovm",
        "leave_one_out": "loovm",
        "leave_one_patient_out": "loovm",
        "train_all": "all_on_all",
        "train-on-all": "all_on_all",
        "all-on-all": "all_on_all",
        "kfold": "stratified_kfold",
        "stratified-kfold": "stratified_kfold",
    }
    mode = aliases.get(mode, mode)
    supported = {"repeated_stratified_shuffle", "stratified_kfold", "loovm", "all_on_all"}
    if mode not in supported:
        raise ValueError(f"Unsupported evaluation.mode: {mode!r}")
    return mode


def _patient_split_pairs(
    *,
    mode: str,
    base_features: pd.DataFrame,
    y_patients: np.ndarray,
    n_splits: int,
    n_repeats: int,
    test_size: float,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(base_features))
    if mode == "repeated_stratified_shuffle":
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
        )
        return list(splitter.split(base_features, y_patients))
    if mode == "stratified_kfold":
        if n_repeats > 1:
            splitter = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            return list(splitter.split(base_features, y_patients))
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return list(splitter.split(base_features, y_patients))
    if mode == "loovm":
        return [
            (np.delete(indices, test_idx), np.asarray([test_idx]))
            for test_idx in range(len(indices))
        ]
    raise ValueError(f"Unsupported split mode: {mode!r}")


def _evaluate_patient_all_on_all(
    df: pd.DataFrame,
    *,
    profile_column: str,
    label_column: str,
    group_column: str,
    specimen_column: str,
    side_column: str,
    q_column: str,
    age_column: str,
    biopsy_column: str,
    lr1_row_policy: str,
    lr1_logreg_c: float,
    lr2_logreg_c: float,
    random_state: int,
    target_sensitivity: float,
    selected_models: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_table, lr1_rows = _fit_patient_model_input(
        df,
        profile_column=profile_column,
        label_column=label_column,
        group_column=group_column,
        specimen_column=specimen_column,
        side_column=side_column,
        q_column=q_column,
        age_column=age_column,
        biopsy_column=biopsy_column,
        lr1_row_policy=lr1_row_policy,
        lr1_logreg_c=lr1_logreg_c,
        random_state=random_state,
    )
    lr1_model = _profile_logistic(logreg_c=lr1_logreg_c, random_state=random_state)
    lr1_model.fit(profile_matrix(lr1_rows, profile_column), _row_labels(lr1_rows, label_column))
    metrics = []
    predictions = []
    for model_name in selected_models:
        train_score, score = _split_model_scores(
            model_name,
            feature_table,
            feature_table,
            lr2_logreg_c=lr2_logreg_c,
            random_state=random_state,
        )
        thresholds = compute_binary_thresholds(
            feature_table["label"].to_numpy(dtype=int),
            train_score,
            target_sensitivity=target_sensitivity,
        )
        metrics.append(
            _patient_metric_row(
                model_name,
                0,
                feature_table,
                feature_table,
                score,
                thresholds,
                evaluation_mode="all_on_all",
            )
        )
        predictions.append(
            _patient_prediction_frame(
                model_name,
                0,
                feature_table,
                score,
                thresholds,
                evaluation_mode="all_on_all",
            )
        )
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


SK_CORE4_FEATURE_COLUMNS = (
    "sk_wasserstein_distance_full_q2",
    "sk_weightedrms1",
    "sk_weightedrms2",
    "sk_mean_peak_value_abs_delta",
)


def _patient_model_feature_columns() -> dict[str, list[str]]:
    sk_symmetry = [
        "profile_p_cancer_logit_average",
        *SK_CORE4_FEATURE_COLUMNS,
    ]
    reliability = ["profile_p_cancer_n_measurements"]
    return {
        "M0Q": ["profile_p_cancer_logit_average", *reliability],
        "M1": sk_symmetry,
        "M1Q": [*sk_symmetry, *reliability],
        "M2": [*sk_symmetry, "age", "age_available"],
        "M2Q": [*sk_symmetry, *reliability, "age", "age_available"],
    }


def _patient_model_descriptions() -> dict[str, dict[str, Any]]:
    return {
        "M0": {
            "name": "M0 profile only",
            "description": "LR1 profile LogisticRegression, logit-averaged to patient p_cancer.",
        },
        "M0Q": {
            "name": "M0Q profile plus target measurement count",
            "description": "M0 plus target-breast prediction count; no symmetry or age.",
        },
        "M1": {
            "name": "M1 profile plus SK symmetry",
            "description": "LR1 target-breast p_cancer plus same-patient target/contralateral SK symmetry block.",
        },
        "M1Q": {
            "name": "M1Q profile plus SK core4 plus target measurement count",
            "description": "M1 plus the fixed SK core4 and target-breast prediction count.",
        },
        "M2": {
            "name": "M2 profile plus SK symmetry plus age",
            "description": "M1 plus age and age availability flag.",
        },
        "M2Q": {
            "name": "M2Q profile plus SK core4 plus target measurement count plus age",
            "description": "M1Q plus age and age availability flag.",
        },
    }


def _patient_model_feature_schema(selected_models: Sequence[str]) -> dict[str, Any]:
    feature_columns = {"M0": ["profile_p_cancer_logit_average"]}
    feature_columns.update(_patient_model_feature_columns())
    return {
        model_name: {
            "feature_columns": feature_columns[model_name],
            "unit": "patient",
            "label": "BENIGN vs CANCER decision-support class",
        }
        for model_name in selected_models
    }


def _patient_model_warnings(
    config: dict[str, Any],
    selected_models: Sequence[str],
    feature_table: pd.DataFrame,
) -> list[str]:
    warnings = [
        "Research-draft decision support only; requires radiologist review.",
        "Not for autonomous diagnosis.",
    ]
    mode = _evaluation_mode(config.get("evaluation", {}))
    if mode == "all_on_all":
        warnings.append("all_on_all is an optimistic discovery ceiling, not validation.")
    if any(model_name in selected_models for model_name in ["M2", "M2Q"]):
        warnings.append("M2 includes age; age contribution must be reviewed separately.")
    if any(model_name in selected_models for model_name in ["M1Q", "M2Q"]):
        warnings.append(
            "Q models include target-breast prediction count; report reliability separately from p_cancer."
        )
    unavailable = int((feature_table["symmetry_available"] == 0).sum())
    if unavailable:
        warnings.append(
            f"{unavailable} patients have unavailable paired-breast symmetry features."
        )
    low_target = int((feature_table["target_measurements"] < 3).sum())
    if low_target:
        warnings.append(
            f"{low_target} patients have fewer than 3 valid target-breast measurements."
        )
    low_contralateral = int((feature_table["contralateral_measurements"] < 3).sum())
    if low_contralateral:
        warnings.append(
            f"{low_contralateral} patients have fewer than 3 valid contralateral-breast measurements."
        )
    return warnings


def _lr1_training_rows(
    df: pd.DataFrame,
    *,
    label_column: str,
    biopsy_column: str,
    lr1_row_policy: str,
    require_two_classes: bool = True,
) -> pd.DataFrame:
    _require_training_columns(df, [label_column])
    out = df[df[label_column].isin(LABEL_MAP)].copy()
    if lr1_row_policy == "biopsy_only":
        _require_training_columns(out, [biopsy_column])
        out = out[_boolean_series(out[biopsy_column])].copy()
    elif lr1_row_policy != "all_rows":
        raise ValueError(f"Unsupported lr1_row_policy: {lr1_row_policy!r}")
    if require_two_classes and out[label_column].nunique() != 2:
        raise ValueError("LR1 training rows must contain BENIGN and CANCER.")
    return out.reset_index(drop=True)


def _row_labels(df: pd.DataFrame, label_column: str) -> np.ndarray:
    return df[label_column].map(LABEL_MAP).astype(int).to_numpy()


def _score_lr1_rows(
    lr1_model: Pipeline,
    rows: pd.DataFrame,
    *,
    full_df: pd.DataFrame,
    profile_column: str,
    group_column: str,
    side_column: str,
    label_column: str,
    biopsy_column: str,
) -> pd.DataFrame:
    _require_training_columns(rows, [group_column, side_column])
    out = rows[[group_column, side_column]].copy()
    out[group_column] = out[group_column].astype(str)
    out["_side_norm"] = out[side_column].map(_normalize_side)
    out["lr1_measurement_p_cancer"] = lr1_model.predict_proba(
        profile_matrix(rows, profile_column)
    )[:, 1]
    target_lookup = _patient_target_side_lookup(
        full_df,
        group_column=group_column,
        side_column=side_column,
        label_column=label_column,
        biopsy_column=biopsy_column,
    )
    grouped_rows = []
    for patient_id, group in out.groupby(group_column, sort=True):
        target = target_lookup.get(str(patient_id))
        if target is None:
            continue
        target_scores = group.loc[
            group["_side_norm"] == target["inferred_target_side_norm"],
            "lr1_measurement_p_cancer",
        ].to_numpy(dtype=float)
        if target_scores.size == 0:
            raise ValueError(
                f"No LR1 target-side scores for patient {patient_id!r}; "
                "check target-side policy and lr1_row_policy."
            )
        grouped_rows.append(
            {
                "patientId": str(patient_id),
                "profile_p_cancer_probability_mean": float(np.mean(target_scores)),
                "profile_p_cancer_logit_average": _logit_average_probability(
                    target_scores
                ),
                "profile_p_cancer_n_measurements": int(target_scores.size),
            }
        )
    return pd.DataFrame(grouped_rows)


def _empty_lr1_scores(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patientId": sorted(df[group_column].astype(str).unique()),
            "profile_p_cancer_probability_mean": 0.5,
            "profile_p_cancer_logit_average": 0.5,
            "profile_p_cancer_n_measurements": 0,
        }
    )


def _logit_average_probability(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return 0.5
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    mean_logit = float(np.mean(logits))
    return float(1.0 / (1.0 + np.exp(-mean_logit)))


def _patient_feature_table(
    df: pd.DataFrame,
    lr1_scores: pd.DataFrame,
    *,
    profile_column: str,
    label_column: str,
    group_column: str,
    specimen_column: str,
    side_column: str,
    q_column: str,
    age_column: str,
    biopsy_column: str,
    require_two_classes: bool = True,
) -> pd.DataFrame:
    _require_training_columns(
        df,
        [
            group_column,
            specimen_column,
            label_column,
            profile_column,
            side_column,
            q_column,
        ],
    )
    rows = []
    for patient_id, patient_df in df.groupby(group_column, sort=True):
        labels = patient_df[label_column].map(LABEL_MAP).dropna().astype(int)
        if labels.empty:
            continue
        label = int((labels == 1).any())
        inferred_target = _patient_inferred_target_side(
            patient_df,
            side_column=side_column,
            label_column=label_column,
            biopsy_column=biopsy_column,
        )
        symmetry = _target_contralateral_symmetry_features(
            patient_df,
            profile_column=profile_column,
            q_column=q_column,
            side_column=side_column,
            target_side_norm=inferred_target["inferred_target_side_norm"],
            contralateral_side_norm=inferred_target[
                "inferred_contralateral_side_norm"
            ],
        )
        rows.append(
            {
                "patientId": str(patient_id),
                "label": label,
                "label_name": "CANCER" if label == 1 else "BENIGN",
                "inferred_target_side": inferred_target["inferred_target_side"],
                "inferred_contralateral_side": inferred_target[
                    "inferred_contralateral_side"
                ],
                "inferred_target_side_reason": inferred_target[
                    "inferred_target_side_reason"
                ],
                "inferred_target_side_ambiguous": int(
                    inferred_target["inferred_target_side_ambiguous"]
                ),
                "specimens": int(patient_df[specimen_column].astype(str).nunique()),
                "measurements": int(len(patient_df)),
                "age": _numeric_median(patient_df, age_column, default=0.0),
                "age_available": int(_has_numeric(patient_df, age_column)),
                **symmetry,
            }
        )
    feature_table = pd.DataFrame(rows)
    out = feature_table.merge(lr1_scores, on="patientId", how="inner")
    out = _add_patient_reliability_columns(out)
    if require_two_classes and out["label"].nunique() != 2:
        raise ValueError("Patient feature table must contain BENIGN and CANCER.")
    return out.reset_index(drop=True)


def build_patient_prediction_feature_row(
    df: pd.DataFrame,
    model_info: dict[str, Any],
    *,
    patient_id: str,
    target_side: str,
    profile_column: str = "radial_profile_data",
    group_column: str = "patientId",
    specimen_column: str = "specimenId",
    side_column: str = "side",
    q_column: str = "q_range",
    age_column: str = "age",
) -> pd.DataFrame:
    """Build one prediction feature row from clinician-supplied target side.

    Training infers target side from biopsy/status metadata. Prediction must not
    do that: the suspicious breast side comes from the clinical input config.
    """
    _require_training_columns(
        df,
        [group_column, specimen_column, side_column, profile_column, q_column],
    )
    lr1_model = model_info.get("lr1_model")
    if lr1_model is None:
        raise ValueError("Model artifact is missing lr1_model.")

    patient_df = df[df[group_column].astype(str) == str(patient_id)].copy()
    if patient_df.empty:
        raise ValueError(f"Patient not found in prediction DataFrame: {patient_id!r}")

    target_side_norm = _normalize_side(target_side)
    if target_side_norm is None:
        raise ValueError(f"Invalid target_side: {target_side!r}")
    side_norms = patient_df[side_column].map(_normalize_side)
    available_sides = sorted(side for side in side_norms.dropna().unique())
    if target_side_norm not in available_sides:
        raise ValueError(
            f"Target side {target_side!r} is absent for patient {patient_id!r}; "
            f"available sides: {available_sides}"
        )

    contralateral = [side for side in available_sides if side != target_side_norm]
    contralateral_side_norm = contralateral[0] if contralateral else None
    needs_paired_symmetry = any(
        str(column).startswith("sk_")
        for column in model_info.get("feature_columns", [])
    )
    if needs_paired_symmetry and contralateral_side_norm is None:
        raise ValueError(
            "Paired target/contralateral breast measurements are required for "
            f"this model; patient {patient_id!r} has only {available_sides}."
        )
    target_df = patient_df[side_norms == target_side_norm].copy()
    target_scores = lr1_model.predict_proba(
        profile_matrix(target_df, profile_column)
    )[:, 1]
    symmetry = _target_contralateral_symmetry_features(
        patient_df,
        profile_column=profile_column,
        q_column=q_column,
        side_column=side_column,
        target_side_norm=target_side_norm,
        contralateral_side_norm=contralateral_side_norm,
    )
    row = {
        "patientId": str(patient_id),
        "target_side": _display_side(target_side_norm),
        "contralateral_side": _display_side(contralateral_side_norm),
        "specimens": int(patient_df[specimen_column].astype(str).nunique()),
        "measurements": int(len(patient_df)),
        "age": _numeric_median(patient_df, age_column, default=0.0),
        "age_available": int(_has_numeric(patient_df, age_column)),
        "profile_p_cancer_probability_mean": float(np.mean(target_scores)),
        "profile_p_cancer_logit_average": _logit_average_probability(target_scores),
        "profile_p_cancer_n_measurements": int(target_scores.size),
        **symmetry,
    }
    return _add_patient_reliability_columns(pd.DataFrame([row]))


def _add_patient_reliability_columns(feature_table: pd.DataFrame) -> pd.DataFrame:
    out = feature_table.copy()
    out["min_measurements_per_breast"] = np.minimum(
        out["target_measurements"].astype(int),
        out["contralateral_measurements"].astype(int),
    )
    out["target_measurements_ok"] = (out["target_measurements"].astype(int) >= 3).astype(
        int
    )
    out["contralateral_measurements_ok"] = (
        out["contralateral_measurements"].astype(int) >= 3
    ).astype(int)
    out["paired_measurements_ok"] = (
        (out["symmetry_available"].astype(int) == 1)
        & (out["min_measurements_per_breast"].astype(int) >= 3)
    ).astype(int)
    out["profile_measurements_ok"] = (
        out["profile_p_cancer_n_measurements"].astype(int) >= 3
    ).astype(int)
    out["result_reliability"] = np.select(
        [
            out["paired_measurements_ok"].astype(bool),
            out["symmetry_available"].astype(bool),
        ],
        ["high", "medium"],
        default="low",
    )
    out["result_reliability_reason"] = np.select(
        [
            out["paired_measurements_ok"].astype(bool),
            out["symmetry_available"].astype(bool),
        ],
        [
            "at least 3 valid measurements per breast",
            "paired breasts available but fewer than 3 measurements in at least one breast",
        ],
        default="paired breast symmetry unavailable",
    )
    return out


def _patient_target_side_lookup(
    df: pd.DataFrame,
    *,
    group_column: str,
    side_column: str,
    label_column: str,
    biopsy_column: str,
) -> dict[str, dict[str, Any]]:
    return {
        str(patient_id): _patient_inferred_target_side(
            patient_df,
            side_column=side_column,
            label_column=label_column,
            biopsy_column=biopsy_column,
        )
        for patient_id, patient_df in df.groupby(group_column, sort=True)
    }


def _patient_inferred_target_side(
    patient_df: pd.DataFrame,
    *,
    side_column: str,
    label_column: str,
    biopsy_column: str,
) -> dict[str, Any]:
    """Infer training target side from biopsy/status metadata.

    This is training-only logic. Prediction must receive the real target side
    from clinician/config input instead of inferring it from labels.
    """
    _require_training_columns(patient_df, [side_column, label_column])
    work = patient_df[[side_column, label_column]].copy()
    work["_side_norm"] = work[side_column].map(_normalize_side)
    work["_label_value"] = work[label_column].map(LABEL_MAP)
    if biopsy_column in patient_df.columns:
        work["_biopsy"] = _boolean_series(patient_df[biopsy_column])
    else:
        work["_biopsy"] = False

    candidates = [
        ("biopsy_cancer", work["_biopsy"] & (work["_label_value"] == 1)),
        ("biopsy_benign", work["_biopsy"] & (work["_label_value"] == 0)),
        ("cancer", work["_label_value"] == 1),
        ("benign", work["_label_value"] == 0),
        ("available_side", work["_side_norm"].notna()),
    ]
    selected_reason = "available_side"
    selected_sides: list[str] = []
    for reason, mask in candidates:
        selected_sides = sorted(
            side for side in work.loc[mask, "_side_norm"].dropna().unique()
        )
        if selected_sides:
            selected_reason = reason
            break
    if not selected_sides:
        raise ValueError("Cannot infer target side: no valid side values.")
    target_side_norm = selected_sides[0]
    side_values = sorted(side for side in work["_side_norm"].dropna().unique())
    contralateral = [side for side in side_values if side != target_side_norm]
    contralateral_side_norm = contralateral[0] if contralateral else None
    return {
        "inferred_target_side_norm": target_side_norm,
        "inferred_target_side": _display_side(target_side_norm),
        "inferred_contralateral_side_norm": contralateral_side_norm,
        "inferred_contralateral_side": _display_side(contralateral_side_norm),
        "inferred_target_side_reason": selected_reason,
        "inferred_target_side_ambiguous": len(selected_sides) > 1,
    }


def _target_contralateral_symmetry_features(
    patient_df: pd.DataFrame,
    *,
    profile_column: str,
    q_column: str,
    side_column: str,
    target_side_norm: str,
    contralateral_side_norm: str | None,
) -> dict[str, float | int]:
    """Measure paired-breast asymmetry with target-side context.

    The cosine asymmetry itself is symmetric: it says whether the two breasts
    differ more than expected from within-breast replicate variability. The
    target-side context comes from separate target and contralateral counters,
    within-breast distances, and LR1 scoring of the suspicious breast.
    """
    target = _side_profiles(patient_df, profile_column, side_column, target_side_norm)
    contralateral = (
        _side_profiles(patient_df, profile_column, side_column, contralateral_side_norm)
        if contralateral_side_norm is not None
        else []
    )
    target_within = _mean_pairwise_cosine(target)
    contralateral_within = _mean_pairwise_cosine(contralateral)
    if not target or not contralateral:
        out = {
            "symmetry_available": 0,
            "target_measurements": int(len(target)),
            "contralateral_measurements": int(len(contralateral)),
            "target_within_cosine_distance_mean": _finite_or_zero(target_within),
            "contralateral_within_cosine_distance_mean": _finite_or_zero(
                contralateral_within
            ),
            "between_breasts_cosine_distance_mean": 0.0,
            "symmetry_cosine_score": 0.0,
        }
        out.update(_empty_sk_symmetry_features())
        return out
    target_mean = np.mean(np.vstack(target), axis=0)
    contralateral_mean = np.mean(np.vstack(contralateral), axis=0)
    between = _cosine_distance(target_mean, contralateral_mean)
    within_values = [
        value
        for value in (target_within, contralateral_within)
        if np.isfinite(value)
    ]
    within_mean = float(np.mean(within_values)) if within_values else 0.0
    out = {
        "symmetry_available": 1,
        "target_measurements": int(len(target)),
        "contralateral_measurements": int(len(contralateral)),
        "target_within_cosine_distance_mean": _finite_or_zero(target_within),
        "contralateral_within_cosine_distance_mean": _finite_or_zero(
            contralateral_within
        ),
        "between_breasts_cosine_distance_mean": _finite_or_zero(between),
        "symmetry_cosine_score": _finite_or_zero(between - within_mean),
    }
    out.update(
        _sk_target_contralateral_symmetry_features(
            patient_df,
            profile_column=profile_column,
            q_column=q_column,
            side_column=side_column,
            target_side_norm=target_side_norm,
            contralateral_side_norm=contralateral_side_norm,
        )
    )
    return out


def _empty_sk_symmetry_features() -> dict[str, float]:
    return {column: 0.0 for column in _sk_symmetry_columns()}


def _sk_symmetry_columns() -> list[str]:
    return [
        "sk_meanrms1",
        "sk_weightedrms1",
        "sk_sigma_target1",
        "sk_sigma_contralateral1",
        "sk_mahalanobis1",
        "sk_meanrms2",
        "sk_weightedrms2",
        "sk_sigma_target2",
        "sk_sigma_contralateral2",
        "sk_mahalanobis2",
        "sk_peak14_intensity_abs_delta",
        "sk_mean_peak_value_abs_delta",
        "sk_wasserstein_distance_mu_tc",
        "sk_cosine_distance_full_q2",
        "sk_wasserstein_distance_full_q2",
    ]


def _sk_target_contralateral_symmetry_features(
    patient_df: pd.DataFrame,
    *,
    profile_column: str,
    q_column: str,
    side_column: str,
    target_side_norm: str,
    contralateral_side_norm: str | None,
) -> dict[str, float]:
    if contralateral_side_norm is None:
        return _empty_sk_symmetry_features()
    metrics = _sk_side_mean_metrics(
        patient_df,
        profile_column=profile_column,
        q_column=q_column,
        side_column=side_column,
        target_side_norm=target_side_norm,
        contralateral_side_norm=contralateral_side_norm,
        q_roi=(7.5, 23.0),
    )
    metrics_full = _sk_side_mean_metrics(
        patient_df,
        profile_column=profile_column,
        q_column=q_column,
        side_column=side_column,
        target_side_norm=target_side_norm,
        contralateral_side_norm=contralateral_side_norm,
        q_roi=(2.0, 23.0),
    )
    if not metrics or not metrics_full:
        return _empty_sk_symmetry_features()
    q = metrics["q"]
    mu_t = metrics["mu_target"]
    mu_c = metrics["mu_contralateral"]
    std_t = metrics["std_target"]
    std_c = metrics["std_contralateral"]
    mask1 = (q >= 7.0) & (q <= 15.0)
    mask2 = (q >= 15.0) & (q <= 23.0)
    return {
        "sk_meanrms1": _finite_or_zero(_rms_difference(mu_t, mu_c, mask1)),
        "sk_weightedrms1": _finite_or_zero(
            _weighted_rms_difference(mu_t, mu_c, std_t, std_c, mask1)
        ),
        "sk_sigma_target1": _finite_or_zero(_sigma_rms(std_t, mask1)),
        "sk_sigma_contralateral1": _finite_or_zero(_sigma_rms(std_c, mask1)),
        "sk_mahalanobis1": _finite_or_zero(
            _mahalanobis_difference(mu_t, mu_c, std_t, std_c, mask1)
        ),
        "sk_meanrms2": _finite_or_zero(_rms_difference(mu_t, mu_c, mask2)),
        "sk_weightedrms2": _finite_or_zero(
            _weighted_rms_difference(mu_t, mu_c, std_t, std_c, mask2)
        ),
        "sk_sigma_target2": _finite_or_zero(_sigma_rms(std_t, mask2)),
        "sk_sigma_contralateral2": _finite_or_zero(_sigma_rms(std_c, mask2)),
        "sk_mahalanobis2": _finite_or_zero(
            _mahalanobis_difference(mu_t, mu_c, std_t, std_c, mask2)
        ),
        "sk_peak14_intensity_abs_delta": _finite_or_zero(
            _peak14_intensity_abs_delta(q, mu_t, mu_c)
        ),
        "sk_mean_peak_value_abs_delta": _finite_or_zero(
            _mean_peak_value_abs_delta(
                patient_df,
                q_column=q_column,
                profile_column=profile_column,
                side_column=side_column,
                target_side_norm=target_side_norm,
                contralateral_side_norm=contralateral_side_norm,
            )
        ),
        "sk_wasserstein_distance_mu_tc": _finite_or_zero(
            _profile_wasserstein(q, mu_t, mu_c)
        ),
        "sk_cosine_distance_full_q2": _finite_or_zero(
            _cosine_distance(metrics_full["mu_target"], metrics_full["mu_contralateral"])
        ),
        "sk_wasserstein_distance_full_q2": _finite_or_zero(
            _profile_wasserstein(
                metrics_full["q"],
                metrics_full["mu_target"],
                metrics_full["mu_contralateral"],
            )
        ),
    }


def _side_profiles(
    df: pd.DataFrame,
    profile_column: str,
    side_column: str,
    side_norm: str | None,
) -> list[np.ndarray]:
    if side_norm is None:
        return []
    rows = df[df[side_column].map(_normalize_side) == side_norm]
    return [np.asarray(value, dtype=float).ravel() for value in rows[profile_column]]


def _sk_side_mean_metrics(
    df: pd.DataFrame,
    *,
    profile_column: str,
    q_column: str,
    side_column: str,
    target_side_norm: str,
    contralateral_side_norm: str,
    q_roi: tuple[float, float],
) -> dict[str, np.ndarray] | None:
    target_profiles: list[np.ndarray] = []
    contralateral_profiles: list[np.ndarray] = []
    q_common: np.ndarray | None = None
    for row in df.itertuples(index=False):
        side = _normalize_side(getattr(row, side_column))
        if side not in {target_side_norm, contralateral_side_norm}:
            continue
        q = np.asarray(getattr(row, q_column), dtype=float).ravel()
        y = np.asarray(getattr(row, profile_column), dtype=float).ravel()
        q, y = _profile_roi(q, y, q_roi)
        y = _normalize_profile_near_minimum(q, _smooth_profile(y))
        if q_common is None:
            q_common = q
        y_common = np.interp(q_common, q, y)
        if side == target_side_norm:
            target_profiles.append(y_common)
        else:
            contralateral_profiles.append(y_common)
    if q_common is None or not target_profiles or not contralateral_profiles:
        return None
    target = np.vstack(target_profiles)
    contralateral = np.vstack(contralateral_profiles)
    return {
        "q": q_common,
        "mu_target": np.mean(target, axis=0),
        "mu_contralateral": np.mean(contralateral, axis=0),
        "std_target": _profile_std(target),
        "std_contralateral": _profile_std(contralateral),
    }


def _profile_roi(
    q: np.ndarray,
    y: np.ndarray,
    q_roi: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    mask = (q >= float(q_roi[0])) & (q <= float(q_roi[1]))
    if int(mask.sum()) < 5:
        return q, y
    return q[mask], y[mask]


def _smooth_profile(y: np.ndarray) -> np.ndarray:
    if y.size < 7:
        return y
    window = min(11, y.size if y.size % 2 else y.size - 1)
    if window < 5:
        return y
    return savgol_filter(y, window_length=window, polyorder=min(3, window - 2))


def _normalize_profile_near_minimum(
    q: np.ndarray,
    y: np.ndarray,
    *,
    q0: float = 6.7,
    halfwidth: float = 0.25,
) -> np.ndarray:
    mask = (q >= q0 - halfwidth) & (q <= q0 + halfwidth) & np.isfinite(y)
    baseline = (
        float(np.nanpercentile(y[mask], 5))
        if int(mask.sum()) >= 2
        else float(np.nanpercentile(y, 5))
    )
    if not np.isfinite(baseline) or abs(baseline) < 1e-12:
        baseline = 1.0
    return y / baseline


def _profile_std(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros(values.shape[1])
    return np.std(values, axis=0, ddof=1)


def _rms_difference(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = np.asarray(a - b, dtype=float)
    good = mask & np.isfinite(diff)
    return float(np.sqrt(np.mean(diff[good] ** 2))) if int(good.sum()) >= 5 else np.nan


def _weighted_rms_difference(
    a: np.ndarray,
    b: np.ndarray,
    std_a: np.ndarray,
    std_b: np.ndarray,
    mask: np.ndarray,
) -> float:
    diff = np.asarray(a - b, dtype=float)
    var = np.asarray(std_a**2 + std_b**2, dtype=float)
    good = mask & np.isfinite(diff) & np.isfinite(var)
    if int(good.sum()) < 5:
        return np.nan
    floor = float(np.nanpercentile(var[good], 5))
    weight = 1.0 / np.maximum(var[good], floor + 1e-12)
    return float(np.sqrt(np.sum(weight * diff[good] ** 2) / np.sum(weight)))


def _mahalanobis_difference(
    a: np.ndarray,
    b: np.ndarray,
    std_a: np.ndarray,
    std_b: np.ndarray,
    mask: np.ndarray,
) -> float:
    diff = np.asarray(a - b, dtype=float)
    var = np.asarray(std_a**2 + std_b**2, dtype=float)
    good = mask & np.isfinite(diff) & np.isfinite(var)
    if int(good.sum()) < 5:
        return np.nan
    return float(np.sqrt(np.sum(diff[good] ** 2 / (var[good] + 1e-12))))


def _sigma_rms(std: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(std)
    return float(np.sqrt(np.mean(std[good] ** 2))) if int(good.sum()) >= 5 else np.nan


def _peak14_intensity_abs_delta(
    q: np.ndarray,
    target: np.ndarray,
    contralateral: np.ndarray,
) -> float:
    target_peak = _peak_value(q, target, q_min=13.5, q_max=14.5)
    contralateral_peak = _peak_value(q, contralateral, q_min=13.5, q_max=14.5)
    if not np.isfinite(target_peak) or not np.isfinite(contralateral_peak):
        return np.nan
    return float(abs(target_peak - contralateral_peak))


def _peak_value(
    q: np.ndarray,
    y: np.ndarray,
    *,
    q_min: float,
    q_max: float,
) -> float:
    mask = (q >= q_min) & (q <= q_max) & np.isfinite(y)
    return float(np.nanmax(y[mask])) if int(mask.sum()) >= 3 else np.nan


def _mean_peak_value_abs_delta(
    df: pd.DataFrame,
    *,
    q_column: str,
    profile_column: str,
    side_column: str,
    target_side_norm: str,
    contralateral_side_norm: str,
) -> float:
    target_peak = _mean_peak_value_for_side(
        df,
        q_column=q_column,
        profile_column=profile_column,
        side_column=side_column,
        side_norm=target_side_norm,
    )
    contralateral_peak = _mean_peak_value_for_side(
        df,
        q_column=q_column,
        profile_column=profile_column,
        side_column=side_column,
        side_norm=contralateral_side_norm,
    )
    if not np.isfinite(target_peak) or not np.isfinite(contralateral_peak):
        return np.nan
    return float(abs(target_peak - contralateral_peak))


def _mean_peak_value_for_side(
    df: pd.DataFrame,
    *,
    q_column: str,
    profile_column: str,
    side_column: str,
    side_norm: str,
) -> float:
    values = []
    subset = df[[side_column, q_column, profile_column]]
    for side, q_raw, y_raw in subset.itertuples(index=False, name=None):
        if _normalize_side(side) != side_norm:
            continue
        q = np.asarray(q_raw, dtype=float).ravel()
        y = np.asarray(y_raw, dtype=float).ravel()
        peak = _peak_value(q, y, q_min=13.0, q_max=14.8)
        if np.isfinite(peak):
            values.append(peak)
    return float(np.mean(values)) if values else np.nan


def _profile_wasserstein(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(q) & np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 5:
        return np.nan
    qv = q[good]
    av = np.clip(a[good], 0.0, None)
    bv = np.clip(b[good], 0.0, None)
    if float(av.sum()) <= 1e-12 or float(bv.sum()) <= 1e-12:
        return np.nan
    order = np.argsort(qv)
    qv = qv[order]
    av = av[order] / float(av.sum())
    bv = bv[order] / float(bv.sum())
    return float(np.sum(np.abs(np.cumsum(av)[:-1] - np.cumsum(bv)[:-1]) * np.diff(qv)))


def _normalize_side(value: Any) -> str | None:
    clean = str(value).strip().upper()
    if clean.startswith("LEFT"):
        return "LEFT"
    if clean.startswith("RIGHT"):
        return "RIGHT"
    return None


def _display_side(side_norm: str | None) -> str:
    if side_norm == "LEFT":
        return "Left"
    if side_norm == "RIGHT":
        return "Right"
    return ""


def _mean_pairwise_cosine(profiles: list[np.ndarray]) -> float:
    if len(profiles) < 2:
        return np.nan
    values = [
        _cosine_distance(profiles[i], profiles[j])
        for i in range(len(profiles))
        for j in range(i + 1, len(profiles))
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 3:
        return np.nan
    av = np.asarray(a[good], dtype=float)
    bv = np.asarray(b[good], dtype=float)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - np.clip(np.dot(av, bv) / denom, -1.0, 1.0))


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _numeric_median(df: pd.DataFrame, column: str, *, default: float) -> float:
    if column not in df.columns:
        return float(default)
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.median()) if values.notna().any() else float(default)


def _has_numeric(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()


def _boolean_series(values: pd.Series) -> pd.Series:
    clean = values.astype("object").where(values.notna(), False)
    if clean.dtype == bool:
        return clean
    return clean.astype(str).str.lower().isin(["true", "1", "yes"])


def _paired_breast_training_dataframe(
    df: pd.DataFrame,
    *,
    group_column: str,
    side_column: str,
    required: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Optionally retain only patients with valid left and right breast data."""
    _require_training_columns(df, [group_column, side_column])
    side_norm = df[side_column].map(_normalize_side)
    side_frame = pd.DataFrame(
        {
            "patient_id": df[group_column].astype(str),
            "side": side_norm,
        }
    ).dropna(subset=["side"])
    paired_ids = set(
        side_frame.groupby("patient_id")["side"]
        .agg(lambda values: set(values))
        .loc[lambda sides: sides.map(lambda values: values == {"LEFT", "RIGHT"})]
        .index
    )
    patient_ids = set(df[group_column].astype(str))
    if required:
        keep = df[group_column].astype(str).isin(paired_ids) & side_norm.notna()
        out = df.loc[keep].copy()
    else:
        out = df.copy()
    audit = {
        "required": required,
        "rule": "exactly LEFT and RIGHT breast sides" if required else "not applied",
        "patients_before": len(patient_ids),
        "patients_after": int(out[group_column].astype(str).nunique()),
        "patients_dropped": len(patient_ids) - int(out[group_column].astype(str).nunique()),
        "measurements_before": int(len(df)),
        "measurements_after": int(len(out)),
        "measurements_dropped": int(len(df) - len(out)),
    }
    if required and out.empty:
        raise ValueError("Paired-breast eligibility removed every training patient.")
    return out.reset_index(drop=True), audit


def _profile_logistic(*, logreg_c: float, random_state: int) -> Pipeline:
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


def _scalar_logistic(*, logreg_c: float, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
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


def _patient_metric_row(
    model_name: str,
    split_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score: np.ndarray,
    thresholds: dict[str, Any],
    *,
    evaluation_mode: str,
) -> dict[str, Any]:
    y = test_df["label"].to_numpy(dtype=int)
    pred = (score >= float(thresholds["threshold_target"])).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "model_name": model_name,
        "split_id": int(split_id),
        "evaluation_mode": evaluation_mode,
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "sensitivity_target": sensitivity,
        "specificity_target": specificity,
        "balanced_accuracy_target": _mean_finite([sensitivity, specificity]),
        "ppv_target": _ratio(tp, tp + fp),
        "npv_target": _ratio(tn, tn + fn),
        "tp_target": int(tp),
        "tn_target": int(tn),
        "fp_target": int(fp),
        "fn_target": int(fn),
        "train_patients": int(train_df["patientId"].nunique()),
        "test_patients": int(test_df["patientId"].nunique()),
        "train_cancer_patients": int((train_df["label"] == 1).sum()),
        "test_cancer_patients": int((test_df["label"] == 1).sum()),
        **thresholds,
    }


def _patient_prediction_frame(
    model_name: str,
    split_id: int,
    test_df: pd.DataFrame,
    score: np.ndarray,
    thresholds: dict[str, Any],
    *,
    evaluation_mode: str,
) -> pd.DataFrame:
    out = test_df[["patientId", "label", "label_name"]].copy()
    out["model_name"] = model_name
    out["split_id"] = int(split_id)
    out["evaluation_mode"] = evaluation_mode
    out["p_cancer"] = np.asarray(score, dtype=float)
    out["threshold_youden"] = float(thresholds["threshold_youden"])
    out["threshold_target"] = float(thresholds["threshold_target"])
    out["y_pred_target"] = (out["p_cancer"] >= out["threshold_target"]).astype(int)
    return out


def _pooled_patient_metrics(
    predictions: pd.DataFrame,
    *,
    evaluation_mode: str,
) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model_name", sort=False):
        y = group["label"].to_numpy(dtype=int)
        score = group["p_cancer"].to_numpy(dtype=float)
        pred = group["y_pred_target"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = _ratio(tp, tp + fn)
        specificity = _ratio(tn, tn + fp)
        rows.append(
            {
                "model_name": model_name,
                "split_id": -1,
                "evaluation_mode": evaluation_mode,
                "roc_auc": float(roc_auc_score(y, score)),
                "pr_auc": float(average_precision_score(y, score)),
                "sensitivity_target": sensitivity,
                "specificity_target": specificity,
                "balanced_accuracy_target": _mean_finite([sensitivity, specificity]),
                "ppv_target": _ratio(tp, tp + fp),
                "npv_target": _ratio(tn, tn + fn),
                "tp_target": int(tp),
                "tn_target": int(tn),
                "fp_target": int(fp),
                "fn_target": int(fn),
                "train_patients": None,
                "test_patients": int(group["patientId"].nunique()),
                "train_cancer_patients": None,
                "test_cancer_patients": int((group["label"] == 1).sum()),
                "threshold_youden": float(group["threshold_youden"].median()),
                "threshold_target": float(group["threshold_target"].median()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_patient_model_metrics(split_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in split_metrics.groupby("model_name", sort=False):
        evaluation_modes = sorted(group["evaluation_mode"].dropna().astype(str).unique())
        rows.append(
            {
                "model_name": model_name,
                "evaluation_mode": evaluation_modes[0] if len(evaluation_modes) == 1 else ",".join(evaluation_modes),
                "splits": int(len(group)),
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_std": float(group["roc_auc"].std(ddof=0)),
                "pr_auc_mean": float(group["pr_auc"].mean()),
                "sensitivity_target_mean": float(group["sensitivity_target"].mean()),
                "sensitivity_target_std": float(group["sensitivity_target"].std(ddof=0)),
                "specificity_target_mean": float(group["specificity_target"].mean()),
                "specificity_target_std": float(group["specificity_target"].std(ddof=0)),
                "balanced_accuracy_target_mean": float(
                    group["balanced_accuracy_target"].mean()
                ),
                "ppv_target_mean": float(group["ppv_target"].mean()),
                "npv_target_mean": float(group["npv_target"].mean()),
                "true_positives_mean": float(group["tp_target"].mean()),
                "true_negatives_mean": float(group["tn_target"].mean()),
                "false_negatives_mean": float(group["fn_target"].mean()),
                "false_positives_mean": float(group["fp_target"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _patient_dataset_summary(
    df: pd.DataFrame,
    feature_table: pd.DataFrame,
    lr1_rows: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "patients": int(df["patientId"].astype(str).nunique()),
                "specimens": int(df["specimenId"].astype(str).nunique()),
                "lr1_rows": int(len(lr1_rows)),
                "lr1_patients": int(lr1_rows["patientId"].astype(str).nunique()),
                "final_patients": int(len(feature_table)),
                "final_cancer_patients": int((feature_table["label"] == 1).sum()),
                "final_benign_patients": int((feature_table["label"] == 0).sum()),
            }
        ]
    )


def _require_training_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing training columns: {missing}")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _mean_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _prediction_preprocessing_payload(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    config_text = config_path.read_text(encoding="utf-8")
    config = load_preprocessing_config(config_path)
    return {
        "path": str(config_path),
        "config": config,
        "config_text": config_text,
        "config_sha256": sha256(config_text.encode("utf-8")).hexdigest(),
    }


def _preprocessing_lineage_fields(
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None,
) -> dict[str, Any]:
    training_config = preprocessing_artifact.get("preprocessing_config")
    training_config_text = preprocessing_artifact.get("preprocessing_config_text")
    training_config_sha256 = preprocessing_artifact.get("preprocessing_config_sha256")
    fields = {
        "preprocessing_config_sha256": training_config_sha256,
        "training_preprocessing_config": training_config,
        "training_preprocessing_config_text": training_config_text,
        "training_preprocessing_config_sha256": training_config_sha256,
        "preprocessing_metadata": preprocessing_artifact.get("metadata", {}),
    }
    if prediction_preprocessing is None:
        fields.update(
            {
                "prediction_preprocessing_config": None,
                "prediction_preprocessing_config_text": None,
                "prediction_preprocessing_config_sha256": None,
                "prediction_preprocessing_config_path": None,
            }
        )
        return fields
    fields.update(
        {
            "prediction_preprocessing_config": prediction_preprocessing["config"],
            "prediction_preprocessing_config_text": prediction_preprocessing[
                "config_text"
            ],
            "prediction_preprocessing_config_sha256": prediction_preprocessing[
                "config_sha256"
            ],
            "prediction_preprocessing_config_path": prediction_preprocessing["path"],
        }
    )
    return fields


def _validate_training_config(config: dict[str, Any], config_path: Path) -> None:
    if not isinstance(config, dict):
        raise TypeError(f"Training config must be a mapping: {config_path}")
    missing = [
        section
        for section in ("training", "io", "model", "evaluation")
        if section not in config
    ]
    if missing:
        raise ValueError(f"Missing training config sections: {missing}")
    if not config.get("io", {}).get("input_dataframe_joblib_path"):
        raise ValueError(f"Missing io.input_dataframe_joblib_path in {config_path}")
    if not config.get("io", {}).get("output_model_joblib_path"):
        raise ValueError(f"Missing io.output_model_joblib_path in {config_path}")
    if not config.get("io", {}).get("prediction_preprocessing_config_path"):
        raise ValueError(
            f"Missing io.prediction_preprocessing_config_path in {config_path}"
        )


def _config_path(config: dict[str, Any], config_path: Path, key: str) -> Path:
    value = config.get("io", {}).get(key)
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _optional_config_path(
    config: dict[str, Any],
    config_path: Path,
    key: str,
) -> Path | None:
    value = config.get("io", {}).get(key)
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _write_json_summary(artifact: dict[str, Any], output_path: Path) -> None:
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_training_summary_payload(artifact), indent=2),
        encoding="utf-8",
    )


def _write_yaml_description(artifact: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(_training_summary_payload(artifact), sort_keys=False),
        encoding="utf-8",
    )


def _training_summary_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": artifact.get("kind"),
        "version": artifact.get("version"),
        "created_at": artifact.get("created_at"),
        "model_type": artifact.get("model_type"),
        "model_descriptions": artifact.get("model_descriptions", {}),
        "dataset_summary": _records(artifact.get("dataset_summary")),
        "metric_summary": _records(artifact.get("metric_summary")),
        "training_config_sha256": artifact.get("training_config_sha256"),
        "preprocessing_config_sha256": artifact.get("preprocessing_config_sha256"),
        "training_preprocessing_config_sha256": artifact.get(
            "training_preprocessing_config_sha256"
        ),
        "prediction_preprocessing_config_sha256": artifact.get(
            "prediction_preprocessing_config_sha256"
        ),
        "input_dataframe_joblib_sha256": artifact.get("input_dataframe_joblib_sha256"),
        "metadata": artifact.get("metadata", {}),
    }


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aramis_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.exists():
        with pyproject_path.open("rb") as handle:
            pyproject = tomllib.load(handle)
        return str(pyproject.get("project", {}).get("version", "unknown"))
    try:
        return version("aramis")
    except PackageNotFoundError:
        return "unknown"


def _aramis_git_sha() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return "unavailable"
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
