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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
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

TARGET_CASE_ID = "target_case_id"


class PatientModelInputBuilder(BaseEstimator):
    """Build target-breast cases used by Aramis M0/M1/M2 models.

    The input is a preprocessed measurement-level DataFrame. The builder first
    selects LR1 training rows according to the product policy, trains the profile
    LogisticRegression, scores measurements, and aggregates target-breast scores
    to one `p_cancer` value per target case by averaging LR1 evidence in logit
    space. Each biopsied breast is one historical target case; a bilateral
    biopsy patient therefore contributes two cases. Contralateral measurements
    supply optional symmetry context only. Patient-safe splitting keeps every
    patient's cases in one fold.
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
        """Return one row per biopsied target breast with model feature fields."""
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
        """Fit LR1 and return the target-breast feature table."""
        return self.fit(x, y).transform(x)


class PatientModelSetTrainer(BaseEstimator):
    """Train selected Aramis target-breast models.

    The trainer consumes the target-case feature table from
    `PatientModelInputBuilder` and its retained LR1 measurement rows. M1/M1Q
    and M2/M2Q use one final LogisticRegression. It receives SK terms only as
    a gated optional refinement: all SK terms are zero when contralateral data
    is unavailable. Reliability remains a report field, not a model feature.
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
        """Fit final target-breast models and store them in `models_`."""
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
        """Fit the full target-breast training route and build `artifact_`."""
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
        random_state = int(evaluation_config.get("random_state", 42))
        target_sensitivity = float(evaluation_config.get("target_sensitivity", 0.95))
        self.hyperparameter_selection_ = None
        if evaluation_config.get("nested", {}).get("enabled", False):
            self.hyperparameter_selection_ = _select_nested_hyperparameters(
                x,
                selected_models=selected_models,
                evaluation_config=evaluation_config,
                profile_column=profile_column,
                label_column=label_column,
                group_column=group_column,
                specimen_column=specimen_column,
                side_column=side_column,
                q_column=q_column,
                age_column=age_column,
                biopsy_column=biopsy_column,
                lr1_row_policy=lr1_row_policy,
                random_state=random_state,
                target_sensitivity=target_sensitivity,
            )
            lr1_logreg_c = float(self.hyperparameter_selection_["lr1_c"])
            lr2_logreg_c = float(self.hyperparameter_selection_["lr2_c"])

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
        self.feature_table_ = self.input_builder_.fit_transform(x)
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
        if self.hyperparameter_selection_ is not None:
            _apply_oof_thresholds(
                self.model_trainer_.models_,
                self.hyperparameter_selection_["thresholds"],
            )
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
        self.evaluator_.fit(x)
        self.artifact_ = _patient_training_artifact(
            df=x,
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
            hyperparameter_selection=self.hyperparameter_selection_,
        )
        return self


def build_patient_training_pipeline(
    *,
    config: dict[str, Any],
    config_text: str,
    input_dataframe_joblib_path: str | Path,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None = None,
) -> Pipeline:
    """Return one sklearn Pipeline for patient-safe target-breast training."""
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
    """Train one artifact containing M0, M1, and M2 target-breast models."""
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
    hyperparameter_selection: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the traceable joblib payload for target-breast model training."""
    evaluation_config = config.get("evaluation", {})
    metric_summary = _summarize_patient_model_metrics(
        split_metrics,
        split_predictions,
        random_state=int(evaluation_config.get("random_state", 42)),
        bootstrap_samples=int(evaluation_config.get("bootstrap_samples", 0)),
    )
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
        "hyperparameter_selection": hyperparameter_selection,
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
    for model_name in ("M0", "M0Q"):
        if model_name not in selected_models:
            continue
        models[model_name] = {
            "name": _patient_model_descriptions()[model_name]["name"],
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
        if model_name == "M0Q":
            continue
        if model_name in _gated_model_names():
            final_model = GatedSymmetryLogistic(
                include_age=model_name in {"M2", "M2Q"},
                logreg_c=lr2_logreg_c,
                random_state=random_state,
            ).fit(feature_table, y)
            score = final_model.predict_proba(feature_table)[:, 1]
            models[model_name] = {
                "name": _patient_model_descriptions()[model_name]["name"],
                "lr1_model": lr1_model,
                "final_model": final_model,
                "feature_columns": _gated_model_input_columns(model_name),
                "symmetry_policy": "single_model_gated_optional_refinement",
                "symmetry_gate": "symmetry_available",
                "thresholds": compute_binary_thresholds(
                    y,
                    score,
                    target_sensitivity=target_sensitivity,
                ),
            }
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


def _apply_oof_thresholds(
    models: dict[str, Any],
    thresholds_by_model: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for model_name, route_thresholds in thresholds_by_model.items():
        models[model_name]["thresholds"] = dict(route_thresholds["default"])


def _fit_split_feature_tables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
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
        random_state=random_state,
    )
    lr1_model = _profile_logistic(
        logreg_c=lr1_logreg_c,
        random_state=random_state,
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
    return train_features, test_features


def _select_nested_hyperparameters(
    df: pd.DataFrame,
    *,
    selected_models: Sequence[str],
    evaluation_config: dict[str, Any],
    profile_column: str,
    label_column: str,
    group_column: str,
    specimen_column: str,
    side_column: str,
    q_column: str,
    age_column: str,
    biopsy_column: str,
    lr1_row_policy: str,
    random_state: int,
    target_sensitivity: float,
) -> dict[str, Any]:
    nested = evaluation_config.get("nested", {})
    c1_grid = [float(value) for value in nested.get("lr1_c_grid", [0.1, 0.3])]
    c2_grid = [float(value) for value in nested.get("lr2_c_grid", [0.1, 0.3])]
    inner_n_splits = int(nested.get("inner_n_splits", 4))
    inner_n_repeats = int(nested.get("inner_n_repeats", 1))
    candidates = []
    for lr1_c in c1_grid:
        for lr2_c in c2_grid:
            oof = _inner_oof_model_scores(
                df,
                selected_models=["M2Q"],
                profile_column=profile_column,
                label_column=label_column,
                group_column=group_column,
                specimen_column=specimen_column,
                side_column=side_column,
                q_column=q_column,
                age_column=age_column,
                biopsy_column=biopsy_column,
                lr1_row_policy=lr1_row_policy,
                lr1_logreg_c=lr1_c,
                lr2_logreg_c=lr2_c,
                n_splits=inner_n_splits,
                n_repeats=inner_n_repeats,
                random_state=random_state,
            )["M2Q"]
            candidates.append(
                {
                    "lr1_c": lr1_c,
                    "lr2_c": lr2_c,
                    "roc_auc": float(
                        roc_auc_score(oof["label"], oof["operational_score"])
                    ),
                }
            )
    selected = max(
        candidates,
        key=lambda row: (row["roc_auc"], -row["lr1_c"], -row["lr2_c"]),
    )
    selected_oof = _inner_oof_model_scores(
        df,
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
        lr1_logreg_c=selected["lr1_c"],
        lr2_logreg_c=selected["lr2_c"],
        n_splits=inner_n_splits,
        n_repeats=inner_n_repeats,
        random_state=random_state,
    )
    return {
        **selected,
        "selection_metric": "inner_oof_operational_roc_auc",
        "candidate_metrics": candidates,
        "thresholds": {
            model_name: _oof_thresholds(
                frame,
                target_sensitivity=target_sensitivity,
            )
            for model_name, frame in selected_oof.items()
        },
    }


def _inner_oof_model_scores(
    df: pd.DataFrame,
    *,
    selected_models: Sequence[str],
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
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    base_features = _patient_feature_table(
        df,
        _empty_lr1_scores(
            df,
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
    )
    records: dict[str, list[pd.DataFrame]] = {
        model_name: [] for model_name in selected_models
    }
    for split_id, (train_idx, test_idx) in enumerate(
        _patient_split_pairs(
            mode="stratified_kfold",
            base_features=base_features,
            y_patients=base_features["label"].to_numpy(dtype=int),
            n_splits=n_splits,
            n_repeats=n_repeats,
            test_size=0.30,
            random_state=random_state,
        )
    ):
        train_patients = set(base_features.iloc[train_idx]["patientId"].astype(str))
        test_patients = set(base_features.iloc[test_idx]["patientId"].astype(str))
        train_df = df[df[group_column].astype(str).isin(train_patients)].copy()
        test_df = df[df[group_column].astype(str).isin(test_patients)].copy()
        train_features, test_features = _fit_split_feature_tables(
            train_df,
            test_df,
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
        for model_name in selected_models:
            (
                _,
                test_score,
                _,
                test_routes,
                _,
                test_route_scores,
            ) = _split_model_scores(
                model_name,
                train_features,
                test_features,
                lr2_logreg_c=lr2_logreg_c,
                random_state=random_state + split_id,
            )
            out = test_features[[TARGET_CASE_ID, "patientId", "label"]].copy()
            out["operational_score"] = test_score
            out["model_route"] = test_routes
            records[model_name].append(out)
    return {
        model_name: _average_oof_target_case_scores(pd.concat(frames, ignore_index=True))
        for model_name, frames in records.items()
    }


def _average_oof_target_case_scores(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "patientId": ("patientId", "first"),
        "label": ("label", "first"),
        "operational_score": ("operational_score", "mean"),
        "model_route": ("model_route", "first"),
    }
    return frame.groupby(TARGET_CASE_ID, as_index=False).agg(**aggregations)


def _oof_thresholds(
    oof: pd.DataFrame,
    *,
    target_sensitivity: float,
) -> dict[str, dict[str, Any]]:
    y = oof["label"].to_numpy(dtype=int)
    return {
        "default": compute_binary_thresholds(
            y,
            oof["operational_score"].to_numpy(dtype=float),
            target_sensitivity=target_sensitivity,
        )
    }


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
    nested_enabled = bool(evaluation_config.get("nested", {}).get("enabled", False))
    base_features = _patient_feature_table(
        df,
        _empty_lr1_scores(
            df,
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
        nested_selection = None
        split_lr1_c = lr1_logreg_c
        split_lr2_c = lr2_logreg_c
        if nested_enabled:
            nested_selection = _select_nested_hyperparameters(
                train_df,
                selected_models=selected_models,
                evaluation_config=evaluation_config,
                profile_column=profile_column,
                label_column=label_column,
                group_column=group_column,
                specimen_column=specimen_column,
                side_column=side_column,
                q_column=q_column,
                age_column=age_column,
                biopsy_column=biopsy_column,
                lr1_row_policy=lr1_row_policy,
                random_state=random_state + split_id,
                target_sensitivity=target_sensitivity,
            )
            split_lr1_c = float(nested_selection["lr1_c"])
            split_lr2_c = float(nested_selection["lr2_c"])
        train_features, test_features = _fit_split_feature_tables(
            train_df,
            test_df,
            profile_column=profile_column,
            label_column=label_column,
            group_column=group_column,
            specimen_column=specimen_column,
            side_column=side_column,
            q_column=q_column,
            age_column=age_column,
            biopsy_column=biopsy_column,
            lr1_row_policy=lr1_row_policy,
            lr1_logreg_c=split_lr1_c,
            random_state=random_state + split_id,
        )
        for model_name in selected_models:
            (
                train_score,
                test_score,
                train_routes,
                test_routes,
                train_route_scores,
                test_route_scores,
            ) = _split_model_scores(
                model_name,
                train_features,
                test_features,
                lr2_logreg_c=split_lr2_c,
                random_state=random_state + split_id,
            )
            route_thresholds = (
                nested_selection["thresholds"][model_name]
                if nested_selection is not None
                else _route_thresholds(
                    train_features["label"].to_numpy(dtype=int),
                    train_route_scores,
                    target_sensitivity=target_sensitivity,
                )
            )
            thresholds = _threshold_summary(route_thresholds)
            thresholds["selected_lr1_c"] = split_lr1_c
            thresholds["selected_lr2_c"] = split_lr2_c
            test_thresholds = _route_threshold_values(test_routes, route_thresholds)
            if mode != "loovm":
                metrics.append(
                    _patient_metric_row(
                        model_name,
                        split_id,
                        train_features,
                        test_features,
                        test_score,
                        thresholds,
                        test_thresholds,
                        evaluation_mode=mode,
                        evaluation_view="operational",
                    )
                )
            predictions.append(
                _patient_prediction_frame(
                    model_name,
                    split_id,
                    test_features,
                    test_score,
                    thresholds,
                    test_routes,
                    test_thresholds,
                    evaluation_mode=mode,
                    evaluation_view="operational",
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
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    if model_name in {"M0", "M0Q"}:
        train_score = train_features["profile_p_cancer_logit_average"].to_numpy(dtype=float)
        test_score = test_features["profile_p_cancer_logit_average"].to_numpy(dtype=float)
        return (
            train_score,
            test_score,
            _default_routes(train_score),
            _default_routes(test_score),
            {"default": train_score},
            {"default": test_score},
        )
    if model_name in _gated_model_names():
        model = GatedSymmetryLogistic(
            include_age=model_name in {"M2", "M2Q"},
            logreg_c=lr2_logreg_c,
            random_state=random_state,
        ).fit(train_features, train_features["label"].to_numpy(dtype=int))
        train_score = model.predict_proba(train_features)[:, 1]
        test_score = model.predict_proba(test_features)[:, 1]
        return (
            train_score,
            test_score,
            _default_routes(train_features),
            _default_routes(test_features),
            {"default": train_score},
            {"default": test_score},
        )
    columns = _patient_model_feature_columns()[model_name]
    model = _scalar_logistic(logreg_c=lr2_logreg_c, random_state=random_state)
    model.fit(train_features[columns], train_features["label"].to_numpy(dtype=int))
    return (
        model.predict_proba(train_features[columns])[:, 1],
        model.predict_proba(test_features[columns])[:, 1],
        _default_routes(train_features),
        _default_routes(test_features),
        {"default": model.predict_proba(train_features[columns])[:, 1]},
        {"default": model.predict_proba(test_features[columns])[:, 1]},
    )


def _selected_patient_models(model_config: dict[str, Any]) -> list[str]:
    selected = model_config.get(
        "selected_models",
        ["M1Q"],
    )
    if isinstance(selected, str):
        selected = [selected]
    out = [str(model_name).upper() for model_name in selected]
    supported = {"A0", "M0", "M0Q", "M1", "M1Q", "M2", "M2Q"}
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
    patient_table = (
        base_features.groupby("patientId", as_index=False)["label"]
        .max()
        .assign(patientId=lambda frame: frame["patientId"].astype(str))
    )
    patient_indices = np.arange(len(patient_table))
    patient_labels = patient_table["label"].to_numpy(dtype=int)

    def case_indices(patient_index: np.ndarray) -> np.ndarray:
        patient_ids = set(patient_table.iloc[patient_index]["patientId"])
        return base_features.index[
            base_features["patientId"].astype(str).isin(patient_ids)
        ].to_numpy()

    if mode == "repeated_stratified_shuffle":
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
        )
        return [
            (case_indices(train_index), case_indices(test_index))
            for train_index, test_index in splitter.split(patient_table, patient_labels)
        ]
    if mode == "stratified_kfold":
        if n_repeats > 1:
            splitter = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            return [
                (case_indices(train_index), case_indices(test_index))
                for train_index, test_index in splitter.split(patient_table, patient_labels)
            ]
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return [
            (case_indices(train_index), case_indices(test_index))
            for train_index, test_index in splitter.split(patient_table, patient_labels)
        ]
    if mode == "loovm":
        return [
            (case_indices(np.delete(patient_indices, test_idx)), case_indices(np.asarray([test_idx])))
            for test_idx in range(len(patient_indices))
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
        (
            train_score,
            score,
            train_routes,
            routes,
            train_route_scores,
            route_scores,
        ) = _split_model_scores(
            model_name,
            feature_table,
            feature_table,
            lr2_logreg_c=lr2_logreg_c,
            random_state=random_state,
        )
        route_thresholds = _route_thresholds(
            feature_table["label"].to_numpy(dtype=int),
            train_route_scores,
            target_sensitivity=target_sensitivity,
        )
        thresholds = _threshold_summary(route_thresholds)
        decision_thresholds = _route_threshold_values(routes, route_thresholds)
        metrics.append(
            _patient_metric_row(
                model_name,
                0,
                feature_table,
                feature_table,
                score,
                thresholds,
                decision_thresholds,
                evaluation_mode="all_on_all",
                evaluation_view="operational",
            )
        )
        predictions.append(
            _patient_prediction_frame(
                model_name,
                0,
                feature_table,
                score,
                thresholds,
                routes,
                decision_thresholds,
                evaluation_mode="all_on_all",
                evaluation_view="operational",
            )
        )
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


SK_CORE4_FEATURE_COLUMNS = (
    "sk_wasserstein_distance_full_q2",
    "sk_weightedrms1",
    "sk_weightedrms2",
    "sk_mean_peak_value_abs_delta",
)


class GatedSymmetryLogistic(BaseEstimator):
    """One LR2 model with a neutral symmetry correction when pairing is absent.

    The base profile and optional age terms are always evaluated. SK Core4
    terms are standardized from paired training cases only and are set to zero
    whenever `symmetry_available` is false. The availability flag is therefore
    a gate, not a learned diagnostic predictor and not a second LR2 route.
    """

    def __init__(
        self,
        *,
        include_age: bool,
        logreg_c: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.include_age = include_age
        self.logreg_c = logreg_c
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "GatedSymmetryLogistic":
        base = x.loc[:, self.base_feature_columns_].apply(pd.to_numeric, errors="coerce")
        self.base_fill_values_ = base.median().fillna(0.0)
        base_values = base.fillna(self.base_fill_values_).to_numpy(dtype=float)
        self.base_scaler_ = StandardScaler().fit(base_values)

        paired = x["symmetry_available"].astype(bool).to_numpy()
        symmetry = x.loc[:, SK_CORE4_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
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

    @property
    def base_feature_columns_(self) -> list[str]:
        return [
            "profile_p_cancer_logit_average",
            *( ["age", "age_available"] if self.include_age else [] ),
        ]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.logreg_.predict_proba(self._matrix(x))

    def _matrix(self, x: pd.DataFrame) -> np.ndarray:
        base = x.loc[:, self.base_feature_columns_].apply(pd.to_numeric, errors="coerce")
        base_scaled = self.base_scaler_.transform(
            base.fillna(self.base_fill_values_).to_numpy(dtype=float)
        )
        paired = x["symmetry_available"].astype(bool).to_numpy()
        symmetry = x.loc[:, SK_CORE4_FEATURE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(self.symmetry_means_)
        symmetry_scaled = (
            symmetry.to_numpy(dtype=float) - self.symmetry_means_.to_numpy(dtype=float)
        ) / self.symmetry_scales_.to_numpy(dtype=float)
        symmetry_scaled[~paired, :] = 0.0
        return np.hstack([base_scaled, symmetry_scaled])


def _gated_model_names() -> set[str]:
    return {"M1", "M1Q", "M2", "M2Q"}


def _gated_model_input_columns(model_name: str) -> list[str]:
    include_age = model_name in {"M2", "M2Q"}
    return [
        "profile_p_cancer_logit_average",
        *(["age", "age_available"] if include_age else []),
        *SK_CORE4_FEATURE_COLUMNS,
        "symmetry_available",
    ]


def _default_routes(values: Any) -> np.ndarray:
    return np.full(len(values), "default", dtype=object)


def _route_thresholds(
    labels: np.ndarray,
    route_scores: dict[str, np.ndarray],
    *,
    target_sensitivity: float,
) -> dict[str, dict[str, Any]]:
    return {
        "default": compute_binary_thresholds(
            labels,
            route_scores["default"],
            target_sensitivity=target_sensitivity,
        )
    }


def _route_threshold_values(
    routes: np.ndarray,
    route_thresholds: dict[str, dict[str, Any]],
) -> np.ndarray:
    return np.asarray(
        [route_thresholds[str(route)]["threshold_target"] for route in routes],
        dtype=float,
    )


def _threshold_summary(route_thresholds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return dict(route_thresholds["default"])


def _patient_model_feature_columns() -> dict[str, list[str]]:
    sk_symmetry = [
        "profile_p_cancer_logit_average",
        *SK_CORE4_FEATURE_COLUMNS,
    ]
    return {
        "A0": ["age", "age_available"],
        "M0Q": ["profile_p_cancer_logit_average"],
        "M1": sk_symmetry,
        "M1Q": sk_symmetry,
        "M2": [*sk_symmetry, "age", "age_available"],
        "M2Q": [*sk_symmetry, "age", "age_available"],
    }


def _patient_model_descriptions() -> dict[str, dict[str, Any]]:
    return {
        "M0": {
            "name": "M0 profile only",
            "description": "LR1 profile LogisticRegression, logit-averaged to patient p_cancer.",
        },
        "A0": {
            "name": "A0 age only",
            "description": "Age and age availability only; shortcut-risk control model.",
        },
        "M0Q": {
            "name": "M0Q profile with separate reliability reporting",
            "description": "Same prediction as M0; measurement count affects reliability reporting only.",
        },
        "M1": {
            "name": "M1 profile plus SK symmetry",
            "description": "LR1 target-breast p_cancer plus same-patient target/contralateral SK symmetry block.",
        },
        "M1Q": {
            "name": "M1Q profile plus gated SK Core4 with reliability reporting",
            "description": "One final model: SK Core4 refines the profile score only when paired symmetry is available; measurement counts affect reliability reporting only.",
        },
        "M2": {
            "name": "M2 profile plus SK symmetry plus age",
            "description": "M1 plus age and age availability flag.",
        },
        "M2Q": {
            "name": "M2Q profile, gated SK Core4 refinement, and age with reliability reporting",
            "description": "One final model: profile and age are always evaluated; SK Core4 adds a neutral-gated refinement only when paired symmetry is available. Measurement counts affect reliability reporting only.",
        },
    }


def _patient_model_feature_schema(selected_models: Sequence[str]) -> dict[str, Any]:
    feature_columns = {"M0": ["profile_p_cancer_logit_average"]}
    feature_columns.update(_patient_model_feature_columns())
    schema = {
        model_name: {
            "feature_columns": feature_columns[model_name],
            "unit": "target_breast_case",
            "label": "BENIGN vs CANCER decision-support class",
        }
        for model_name in selected_models
    }
    for model_name in _gated_model_names().intersection(schema):
        schema[model_name] = {
            "feature_columns": _gated_model_input_columns(model_name),
            "learned_feature_columns": _patient_model_feature_columns()[model_name],
            "symmetry_gate": "symmetry_available",
            "symmetry_policy": "single_model_gated_optional_refinement",
            "reliability_fields": [
                "profile_p_cancer_n_measurements",
                "target_measurements",
                "contralateral_measurements",
                "symmetry_available",
            ],
            "unit": "target_breast_case",
            "label": "BENIGN vs CANCER decision-support class",
        }
    return schema


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
            "Q models report measurement sufficiency separately; reliability fields are not model predictors."
        )
    unavailable = int((feature_table["symmetry_available"] == 0).sum())
    if unavailable:
        warnings.append(
            f"{unavailable} target-breast cases have unavailable paired-breast symmetry features."
        )
    low_target = int((feature_table["target_measurements"] < 3).sum())
    if low_target:
        warnings.append(
            f"{low_target} target-breast cases have fewer than 3 valid target-breast measurements."
        )
    low_contralateral = int((feature_table["contralateral_measurements"] < 3).sum())
    if low_contralateral:
        warnings.append(
            f"{low_contralateral} target-breast cases have fewer than 3 valid contralateral-breast measurements."
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


def _target_breast_cases(
    df: pd.DataFrame,
    *,
    group_column: str,
    side_column: str,
    label_column: str,
    biopsy_column: str,
) -> pd.DataFrame:
    """Return one historical target case for every biopsied breast."""
    biopsy_rows = df[
        df[label_column].isin(LABEL_MAP) & _boolean_series(df[biopsy_column])
    ].copy()
    records: list[dict[str, Any]] = []
    for patient_id, patient_df in df.groupby(group_column, sort=True):
        patient_biopsy = biopsy_rows[
            biopsy_rows[group_column].astype(str) == str(patient_id)
        ]
        for target_side in sorted(
            side
            for side in patient_biopsy[side_column].map(_normalize_side).dropna().unique()
        ):
            target_rows = patient_biopsy[
                patient_biopsy[side_column].map(_normalize_side) == target_side
            ]
            labels = target_rows[label_column].map(LABEL_MAP).dropna().unique()
            if len(labels) != 1:
                raise ValueError(
                    f"Target breast {patient_id!r}/{target_side!r} has ambiguous labels."
                )
            sides = set(patient_df[side_column].map(_normalize_side).dropna())
            contralateral = next((side for side in sides if side != target_side), None)
            records.append(
                {
                    TARGET_CASE_ID: f"{patient_id}::{target_side}",
                    group_column: str(patient_id),
                    "target_side_norm": target_side,
                    "target_side": _display_side(target_side),
                    "contralateral_side_norm": contralateral,
                    "contralateral_side": _display_side(contralateral),
                    "label": int(labels[0]),
                }
            )
    cases = pd.DataFrame(records)
    if cases.empty:
        raise ValueError("No biopsied target-breast cases are available.")
    return cases


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
    target_cases = _target_breast_cases(
        full_df,
        group_column=group_column,
        side_column=side_column,
        label_column=label_column,
        biopsy_column=biopsy_column,
    )
    grouped_rows = []
    for target in target_cases.itertuples(index=False):
        group = out[out[group_column].astype(str) == str(getattr(target, group_column))]
        target_scores = group.loc[
            group["_side_norm"] == target.target_side_norm,
            "lr1_measurement_p_cancer",
        ].to_numpy(dtype=float)
        if target_scores.size == 0:
            raise ValueError(
                f"No LR1 target-side scores for {target.target_case_id!r}; "
                "check target-side policy and lr1_row_policy."
            )
        grouped_rows.append(
            {
                TARGET_CASE_ID: target.target_case_id,
                "profile_p_cancer_probability_mean": float(np.mean(target_scores)),
                "profile_p_cancer_logit_average": _logit_average_probability(
                    target_scores
                ),
                "profile_p_cancer_n_measurements": int(target_scores.size),
            }
        )
    return pd.DataFrame(grouped_rows)


def _empty_lr1_scores(
    df: pd.DataFrame,
    *,
    group_column: str,
    side_column: str,
    label_column: str,
    biopsy_column: str,
) -> pd.DataFrame:
    cases = _target_breast_cases(
        df,
        group_column=group_column,
        side_column=side_column,
        label_column=label_column,
        biopsy_column=biopsy_column,
    )
    return cases[[TARGET_CASE_ID]].assign(
        profile_p_cancer_probability_mean=0.5,
        profile_p_cancer_logit_average=0.5,
        profile_p_cancer_n_measurements=0,
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
            biopsy_column,
        ],
    )
    rows = []
    target_cases = _target_breast_cases(
        df,
        group_column=group_column,
        side_column=side_column,
        label_column=label_column,
        biopsy_column=biopsy_column,
    )
    for target_case in target_cases.itertuples(index=False):
        patient_id = str(getattr(target_case, group_column))
        patient_df = df[df[group_column].astype(str) == patient_id]
        symmetry = _target_contralateral_symmetry_features(
            patient_df,
            profile_column=profile_column,
            q_column=q_column,
            side_column=side_column,
            target_side_norm=target_case.target_side_norm,
            contralateral_side_norm=target_case.contralateral_side_norm,
        )
        rows.append(
            {
                TARGET_CASE_ID: target_case.target_case_id,
                group_column: patient_id,
                "label": int(target_case.label),
                "label_name": "CANCER" if int(target_case.label) == 1 else "BENIGN",
                "target_side": target_case.target_side,
                "contralateral_side": target_case.contralateral_side,
                "specimens": int(patient_df[specimen_column].astype(str).nunique()),
                "measurements": int(len(patient_df)),
                "age": _numeric_median(patient_df, age_column, default=0.0),
                "age_available": int(_has_numeric(patient_df, age_column)),
                **symmetry,
            }
    )
    feature_table = pd.DataFrame(rows)
    out = feature_table.merge(lr1_scores, on=TARGET_CASE_ID, how="inner")
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
        TARGET_CASE_ID: f"{patient_id}::{target_side_norm}",
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
    decision_thresholds: np.ndarray,
    *,
    evaluation_mode: str,
    evaluation_view: str,
) -> dict[str, Any]:
    y = test_df["label"].to_numpy(dtype=int)
    pred = (score >= decision_thresholds).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    calibration_intercept, calibration_slope = _calibration_parameters(y, score)
    return {
        "model_name": model_name,
        "split_id": int(split_id),
        "evaluation_mode": evaluation_mode,
        "evaluation_view": evaluation_view,
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
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
        "train_target_cases": int(len(train_df)),
        "test_target_cases": int(len(test_df)),
        "train_cancer_target_cases": int((train_df["label"] == 1).sum()),
        "test_cancer_target_cases": int((test_df["label"] == 1).sum()),
        **thresholds,
    }


def _patient_prediction_frame(
    model_name: str,
    split_id: int,
    test_df: pd.DataFrame,
    score: np.ndarray,
    thresholds: dict[str, Any],
    routes: np.ndarray,
    decision_thresholds: np.ndarray,
    *,
    evaluation_mode: str,
    evaluation_view: str,
) -> pd.DataFrame:
    out = test_df[[TARGET_CASE_ID, "patientId", "label", "label_name"]].copy()
    out["model_name"] = model_name
    out["split_id"] = int(split_id)
    out["evaluation_mode"] = evaluation_mode
    out["evaluation_view"] = evaluation_view
    out["p_cancer"] = np.asarray(score, dtype=float)
    out["model_route"] = np.asarray(routes, dtype=str)
    out["threshold_youden"] = float(thresholds["threshold_youden"])
    out["threshold_target"] = np.asarray(decision_thresholds, dtype=float)
    out["y_pred_target"] = (out["p_cancer"] >= out["threshold_target"]).astype(int)
    return out


def _pooled_patient_metrics(
    predictions: pd.DataFrame,
    *,
    evaluation_mode: str,
) -> pd.DataFrame:
    rows = []
    for (model_name, evaluation_view), group in predictions.groupby(
        ["model_name", "evaluation_view"], sort=False
    ):
        y = group["label"].to_numpy(dtype=int)
        score = group["p_cancer"].to_numpy(dtype=float)
        pred = group["y_pred_target"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = _ratio(tp, tp + fn)
        specificity = _ratio(tn, tn + fp)
        calibration_intercept, calibration_slope = _calibration_parameters(y, score)
        rows.append(
            {
                "model_name": model_name,
                "split_id": -1,
                "evaluation_mode": evaluation_mode,
                "evaluation_view": evaluation_view,
                "roc_auc": float(roc_auc_score(y, score)),
                "pr_auc": float(average_precision_score(y, score)),
                "brier_score": float(brier_score_loss(y, score)),
                "log_loss": float(log_loss(y, score, labels=[0, 1])),
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
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
                "train_target_cases": None,
                "test_target_cases": int(len(group)),
                "train_cancer_target_cases": None,
                "test_cancer_target_cases": int((group["label"] == 1).sum()),
                "threshold_youden": float(group["threshold_youden"].median()),
                "threshold_target": float(group["threshold_target"].median()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_patient_model_metrics(
    split_metrics: pd.DataFrame,
    split_predictions: pd.DataFrame,
    *,
    random_state: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    for (model_name, evaluation_view), group in split_metrics.groupby(
        ["model_name", "evaluation_view"], sort=False
    ):
        evaluation_modes = sorted(group["evaluation_mode"].dropna().astype(str).unique())
        rows.append(
            {
                "model_name": model_name,
                "evaluation_mode": evaluation_modes[0] if len(evaluation_modes) == 1 else ",".join(evaluation_modes),
                "evaluation_view": evaluation_view,
                "evidence_status": (
                    "fit_diagnostic_only"
                    if evaluation_modes == ["all_on_all"]
                    else "patient_safe_validation"
                ),
                "splits": int(len(group)),
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_std": float(group["roc_auc"].std(ddof=0)),
                "pr_auc_mean": float(group["pr_auc"].mean()),
                "pr_auc_std": float(group["pr_auc"].std(ddof=0)),
                "brier_score_mean": float(group["brier_score"].mean()),
                "brier_score_std": float(group["brier_score"].std(ddof=0)),
                "log_loss_mean": float(group["log_loss"].mean()),
                "log_loss_std": float(group["log_loss"].std(ddof=0)),
                "calibration_intercept_mean": float(
                    group["calibration_intercept"].mean()
                ),
                "calibration_slope_mean": float(group["calibration_slope"].mean()),
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
    summary = pd.DataFrame(rows)
    intervals = _patient_bootstrap_intervals(
        split_predictions,
        random_state=random_state,
        bootstrap_samples=bootstrap_samples,
    )
    return summary.merge(
        intervals,
        on=["model_name", "evaluation_view"],
        how="left",
    )


def _patient_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    random_state: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_state)
    for (model_name, evaluation_view), group in predictions.groupby(
        ["model_name", "evaluation_view"], sort=False
    ):
        patient = (
            group.groupby("patientId", as_index=False)
            .agg(
                label=("label", "first"),
                p_cancer=("p_cancer", "mean"),
                threshold_target=("threshold_target", "mean"),
            )
            .reset_index(drop=True)
        )
        y = patient["label"].to_numpy(dtype=int)
        score = patient["p_cancer"].to_numpy(dtype=float)
        threshold = patient["threshold_target"].to_numpy(dtype=float)
        point = _binary_metric_values(y, score, threshold)
        sampled = {name: [] for name in point}
        for _ in range(max(0, bootstrap_samples)):
            indices = rng.integers(0, len(patient), size=len(patient))
            if np.unique(y[indices]).size != 2:
                continue
            values = _binary_metric_values(y[indices], score[indices], threshold[indices])
            for name, value in values.items():
                if np.isfinite(value):
                    sampled[name].append(value)
        row: dict[str, Any] = {
            "model_name": model_name,
            "evaluation_view": evaluation_view,
            "pooled_patients": int(len(patient)),
        }
        for name, value in point.items():
            values = sampled[name]
            row[f"{name}_pooled"] = value
            row[f"{name}_ci_low"] = (
                float(np.quantile(values, 0.025)) if values else float("nan")
            )
            row[f"{name}_ci_high"] = (
                float(np.quantile(values, 0.975)) if values else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _binary_metric_values(
    y: np.ndarray,
    score: np.ndarray,
    threshold: np.ndarray,
) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    calibration_intercept, calibration_slope = _calibration_parameters(y, score)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": _mean_finite([sensitivity, specificity]),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "brier_score": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


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
                "final_patients": int(feature_table["patientId"].astype(str).nunique()),
                "final_target_cases": int(len(feature_table)),
                "final_cancer_target_cases": int((feature_table["label"] == 1).sum()),
                "final_benign_target_cases": int((feature_table["label"] == 0).sum()),
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


def _calibration_parameters(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if np.unique(y).size != 2:
        return float("nan"), float("nan")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    calibrator.fit(logits, y)
    return (
        float(calibrator.intercept_[0]),
        float(calibrator.coef_[0, 0]),
    )


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
    nested = config.get("evaluation", {}).get("nested", {})
    if nested.get("enabled", False):
        selected_models = _selected_patient_models(config.get("model", {}))
        if "M2Q" not in selected_models:
            raise ValueError("Nested model selection currently requires selected_models M2Q.")
        for key in ("lr1_c_grid", "lr2_c_grid"):
            values = nested.get(key)
            if not isinstance(values, list) or not values:
                raise ValueError(f"evaluation.nested.{key} must be a non-empty list.")
            if any(float(value) <= 0.0 for value in values):
                raise ValueError(f"evaluation.nested.{key} values must be positive.")
        if int(nested.get("inner_n_splits", 4)) < 2:
            raise ValueError("evaluation.nested.inner_n_splits must be at least 2.")
    if int(config.get("evaluation", {}).get("bootstrap_samples", 0)) < 0:
        raise ValueError("evaluation.bootstrap_samples must be non-negative.")


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
        "model_registry": _model_registry_summary(artifact.get("models", {})),
        "feature_schema": _jsonable(artifact.get("feature_schema", {})),
        "dataset_summary": _records(artifact.get("dataset_summary")),
        "metric_summary": _records(artifact.get("metric_summary")),
        "hyperparameter_selection": _jsonable(
            artifact.get("hyperparameter_selection")
        ),
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


def _model_registry_summary(models: dict[str, Any]) -> dict[str, Any]:
    """Return the serializable trained-model details for the training YAML."""
    return {
        model_name: _model_summary(model_info)
        for model_name, model_info in models.items()
    }


def _model_summary(model_info: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "name": model_info.get("name"),
        "lr1_profile_model": _pipeline_summary(model_info.get("lr1_model")),
        "thresholds": _jsonable(model_info.get("thresholds", {})),
    }
    if "routes" not in model_info:
        summary["feature_columns"] = list(model_info.get("feature_columns", []))
        summary["final_model"] = _pipeline_summary(model_info.get("final_model"))
        if "symmetry_policy" in model_info:
            summary["symmetry_policy"] = model_info["symmetry_policy"]
            summary["symmetry_gate"] = model_info["symmetry_gate"]
        return summary

    summary["routing_field"] = model_info.get("routing_field")
    summary["routing_policy"] = model_info.get("routing_policy")
    summary["routes"] = {
        route_name: {
            "feature_columns": list(route_info.get("feature_columns", [])),
            "training_patients": route_info.get("training_patients"),
            "thresholds": _jsonable(route_info.get("thresholds", {})),
            "final_model": _pipeline_summary(route_info.get("final_model")),
        }
        for route_name, route_info in model_info["routes"].items()
    }
    return summary


def _pipeline_summary(model: Pipeline | GatedSymmetryLogistic | None) -> dict[str, Any] | None:
    """Describe fitted sklearn pipeline parameters without serializing estimators."""
    if model is None:
        return None
    if isinstance(model, GatedSymmetryLogistic):
        return {
            "type": type(model).__name__,
            "include_age": bool(model.include_age),
            "base_feature_columns": list(model.base_feature_columns_),
            "symmetry_feature_columns": list(SK_CORE4_FEATURE_COLUMNS),
            "symmetry_gate": "symmetry_available",
            "symmetry_means": _jsonable(model.symmetry_means_.to_numpy()),
            "symmetry_scales": _jsonable(model.symmetry_scales_.to_numpy()),
            "logreg": {
                "C": float(model.logreg_.C),
                "class_weight": model.logreg_.class_weight,
                "solver": model.logreg_.solver,
                "max_iter": int(model.logreg_.max_iter),
                "random_state": model.logreg_.random_state,
                "classes": _jsonable(model.logreg_.classes_),
                "coef": _jsonable(model.logreg_.coef_),
                "intercept": _jsonable(model.logreg_.intercept_),
            },
        }
    summary: dict[str, Any] = {"type": type(model).__name__, "steps": {}}
    for step_name, step in model.named_steps.items():
        step_summary: dict[str, Any] = {"type": type(step).__name__}
        if isinstance(step, SimpleImputer):
            step_summary["strategy"] = step.strategy
            step_summary["statistics"] = _jsonable(step.statistics_)
        elif isinstance(step, StandardScaler):
            step_summary["mean"] = _jsonable(step.mean_)
            step_summary["scale"] = _jsonable(step.scale_)
        elif isinstance(step, LogisticRegression):
            step_summary.update(
                {
                    "C": float(step.C),
                    "class_weight": step.class_weight,
                    "solver": step.solver,
                    "max_iter": int(step.max_iter),
                    "random_state": step.random_state,
                    "classes": _jsonable(step.classes_),
                    "coef": _jsonable(step.coef_),
                    "intercept": _jsonable(step.intercept_),
                }
            )
        summary["steps"][step_name] = step_summary
    return summary


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
