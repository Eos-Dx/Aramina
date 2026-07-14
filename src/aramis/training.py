"""Training entrypoints for Aramis research-draft models."""

from __future__ import annotations

import json
import platform
import subprocess
import tomllib
from collections.abc import Sequence
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any
from uuid import uuid4

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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xrd_preprocessing import (
    load_preprocessing_artifact,
    load_preprocessing_dataframe,
)

from .model_utils import (
    LABEL_MAP,
    compute_binary_thresholds,
    profile_matrix,
)
from .m2q_model import GatedSymmetryLogistic, SK_CORE4_FEATURE_COLUMNS
from .training_config import (
    load_training_config,
    resolve_training_recipe,
    resolved_recipe_path,
)

TARGET_CASE_ID = "target_case_id"
PATIENT_BOOTSTRAP_SAMPLES = 2_000


class PatientModelInputBuilder(BaseEstimator):
    """Build target-breast cases used by the fixed Aramis M2Q model.

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


class M2QModelTrainer(BaseEstimator):
    """Train the fixed M2Q target-breast decision-support model.

    The trainer consumes the target-case feature table from
    `PatientModelInputBuilder` and its retained LR1 measurement rows. M2Q uses
    one final LogisticRegression. It receives SK terms only as
    a gated optional refinement: all SK terms are zero when contralateral data
    is unavailable. Reliability remains a report field, not a model feature.
    """

    def __init__(
        self,
        *,
        profile_column: str = "radial_profile_data",
        label_column: str = "product_status_group",
        lr1_logreg_c: float = 1.0,
        lr2_logreg_c: float = 1.0,
        random_state: int = 42,
        target_sensitivity: float = 0.95,
    ) -> None:
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
    ) -> "M2QModelTrainer":
        """Fit the final M2Q model and store it in `models_`."""
        self.models_ = {
            "M2Q": _fit_m2q_model(
                feature_table,
                lr1_rows,
                profile_column=self.profile_column,
                label_column=self.label_column,
                lr1_logreg_c=self.lr1_logreg_c,
                lr2_logreg_c=self.lr2_logreg_c,
                random_state=self.random_state,
                target_sensitivity=self.target_sensitivity,
            )
        }
        return self


class M2QModelEvaluator(BaseEstimator):
    """Evaluate M2Q with patient-safe repeated stratified K-fold.

    All splits operate at patient level, so measurements from one patient cannot
    appear in both train and test.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
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

    def fit(self, x: pd.DataFrame, y: Any = None) -> "M2QModelEvaluator":
        """Store M2Q fold metrics and held-out predictions."""
        self.split_metrics_, self.split_predictions_ = _evaluate_m2q_model(
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
        )
        return self


class AramisPatientTrainingPipeline(BaseEstimator):
    """Complete sklearn-compatible training estimator for Aramis patient models.

    This estimator is the product-level training unit. It receives a
    measurement-level preprocessing DataFrame, builds target-breast features,
    evaluates fixed M2Q, and exposes the final traceable model artifact.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        config_text: str,
        input_dataframe_joblib_path: str | Path,
        preprocessing_artifact: dict[str, Any],
        prediction_preprocessing: dict[str, Any] | None = None,
        workflow_config_yaml: str | None = None,
    ) -> None:
        self.config = config
        self.config_text = config_text
        self.input_dataframe_joblib_path = input_dataframe_joblib_path
        self.preprocessing_artifact = preprocessing_artifact
        self.prediction_preprocessing = prediction_preprocessing
        self.workflow_config_yaml = workflow_config_yaml

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
        default_logreg_c = float(model_config.get("logreg_c", 1.0))
        lr1_logreg_c = float(model_config.get("lr1_logreg_c", default_logreg_c))
        lr2_logreg_c = float(model_config.get("lr2_logreg_c", default_logreg_c))
        random_state = int(evaluation_config.get("random_state", 42))
        target_sensitivity = float(evaluation_config.get("target_sensitivity", 0.95))
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
        self.evaluator_ = M2QModelEvaluator(
            config=self.config,
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
        self.model_trainer_ = M2QModelTrainer(
            profile_column=profile_column,
            label_column=label_column,
            lr1_logreg_c=lr1_logreg_c,
            lr2_logreg_c=lr2_logreg_c,
            random_state=random_state,
            target_sensitivity=target_sensitivity,
        )
        self.model_trainer_.fit(self.feature_table_, self.input_builder_.lr1_rows_)
        self.artifact_ = _patient_training_artifact(
            df=x,
            config=self.config,
            config_text=self.config_text,
            input_dataframe_joblib_path=self.input_dataframe_joblib_path,
            preprocessing_artifact=self.preprocessing_artifact,
            prediction_preprocessing=self.prediction_preprocessing,
            models=self.model_trainer_.models_,
            feature_table=self.feature_table_,
            lr1_rows=self.input_builder_.lr1_rows_,
            split_metrics=self.evaluator_.split_metrics_,
            split_predictions=self.evaluator_.split_predictions_,
            workflow_config_yaml=self.workflow_config_yaml,
        )
        return self


def build_patient_training_pipeline(
    *,
    config: dict[str, Any],
    config_text: str,
    input_dataframe_joblib_path: str | Path,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None = None,
    workflow_config_yaml: str | None = None,
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
                    workflow_config_yaml=workflow_config_yaml,
                ),
            )
        ]
    )


def run_training_from_config(
    config_path: str | Path,
    *,
    dataframe: pd.DataFrame | None = None,
    preprocessing_artifact: dict[str, Any] | None = None,
    dataframe_joblib_path: str | Path | None = None,
    output_folder: str | Path | None = None,
    workflow_config_yaml: str | None = None,
) -> dict[str, Any]:
    """Evaluate or final-fit one immutable Aramis model recipe."""
    config_path = Path(config_path).expanduser().resolve()
    public_config, config_text = load_training_config(config_path)
    recipe_id = str(public_config["model"]["recipe"])
    recipe, registry_path = resolve_training_recipe(recipe_id)
    config = _effective_training_config(public_config, recipe)
    input_path = Path(dataframe_joblib_path).resolve() if dataframe_joblib_path else (
        _public_config_path(
            public_config, config_path, section="input", key="dataframe_joblib_path"
        )
    )
    output_root = Path(output_folder).resolve() if output_folder else (
        _public_config_path(
            public_config, config_path, section="output", key="folder"
        )
    )
    run_folder = _new_training_run_folder(output_root, public_config["training"])
    prediction_preprocessing_config_path = resolved_recipe_path(
        str(recipe["prediction_preprocessing_config_path"]), registry_path
    )
    prediction_preprocessing = _prediction_preprocessing_payload(
        prediction_preprocessing_config_path
    )
    if dataframe is None:
        dataframe, loaded_artifact = _load_training_dataframe(input_path)
        preprocessing_artifact = preprocessing_artifact or loaded_artifact
    df = dataframe
    _require_preprocessing_lineage(preprocessing_artifact)

    model_type = str(config["model"]["type"])
    if model_type != "m2q_gated_target_case":
        raise ValueError(f"Unsupported training model.type: {model_type!r}")
    artifact = train_m2q_model_artifact(
        df,
        config=config,
        config_text=config_text,
        input_dataframe_joblib_path=input_path,
        preprocessing_artifact=preprocessing_artifact,
        prediction_preprocessing=prediction_preprocessing,
        workflow_config_yaml=workflow_config_yaml,
    )
    evaluation_artifact = _evaluation_artifact(
        artifact,
        recipe_id=recipe_id,
        training_config_yaml=config_text,
    )
    _write_evaluation_outputs(evaluation_artifact, run_folder)
    if public_config["training"]["mode"] == "evaluation":
        evaluation_artifact["run_folder"] = str(run_folder)
        return evaluation_artifact

    model_artifact = _final_model_artifact(
        artifact,
        public_config=public_config,
        recipe_id=recipe_id,
        training_config_yaml=config_text,
    )
    model_path = run_folder / "model.joblib"
    joblib.dump(model_artifact, model_path)
    model_sha = _file_sha256(model_path)
    model_id = _model_artifact_id(public_config["training"], model_sha)
    description = _model_description(
        model_artifact,
        model_id=model_id,
        model_sha=model_sha,
        model_path=model_path,
    )
    _write_yaml(run_folder / "model_description.yaml", description)
    model_artifact["run_folder"] = str(run_folder)
    model_artifact["model_path"] = str(model_path)
    model_artifact["model_id"] = model_id
    return model_artifact


def train_m2q_model_artifact(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    config_text: str,
    input_dataframe_joblib_path: str | Path,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None = None,
    workflow_config_yaml: str | None = None,
) -> dict[str, Any]:
    """Train one traceable M2Q target-breast model artifact."""
    pipeline = build_patient_training_pipeline(
        config=config,
        config_text=config_text,
        input_dataframe_joblib_path=input_dataframe_joblib_path,
        preprocessing_artifact=preprocessing_artifact,
        prediction_preprocessing=prediction_preprocessing,
        workflow_config_yaml=workflow_config_yaml,
    )
    pipeline.fit(df)
    return pipeline.named_steps["patient_training"].artifact_


def _load_training_dataframe(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a traceable preprocessing artifact for product training."""
    value = joblib.load(path)
    if isinstance(value, pd.DataFrame):
        raise ValueError(
            "Training requires a preprocessing artifact joblib, not a plain DataFrame. "
            "Run `aramis preprocess` or `aramis preprocess-train` first."
        )
    artifact = load_preprocessing_artifact(path)
    _require_preprocessing_lineage(artifact)
    return load_preprocessing_dataframe(path), artifact


def _require_preprocessing_lineage(artifact: dict[str, Any] | None) -> None:
    """Require the resolved preprocessing YAML and input H5 checksum."""
    if not isinstance(artifact, dict):
        raise ValueError("Training requires a preprocessing artifact with provenance.")
    if not isinstance(artifact.get("preprocessing_config_yaml"), str):
        raise ValueError("Training artifact is missing preprocessing_config_yaml.")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("input_h5_sha256"):
        raise ValueError("Training artifact is missing metadata.input_h5_sha256.")


def _patient_training_artifact(
    *,
    df: pd.DataFrame,
    config: dict[str, Any],
    config_text: str,
    input_dataframe_joblib_path: str | Path,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None,
    models: dict[str, Any],
    feature_table: pd.DataFrame,
    lr1_rows: pd.DataFrame,
    split_metrics: pd.DataFrame,
    split_predictions: pd.DataFrame,
    workflow_config_yaml: str | None,
) -> dict[str, Any]:
    """Build the traceable joblib payload for target-breast model training."""
    evaluation_config = config.get("evaluation", {})
    metric_summary = _summarize_patient_model_metrics(
        split_metrics,
        split_predictions,
        random_state=int(evaluation_config.get("random_state", 42)),
        bootstrap_samples=int(
            evaluation_config.get("bootstrap_samples", PATIENT_BOOTSTRAP_SAMPLES)
        ),
    )
    dataset_summary = _patient_dataset_summary(df, feature_table, lr1_rows)
    model_descriptions = {
        "M2Q": {
            "name": "M2Q profile, gated SK Core4 refinement, and age",
            "description": (
                "One final model: profile and age are always evaluated; SK Core4 "
                "adds a neutral-gated refinement only when paired symmetry is available."
            ),
        }
    }
    return {
        "kind": "aramis_training_artifact",
        "version": "0.3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "m2q_gated_target_case",
        "model_columns": {
            key: config["model"][key]
            for key in (
                "profile_column",
                "group_column",
                "specimen_column",
                "side_column",
                "age_column",
            )
        },
        "models": models,
        "model_descriptions": model_descriptions,
        "feature_schema": _m2q_feature_schema(),
        "warnings": _m2q_warnings(config, feature_table),
        "training_config_yaml": config_text,
        "prediction_contract_yaml": yaml.safe_dump(
            config["prediction_contract"], sort_keys=False
        ),
        **_preprocessing_lineage_fields(
            preprocessing_artifact,
            prediction_preprocessing,
        ),
        "input_dataframe_joblib_sha256": _file_sha256(input_dataframe_joblib_path),
        "dataset_summary": dataset_summary,
        "metric_summary": metric_summary,
        "split_metrics": split_metrics,
        "split_predictions": split_predictions,
        "metadata": {
            "aramis_version": _aramis_version(),
            "aramis_git_sha": _aramis_git_sha(),
        },
        "reproducibility": _reproducibility_manifest(
            preprocessing_artifact=preprocessing_artifact,
            training_config_yaml=config_text,
            prediction_preprocessing=prediction_preprocessing,
            workflow_config_yaml=workflow_config_yaml,
        ),
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


def _fit_m2q_model(
    feature_table: pd.DataFrame,
    lr1_rows: pd.DataFrame,
    *,
    profile_column: str,
    label_column: str,
    lr1_logreg_c: float,
    lr2_logreg_c: float,
    random_state: int,
    target_sensitivity: float,
) -> dict[str, Any]:
    """Fit the fixed two-layer M2Q model on all accepted target cases."""
    lr1_model = _profile_logistic(logreg_c=lr1_logreg_c, random_state=random_state)
    lr1_model.fit(profile_matrix(lr1_rows, profile_column), _row_labels(lr1_rows, label_column))
    y = feature_table["label"].to_numpy(dtype=int)
    final_model = GatedSymmetryLogistic(
        logreg_c=lr2_logreg_c,
        random_state=random_state,
    ).fit(feature_table, y)
    score = final_model.predict_proba(feature_table)[:, 1]
    return {
        "name": "M2Q profile, gated SK Core4 refinement, and age",
        "lr1_model": lr1_model,
        "final_model": final_model,
        "feature_columns": _m2q_model_input_columns(),
        "symmetry_policy": "single_model_gated_optional_refinement",
        "symmetry_gate": "symmetry_available",
        "thresholds": compute_binary_thresholds(
            y,
            score,
            target_sensitivity=target_sensitivity,
        ),
    }


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


def _evaluate_m2q_model(
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_config = config.get("evaluation", {})
    mode = _evaluation_mode(evaluation_config)
    n_splits = int(evaluation_config["n_splits"])
    n_repeats = int(evaluation_config["n_repeats"])
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
    metrics = []
    predictions = []
    y_patients = base_features["label"].to_numpy(dtype=int)
    split_pairs = _patient_split_pairs(
        mode=mode,
        base_features=base_features,
        y_patients=y_patients,
        n_splits=n_splits,
        n_repeats=n_repeats,
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
        final_model = GatedSymmetryLogistic(
            logreg_c=lr2_logreg_c,
            random_state=random_state + split_id,
        ).fit(train_features, train_features["label"].to_numpy(dtype=int))
        train_score = final_model.predict_proba(train_features)[:, 1]
        test_score = final_model.predict_proba(test_features)[:, 1]
        thresholds = compute_binary_thresholds(
            train_features["label"].to_numpy(dtype=int),
            train_score,
            target_sensitivity=target_sensitivity,
        )
        thresholds["selected_lr1_c"] = lr1_logreg_c
        thresholds["selected_lr2_c"] = lr2_logreg_c
        test_thresholds = np.full(
            len(test_features),
            thresholds["threshold_target"],
            dtype=float,
        )
        metrics.append(
            _patient_metric_row(
                "M2Q",
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
                "M2Q",
                split_id,
                test_features,
                test_score,
                thresholds,
                _default_routes(test_features),
                test_thresholds,
                evaluation_mode=mode,
                evaluation_view="operational",
            )
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    return pd.DataFrame(metrics), prediction_frame


def _evaluation_mode(evaluation_config: dict[str, Any]) -> str:
    if evaluation_config.get("mode") != "stratified_kfold":
        raise ValueError("The product evaluator supports only stratified_kfold.")
    return "stratified_kfold"


def _patient_split_pairs(
    *,
    mode: str,
    base_features: pd.DataFrame,
    y_patients: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    patient_table = (
        base_features.groupby("patientId", as_index=False)["label"]
        .max()
        .assign(patientId=lambda frame: frame["patientId"].astype(str))
    )
    patient_labels = patient_table["label"].to_numpy(dtype=int)

    def case_indices(patient_index: np.ndarray) -> np.ndarray:
        patient_ids = set(patient_table.iloc[patient_index]["patientId"])
        return base_features.index[
            base_features["patientId"].astype(str).isin(patient_ids)
        ].to_numpy()

    if mode != "stratified_kfold":
        raise ValueError(f"Unsupported product split mode: {mode!r}")
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return [
        (case_indices(train_index), case_indices(test_index))
        for train_index, test_index in splitter.split(patient_table, patient_labels)
    ]


def _m2q_model_input_columns() -> list[str]:
    """Return the fixed M2Q feature contract in report order."""
    return [
        "profile_p_cancer_logit_average",
        "age",
        "age_available",
        *SK_CORE4_FEATURE_COLUMNS,
        "symmetry_available",
    ]


def _default_routes(values: Any) -> np.ndarray:
    return np.full(len(values), "default", dtype=object)


def _m2q_feature_schema() -> dict[str, Any]:
    return {
        "M2Q": {
            "feature_columns": _m2q_model_input_columns(),
            "learned_feature_columns": [
                "profile_p_cancer_logit_average",
                "age",
                "age_available",
                *SK_CORE4_FEATURE_COLUMNS,
            ],
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
    }


def _m2q_warnings(
    config: dict[str, Any],
    feature_table: pd.DataFrame,
) -> list[str]:
    warnings = [
        "Research-draft decision support only; requires radiologist review.",
        "Not for autonomous diagnosis.",
    ]
    _ = config
    warnings.extend(
        [
            "M2Q includes age; its contribution must be reviewed separately.",
            "Measurement sufficiency is reported separately; reliability fields are not model predictors.",
        ]
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
                "evidence_status": "patient_safe_validation",
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
        cases = (
            group.groupby(TARGET_CASE_ID, as_index=False)
            .agg(
                patientId=("patientId", "first"),
                label=("label", "first"),
                p_cancer=("p_cancer", "mean"),
                threshold_target=("threshold_target", "mean"),
            )
            .reset_index(drop=True)
        )
        y = cases["label"].to_numpy(dtype=int)
        score = cases["p_cancer"].to_numpy(dtype=float)
        threshold = cases["threshold_target"].to_numpy(dtype=float)
        point = _binary_metric_values(y, score, threshold)
        sampled = {name: [] for name in point}
        for _ in range(max(0, bootstrap_samples)):
            patient_ids = cases["patientId"].drop_duplicates().to_numpy()
            sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
            sample = pd.concat(
                [cases.loc[cases["patientId"] == patient_id] for patient_id in sampled_ids],
                ignore_index=True,
            )
            sample_y = sample["label"].to_numpy(dtype=int)
            if np.unique(sample_y).size != 2:
                continue
            values = _binary_metric_values(
                sample_y,
                sample["p_cancer"].to_numpy(dtype=float),
                sample["threshold_target"].to_numpy(dtype=float),
            )
            for name, value in values.items():
                if np.isfinite(value):
                    sampled[name].append(value)
        row: dict[str, Any] = {
            "model_name": model_name,
            "evaluation_view": evaluation_view,
            "pooled_patients": int(cases["patientId"].nunique()),
            "pooled_target_cases": int(len(cases)),
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
    from xrd_preprocessing import load_preprocessing_config

    config = load_preprocessing_config(config_path)
    return {
        "path": str(config_path),
        "yaml": yaml.safe_dump(config, sort_keys=False),
    }


def _preprocessing_lineage_fields(
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None,
) -> dict[str, Any]:
    training_config_yaml = preprocessing_artifact.get("preprocessing_config_yaml")
    fields = {
        "historical_preprocessing_yaml": training_config_yaml,
        "preprocessing_metadata": preprocessing_artifact.get("metadata", {}),
    }
    if prediction_preprocessing is None:
        fields["prediction_preprocessing_yaml"] = None
        return fields
    fields["prediction_preprocessing_yaml"] = prediction_preprocessing["yaml"]
    return fields


def _reproducibility_manifest(
    *,
    preprocessing_artifact: dict[str, Any],
    training_config_yaml: str,
    prediction_preprocessing: dict[str, Any] | None,
    workflow_config_yaml: str | None,
) -> dict[str, Any]:
    """Record inputs required to repeat this research-draft training run."""
    historical_preprocessing_yaml = str(
        preprocessing_artifact["preprocessing_config_yaml"]
    )
    prediction_preprocessing_yaml = (
        str(prediction_preprocessing["yaml"])
        if prediction_preprocessing is not None
        else None
    )
    source_metadata = dict(preprocessing_artifact.get("metadata", {}))
    source_path = source_metadata.get("input_h5_path")
    configs = {
        "workflow_yaml": workflow_config_yaml,
        "training_yaml": training_config_yaml,
        "historical_preprocessing_yaml": historical_preprocessing_yaml,
        "prediction_preprocessing_yaml": prediction_preprocessing_yaml,
    }
    return {
        "contract": "aramis_reproducibility_v0_1",
        "reproduction_mode": (
            "raw_h5_preprocess_train"
            if workflow_config_yaml is not None
            else "preprocessed_artifact_train"
        ),
        "source_h5": {
            "filename": Path(str(source_path)).name if source_path else "unknown",
            "sha256": str(source_metadata["input_h5_sha256"]),
        },
        "source_code": {
            "aramis": {
                "version": _aramis_version(),
                "git_sha": _aramis_git_sha(),
            },
            "xrd_preprocessing": _distribution_provenance("xrd-preprocessing"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _installed_version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "joblib",
                    "h5py",
                    "pyFAI",
                    "fabio",
                    "PyYAML",
                )
            },
        },
        "configs": configs,
        "checksums": {
            f"{name}_sha256": _text_sha256(text)
            for name, text in configs.items()
            if text is not None
        },
    }


def _distribution_provenance(distribution_name: str) -> dict[str, Any]:
    """Return installed package identity and pip's VCS provenance when present."""
    result: dict[str, Any] = {"version": _installed_version(distribution_name)}
    try:
        payload = distribution(distribution_name).read_text("direct_url.json")
    except PackageNotFoundError:
        return result
    if payload is None:
        return result
    direct_url = json.loads(payload)
    result["url"] = direct_url.get("url")
    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, dict):
        result["requested_revision"] = vcs_info.get("requested_revision")
        result["git_commit"] = vcs_info.get("commit_id")
    return result


def _installed_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unavailable"


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _validate_training_config(config: dict[str, Any], config_path: Path) -> None:
    """Backward-compatible import target for the current public validator."""
    from .training_config import validate_training_config

    validate_training_config(config, config_path)


def _effective_training_config(
    public_config: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    evaluation = public_config["evaluation"]
    return {
        "training": dict(public_config["training"]),
        "model": dict(recipe["model"]),
        "evaluation": {
            "mode": "stratified_kfold",
            "n_splits": int(evaluation["folds"]),
            "n_repeats": int(evaluation["repeats"]),
            "random_state": int(evaluation["random_seed"]),
            "target_sensitivity": float(recipe["target_sensitivity"]),
            "bootstrap_samples": PATIENT_BOOTSTRAP_SAMPLES,
        },
        "prediction_contract": dict(recipe["prediction_contract"]),
    }


def _public_config_path(
    config: dict[str, Any],
    config_path: Path,
    *,
    section: str,
    key: str,
) -> Path:
    value = config[section][key]
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _new_training_run_folder(output_root: Path, training: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = _safe_artifact_stem(f"{training['name']}_{training['version']}")
    folder = output_root / f"{stem}_{stamp}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _safe_artifact_stem(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    ).strip("_")


def _evaluation_artifact(
    artifact: dict[str, Any],
    *,
    recipe_id: str,
    training_config_yaml: str,
) -> dict[str, Any]:
    return {
        "kind": "aramis_evaluation_artifact",
        "version": "0.1",
        "created_at": artifact["created_at"],
        "recipe": recipe_id,
        "training_config_yaml": training_config_yaml,
        "historical_preprocessing_yaml": artifact.get(
            "historical_preprocessing_yaml"
        ),
        "dataset_summary": artifact["dataset_summary"],
        "metric_summary": artifact["metric_summary"],
        "split_metrics": artifact["split_metrics"],
        "split_predictions": artifact["split_predictions"],
        "metadata": artifact["metadata"],
    }


def _write_evaluation_outputs(artifact: dict[str, Any], folder: Path) -> None:
    joblib.dump(artifact, folder / "evaluation.joblib")
    artifact["split_metrics"].to_csv(folder / "evaluation_metrics.csv", index=False)
    artifact["split_predictions"].to_csv(
        folder / "evaluation_predictions.csv", index=False
    )
    summary = {
        "kind": artifact["kind"],
        "version": artifact["version"],
        "created_at": artifact["created_at"],
        "recipe": artifact["recipe"],
        "dataset_summary": _records(artifact["dataset_summary"]),
        "metric_summary": _records(artifact["metric_summary"]),
        "files": {
            "joblib": "evaluation.joblib",
            "metrics": "evaluation_metrics.csv",
            "predictions": "evaluation_predictions.csv",
        },
    }
    _write_json(folder / "evaluation.json", summary)
    _write_yaml(folder / "evaluation.yaml", summary)


def _final_model_artifact(
    artifact: dict[str, Any],
    *,
    public_config: dict[str, Any],
    recipe_id: str,
    training_config_yaml: str,
) -> dict[str, Any]:
    return {
        "kind": "aramis_training_artifact",
        "version": "0.3",
        "created_at": artifact["created_at"],
        "model_type": artifact["model_type"],
        "model_columns": artifact["model_columns"],
        "model_identity": {
            "name": public_config["training"]["name"],
            "version": str(public_config["training"]["version"]),
            "recipe": recipe_id,
        },
        "models": artifact["models"],
        "model_descriptions": artifact["model_descriptions"],
        "feature_schema": artifact["feature_schema"],
        "warnings": artifact["warnings"],
        "dataset_summary": artifact["dataset_summary"],
        "training_config_yaml": training_config_yaml,
        "historical_preprocessing_yaml": artifact.get(
            "historical_preprocessing_yaml"
        ),
        "prediction_preprocessing_yaml": artifact["prediction_preprocessing_yaml"],
        "prediction_contract_yaml": artifact["prediction_contract_yaml"],
        "evaluation": {
            "protocol": dict(public_config["evaluation"]),
            "summary": _records(artifact["metric_summary"]),
            "artifacts": {
                "summary": "evaluation.yaml",
                "metrics": "evaluation_metrics.csv",
                "predictions": "evaluation_predictions.csv",
            },
        },
        "input_dataframe_joblib_sha256": artifact.get(
            "input_dataframe_joblib_sha256"
        ),
        "preprocessing_metadata": artifact.get("preprocessing_metadata", {}),
        "metadata": artifact["metadata"],
        "reproducibility": artifact["reproducibility"],
    }


def _model_artifact_id(training: dict[str, Any], model_sha: str) -> str:
    return _safe_artifact_stem(
        f"{training['name']}_{training['version']}_{model_sha[:12]}"
    )


def _model_description(
    artifact: dict[str, Any],
    *,
    model_id: str,
    model_sha: str,
    model_path: Path,
) -> dict[str, Any]:
    model_name = next(iter(artifact["models"]))
    model = artifact["models"][model_name]
    return {
        "kind": "aramis_model_description",
        "version": "0.1",
        "model_id": model_id,
        "model_name": artifact["model_identity"]["name"],
        "model_version": artifact["model_identity"]["version"],
        "model_recipe": artifact["model_identity"]["recipe"],
        "selected_model": model_name,
        "model_summary": _model_summary(model),
        "model_joblib": model_path.name,
        "model_joblib_sha256": model_sha,
        "decision_thresholds": _jsonable(model.get("thresholds", {})),
        "feature_schema": _jsonable(artifact["feature_schema"]),
        "dataset_summary": _records(artifact["dataset_summary"]),
        "evaluation_artifacts": {
            "summary": "evaluation.yaml",
            "metrics": "evaluation_metrics.csv",
            "predictions": "evaluation_predictions.csv",
        },
        "clinical_stage": "research draft",
        "requires_radiologist_review": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(_jsonable(payload), sort_keys=False), encoding="utf-8"
    )


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
