"""Training entrypoints for Aramina research-draft models."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from xrd_preprocessing import (
    load_preprocessing_artifact,
    load_preprocessing_dataframe,
)

from .model_utils import profile_matrix
from .config_paths import resolve_config_path
from .patient_features import (
    lr1_training_rows as _lr1_training_rows,
    patient_feature_table as _patient_feature_table,
    row_labels as _row_labels,
    score_lr1_rows as _score_lr1_rows,
)
from .patient_features import build_patient_prediction_feature_row  # noqa: F401
from .patient_features import logit_average_probability as _logit_average_probability  # noqa: F401
from .training_config import (
    PRODUCT_MODEL_NAME,
    load_training_config,
    resolve_model_definition,
)
from .training_artifacts import (
    _evaluation_artifact,
    _final_model_artifact,
    _patient_training_artifact,
    _prediction_preprocessing_payload,
    _project_owned_path,
    _write_evaluation_outputs,
)
from .training_evaluation import (
    _evaluate_m2q_model,
)
from .training_evaluation import _patient_split_pairs  # noqa: F401
from .training_model import (
    _fit_m2q_model,
    _profile_logistic,
)
from .model_description import (
    _decision_threshold_record,
    _file_sha256,
    _model_artifact_id,
    _model_description,
    _model_reference,
    _write_model_input_snapshots,
    _write_yaml,
)
from .preprocessing_lineage import require_training_preprocessing_artifact

PATIENT_BOOTSTRAP_SAMPLES = 2_000
logger = logging.getLogger(__name__)


class PatientModelInputBuilder(BaseEstimator):
    """Build target-breast cases used by the fixed Aramina M2Q model.

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
            PRODUCT_MODEL_NAME: _fit_m2q_model(
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


class AraminaPatientTrainingPipeline(BaseEstimator):
    """Complete sklearn-compatible training estimator for Aramina patient models.

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
        preprocess_train_config_yaml: str | None = None,
    ) -> None:
        self.config = config
        self.config_text = config_text
        self.input_dataframe_joblib_path = input_dataframe_joblib_path
        self.preprocessing_artifact = preprocessing_artifact
        self.prediction_preprocessing = prediction_preprocessing
        self.preprocess_train_config_yaml = preprocess_train_config_yaml

    def fit(self, x: pd.DataFrame, y: Any = None) -> "AraminaPatientTrainingPipeline":
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
        if bool(self.config.get("run", {}).get("evaluation", False)):
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
            ).fit(x)
            split_metrics = self.evaluator_.split_metrics_
            split_predictions = self.evaluator_.split_predictions_
        else:
            self.evaluator_ = None
            split_metrics = pd.DataFrame()
            split_predictions = pd.DataFrame()
        self.model_trainer_ = M2QModelTrainer(
            profile_column=profile_column,
            label_column=label_column,
            lr1_logreg_c=lr1_logreg_c,
            lr2_logreg_c=lr2_logreg_c,
            random_state=random_state,
            target_sensitivity=target_sensitivity,
        )
        self.model_trainer_.fit(self.feature_table_, self.input_builder_.lr1_rows_)
        for model_info in self.model_trainer_.models_.values():
            model_info["class_definition"] = dict(self.config["class_definition"])
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
            split_metrics=split_metrics,
            split_predictions=split_predictions,
            preprocess_train_config_yaml=self.preprocess_train_config_yaml,
        )
        return self


def build_patient_training_pipeline(
    *,
    config: dict[str, Any],
    config_text: str,
    input_dataframe_joblib_path: str | Path,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any] | None = None,
    preprocess_train_config_yaml: str | None = None,
) -> Pipeline:
    """Return one sklearn Pipeline for patient-safe target-breast training."""
    return Pipeline(
        [
            (
                "patient_training",
                AraminaPatientTrainingPipeline(
                    config=config,
                    config_text=config_text,
                    input_dataframe_joblib_path=input_dataframe_joblib_path,
                    preprocessing_artifact=preprocessing_artifact,
                    prediction_preprocessing=prediction_preprocessing,
                    preprocess_train_config_yaml=preprocess_train_config_yaml,
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
    preprocess_train_config_yaml: str | None = None,
) -> dict[str, Any]:
    """Run requested evaluation and/or final fit for one fixed product model."""
    config_path = Path(config_path).expanduser().resolve()
    public_config, config_text = load_training_config(config_path)
    model_identity = public_config["model"]
    model_definition = resolve_model_definition(str(model_identity["name"]))
    config = _effective_training_config(public_config, model_definition)
    input_path = (
        Path(dataframe_joblib_path).resolve()
        if dataframe_joblib_path
        else (
            _public_config_path(
                public_config, config_path, section="input", key="dataframe_joblib_path"
            )
        )
    )
    output_root = (
        Path(output_folder).resolve()
        if output_folder
        else (
            _public_config_path(
                public_config, config_path, section="output", key="folder"
            )
        )
    )
    run_folder = _new_training_run_folder(output_root, model_identity)
    prediction_preprocessing_config_path = _project_owned_path(
        str(model_definition["prediction_preprocessing_config_path"]),
        config_path,
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
    logger.info(
        "Training model=%s rows=%d patients=%d",
        model_identity["name"],
        len(df),
        df["patientId"].astype(str).nunique(),
    )
    if public_config["run"]["evaluation"]:
        logger.info(
            "Evaluation: repeated stratified %d-fold x%d (seed=%d)",
            config["evaluation"]["n_splits"],
            config["evaluation"]["n_repeats"],
            config["evaluation"]["random_state"],
        )
    artifact = train_m2q_model_artifact(
        df,
        config=config,
        config_text=config_text,
        input_dataframe_joblib_path=input_path,
        preprocessing_artifact=preprocessing_artifact,
        prediction_preprocessing=prediction_preprocessing,
        preprocess_train_config_yaml=preprocess_train_config_yaml,
    )
    evaluation_artifact = _evaluation_artifact(
        artifact,
        model_identity=model_identity,
        target_sensitivity=float(model_definition["target_sensitivity"]),
        training_config_yaml=config_text,
    )
    if not public_config["run"]["train_on_all"]:
        if public_config["run"]["evaluation"]:
            _write_evaluation_outputs(evaluation_artifact, run_folder)
            logger.info("Evaluation artifacts written: %s", run_folder)
        evaluation_artifact["run_folder"] = str(run_folder)
        return evaluation_artifact

    model_artifact = _final_model_artifact(
        artifact,
        public_config=public_config,
        model_definition=model_definition,
        training_config_yaml=config_text,
    )
    model_path = run_folder / "model.joblib"
    joblib.dump(model_artifact, model_path)
    _write_model_input_snapshots(model_artifact, run_folder)
    model_sha = _file_sha256(model_path)
    model_id = _model_artifact_id(model_identity, model_sha)
    if public_config["run"]["evaluation"]:
        _write_evaluation_outputs(
            evaluation_artifact,
            run_folder,
            model=_model_reference(
                model_identity,
                model_id=model_id,
                artifact_sha256=model_sha,
            ),
            decision_threshold=_decision_threshold_record(model_artifact),
        )
        logger.info("Evaluation artifacts written: %s", run_folder)
    description = _model_description(
        model_artifact,
        model_id=model_id,
        model_sha=model_sha,
        model_path=model_path,
    )
    _write_yaml(run_folder / "model_description.yaml", description)
    logger.info("Final model written: %s", model_path)
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
    preprocess_train_config_yaml: str | None = None,
) -> dict[str, Any]:
    """Train one traceable M2Q target-breast model artifact."""
    pipeline = build_patient_training_pipeline(
        config=config,
        config_text=config_text,
        input_dataframe_joblib_path=input_dataframe_joblib_path,
        preprocessing_artifact=preprocessing_artifact,
        prediction_preprocessing=prediction_preprocessing,
        preprocess_train_config_yaml=preprocess_train_config_yaml,
    )
    pipeline.fit(df)
    return pipeline.named_steps["patient_training"].artifact_


def _load_training_dataframe(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a traceable preprocessing artifact for product training."""
    value = joblib.load(path)
    if isinstance(value, pd.DataFrame):
        raise ValueError(
            "Training requires a preprocessing artifact joblib, not a plain DataFrame. "
            "Run `aramina preprocess` or `aramina preprocess-train` first."
        )
    artifact = load_preprocessing_artifact(path)
    _require_preprocessing_lineage(artifact)
    return load_preprocessing_dataframe(path), artifact


def _require_preprocessing_lineage(artifact: dict[str, Any] | None) -> None:
    """Require the v0.2 resolved product preprocessing identity."""
    require_training_preprocessing_artifact(artifact)
    metadata = artifact["metadata"]
    if not metadata.get("input_h5_sha256"):
        raise ValueError("Training artifact is missing metadata.input_h5_sha256.")


def _validate_training_config(config: dict[str, Any], config_path: Path) -> None:
    """Backward-compatible import target for the current public validator."""
    from .training_config import validate_training_config

    validate_training_config(config, config_path)


def _effective_training_config(
    public_config: dict[str, Any],
    model_definition: dict[str, Any],
) -> dict[str, Any]:
    evaluation = public_config["evaluation"]
    return {
        "model_identity": dict(public_config["model"]),
        "run": dict(public_config["run"]),
        "model": dict(model_definition["model"]),
        "class_definition": dict(model_definition["class_definition"]),
        "evaluation": {
            "mode": "stratified_kfold",
            "n_splits": int(evaluation["folds"]),
            "n_repeats": int(evaluation["repeats"]),
            "random_state": int(evaluation["random_seed"]),
            "target_sensitivity": float(model_definition["target_sensitivity"]),
            "bootstrap_samples": PATIENT_BOOTSTRAP_SAMPLES,
        },
        "prediction_contract": dict(model_definition["prediction_contract"]),
    }


def _public_config_path(
    config: dict[str, Any],
    config_path: Path,
    *,
    section: str,
    key: str,
) -> Path:
    value = config[section][key]
    return resolve_config_path(value, config_path)


def _new_training_run_folder(output_root: Path, model: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = _safe_artifact_stem(f"{model['name']}_{model['version']}")
    folder = output_root / f"{stem}_{stamp}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _safe_artifact_stem(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    ).strip("_")
