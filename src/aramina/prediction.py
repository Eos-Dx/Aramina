"""Public prediction entrypoint for Aramina research-draft decision support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from .pipelines import run_preprocessing_pipeline
from .preprocessing_contract import validate_aramina_preprocessing_config
from .prediction_contract import (
    _config_path,
    _model_identity,
    _model_info,
    _prediction_contract,
    _prediction_output_paths,
    _validate_h5_container_contract,
    _validate_prediction_config,
    _write_text,
)
from .prediction_reports import _prediction_reports
from .prediction_reports import _metadata_value  # noqa: F401
from .prediction_scoring import (
    _normalize_side,
    _model_class_definition,
    _prediction_columns,
    _side_prediction,
    _unavailable_side_prediction,
)


def run_prediction_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run one H5-backed, one-patient Aramina prediction from YAML."""
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_prediction_config(config, config_path)
    model_path = _config_path(
        config,
        config_path,
        section="io",
        key="input_model_joblib_path",
    )
    model_artifact = joblib.load(model_path)
    model_id, model_name, model_version = _model_identity(model_artifact, model_path)
    output_paths = _prediction_output_paths(config, config_path, model_id=model_id)
    model_info = _model_info(model_artifact, model_name)
    dataframe = _prediction_dataframe(
        config,
        config_path,
        model_artifact,
        output_paths["dataframe_joblib"],
    )
    patient_id = str(config["patient"]["patient_id"])
    target_side = _prediction_target_side(config, patient_id)
    threshold_key = _prediction_contract(model_artifact)["decision"]["threshold_key"]
    columns = _prediction_columns(model_artifact)
    target_prediction = _side_prediction(
        dataframe,
        model_info,
        patient_id=patient_id,
        target_side=target_side,
        columns=columns,
        model_name=model_name,
        threshold_key=threshold_key,
    )
    target_prediction["is_target"] = True
    contralateral_side = target_prediction["feature_row"].get("contralateral_side")
    contralateral_prediction = (
        _side_prediction(
            dataframe,
            model_info,
            patient_id=patient_id,
            target_side=contralateral_side,
            columns=columns,
            model_name=model_name,
            threshold_key=threshold_key,
            force_no_symmetry=True,
        )
        if _normalize_side(contralateral_side) is not None
        else _unavailable_side_prediction(_model_class_definition(model_info))
    )
    reports = _prediction_reports(
        config=config,
        output_paths=output_paths,
        model_path=model_path,
        model_artifact=model_artifact,
        model_id=model_id,
        model_name=model_name,
        model_version=model_version,
        target_prediction=target_prediction,
        contralateral_prediction=contralateral_prediction,
    )
    _write_reports(output_paths, reports)
    return reports


def _prediction_dataframe(
    config: dict[str, Any],
    config_path: Path,
    model_artifact: dict[str, Any],
    output_dataframe_path: Path,
) -> pd.DataFrame:
    dataframe_value = config.get("io", {}).get("input_dataframe_joblib_path")
    if dataframe_value not in {None, ""}:
        dataframe_path = _config_path(
            config,
            config_path,
            section="io",
            key="input_dataframe_joblib_path",
        )
        from xrd_preprocessing import load_preprocessing_dataframe

        return load_preprocessing_dataframe(dataframe_path)

    preprocessing_config = _prediction_preprocessing_config(model_artifact)
    validate_aramina_preprocessing_config(preprocessing_config)
    h5_path = _config_path(config, config_path, section="io", key="input_h5_path")
    _validate_h5_container_contract(
        model_artifact,
        h5_path,
        expected_patient_id=str(config["patient"]["patient_id"]),
    )
    preprocessing_config.setdefault("io", {})
    preprocessing_config["io"]["input_h5_path"] = str(h5_path)
    preprocessing_config["io"]["output_joblib_path"] = str(output_dataframe_path)
    return run_preprocessing_pipeline(
        h5_path,
        preprocessing_config,
        output_joblib_path=output_dataframe_path,
    )


def _prediction_preprocessing_config(model_artifact: dict[str, Any]) -> dict[str, Any]:
    """Return model-held prediction preprocessing without mutating it."""
    config_yaml = model_artifact.get("prediction_preprocessing_yaml")
    if not isinstance(config_yaml, str):
        raise ValueError(
            "Model artifact has no prediction_preprocessing_yaml. Retrain model."
        )
    config = yaml.safe_load(config_yaml)
    if not isinstance(config, dict):
        raise ValueError("Model prediction_preprocessing_yaml is not a YAML mapping.")
    return deepcopy(config)


def _prediction_target_side(config: dict[str, Any], patient_id: str) -> str:
    target_side = config.get("patient", {}).get("target_side")
    if target_side not in {None, ""}:
        return str(target_side)
    raise ValueError(
        "Prediction target side is missing: set patient.target_side in predict YAML. "
        f"Target side is not inferred from H5 metadata for patient {patient_id!r}."
    )


def _write_reports(output_paths: dict[str, Path], reports: dict[str, Any]) -> None:
    for report_name in ("external_report", "internal_report"):
        prefix = "external" if report_name == "external_report" else "internal"
        report = reports[report_name]
        _write_text(
            output_paths[f"{prefix}_yaml"],
            yaml.safe_dump(report, sort_keys=False),
        )
