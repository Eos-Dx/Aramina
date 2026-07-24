"""Local HTTP API for one immutable Aramis prediction artifact."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .prediction import run_prediction_from_config


MODEL_PATH = Path(os.environ.get("ARAMIS_MODEL_PATH", "")).expanduser().resolve()

app = FastAPI(title="Aramis Immutable Model Service", version="0.1")


@app.get("/health")
def health() -> dict[str, str]:
    """Report service readiness without exposing model internals."""
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Configured model artifact is unavailable.",
        )
    return {"status": "ready", "model_artifact": MODEL_PATH.name}


@app.post("/predict")
async def predict(
    input_h5: UploadFile = File(...),
    request_json: str = Form(...),
) -> dict[str, Any]:
    """Score one H5 v0.3 patient container using the frozen artifact."""
    request = _validated_request(request_json)
    if not input_h5.filename or not input_h5.filename.endswith(".h5"):
        raise HTTPException(status_code=400, detail="input_h5 must be an .h5 file.")
    try:
        with tempfile.TemporaryDirectory(prefix="aramis-api-") as temp:
            temp_path = Path(temp)
            h5_path = temp_path / "one_patient.h5"
            h5_path.write_bytes(await input_h5.read())
            config_path = temp_path / "prediction.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "run": {
                            "analysis_author": request["analysis_author"],
                            "prediction_comment": request["prediction_comment"],
                        },
                        "io": {
                            "input_h5_path": str(h5_path),
                            "input_model_joblib_path": str(MODEL_PATH),
                            "output_folder": str(temp_path / "reports"),
                        },
                        "patient": {
                            "patient_id": request["patient_id"],
                            "target_side": request["target_side"],
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            return run_prediction_from_config(config_path)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validated_request(raw: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="request_json must be valid JSON.",
        ) from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="request_json must be an object.")
    allowed = {"analysis_author", "prediction_comment", "patient_id", "target_side"}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unsupported request fields: {unknown}")
    target_side = _required_text(value, "target_side").lower()
    if target_side not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="target_side must be left or right.")
    return {
        "analysis_author": _required_text(value, "analysis_author"),
        "prediction_comment": str(value.get("prediction_comment", "")).strip(),
        "patient_id": _required_text(value, "patient_id"),
        "target_side": target_side,
    }


def _required_text(value: dict[str, Any], key: str) -> str:
    text = str(value.get(key, "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{key} is required.")
    return text
