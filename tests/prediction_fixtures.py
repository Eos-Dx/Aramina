from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from xrd_preprocessing import save_preprocessing_artifact

from aramina.training import run_training_from_config
from aramina.training_config import PRODUCT_MODEL_NAME


def patient_frame() -> pd.DataFrame:
    rows = []
    q = np.linspace(2.0, 23.0, 256)
    for patient_idx in range(18):
        cancer = patient_idx % 3 == 0
        patient_label = "CANCER" if cancer else "BENIGN"
        for side in ("Left", "Right"):
            specimen_id = f"P{patient_idx:02d}_{side}"
            specimen_label = patient_label if side == "Left" else "BENIGN"
            for measurement_idx in range(3):
                shift = 0.8 if specimen_label == "CANCER" else -0.4
                rows.append(
                    {
                        "patientId": f"P{patient_idx:02d}",
                        "specimenId": specimen_id,
                        "measurementId": f"{specimen_id}_M{measurement_idx}",
                        "side": side,
                        "product_status_group": specimen_label,
                        "radial_profile_data": shift
                        + np.sin(q / 3.0)
                        + measurement_idx * 0.01,
                        "q_range": q,
                        "age": 45 + patient_idx,
                        "biopsy": side == "Left",
                    }
                )
    return pd.DataFrame(rows)


def training_config(input_path: Path, output_folder: Path) -> dict:
    return {
        "contract": "aramina_training_config_v0_3",
        "model": {
            "name": PRODUCT_MODEL_NAME,
            "version": "0.1-beta",
            "model_author": "test",
            "clinical_stage": "research draft",
            "intended_use": "Synthetic decision-support test.",
        },
        "run": {"evaluation": True, "train_on_all": True},
        "input": {"dataframe_joblib_path": str(input_path)},
        "output": {"folder": str(output_folder)},
        "evaluation": {
            "method": "repeated_stratified_kfold",
            "folds": 5,
            "repeats": 20,
            "random_seed": 42,
        },
    }


def prediction_config(
    dataframe_path: Path,
    model_path: Path,
    output_folder: Path,
    *,
    patient_id: str = "P00",
    target_side: str = "Left",
) -> dict:
    return {
        "run": {
            "analysis_author": "Test Author",
            "prediction_comment": "synthetic test",
            "synthetic_test_mode": True,
        },
        "io": {
            "input_dataframe_joblib_path": str(dataframe_path),
            "input_model_joblib_path": str(model_path),
            "output_folder": str(output_folder),
        },
        "patient": {"patient_id": patient_id, "target_side": target_side},
    }


def train_model(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("prediction_model")
    dataframe_path = root / "training.joblib"
    config_path = root / "train.yaml"
    save_preprocessing_artifact(
        patient_frame(),
        dataframe_path,
        preprocessing_config_text=(
            "pipeline:\n"
            "  steps:\n"
            "  - name: test\n"
            "    transformer: H5ToDataFrameTransformer\n"
        ),
        metadata={"input_h5_sha256": "test-h5"},
    )
    config_path.write_text(
        yaml.safe_dump(training_config(dataframe_path, root / "runs")),
        encoding="utf-8",
    )
    result = run_training_from_config(config_path)
    return Path(result["model_path"]), dataframe_path
