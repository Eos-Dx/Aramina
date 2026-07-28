from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from xrd_preprocessing import load_preprocessing_config, save_preprocessing_artifact

from aramina.preprocessing_lineage import build_preprocessing_lineage


def save_training_preprocessing_artifact(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    input_h5_sha256: str,
) -> dict:
    root = Path(__file__).parents[1]
    config = load_preprocessing_config(
        root
        / "config"
        / "preprocessing"
        / "config_preprocessing_biopsy_patients_v0_2.yaml"
    )
    return save_preprocessing_artifact(
        dataframe,
        path,
        preprocessing_config_text=yaml.safe_dump(config, sort_keys=False),
        preprocessing_config=config,
        metadata={
            "input_h5_sha256": input_h5_sha256,
            "aramina_preprocessing_lineage": build_preprocessing_lineage(config),
        },
    )
