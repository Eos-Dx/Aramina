from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from aramis.training import run_training_from_config


ROOT = Path(__file__).resolve().parents[1]
INPUT_JOBLIB = (
    ROOT
    / "examples"
    / "outputs"
    / "model_selection_m1q_v0_1"
    / "preprocessing"
    / "aramis_t100_biopsy_patients_model_input.joblib"
)
PREDICTION_PREPROCESSING_CONFIG = (
    ROOT / "config" / "preprocessing" / "aramis_prediction_patient_model_input_v0_1.yaml"
)
BASE_CONFIG = (
    ROOT
    / "config"
    / "training"
    / "t100_peak_delta_experiment_v0_1"
    / "aramis_t100_peak_delta_stratified_5fold_c1_1_c2_1.yaml"
)
OUTPUT_ROOT = ROOT / "examples" / "outputs" / "t100_peak_delta_experiment_v0_1"
GRID_CONFIG_DIR = OUTPUT_ROOT / "regularization_configs"
GRID_TRAINING_DIR = OUTPUT_ROOT / "regularization_training"
RESULT_CSV = (
    ROOT
    / "docs"
    / "modeling"
    / "results"
    / "t100_peak_delta_regularization_grid_v0_1.csv"
)

C_VALUES = [0.1, 0.3, 1.0, 3.0]
SELECTED_MODELS = ["M0Q", "M1Q", "M2Q"]


def _grid_config(base: dict, *, lr1_c: float, lr2_c: float) -> dict:
    config = deepcopy(base)
    name = f"aramis_t100_peak_delta_stratified_5fold_c1_{lr1_c:g}_c2_{lr2_c:g}"
    name = name.replace(".", "p")
    config["training"]["name"] = name
    config["training"]["role"] = "peak_delta_regularization_grid"
    config["model"]["selected_models"] = SELECTED_MODELS
    config["model"]["lr1_logreg_c"] = lr1_c
    config["model"]["lr2_logreg_c"] = lr2_c
    config["model"].pop("logreg_c", None)
    config["io"]["input_dataframe_joblib_path"] = str(INPUT_JOBLIB)
    config["io"]["prediction_preprocessing_config_path"] = str(
        PREDICTION_PREPROCESSING_CONFIG
    )
    config["io"]["output_model_joblib_path"] = str(
        GRID_TRAINING_DIR / f"{name}.joblib"
    )
    config["io"]["output_json_path"] = str(
        GRID_TRAINING_DIR / f"{name}_summary.json"
    )
    config["io"]["output_yaml_path"] = str(
        GRID_TRAINING_DIR / f"{name}_description.yaml"
    )
    return config


def main() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    GRID_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    GRID_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for lr1_c, lr2_c in product(C_VALUES, C_VALUES):
        config = _grid_config(base, lr1_c=lr1_c, lr2_c=lr2_c)
        config_path = GRID_CONFIG_DIR / f"{config['training']['name']}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        artifact = run_training_from_config(config_path)
        summary = artifact["metric_summary"].copy()
        summary.insert(0, "lr1_logreg_c", lr1_c)
        summary.insert(1, "lr2_logreg_c", lr2_c)
        summary["training_config_path"] = str(config_path)
        summary["model_joblib_path"] = str(
            GRID_TRAINING_DIR / f"{config['training']['name']}.joblib"
        )
        rows.extend(summary.to_dict(orient="records"))

    result = pd.DataFrame(rows)
    result.to_csv(RESULT_CSV, index=False)
    print(RESULT_CSV)
    print(
        result.sort_values(
            ["model_name", "roc_auc_mean", "specificity_target_mean"],
            ascending=[True, False, False],
        )
        .groupby("model_name")
        .head(5)[
            [
                "model_name",
                "lr1_logreg_c",
                "lr2_logreg_c",
                "roc_auc_mean",
                "sensitivity_target_mean",
                "specificity_target_mean",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
