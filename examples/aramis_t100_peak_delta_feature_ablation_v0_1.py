from __future__ import annotations

from copy import deepcopy
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
BASE_STRATIFIED_CONFIG = (
    ROOT
    / "config"
    / "training"
    / "t100_peak_delta_experiment_v0_1"
    / "aramis_t100_peak_delta_stratified_5fold_c1_1_c2_1.yaml"
)
BASE_TRAIN_ALL_CONFIG = (
    ROOT
    / "config"
    / "training"
    / "t100_peak_delta_experiment_v0_1"
    / "aramis_t100_peak_delta_train_all_c1_1_c2_1.yaml"
)
OUTPUT_ROOT = ROOT / "examples" / "outputs" / "t100_peak_delta_experiment_v0_1"
CONFIG_DIR = OUTPUT_ROOT / "feature_ablation_configs"
TRAINING_DIR = OUTPUT_ROOT / "feature_ablation_training"
ABLATION_CSV = (
    ROOT
    / "docs"
    / "modeling"
    / "results"
    / "t100_peak_delta_m2q_feature_ablation_v0_1.csv"
)
COEFFICIENT_CSV = (
    ROOT
    / "docs"
    / "modeling"
    / "results"
    / "t100_peak_delta_m2q_train_all_coefficients_v0_1.csv"
)

LR1_C = 0.3
LR2_C = 0.1

FEATURE_GROUPS = {
    "full": [],
    "drop_peak_delta": [
        "sk_peak14_intensity_abs_delta",
        "sk_mean_peak_value_abs_delta",
    ],
    "drop_sigma": [
        "sk_sigma_target1",
        "sk_sigma_contralateral1",
        "sk_sigma_target2",
        "sk_sigma_contralateral2",
    ],
    "drop_rms": ["sk_meanrms1", "sk_meanrms2"],
    "drop_weighted_rms": ["sk_weightedrms1", "sk_weightedrms2"],
    "drop_mahalanobis": ["sk_mahalanobis1", "sk_mahalanobis2"],
    "drop_distance_tail": [
        "sk_wasserstein_distance_mu_tc",
        "sk_cosine_distance_full_q2",
        "sk_wasserstein_distance_full_q2",
    ],
    "drop_all_sk_symmetry": [
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
    ],
}


def _base_io(config: dict, *, name: str) -> dict:
    out = deepcopy(config)
    out["training"]["name"] = name
    out["training"]["role"] = "peak_delta_feature_ablation"
    out["model"]["selected_models"] = ["M2Q"]
    out["model"]["lr1_logreg_c"] = LR1_C
    out["model"]["lr2_logreg_c"] = LR2_C
    out["model"].pop("logreg_c", None)
    out["io"]["input_dataframe_joblib_path"] = str(INPUT_JOBLIB)
    out["io"]["prediction_preprocessing_config_path"] = str(
        PREDICTION_PREPROCESSING_CONFIG
    )
    out["io"]["output_model_joblib_path"] = str(TRAINING_DIR / f"{name}.joblib")
    out["io"]["output_json_path"] = str(TRAINING_DIR / f"{name}_summary.json")
    out["io"]["output_yaml_path"] = str(TRAINING_DIR / f"{name}_description.yaml")
    return out


def _run_config(config: dict, *, name: str) -> dict:
    config_path = CONFIG_DIR / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return run_training_from_config(config_path)


def _write_coefficients(artifact: dict) -> None:
    model = artifact["models"]["M2Q"]["final_model"]
    columns = artifact["models"]["M2Q"]["feature_columns"]
    coefficients = model.named_steps["logreg"].coef_[0]
    rows = [
        {
            "feature": feature,
            "coefficient": coefficient,
            "abs_coefficient": abs(coefficient),
        }
        for feature, coefficient in zip(columns, coefficients, strict=True)
    ]
    (
        pd.DataFrame(rows)
        .sort_values("abs_coefficient", ascending=False)
        .to_csv(COEFFICIENT_CSV, index=False)
    )


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_CSV.parent.mkdir(parents=True, exist_ok=True)

    stratified_base = yaml.safe_load(BASE_STRATIFIED_CONFIG.read_text(encoding="utf-8"))
    rows = []
    for group_name, drop_columns in FEATURE_GROUPS.items():
        name = f"aramis_t100_peak_delta_m2q_{group_name}_stratified_5fold"
        config = _base_io(stratified_base, name=name)
        config["model"]["drop_feature_columns"] = drop_columns
        artifact = _run_config(config, name=name)
        summary = artifact["metric_summary"].copy()
        summary.insert(0, "feature_group", group_name)
        summary.insert(1, "dropped_features", ";".join(drop_columns))
        rows.extend(summary.to_dict(orient="records"))

    result = pd.DataFrame(rows)
    result.to_csv(ABLATION_CSV, index=False)

    train_all_base = yaml.safe_load(BASE_TRAIN_ALL_CONFIG.read_text(encoding="utf-8"))
    train_all_config = _base_io(
        train_all_base,
        name="aramis_t100_peak_delta_m2q_full_train_all_c1_0p3_c2_0p1",
    )
    train_all_config["model"]["drop_feature_columns"] = []
    train_all_artifact = _run_config(
        train_all_config,
        name="aramis_t100_peak_delta_m2q_full_train_all_c1_0p3_c2_0p1",
    )
    _write_coefficients(train_all_artifact)

    print(ABLATION_CSV)
    print(COEFFICIENT_CSV)
    print(
        result[
            [
                "feature_group",
                "roc_auc_mean",
                "sensitivity_target_mean",
                "specificity_target_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
