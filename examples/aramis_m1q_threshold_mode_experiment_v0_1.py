"""Run M1Q threshold and validation-mode comparison.

This experiment compares T70, T100, and T130 monochromaticity pools using the
same biopsy-patient cohort rule:

patient is kept if any row has biopsy=True
contralateral rows are kept for symmetry
NORMAL is mapped to BENIGN
EXCLUDE is dropped before training

The model is fixed to M1Q. Validation modes are all-on-all, LOOVM, stratified
5-fold, and repeated patient-safe 80/20 splits.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from xrd_preprocessing import save_preprocessing_artifact

from aramis.training import run_training_from_config


ROOT = Path(__file__).resolve().parents[1]
WIDE_POOL_DIR = ROOT / "examples" / "outputs" / "threshold_grid_patient_cohorts" / "wide_pools"
PREPROCESSING_OUTPUT_DIR = ROOT / "examples" / "outputs" / "model_selection_m1q_v0_1" / "preprocessing"
TRAINING_OUTPUT_DIR = ROOT / "examples" / "outputs" / "model_selection_m1q_v0_1" / "training"
TRAINING_CONFIG_DIR = ROOT / "config" / "training" / "model_selection_m1q_v0_1"
RESULTS_DIR = ROOT / "docs" / "modeling" / "results"
SUMMARY_CSV = RESULTS_DIR / "m1q_threshold_mode_comparison_v0_1.csv"
SUMMARY_MD = ROOT / "docs" / "modeling" / "m1q_threshold_mode_comparison_v0_1.md"


THRESHOLDS = {
    "T70": {
        "wide_pool": WIDE_POOL_DIR / "aramis_wide_t70.joblib",
        "threshold_factor": 0.7,
        "monochromaticity_max_score": 0.00525,
    },
    "T100": {
        "wide_pool": WIDE_POOL_DIR / "aramis_wide_t100.joblib",
        "threshold_factor": 1.0,
        "monochromaticity_max_score": 0.0075,
    },
    "T130": {
        "wide_pool": WIDE_POOL_DIR / "aramis_wide_t130.joblib",
        "threshold_factor": 1.3,
        "monochromaticity_max_score": 0.00975,
    },
}


MODES = {
    "train_all": {"mode": "all_on_all", "n_splits": 1},
    "loovm": {"mode": "loovm", "n_splits": 1},
    "stratified_5fold": {"mode": "stratified_kfold", "n_splits": 5},
    "patient_80_20_x50": {
        "mode": "repeated_stratified_shuffle",
        "n_splits": 50,
        "test_size": 0.20,
    },
}


MODEL_INPUT_COLUMNS = [
    "patientId",
    "specimenId",
    "side",
    "position",
    "started_at",
    "measurementDate",
    "specimen_status",
    "product_status_group",
    "product_diagnosis",
    "patient_product_diagnosis",
    "age",
    "biopsy",
    "sample_biopsy",
    "sample_biopsy_type",
    "sample_height_in",
    "sample_weight_lb",
    "breast_density",
    "birads",
    "sample_thickness_mm",
    "calibrant_thickness_mm",
    "poni_q_max_nm_inv",
    "measurement_data_source",
    "q_range",
    "radial_profile_data",
    "snr_db",
    "specimen_measurement_count",
    "radial_profile_value_at_q",
    "radial_profile_nearest_q_nm_inv",
    "radial_profile_q_delta_nm_inv",
    "radial_profile_value_pass",
]


def main() -> None:
    PREPROCESSING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preprocessing_paths = {
        threshold: _write_preprocessing_artifact(threshold, spec)
        for threshold, spec in THRESHOLDS.items()
    }
    summary_rows = []
    for threshold, preprocessing_path in preprocessing_paths.items():
        for mode_name, mode_spec in MODES.items():
            config_path = _write_training_config(threshold, preprocessing_path, mode_name, mode_spec)
            artifact = _load_or_run_training(config_path)
            summary_rows.extend(_summary_rows(threshold, mode_name, artifact, config_path))

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["threshold", "mode"])
    summary.to_csv(SUMMARY_CSV, index=False)
    _write_markdown_summary(summary)
    print(summary.to_string(index=False))
    print(SUMMARY_CSV)
    print(SUMMARY_MD)


def _write_preprocessing_artifact(threshold: str, spec: dict[str, Any]) -> Path:
    source = spec["wide_pool"]
    if not source.exists():
        raise FileNotFoundError(f"Missing wide pool joblib: {source}")
    artifact = joblib.load(source)
    df = artifact["dataframe"] if isinstance(artifact, dict) else artifact
    model_df = _biopsy_patient_model_input(df)
    output_path = PREPROCESSING_OUTPUT_DIR / f"aramis_{threshold.lower()}_biopsy_patients_model_input.joblib"
    config = {
        "aramis_preprocessing": {
            "name": f"aramis_{threshold.lower()}_biopsy_patients_model_input",
            "version": "0.1",
            "branch": "one_to_many",
            "clinical_stage": "research draft",
        },
        "cohort_rule": {
            "patient_filter": "keep patient if any row has biopsy=True",
            "contralateral_rows": "kept for symmetry",
            "normal_mapping": "NORMAL -> BENIGN",
            "excluded_rows": "EXCLUDE and unlabeled rows",
        },
        "experiment": {
            "threshold": threshold,
            "threshold_factor": spec["threshold_factor"],
            "monochromaticity_max_score": spec["monochromaticity_max_score"],
            "source_wide_pool": str(source),
        },
    }
    save_preprocessing_artifact(
        model_df,
        output_path,
        preprocessing_config=config,
        metadata={
            "experiment": "m1q_threshold_mode_comparison_v0_1",
            "threshold": threshold,
            "source_wide_pool": str(source),
            "rows": int(len(model_df)),
            "patients": int(model_df["patientId"].nunique()),
            "specimens": int(model_df["specimenId"].nunique()),
        },
    )
    return output_path


def _biopsy_patient_model_input(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    _require_columns(work, ["patientId", "specimenId", "product_status_group", "biopsy"])
    biopsy_patients = set(work.loc[_boolean_series(work["biopsy"]), "patientId"].astype(str))
    work = work[work["patientId"].astype(str).isin(biopsy_patients)].copy()
    work["product_status_group"] = work["product_status_group"].replace({"NORMAL": "BENIGN"})
    work = work[work["product_status_group"].isin(["BENIGN", "CANCER"])].copy()
    labelled_biopsy_patients = set(
        work.loc[_boolean_series(work["biopsy"]), "patientId"].astype(str)
    )
    work = work[work["patientId"].astype(str).isin(labelled_biopsy_patients)].copy()
    _fill_missing_audit_columns(work)
    _require_columns(work, MODEL_INPUT_COLUMNS)
    return work[MODEL_INPUT_COLUMNS].reset_index(drop=True)


def _write_training_config(
    threshold: str,
    preprocessing_path: Path,
    mode_name: str,
    mode_spec: dict[str, Any],
) -> Path:
    config = _base_training_config()
    run_name = f"aramis_m1q_{threshold.lower()}_{mode_name}"
    config["training"]["name"] = run_name
    config["io"]["input_dataframe_joblib_path"] = _relative_to(preprocessing_path, TRAINING_CONFIG_DIR)
    config["io"]["output_model_joblib_path"] = _relative_to(
        TRAINING_OUTPUT_DIR / f"{run_name}.joblib",
        TRAINING_CONFIG_DIR,
    )
    config["io"]["output_json_path"] = _relative_to(
        TRAINING_OUTPUT_DIR / f"{run_name}_summary.json",
        TRAINING_CONFIG_DIR,
    )
    config["io"]["output_yaml_path"] = _relative_to(
        TRAINING_OUTPUT_DIR / f"{run_name}_description.yaml",
        TRAINING_CONFIG_DIR,
    )
    config["evaluation"].update(mode_spec)
    config_path = TRAINING_CONFIG_DIR / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _load_or_run_training(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary_path = (config_path.parent / config["io"]["output_json_path"]).resolve()
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return run_training_from_config(config_path)


def _base_training_config() -> dict[str, Any]:
    return {
        "training": {
            "name": "aramis_m1q_threshold_mode",
            "version": "0.1",
            "branch": "one_to_many",
            "clinical_stage": "research draft",
            "intended_use": (
                "decision-support p_cancer research draft; requires radiologist review"
            ),
            "role": "model_selection_experiment",
        },
        "io": {
            "input_dataframe_joblib_path": "",
            "output_model_joblib_path": "",
            "output_json_path": "",
            "output_yaml_path": "",
            "prediction_preprocessing_config_path": (
                "../../preprocessing/aramis_prediction_patient_model_input_v0_1.yaml"
            ),
        },
        "model": {
            "type": "patient_m0_m1_m2_logistic_set",
            "profile_column": "radial_profile_data",
            "label_column": "product_status_group",
            "group_column": "patientId",
            "specimen_column": "specimenId",
            "side_column": "side",
            "age_column": "age",
            "biopsy_column": "biopsy",
            "lr1_row_policy": "biopsy_only",
            "selected_models": ["M1Q"],
            "logreg_c": 1.0,
        },
        "evaluation": {
            "mode": "repeated_stratified_shuffle",
            "n_splits": 50,
            "test_size": 0.20,
            "random_state": 42,
            "target_sensitivity": 0.95,
        },
    }


def _summary_rows(
    threshold: str,
    mode_name: str,
    artifact: dict[str, Any],
    config_path: Path,
) -> list[dict[str, Any]]:
    dataset_summary = _first_record(artifact["dataset_summary"])
    rows = []
    for row in _records(artifact["metric_summary"]):
        if row["model_name"] != "M1Q":
            continue
        out = deepcopy(row)
        out.update(
            {
                "threshold": threshold,
                "mode": mode_name,
                "patients": dataset_summary["final_patients"],
                "cancer_patients": dataset_summary["final_cancer_patients"],
                "benign_patients": dataset_summary["final_benign_patients"],
                "measurement_rows": dataset_summary["rows"],
                "lr1_rows": dataset_summary["lr1_rows"],
                "config_path": str(config_path),
            }
        )
        rows.append(out)
    return rows


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return list(value)


def _first_record(value: Any) -> dict[str, Any]:
    records = _records(value)
    if not records:
        raise ValueError("Expected at least one summary record.")
    return records[0]


def _write_markdown_summary(summary: pd.DataFrame) -> None:
    display = summary[
        [
            "threshold",
            "mode",
            "patients",
            "cancer_patients",
            "benign_patients",
            "roc_auc_mean",
            "roc_auc_std",
            "sensitivity_target_mean",
            "sensitivity_target_std",
            "specificity_target_mean",
            "specificity_target_std",
        ]
    ].copy()
    for column in [
        "roc_auc_mean",
        "roc_auc_std",
        "sensitivity_target_mean",
        "sensitivity_target_std",
        "specificity_target_mean",
        "specificity_target_std",
    ]:
        display[column] = display[column].map(lambda value: f"{float(value):.3f}")
    text = [
        "# M1Q Threshold And Validation-Mode Comparison v0.1",
        "",
        "Status: research draft. Not clinical validation.",
        "",
        "Cohort rule: biopsy-patient cohort; contralateral rows kept for symmetry; NORMAL mapped to BENIGN; EXCLUDE dropped.",
        "",
        "Model: M1Q only.",
        "",
        _markdown_table(display),
        "",
        f"Machine-readable table: `{SUMMARY_CSV.relative_to(ROOT)}`",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(text), encoding="utf-8")


def _boolean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _fill_missing_audit_columns(df: pd.DataFrame) -> None:
    defaults: dict[str, Any] = {
        "radial_profile_value_at_q": float("nan"),
        "radial_profile_nearest_q_nm_inv": float("nan"),
        "radial_profile_q_delta_nm_inv": float("nan"),
        "radial_profile_value_pass": True,
    }
    for column, value in defaults.items():
        if column not in df.columns:
            df[column] = value


def _relative_to(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


if __name__ == "__main__":
    main()
