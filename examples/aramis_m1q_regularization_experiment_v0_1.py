"""M1Q regularization experiment for Aramis v0.1 research draft.

The experiment uses T100 biopsy-patient preprocessing and tunes the shared L2
LogisticRegression strength on repeated patient-safe stratified K-fold. Other
validation modes are rerun only after the K-fold grid selects the conservative
regularization value.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from textwrap import dedent
from typing import Any

import joblib
import pandas as pd
import yaml

from aramis.training import train_patient_m0_m1_m2_model_artifact
from xrd_preprocessing import (
    load_preprocessing_artifact,
    load_preprocessing_config,
    load_preprocessing_dataframe,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_JOBLIB = (
    ROOT
    / "examples"
    / "outputs"
    / "model_selection_m1q_v0_1"
    / "preprocessing"
    / "aramis_t100_biopsy_patients_model_input.joblib"
)
PREDICT_PREPROCESSING_CONFIG = (
    ROOT
    / "config"
    / "preprocessing"
    / "aramis_prediction_patient_model_input_v0_1.yaml"
)
OUTPUT_DIR = (
    ROOT
    / "examples"
    / "outputs"
    / "model_selection_m1q_regularization_v0_1"
)
DOC_PATH = ROOT / "docs" / "modeling" / "m1q_regularization_experiment_v0_1.md"

C_GRID = [0.03, 0.1, 0.3, 1.0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_preprocessing_dataframe(INPUT_JOBLIB)
    preprocessing_artifact = load_preprocessing_artifact(INPUT_JOBLIB)
    prediction_preprocessing_text = PREDICT_PREPROCESSING_CONFIG.read_text(
        encoding="utf-8"
    )
    prediction_preprocessing = {
        "path": str(PREDICT_PREPROCESSING_CONFIG),
        "config": load_preprocessing_config(PREDICT_PREPROCESSING_CONFIG),
        "config_text": prediction_preprocessing_text,
        "config_sha256": sha256(
            prediction_preprocessing_text.encode("utf-8")
        ).hexdigest(),
    }

    kfold_path = OUTPUT_DIR / "m1q_t100_regularization_kfold_grid.csv"
    if kfold_path.exists():
        print(f"using cached K-fold grid: {kfold_path}", flush=True)
        kfold_df = pd.read_csv(kfold_path)
    else:
        kfold_rows = []
        for logreg_c in C_GRID:
            print(f"running K-fold grid C={logreg_c}", flush=True)
            config = _training_config(
                name=f"aramis_m1q_t100_kfold20_c{_c_slug(logreg_c)}",
                logreg_c=logreg_c,
                mode="stratified_kfold",
                n_splits=5,
                n_repeats=20,
            )
            artifact = _train(
                df,
                preprocessing_artifact,
                prediction_preprocessing,
                config,
            )
            kfold_rows.append(
                _summary_row(artifact, logreg_c=logreg_c, mode="kfold_5x20")
            )
        kfold_df = pd.DataFrame(kfold_rows)

    kfold_df = kfold_df.sort_values(
        ["roc_auc_mean", "specificity_target_mean"],
        ascending=[False, False],
    )
    selected_c = _select_conservative_c(kfold_df)

    mode_rows = []
    final_artifacts = {}
    for mode_name, kwargs in _validation_modes().items():
        print(f"running selected-C mode={mode_name} C={selected_c}", flush=True)
        config = _training_config(
            name=f"aramis_m1q_t100_{mode_name}_c{_c_slug(selected_c)}",
            logreg_c=selected_c,
            **kwargs,
        )
        artifact = _train(df, preprocessing_artifact, prediction_preprocessing, config)
        final_artifacts[mode_name] = artifact
        mode_rows.append(_summary_row(artifact, logreg_c=selected_c, mode=mode_name))

    mode_df = pd.DataFrame(mode_rows)
    kfold_df.to_csv(kfold_path, index=False)
    mode_df.to_csv(OUTPUT_DIR / "m1q_t100_selected_c_validation_modes.csv", index=False)
    joblib.dump(
        final_artifacts["train_all"],
        OUTPUT_DIR / "aramis_m1q_t100_selected_c_train_all.joblib",
    )
    _write_doc(kfold_df, mode_df, selected_c)

    print(f"selected_c={selected_c}")
    print(kfold_df.to_string(index=False))
    print(mode_df.to_string(index=False))
    print(DOC_PATH)


def _train(
    df: pd.DataFrame,
    preprocessing_artifact: dict[str, Any],
    prediction_preprocessing: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    config_text = yaml.safe_dump(config, sort_keys=False)
    return train_patient_m0_m1_m2_model_artifact(
        df,
        config=config,
        config_text=config_text,
        input_dataframe_joblib_path=INPUT_JOBLIB,
        preprocessing_artifact=preprocessing_artifact,
        prediction_preprocessing=prediction_preprocessing,
    )


def _training_config(
    *,
    name: str,
    logreg_c: float,
    mode: str,
    n_splits: int,
    n_repeats: int = 1,
    test_size: float = 0.2,
) -> dict[str, Any]:
    return {
        "training": {
            "name": name,
            "version": "0.1",
            "branch": "one_to_many",
            "clinical_stage": "research draft",
            "intended_use": (
                "decision-support p_cancer research draft; "
                "requires radiologist review"
            ),
            "role": "m1q_regularization_experiment",
        },
        "io": {
            "input_dataframe_joblib_path": str(INPUT_JOBLIB),
            "output_model_joblib_path": str(OUTPUT_DIR / f"{name}.joblib"),
            "prediction_preprocessing_config_path": str(PREDICT_PREPROCESSING_CONFIG),
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
            "logreg_c": float(logreg_c),
            "penalty": "l2",
        },
        "evaluation": {
            "mode": mode,
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "test_size": float(test_size),
            "random_state": 42,
            "target_sensitivity": 0.95,
        },
    }


def _validation_modes() -> dict[str, dict[str, Any]]:
    return {
        "kfold_5x20": {
            "mode": "stratified_kfold",
            "n_splits": 5,
            "n_repeats": 20,
            "test_size": 0.2,
        },
        "patient_80_20_x50": {
            "mode": "repeated_stratified_shuffle",
            "n_splits": 50,
            "test_size": 0.2,
        },
        "loovm": {
            "mode": "loovm",
            "n_splits": 1,
            "test_size": 0.2,
        },
        "train_all": {
            "mode": "all_on_all",
            "n_splits": 1,
            "test_size": 0.2,
        },
    }


def _summary_row(artifact: dict[str, Any], *, logreg_c: float, mode: str) -> dict[str, Any]:
    row = artifact["metric_summary"].iloc[0].to_dict()
    return {
        "mode": mode,
        "logreg_c": float(logreg_c),
        "splits": int(row["splits"]),
        "roc_auc_mean": float(row["roc_auc_mean"]),
        "roc_auc_std": float(row["roc_auc_std"]),
        "sensitivity_target_mean": float(row["sensitivity_target_mean"]),
        "sensitivity_target_std": float(row["sensitivity_target_std"]),
        "specificity_target_mean": float(row["specificity_target_mean"]),
        "specificity_target_std": float(row["specificity_target_std"]),
        "balanced_accuracy_target_mean": float(row["balanced_accuracy_target_mean"]),
        "ppv_target_mean": float(row["ppv_target_mean"]),
        "npv_target_mean": float(row["npv_target_mean"]),
        "threshold_target_median": float(
            artifact["split_predictions"]["threshold_target"].median()
        ),
    }


def _select_conservative_c(kfold_df: pd.DataFrame) -> float:
    best = kfold_df.iloc[0]
    cutoff = float(best["roc_auc_mean"] - 0.005)
    candidates = kfold_df[kfold_df["roc_auc_mean"] >= cutoff].sort_values("logreg_c")
    return float(candidates.iloc[0]["logreg_c"])


def _write_doc(kfold_df: pd.DataFrame, mode_df: pd.DataFrame, selected_c: float) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# M1Q T100 Regularization Experiment v0.1

Status: research draft.

Purpose: tune L2 regularization for the current Aramis M1Q candidate using the
T100 biopsy-patient model-input DataFrame. Regularization is selected on
repeated patient-safe stratified 5-fold validation, not on train-all metrics.

Dataset:

```text
{INPUT_JOBLIB.relative_to(ROOT)}
```

Selection rule:

```text
primary mode: repeated patient-safe stratified 5-fold x20
target sensitivity: 0.95
penalty: L2 LogisticRegression
C grid: {C_GRID}
selected C: {selected_c}
rule: highest K-fold ROC AUC, then smaller C if ROC AUC differs by less than 0.005
```

## K-fold Regularization Grid

{_markdown_table(kfold_df)}

## Selected-C Validation Modes

{_markdown_table(mode_df)}

## Interpretation

The K-fold grid is used to choose regularization. The final train-all artifact is
an optimistic fitted model candidate, not validation evidence. Thresholds are
selected on training folds for the 0.95 sensitivity target and then applied to
held-out patients for split-based modes.
"""
    DOC_PATH.write_text(dedent(text), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    table = deepcopy(df)
    for column in table.select_dtypes(include=["float"]).columns:
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    columns = [str(column) for column in table.columns]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for values in table.itertuples(index=False):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def _c_slug(value: float) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
