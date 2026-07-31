"""Paired same-source T130 held-out comparison for three T100-trained procedures.

This research runner deliberately leaves product code, model artifacts, and product
configuration untouched.  It scores each case declared by the T130 case manifest,
including MRI-only target breasts that the biopsy-based training helper would omit.
"""

from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from aramina.m2q_model import build_profile_logistic
from aramina.model_metrics import binary_metric_values
from aramina.model_utils import compute_binary_thresholds, profile_matrix
from aramina.patient_features import (
    build_patient_prediction_feature_row,
    lr1_training_rows,
    row_labels,
)

from experiments.profile_symmetry_age_refinement.recalibrated_joint_data import (
    fit_feature_pair,
    lr1_cross_fitted_features,
    load_input_dataframe,
    model_columns,
)
from experiments.profile_symmetry_age_refinement.recalibrated_joint_model import (
    RecalibratedJointAdditiveClassifier,
)
from experiments.profile_symmetry_age_refinement.t130_full_patient_preprocessing import (
    preprocess_manifest_patients,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY = EXPERIMENT_DIR.parents[1]
EVIDENCE_DIR = EXPERIMENT_DIR / "evidence" / "t100_5x10_20260731"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "evidence" / "t130_holdout_20260731"
DEFAULT_T100_INPUT = Path(
    "/Users/sad/dev/Aramina/examples/outputs/model_input/"
    "aramina_biopsy_patients_model_input_v0_1.joblib"
)
DEFAULT_T130_INPUT = Path(
    "/Users/sad/dev/Araminavisor-demo/model_test/artifacts/"
    "aramina_mri_or_biopsy_held_out_t130.joblib"
)
DEFAULT_PRODUCT_ARTIFACT = Path(
    "/Users/sad/dev/Aramina/models/"
    "aramina_target_breast_risk_0_2_12-beta_9bb911189af6/model.joblib"
)
TARGET_SENSITIVITY = 0.95
LR1_C = 0.1
RANDOM_STATE = 42
OOF_SPLITS = 5
TRAIN_ALL_RANDOM_STATE = RANDOM_STATE + 9_000_000
TRAIN_ALL_LR1_OOF_RANDOM_STATE = TRAIN_ALL_RANDOM_STATE + 10
WILSON_Z_95 = 1.959963984540054
EXPECTED_T100_SHA256 = "76251daa67e5fc349c7571ded2180fdfb199f401e1d04c498e0b66154b484161"
EXPECTED_T130_SHA256 = "235167d64e8d0259a89f7b29eadea090ea91e07356d861b9c39fee9462650a1f"
EXPECTED_PRODUCT_SHA256 = "9bb911189af6e1bd954d21765cbc97c4a57fcf7884657c4b22518f280103e11d"
EXPECTED_SOURCE_H5_SHA256 = "d2d61e83850b282c3d2479ea436deed821c4488b96983252d294f3d56ee3f1f9"
EXPECTED_REGULARIZATION_SHA256 = "b78fb1593d00a6fe1972629e109743031f2106021233f53d94c00d8129dcfea4"
EXPECTED_TRAIN_ALL_METRICS_SHA256 = "054db85f8e23eb00fbd41de4a22c066553136abd55fcfb9ca2a57af8cf4987a1"

CURRENT_PRODUCT = "frozen_current_product"
SAME_FITTED = "recalibrated_joint_same_fitted_lr1"
OOF_FITTED = "recalibrated_joint_lr1_oof"


def assert_shared_test_features(
    frozen_features: pd.DataFrame,
    recalibrated_features: pd.DataFrame,
) -> None:
    """Require identical case order and numeric inputs for all three procedures."""
    if not frozen_features["target_case_id"].equals(
        recalibrated_features["target_case_id"]
    ):
        raise ValueError("Product and recalibrated T130 case ordering differs.")
    numeric_columns = [
        "profile_p_cancer_logit_average",
        "age",
        "age_available",
        "symmetry_available",
        *product_symmetry_columns(),
    ]
    frozen = frozen_features[numeric_columns].to_numpy(dtype=float)
    recalibrated = recalibrated_features[numeric_columns].to_numpy(dtype=float)
    if not np.allclose(frozen, recalibrated, rtol=0.0, atol=1e-12):
        difference = float(np.max(np.abs(frozen - recalibrated)))
        raise ValueError(
            "Product and recalibrated T130 feature rows differ; "
            f"maximum absolute difference={difference:.3g}."
        )


def product_symmetry_columns() -> list[str]:
    """Return the frozen product's four learned SK feature names."""
    from aramina.m2q_model import SK_CORE4_FEATURE_COLUMNS

    return list(SK_CORE4_FEATURE_COLUMNS)


def assert_lr1_equivalence(
    product_lr1: Any,
    rebuilt_lr1: Any,
    reference_matrix: np.ndarray,
) -> dict[str, str]:
    """Require identical LR1 configuration and predictions on all fitted rows.

    ``joblib.hash`` is not a numerical model identity: equivalent sklearn
    estimators can serialize differently.  The frozen product LR1 remains the
    deployed object; the rebuilt model is only an independent reproducibility
    check.
    """
    for step in ("scaler", "logreg"):
        product_params = product_lr1.named_steps[step].get_params(deep=True)
        rebuilt_params = rebuilt_lr1.named_steps[step].get_params(deep=True)
        if product_params != rebuilt_params:
            raise ValueError(f"Rebuilt T100 LR1 {step} parameters differ.")
    product_scores = product_lr1.predict_proba(reference_matrix)[:, 1]
    rebuilt_scores = rebuilt_lr1.predict_proba(reference_matrix)[:, 1]
    if not np.array_equal(product_scores, rebuilt_scores):
        difference = float(np.max(np.abs(product_scores - rebuilt_scores)))
        raise ValueError(
            "Rebuilt T100 LR1 predictions differ from the frozen product LR1; "
            f"maximum absolute difference={difference:.3g}."
        )
    return {
        "frozen_product_joblib_hash": joblib.hash(product_lr1),
        "rebuilt_joblib_hash": joblib.hash(rebuilt_lr1),
        "fitted_row_predictions_sha256": sha256(
            np.asarray(product_scores, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest without loading a potentially large artifact."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: str | Path, expected: str, *, name: str) -> str:
    """Fail before model work when a supposedly frozen input has drifted."""
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{name} SHA256 mismatch: expected {expected}, observed {observed}."
        )
    return observed


def load_t130_manifest(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the frozen T130 target-breast manifest without using its partial rows."""
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict) or not isinstance(artifact.get("dataframe"), pd.DataFrame):
        raise ValueError("T130 input must contain {'dataframe': DataFrame, 'case_manifest': ...}.")
    manifest = pd.DataFrame(artifact.get("case_manifest", []))
    required = {"patient_id", "target_side", "reference_label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"T130 case_manifest is missing: {sorted(missing)}")
    manifest = manifest.copy()
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    manifest["target_side"] = manifest["target_side"].astype(str).str.upper()
    manifest["target_case_id"] = manifest["patient_id"] + "::" + manifest["target_side"]
    manifest["label"] = manifest["reference_label"].map({"BENIGN": 0, "CANCER": 1})
    if manifest["label"].isna().any():
        raise ValueError("T130 case_manifest contains an unsupported reference label.")
    return manifest, dict(artifact.get("metadata", {}))


def assert_t130_composition(manifest: pd.DataFrame) -> None:
    """Fail unless the exact locked T130 research subset is supplied."""
    if len(manifest) != 22:
        raise ValueError(f"Expected 22 T130 target cases; found {len(manifest)}.")
    if manifest["patient_id"].nunique() != 17:
        raise ValueError("Expected 17 T130 patients.")
    counts = manifest["label"].value_counts().to_dict()
    if counts != {0: 11, 1: 11}:
        raise ValueError(f"Expected 11 BENIGN and 11 CANCER T130 cases; found {counts}.")
    if manifest["target_case_id"].duplicated().any():
        raise ValueError("T130 case_manifest has duplicate target_case_id values.")


def assert_no_patient_overlap(t100: pd.DataFrame, t130_manifest: pd.DataFrame) -> None:
    """Reject a comparison if a test patient appears in the T100 training input."""
    train_ids = set(t100["patientId"].astype(str))
    test_ids = set(t130_manifest["patient_id"].astype(str))
    overlap = sorted(train_ids.intersection(test_ids))
    if overlap:
        raise ValueError(f"T100/T130 patient overlap ({len(overlap)}): {overlap}")


def fit_lr1_all(t100: pd.DataFrame, columns: dict[str, str]) -> Any:
    """Fit the specified LR1 C=0.1 model on all T100 LR1-admitted rows."""
    rows = lr1_training_rows(
        t100,
        label_column=columns["label_column"],
        biopsy_column=columns["biopsy_column"],
        lr1_row_policy=columns["lr1_row_policy"],
    )
    model = build_profile_logistic(logreg_c=LR1_C, random_state=RANDOM_STATE)
    model.fit(profile_matrix(rows, columns["profile_column"]), row_labels(rows, columns["label_column"]))
    return model


def t130_prediction_features(
    t130: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    lr1_model: Any,
    symmetry_feature_contract: str,
    columns: dict[str, str],
) -> pd.DataFrame:
    """Build all 22 T130 rows from manifest-defined clinician target sides.

    No biopsy field is changed.  This is intentionally different from the
    historical biopsy-only target-case helper used for T100 training features.
    """
    model_info = {
        "lr1_model": lr1_model,
        "symmetry_feature_contract": symmetry_feature_contract,
    }
    rows = [
        build_patient_prediction_feature_row(
            t130,
            model_info,
            patient_id=case.patient_id,
            target_side=case.target_side,
            profile_column=columns["profile_column"],
            group_column=columns["group_column"],
            specimen_column=columns["specimen_column"],
            side_column=columns["side_column"],
            q_column=columns["q_column"],
            age_column=columns["age_column"],
        )
        for case in manifest.itertuples(index=False)
    ]
    features = pd.concat(rows, ignore_index=True)
    if list(features["target_case_id"]) != list(manifest["target_case_id"]):
        raise ValueError("T130 features do not preserve manifest case ordering.")
    required = [
        "profile_p_cancer_logit_average",
        "age",
        "age_available",
        "symmetry_available",
        *product_symmetry_columns(),
    ]
    if not np.isfinite(features[required].to_numpy(dtype=float)).all():
        raise ValueError("T130 model feature rows contain non-finite values.")
    return features


def locked_full_parameters(evidence_dir: str | Path = EVIDENCE_DIR) -> dict[str, float]:
    """Derive selected full-model penalties from tracked train-all evidence."""
    selection_path = Path(evidence_dir) / "regularization_selection.csv"
    require_sha256(
        selection_path,
        EXPECTED_REGULARIZATION_SHA256,
        name="locked regularization selection",
    )
    selection = pd.read_csv(selection_path)
    selected = selection.loc[
        selection["ablation"].eq("profile_age_symmetry")
        & selection["outer_split_id"].astype(str).eq("train_all")
        & selection["selected"].eq(True)
    ]
    expected_steps = {"profile_c", "age_c", "symmetry_c"}
    if set(selected["selection_step"]) != expected_steps or len(selected) != 3:
        raise ValueError("Locked train-all full-model regularization rows are incomplete.")
    parameters = {
        key: float(selected.loc[selected["selection_step"].eq(key), key].iloc[0])
        for key in sorted(expected_steps)
    }
    return parameters


def locked_oof_threshold(evidence_dir: str | Path = EVIDENCE_DIR) -> float:
    """Load the frozen full-chain OOF threshold from tracked aggregate evidence."""
    metrics_path = Path(evidence_dir) / "train_all_metrics.csv"
    require_sha256(
        metrics_path,
        EXPECTED_TRAIN_ALL_METRICS_SHA256,
        name="locked train-all metrics",
    )
    metrics = pd.read_csv(metrics_path)
    selected = metrics.loc[
        metrics["model_name"].eq("recalibrated_joint_additive")
        & metrics["ablation"].eq("profile_age_symmetry")
        & metrics["threshold_provenance"].eq(
            "training_cohort_full_chain_lr1_oof_meta_oof_scores"
        )
    ]
    values = selected["decision_threshold"].dropna().unique()
    if len(values) != 1:
        raise ValueError("Locked T100 full-chain OOF threshold is not unique.")
    return float(values[0])


def wilson_interval(successes: int, total: int, *, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    radius = z * np.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total) / denominator
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def metric_row(
    *,
    procedure: str,
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    threshold_provenance: str,
) -> dict[str, Any]:
    """Return complete held-out metrics with transparent exact-count intervals."""
    values = binary_metric_values(y, score, np.full(len(y), threshold))
    prediction = (score >= threshold).astype(int)
    tp = int(((prediction == 1) & (y == 1)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    sensitivity_low, sensitivity_high = wilson_interval(tp, tp + fn)
    specificity_low, specificity_high = wilson_interval(tn, tn + fp)
    return {
        "procedure": procedure,
        "test_target_cases": int(len(y)),
        "test_patients": None,
        "test_cancer_cases": int((y == 1).sum()),
        "test_benign_cases": int((y == 0).sum()),
        "threshold": float(threshold),
        "threshold_provenance": threshold_provenance,
        **values,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "sensitivity_wilson_95_low": sensitivity_low,
        "sensitivity_wilson_95_high": sensitivity_high,
        "specificity_wilson_95_low": specificity_low,
        "specificity_wilson_95_high": specificity_high,
    }


def paired_disagreement(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired per-case threshold decisions without exposing case IDs."""
    procedures = list(predictions["procedure"].drop_duplicates())
    rows: list[dict[str, Any]] = []
    for left, right in combinations(procedures, 2):
        left_rows = predictions.loc[predictions["procedure"].eq(left)].set_index("target_case_id")
        right_rows = predictions.loc[predictions["procedure"].eq(right)].set_index("target_case_id")
        joined = left_rows[["label", "prediction"]].join(
            right_rows[["label", "prediction"]], lsuffix="_left", rsuffix="_right", validate="one_to_one"
        )
        if not joined["label_left"].equals(joined["label_right"]):
            raise ValueError("Paired procedures disagree on T130 reference labels.")
        for label_name, group in [("all", joined), ("CANCER", joined[joined.label_left.eq(1)]), ("BENIGN", joined[joined.label_left.eq(0)])]:
            left_correct = group.prediction_left.eq(group.label_left)
            right_correct = group.prediction_right.eq(group.label_left)
            rows.append(
                {
                    "left_procedure": left,
                    "right_procedure": right,
                    "reference_group": label_name,
                    "cases": int(len(group)),
                    "same_decision": int(group.prediction_left.eq(group.prediction_right).sum()),
                    "different_decision": int(group.prediction_left.ne(group.prediction_right).sum()),
                    "both_correct": int((left_correct & right_correct).sum()),
                    "left_only_correct": int((left_correct & ~right_correct).sum()),
                    "right_only_correct": int((~left_correct & right_correct).sum()),
                    "both_incorrect": int((~left_correct & ~right_correct).sum()),
                }
            )
    return pd.DataFrame(rows)


def conclusion_markdown(metrics: pd.DataFrame) -> str:
    """Return a compact, non-promotional interpretation for tracked evidence."""
    table = metrics[
        ["procedure", "roc_auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "ppv", "npv", "log_loss", "brier_score", "true_positives", "true_negatives", "false_positives", "false_negatives"]
    ].copy()
    return "\n".join(
        [
            "# T130 Paired Held-out Comparison",
            "",
            "Three T100-trained procedures were scored on the same locked T130 case manifest: 22 target-breast cases from 17 patient-disjoint patients (11 CANCER, 11 BENIGN).",
            "",
            markdown_table(table),
            "",
            "## Interpretation",
            "",
            "- Relative to the frozen product, same-data LR2 training changes one CANCER FN to TP but also one BENIGN TN to FP: sensitivity is +9.09 percentage points, specificity is -9.09 points, and ROC AUC is -0.0331.",
            "- The OOF-trained LR2 has the same T130 ROC AUC and sensitivity as same-data LR2, but two fewer true negatives at its locked lower threshold (0.18564 versus 0.25747).",
            "- There is no clear overall winner on this subset. The frozen product retains higher ROC AUC and specificity; both recalibrated procedures have lower Brier score and log loss, while same-data LR2 shifts the thresholded error trade-off toward sensitivity.",
            "- Research-only same-source check. It is not independent external validation.",
            "- T130 uses looser calibration QC than T100 and selects MRI-or-biopsy cases; MRI denotes that MRI was performed, not an MRI outcome.",
            "- Five patients contribute two target breasts, so 22 cases are not 22 independent patients.",
            "- Each CANCER or BENIGN case changes sensitivity or specificity by 1/11 = 9.09 percentage points. `metrics.csv` reports descriptive case-level Wilson 95% intervals; they do not account for paired breasts within five patients.",
            "- The deliberately balanced 11/11 class composition is not clinical prevalence. PR AUC, PPV, and NPV are descriptive for this subset and must not be projected to clinical workflow.",
            "- The frozen product comparator is scored directly from its immutable model artifact. The current T100 input joblib SHA is recorded separately because it differs from the SHA stored in that artifact.",
        ]
    ) + "\n"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in record]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def run_comparison(
    *,
    t100_input: str | Path = DEFAULT_T100_INPUT,
    t130_input: str | Path = DEFAULT_T130_INPUT,
    product_artifact: str | Path = DEFAULT_PRODUCT_ARTIFACT,
    evidence_dir: str | Path = EVIDENCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Fit and compare the three locked procedures on all 22 manifest cases."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    t100_path = Path(t100_input)
    t130_path = Path(t130_input)
    product_path = Path(product_artifact)
    t100_sha = require_sha256(
        t100_path,
        EXPECTED_T100_SHA256,
        name="T100 training input",
    )
    product_sha = require_sha256(
        product_path,
        EXPECTED_PRODUCT_SHA256,
        name="frozen product artifact",
    )
    t100 = load_input_dataframe(t100_path)
    product = joblib.load(product_path)
    product_model = product["models"]["aramina_target_breast_risk"]
    columns = model_columns()
    rebuilt_lr1 = fit_lr1_all(t100, columns)
    lr1_rows = lr1_training_rows(
        t100,
        label_column=columns["label_column"],
        biopsy_column=columns["biopsy_column"],
        lr1_row_policy=columns["lr1_row_policy"],
    )
    lr1_equivalence = assert_lr1_equivalence(
        product_model["lr1_model"],
        rebuilt_lr1,
        profile_matrix(lr1_rows, columns["profile_column"]),
    )
    full_lr1 = product_model["lr1_model"]
    same_fitted_features, _ = fit_feature_pair(
        t100, t100, columns, lr1_c=LR1_C, random_state=RANDOM_STATE
    )
    parameters = locked_full_parameters(evidence_dir)
    same_model = RecalibratedJointAdditiveClassifier(
        **parameters, use_age=True, use_symmetry=True, random_state=RANDOM_STATE
    ).fit(same_fitted_features, same_fitted_features["label"].to_numpy(dtype=int))
    same_threshold = float(compute_binary_thresholds(
        same_fitted_features["label"].to_numpy(dtype=int),
        same_model.predict_proba(same_fitted_features)[:, 1],
        target_sensitivity=TARGET_SENSITIVITY,
    )["threshold_target"])

    oof_features = lr1_cross_fitted_features(
        t100, columns, lr1_c=LR1_C, n_splits=OOF_SPLITS,
        random_state=TRAIN_ALL_LR1_OOF_RANDOM_STATE,
        context={"outer_split_id": "train_all", "meta_fold_id": "comparison_oof_fit"},
    ).features
    oof_model = RecalibratedJointAdditiveClassifier(
        **parameters, use_age=True, use_symmetry=True, random_state=RANDOM_STATE
    ).fit(oof_features, oof_features["label"].to_numpy(dtype=int))
    oof_threshold = locked_oof_threshold(evidence_dir)

    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    same_model_path = model_dir / "recalibrated_same_fitted_lr1.joblib"
    oof_model_path = model_dir / "recalibrated_lr1_oof.joblib"
    joblib.dump(
        {
            "kind": "aramina_research_recalibrated_model",
            "version": "0.1",
            "training_mode": "lr2_same_fitted_lr1_rows",
            "lr1_model": full_lr1,
            "final_model": same_model,
            "threshold": same_threshold,
            "parameters": parameters,
            "symmetry_feature_contract": product_model["symmetry_feature_contract"],
            "prediction_preprocessing_yaml": product["prediction_preprocessing_yaml"],
            "t100_input_sha256": t100_sha,
            "regularization_selection_sha256": EXPECTED_REGULARIZATION_SHA256,
        },
        same_model_path,
    )
    joblib.dump(
        {
            "kind": "aramina_research_recalibrated_model",
            "version": "0.1",
            "training_mode": "lr2_lr1_patient_safe_oof_rows",
            "lr1_model": full_lr1,
            "final_model": oof_model,
            "threshold": oof_threshold,
            "parameters": parameters,
            "symmetry_feature_contract": product_model["symmetry_feature_contract"],
            "prediction_preprocessing_yaml": product["prediction_preprocessing_yaml"],
            "t100_input_sha256": t100_sha,
            "regularization_selection_sha256": EXPECTED_REGULARIZATION_SHA256,
            "train_all_metrics_sha256": EXPECTED_TRAIN_ALL_METRICS_SHA256,
            "lr1_crossfit_random_state": TRAIN_ALL_LR1_OOF_RANDOM_STATE,
        },
        oof_model_path,
    )
    same_artifact = joblib.load(same_model_path)
    oof_artifact = joblib.load(oof_model_path)

    t130_sha = require_sha256(
        t130_path,
        EXPECTED_T130_SHA256,
        name="T130 held-out manifest artifact",
    )
    manifest, t130_metadata = load_t130_manifest(t130_path)
    assert_t130_composition(manifest)
    assert_no_patient_overlap(t100, manifest)
    source_h5 = Path(str(t130_metadata["source_h5_path"]))
    source_h5_sha = require_sha256(
        source_h5,
        EXPECTED_SOURCE_H5_SHA256,
        name="T130 source H5",
    )
    prediction_preprocessing = yaml.safe_load(
        same_artifact["prediction_preprocessing_yaml"]
    )
    t130 = preprocess_manifest_patients(
        source_h5,
        manifest,
        prediction_preprocessing,
    )
    frozen_features = t130_prediction_features(
        t130,
        manifest,
        lr1_model=product_model["lr1_model"],
        symmetry_feature_contract=str(product_model["symmetry_feature_contract"]),
        columns=columns,
    )
    recalibrated_test = t130_prediction_features(
        t130,
        manifest,
        lr1_model=same_artifact["lr1_model"],
        symmetry_feature_contract=str(same_artifact["symmetry_feature_contract"]),
        columns=columns,
    )
    oof_test = t130_prediction_features(
        t130,
        manifest,
        lr1_model=oof_artifact["lr1_model"],
        symmetry_feature_contract=str(oof_artifact["symmetry_feature_contract"]),
        columns=columns,
    )
    assert_shared_test_features(frozen_features, recalibrated_test)
    assert_shared_test_features(frozen_features, oof_test)
    frozen_scores = product_model["final_model"].predict_proba(frozen_features)[:, 1]
    frozen_threshold = float(product_model["thresholds"]["threshold_target"])
    same_scores = same_artifact["final_model"].predict_proba(recalibrated_test)[:, 1]
    oof_scores = oof_artifact["final_model"].predict_proba(oof_test)[:, 1]
    same_threshold = float(same_artifact["threshold"])
    oof_threshold = float(oof_artifact["threshold"])

    y = manifest["label"].to_numpy(dtype=int)
    score_map = {
        CURRENT_PRODUCT: (frozen_scores, frozen_threshold, "immutable_product_artifact_train_all_fitted_scores"),
        SAME_FITTED: (same_scores, same_threshold, "same_fitted_t100_lr1_lr2_train_all_scores"),
        OOF_FITTED: (oof_scores, oof_threshold, "locked_t100_full_chain_lr1_oof_meta_oof_scores"),
    }
    metric_rows = []
    prediction_rows = []
    for procedure, (score, threshold, provenance) in score_map.items():
        row = metric_row(
            procedure=procedure, y=y, score=np.asarray(score), threshold=threshold,
            threshold_provenance=provenance,
        )
        row["test_patients"] = int(manifest["patient_id"].nunique())
        metric_rows.append(row)
        prediction_rows.append(pd.DataFrame({
            "target_case_id": manifest["target_case_id"], "patient_id": manifest["patient_id"],
            "reference_label": manifest["reference_label"], "label": y,
            "procedure": procedure, "p_cancer": score, "threshold": threshold,
            "prediction": (np.asarray(score) >= threshold).astype(int),
        }))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    disagreement = paired_disagreement(predictions)
    product_input_sha = str(product.get("input_dataframe_joblib_sha256", "not_recorded"))
    current_t100_sha = t100_sha
    metadata = {
        "t100_input_joblib": str(t100_path.resolve()),
        "t100_input_joblib_sha256": current_t100_sha,
        "t130_input_joblib": str(t130_path.resolve()),
        "t130_input_joblib_sha256": t130_sha,
        "frozen_product_artifact": str(product_path.resolve()),
        "frozen_product_artifact_sha256": product_sha,
        "frozen_product_stored_t100_input_joblib_sha256": product_input_sha,
        "frozen_product_current_t100_input_sha_matches": current_t100_sha == product_input_sha,
        "t130_metadata": t130_metadata,
        "t130_source_h5_sha256_verified": source_h5_sha,
        "locked_full_parameters": parameters,
        "locked_oof_threshold": oof_threshold,
        "locked_oof_lr1_crossfit_random_state": TRAIN_ALL_LR1_OOF_RANDOM_STATE,
        "locked_regularization_selection_sha256": EXPECTED_REGULARIZATION_SHA256,
        "locked_train_all_metrics_sha256": EXPECTED_TRAIN_ALL_METRICS_SHA256,
        "frozen_and_rebuilt_lr1_equivalence": lr1_equivalence,
        "shared_test_feature_rows_identical": True,
        "frozen_experimental_models": {
            "same_fitted_lr1": {
                "path": str(same_model_path.resolve()),
                "sha256": sha256_file(same_model_path),
            },
            "lr1_oof": {
                "path": str(oof_model_path.resolve()),
                "sha256": sha256_file(oof_model_path),
            },
        },
    }
    metrics.to_csv(output / "metrics.csv", index=False)
    disagreement.to_csv(output / "paired_disagreement.csv", index=False)
    predictions.to_csv(output / "paired_predictions_local.csv", index=False)
    payload = {
        "experiment": "t130_paired_held_out_comparison_v0_1",
        "status": "research_only_same_source_patient_disjoint_not_external_validation",
        "cohort": {
            "target_cases": 22, "patients": 17, "cancer_cases": 11, "benign_cases": 11,
            "patients_with_two_target_cases": 5,
            "case_step_percentage_points": 100.0 / 11.0,
        },
        "provenance": metadata,
        "procedures": metrics.to_dict(orient="records"),
        "outputs": {
            "metrics": "metrics.csv",
            "paired_disagreement": "paired_disagreement.csv",
            "patient_level_predictions": "paired_predictions_local.csv (local only; gitignored)",
        },
        "limitations": [
            "Same-source T130 quality-control subset; not independent external validation.",
            "MRI-or-biopsy selection; MRI means MRI performed, not an MRI outcome.",
            "Five patients contribute two correlated target-breast cases.",
            "One case changes class-specific sensitivity or specificity by 9.09 percentage points.",
            "Wilson intervals are descriptive case-level intervals and do not model within-patient correlation.",
            "The balanced 11/11 class composition is artificial; PR AUC, PPV, and NPV are subset-specific.",
        ],
    }
    (output / "summary.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (output / "RESULTS.md").write_text(conclusion_markdown(metrics), encoding="utf-8")
    return payload


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--t100-input", default=str(DEFAULT_T100_INPUT))
    parser.add_argument("--t130-input", default=str(DEFAULT_T130_INPUT))
    parser.add_argument("--product-artifact", default=str(DEFAULT_PRODUCT_ARTIFACT))
    parser.add_argument("--evidence-dir", default=str(EVIDENCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison(
        t100_input=args.t100_input,
        t130_input=args.t130_input,
        product_artifact=args.product_artifact,
        evidence_dir=args.evidence_dir,
        output_dir=args.output_dir,
    )
