"""Fixed contract values for paired model evaluation."""

from __future__ import annotations

CONTRACT = "aramina_paired_model_comparison_v0_1"
RAW100_MODEL = "raw100_product"
FPCA30_MODEL = "fpca30_product"
ADDITIVE_MODEL = "additive_recalibration_full"
MODEL_NAMES = (RAW100_MODEL, FPCA30_MODEL, ADDITIVE_MODEL)
PAIRED_COMPARISONS = (
    ("encoder_effect", FPCA30_MODEL, RAW100_MODEL),
    ("architecture_effect", ADDITIVE_MODEL, FPCA30_MODEL),
    ("total_effect", ADDITIVE_MODEL, RAW100_MODEL),
)
MEASUREMENT_ID_COLUMNS = (
    "patientId",
    "specimenId",
    "side",
    "position",
    "started_at",
)
MATCHED_METADATA_COLUMNS = ("product_status_group", "biopsy", "age")
SUMMARY_METRICS = (
    "roc_auc",
    "pr_auc",
    "brier_score",
    "log_loss",
    "sensitivity_target",
    "specificity_target",
    "balanced_accuracy_target",
    "ppv_target",
    "npv_target",
)
POOLED_METRIC_NAMES = {
    "roc_auc": "roc_auc",
    "pr_auc": "pr_auc",
    "brier_score": "brier_score",
    "log_loss": "log_loss",
    "sensitivity_target": "sensitivity",
    "specificity_target": "specificity",
    "balanced_accuracy_target": "balanced_accuracy",
    "ppv_target": "ppv",
    "npv_target": "npv",
}
PROFILE_SCORE_COLUMNS = (
    "profile_p_cancer_probability_mean",
    "profile_p_cancer_logit_average",
    "profile_p_cancer_n_measurements",
)
ADDITIVE_REGULARIZATION = {
    "profile_c": 0.001,
    "age_c": 0.3,
    "symmetry_c": 0.001,
}
ADDITIVE_SOURCE_COMMIT = "543a8319108aebee420b39fbcda888234b8045a6"
ADDITIVE_SOURCE_RECORD = (
    "experiment2:experiments/profile_symmetry_age_refinement/evidence/"
    "t100_5x10_20260731/regularization_selection.csv"
)
