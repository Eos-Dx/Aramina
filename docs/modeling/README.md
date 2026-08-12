# Model Documentation

Status: research-draft product model documentation.

Read in this order:

1. [Current FPCA30 model](aramina_fpca30_target_case_model_v0_3.md): source
   architecture, cohort, fixed regularization, evaluation, and limitations.
2. [Preserved 0.2.12 baseline](aramina_t100_target_case_model_v0_1.md).
3. [Preserved 0.2.13 retraining](aramina_t100_target_case_model_v0_2.md).
4. [Current DataFrame](current_model_dataframe_v0_1.md): measurement, breast,
   target-case, and label rules.
5. [Prediction pipeline](prediction_pipeline_v0_1.md): one-patient scoring and
   report generation.
6. [Symmetry features](sk_symmetry_features_v0_1.md): Core4 definitions.
7. [TRA decision](tra_decision_record_v0_2.md): threshold-centred internal
   score levels.

Preserved executable models live under [`models/<model_id>/`](../../models/README.md).
Current source trains `0.3.1-beta`, but its generated joblib is intentionally
not tracked. Experimental candidates and model-selection tables remain on
experiment branches.
