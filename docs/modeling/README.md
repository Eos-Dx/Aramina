# Model Documentation

Status: research-draft product model documentation.

Read in this order:

1. [Paired model comparison](paired_model_comparison_v0_1.md): runnable matched-
   cohort raw100, FPCA30, and full additive evaluation.
2. [Current FPCA30 model](aramina_fpca30_target_case_model_v0_3.md): source
   architecture, cohort, fixed regularization, evaluation, and limitations.
3. [FPCA30 basis decision](fpca30_component_selection_v0_1.md): why the model
   moved from point-wise predictors to a provisional basis and why 30 rather
   than 10 coefficients were retained for this basis.
4. [Preserved 0.2.12 baseline](aramina_t100_target_case_model_v0_1.md).
5. [Preserved 0.2.13 retraining](aramina_t100_target_case_model_v0_2.md).
6. [Current DataFrame](current_model_dataframe_v0_1.md): measurement, breast,
   target-case, and label rules.
7. [Prediction pipeline](prediction_pipeline_v0_1.md): one-patient scoring and
   report generation.
8. [Symmetry features](sk_symmetry_features_v0_1.md): Core4 definitions.
9. [TRA decision](tra_decision_record_v0_2.md): threshold-centred internal
   score levels.

Preserved executable models live under [`models/<model_id>/`](../../models/README.md).
Current source trains `0.3.1-beta`, but its generated joblib is intentionally
not tracked. Experimental candidates and model-selection tables remain on
experiment branches.
