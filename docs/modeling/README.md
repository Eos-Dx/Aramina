# Model Documentation

Status: frozen research-draft product model.

Read in this order:

1. [Current frozen product model](aramina_t100_target_case_model_v0_1.md): architecture,
   cohort, fixed regularization, evaluation, threshold, and limitations.
2. [Retrained candidate](aramina_t100_target_case_model_v0_2.md): same model
   architecture retrained with XRD-preprocessing `v0.1.9-beta`.
3. [Current DataFrame](current_model_dataframe_v0_1.md): measurement, breast,
   target-case, and label rules.
4. [Prediction pipeline](prediction_pipeline_v0_1.md): one-patient scoring and
   report generation.
5. [Symmetry features](sk_symmetry_features_v0_1.md): Core4 definitions.
6. [TRA decision](tra_decision_record_v0_2.md): threshold-centred internal
   score levels.

The executable model and generated training records live under
[`models/<model_id>/`](../../models/README.md). `0.2.12-beta` remains the
frozen product artifact. The separately retrained `0.2.13-beta` candidate is
retained for comparison; experimental candidates and model-selection tables
remain on experiment branches.
