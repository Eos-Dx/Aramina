# Aramis Modeling Documentation

Status: research draft.

This folder contains model rationale, selected-candidate evidence, and
prediction/training contracts. Product readers should start here:

```text
../machine_learning_concept.md
m2q_gated_target_case_model_v0_1.md
prediction_pipeline_v0_1.md
training_pipeline_classes_v0_1.md
current_model_pipeline_and_risks_v0_1.md
```

## Development Candidate

```text
architecture_id: m2q_gated_target_case_v0_1
selected_model: M2Q
preprocessing: T100 biopsy-patient model input
architecture: one LR2 with gated optional SK Core4 refinement
regularization: LR1 L2 C=0.1; LR2 L2 C=0.3
measurement counts: reliability only, not model inputs
```

Current development decision record:

```text
m2q_gated_target_case_model_v0_1.md
```

The packaged research-draft artifact is
`aramis_m2q_t100_gated_sk_core4_nested_c1_0p1_c2_0p3.joblib`. Synthetic H5
smoke tests use a separate raw-array artifact.

## Evidence Documents

```text
m2q_gated_target_case_model_v0_1.md
  fixed architecture, current nested validation, and train-all diagnostic

age_conditional_incremental_value_v0_1.md
  historical paired-routed age analysis

honest_operational_model_experiment_v0_1.md
  historical paired/fallback experiment

m2q_core4_optional_symmetry_candidate_v0_1_9.md
  historical v0.1.9 smoke-test artifact

m1q_threshold_mode_comparison_v0_1.md
  how threshold / validation modes behave

m2q_age_candidate_update_v0_1.md
  historical rationale for adding age

current_model_dataframe_v0_1.md
  what data the current candidate was trained from

sk_symmetry_features_v0_1.md
  mathematical definitions of SK symmetry features

internal_clinical_report_content_v0_1.md
  internal clinical report field draft and prediction-output requirements
```

## Product Contracts

```text
training_pipeline_classes_v0_1.md
  sklearn-like training classes and artifact structure

prediction_pipeline_v0_1.md
  one-patient H5 prediction route and report schema

current_model_pipeline_and_risks_v0_1.md
  known limitations and required interpretation guards
```

## Archived / Non-Primary Evidence

Older exploratory notebooks and generated tables are not product defaults. They
remain useful as background evidence when comparing alternative model families,
thresholds, or cohorts.

Do not use generated `examples/outputs/**/*.md` tables as API documentation.
Use them only as experiment evidence.
