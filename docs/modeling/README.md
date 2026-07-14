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
`aramis_m2q_t100_0_2_3_beta.joblib`. Synthetic H5
smoke tests use a separate raw-array artifact.

## Current Evidence

```text
m2q_gated_target_case_model_v0_1.md
  fixed architecture, validation, regularization, threshold, and decision record

current_model_dataframe_v0_1.md
  current cohort and target-case definition

sk_symmetry_features_v0_1.md
  mathematical definitions of SK Core4 fields
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

Historical candidate models, threshold comparisons, and generated experiment
tables are retained only in `experiment/aramis-model-selection-v0.1`.
