# Aramis Modeling Documentation

Status: research draft.

This folder contains the current model record, cohort definition, feature
definitions, and prediction-report contract. Product readers should start here:

```text
m2q_gated_target_case_model_v0_1.md
prediction_pipeline_v0_1.md
current_model_dataframe_v0_1.md
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

The packaged research-draft artifact and its `model_description.yaml` are in
`examples/prediction_models/`. The exact artifact ID is derived from the model
joblib SHA256 and recorded in its generated description and reports.

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
../contracts/training_config_v0_1.md
  training YAML, fixed evaluation, final-fit artifact structure

prediction_pipeline_v0_1.md
  one-patient H5 prediction route and report schema

../product_development_rules.md
  product controls, limitations, and change requirements
```

Historical candidate models, threshold comparisons, and generated experiment
tables are retained only in `experiment/aramis-model-selection-v0.1`.
