# Aramina Modeling Documentation

Status: research draft.

This folder contains the current model record, cohort definition, feature
definitions, and prediction-report contract. Product readers should start here:

```text
aramina_t100_target_case_model_v0_1.md
prediction_pipeline_v0_1.md
current_model_dataframe_v0_1.md
tra_decision_record_v0_2.md
```

## Development Candidate

```text
model_name: aramina_target_breast_risk
preprocessing: T100 biopsy-patient model input
architecture: one LR2 with gated optional SK Core4 refinement
regularization: LR1 L2 C=0.1; LR2 L2 C=0.3
measurement counts: reliability only, not model inputs
```

Current development decision record:

```text
aramina_target_breast_risk model record
```

The packaged research-draft artifact and its `model_description.yaml` are in
`models/<model_id>/`. Each training run also writes `evaluation.yaml`
for the complete evaluation footprint. The exact
artifact ID is derived from the model joblib SHA256 and recorded in its generated
description and reports.

## Current Evidence

```text
aramina_t100_target_case_model_v0_1.md
  fixed architecture, validation, regularization, threshold, and decision record

current_model_dataframe_v0_1.md
  current cohort and target-case definition

sk_symmetry_features_v0_1.md
  mathematical definitions of SK Core4 fields

tra_decision_record_v0_2.md
  threshold-centred TRA levels, automatic OOF calibration, and artifact content
```

## Product Contracts

```text
../contracts/training_config_v0_1.md
  training YAML, fixed evaluation, and train-on-all artifact structure

prediction_pipeline_v0_1.md
  one-patient H5 prediction route and report schema

../product_development_rules.md
  product controls, limitations, and change requirements
```

Historical candidate models, threshold comparisons, and generated experiment
tables are retained only in `experiment/aramina-model-selection-v0.1`.
