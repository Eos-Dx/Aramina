# Aramis Modeling Documentation

Status: research draft.

This folder contains model rationale, selected-candidate evidence, and
prediction/training contracts. Product readers should start here:

```text
../machine_learning_concept.md
final_candidate_model_artifact_v0_1.md
prediction_pipeline_v0_1.md
training_pipeline_classes_v0_1.md
current_model_pipeline_and_risks_v0_1.md
```

## Primary Candidate

```text
model_id: aramis_m1q_t100_train_all_c0p1
selected_model: M1Q
preprocessing: T100 biopsy-patient model input
regularization: L2 LogisticRegression, C=0.1
threshold_target: 0.327873
```

Decision record:

```text
final_candidate_model_artifact_v0_1.md
```

## Evidence Documents

```text
m1q_regularization_experiment_v0_1.md
  why C=0.1 was selected

m1q_threshold_mode_comparison_v0_1.md
  how threshold / validation modes behave

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
