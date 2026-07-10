# Aramis Modeling Documentation

Status: research draft.

This folder contains model rationale, selected-candidate evidence, and
prediction/training contracts. Product readers should start here:

```text
../machine_learning_concept.md
final_candidate_model_artifact_v0_1.md
m2q_core4_paired_candidate_v0_1_8.md
prediction_pipeline_v0_1.md
training_pipeline_classes_v0_1.md
current_model_pipeline_and_risks_v0_1.md
```

## Primary Candidate

```text
model_id: aramis_m2q_t100_core4_c1_0p1_c2_0p1
selected_model: M2Q
preprocessing: T100 biopsy-patient model input, strict paired-breast cohort
regularization: LR1 L2 C=0.1; LR2 L2 C=0.1
threshold_target: 0.297674
```

Decision record:

```text
m2q_core4_paired_candidate_v0_1_8.md
```

## Evidence Documents

```text
m2q_core4_paired_candidate_v0_1_8.md
  current fixed model schema, Core4, paired eligibility, and regularization

m1q_threshold_mode_comparison_v0_1.md
  how threshold / validation modes behave

m2q_age_candidate_update_v0_1.md
  why age was added and why M2Q is now the primary candidate

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
