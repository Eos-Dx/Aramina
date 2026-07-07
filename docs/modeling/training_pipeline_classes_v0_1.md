# Aramis Training Pipeline Classes v0.1

This is research-draft decision-support training code. It estimates `p_cancer`
and a suggested BENIGN/CANCER class for radiologist review; it is not autonomous
diagnosis.

The training CLI remains:

```bash
python -m aramis train --config <training.yaml>
```

The YAML selects the input preprocessing joblib, output artifacts, model family,
model subset, and validation mode. Internally, the patient-level M0/M1/M2 route
is organized as sklearn-like classes.

The patient-level route is also wrapped as one sklearn `Pipeline` object:

```text
build_patient_training_pipeline(...)
-> Pipeline([("patient_training", AramisPatientTrainingPipeline(...))])
```

`run_training_from_config()` loads the preprocessing joblib, builds this
pipeline, fits it, then writes the model joblib, JSON summary, and YAML
description.

For the complete product-development route, one workflow YAML can reference both
sub-YAML files:

```bash
python -m aramis run --config config/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml
```

That command executes:

```text
preprocessing YAML -> preprocessing joblib -> training YAML -> model joblib
```

By default, the workflow uses `mode: memory`: preprocessing still writes the
joblib footprint, but the freshly created DataFrame is passed directly to the
training pipeline without reloading the joblib. `mode: artifact` is available
when a run should force training to reload the saved preprocessing artifact.

## AramisPatientTrainingPipeline

Input: measurement-level preprocessing DataFrame.

Output: fitted estimator with `artifact_`.

Responsibilities:

```text
call PatientModelInputBuilder
call PatientModelSetTrainer
call PatientModelSetEvaluator
assemble final traceable training artifact
```

This is the single training pipeline unit used by `python -m aramis train`.

## PatientModelInputBuilder

Input: measurement-level preprocessing DataFrame.

Output: patient-level feature table.

Responsibilities:

```text
select LR1 rows from product labels
fit profile LogisticRegression on radial_profile_data
score measurement-level p_cancer
logit-average measurement scores to patient-level p_cancer
build patient label
infer training target side from biopsy/status metadata
build target/contralateral SK symmetry features
keep target/contralateral cosine symmetry fields for audit
copy age and age_available
record patient/specimen/measurement counters
```

For the primary `biopsy_patients` training cohort:

```text
inferred_target_side = biopsied breast
```

For future prediction, `target_side` must be supplied by the clinician-facing
input config. Prediction must not infer target side from labels.

LR1 aggregation keeps the LogisticRegression evidence scale:

```text
measurement p_cancer -> logit(p_cancer) -> mean logit -> sigmoid(mean logit)
```

The model feature is `profile_p_cancer_logit_average`. The plain probability
mean is retained only as `profile_p_cancer_probability_mean` for audit.

This class owns the first model layer. It preserves the product policy
`lr1_row_policy`, for example `all_rows` or `biopsy_only`.

## PatientModelSetTrainer

Input:

```text
patient feature table
LR1 training rows from PatientModelInputBuilder
```

Output:

```text
models_ dictionary
```

Supported model entries:

```text
M0: LR1 patient p_cancer only
M0Q: M0 + reliability/quality counters
M1: M0 + target/contralateral SK symmetry block
M1Q: M1 + reliability/quality counters
M2: M1 + age + age_available
M2Q: M1Q + age + age_available
```

For v0.1-beta, M1Q is the current primary candidate. M2/M2Q are explicit
age-audit/comparison branches because age can dominate the XRD signal.

Q models keep `p_cancer` as risk and add measurement-count confidence fields as
model features/report fields:

```text
profile_p_cancer_n_measurements
target_measurements
contralateral_measurements
min_measurements_per_breast
target_measurements_ok
contralateral_measurements_ok
paired_measurements_ok
result_reliability
result_reliability_reason
```

Reliability must be reported separately from risk. Example: `p_cancer=0.62`,
`risk_level=high`, `reliability=low`, reason: only one valid target-breast
measurement.

Each entry stores the fitted sklearn model components, selected feature columns,
and thresholds computed at target sensitivity on training scores.

## PatientModelSetEvaluator

Input: measurement-level preprocessing DataFrame.

Output:

```text
split_metrics_
split_predictions_
```

Supported validation modes:

```text
all_on_all: optimistic sanity check; train and score same patient table
loovm: leave-one-patient-out; reports pooled left-out metrics
stratified_kfold: patient-level StratifiedKFold
repeated_stratified_shuffle: repeated patient-level 70/30 split
```

All split-based modes split by `patientId`. Measurements from one patient cannot
appear in both train and test.

## Artifact

The final joblib stores:

```text
models
model_descriptions
training_config
training_config_yaml
training_config_sha256
preprocessing_config_sha256
input_dataframe_joblib_sha256
dataset_summary
feature_table
metric_summary
split_metrics
split_predictions
feature_schema
warnings
metadata
```

This makes the trained artifact traceable to both preprocessing YAML and
training YAML.

`metric_summary` includes ROC AUC, PR AUC, sensitivity, specificity, balanced
accuracy, PPV, NPV, and mean confusion-matrix counts at the configured target
sensitivity threshold. `warnings` records research-draft and validation-mode
limitations that must stay attached to the joblib.
