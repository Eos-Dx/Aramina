# Current Aramis Model Pipeline And Risks v0.1

Status: research draft. This document describes model-development structure and
known weaknesses. It is not clinical validation.

Aramis currently estimates `p_cancer` for breast XRD decision support. The
output is intended for radiologist / qualified clinician review and must not be
used as autonomous diagnosis.

## Current Code Contract

The current command stack is:

```text
python -m aramis preprocess --config <preprocessing.yaml>
python -m aramis train --config <training.yaml>
python -m aramis run --config <workflow.yaml>
```

`preprocess` builds a model-input DataFrame joblib from H5.

`train` consumes one preprocessing joblib and writes one model joblib plus JSON
and YAML summaries.

`run` references one preprocessing YAML and one training YAML, validates that
the preprocessing output path equals the training input path, then executes the
combined workflow.

Default workflow mode is `memory`:

```text
preprocess builds df
preprocess saves preprocessing joblib footprint
the same df is passed directly to train
train writes model joblib
```

The artifact is still written. The speed benefit is that the combined workflow
does not need to reload the DataFrame joblib before training. `mode: artifact`
keeps the stricter reload behavior.

## sklearn Pipeline Status

Preprocessing is a YAML-declared sklearn transformer route owned mostly by
`xrd-preprocessing`.

Training is now wrapped as one sklearn `Pipeline` object:

```text
build_patient_training_pipeline(...)
-> sklearn.pipeline.Pipeline
   -> AramisPatientTrainingPipeline.fit(...)
```

The fitted `AramisPatientTrainingPipeline` stores:

```text
input_builder_
feature_table_
model_trainer_
evaluator_
artifact_
```

This is currently a fit-time training pipeline. It is not yet the final
prediction pipeline. Prediction will need a separate fixed preprocessing +
fixed model inference route.

## Training Classes

### PatientModelInputBuilder

Purpose: convert measurement-level preprocessed rows into patient-level model
features.

Main steps:

```text
select LR1 rows according to lr1_row_policy
fit profile LogisticRegression on radial_profile_data
score measurement-level p_cancer
logit-average measurement probabilities to patient-level p_cancer
build patient-level label
infer training target side from biopsy/status metadata
compute target/contralateral SK symmetry features
keep target/contralateral cosine symmetry fields for audit
copy age and age_available
record patient/specimen/measurement counters
```

### PatientModelSetTrainer

Purpose: fit selected final model variants.

Current variants:

```text
M0:
  profile only
  uses patient-level logit-averaged LR1 p_cancer

M1:
  profile + symmetry
  uses M0 score plus SK target/contralateral symmetry block

M2:
  profile + symmetry + age
  uses M1 features plus age and age_available
```

## Symmetry Relative To Target Breast

Symmetry is partly a patient-level property, not a directional diagnosis. If
the two breasts are very different, that is suspicious context for the patient,
but the symmetry score alone cannot prove which breast created the difference.

Aramis therefore keeps two linked signals:

```text
target-breast profile signal:
  training: LR1 scores only the inferred target breast
  prediction: LR1 must score only clinician-supplied target breast

paired-breast symmetry context:
  compares target breast with contralateral breast for the same patient
```

Current cosine features:

```text
target_within_cosine_distance_mean:
  variability among valid target-breast measurements

contralateral_within_cosine_distance_mean:
  variability among valid contralateral-breast measurements

between_breasts_cosine_distance_mean:
  distance between target-breast mean profile and contralateral-breast mean profile

symmetry_cosine_score:
  between_breasts_cosine_distance_mean
  - mean(target_within_cosine_distance_mean,
         contralateral_within_cosine_distance_mean)
```

Interpretation:

```text
high between-breast distance:
  target and contralateral breasts differ strongly

high target-within distance:
  suspicious breast has internally variable measurements

high contralateral-within distance:
  contralateral breast is also unstable, so symmetry evidence is less clean

high symmetry_cosine_score:
  between-breast difference is larger than within-breast replicate variability
```

In training, the target side is inferred. For the primary `biopsy_patients`
cohort this means:

```text
inferred target breast = biopsied breast
```

For prediction, target side must not be inferred from labels. It must come from
clinician / predict YAML input.

Thus, "relative to suspicious breast" is represented by inferred-target LR1
probability plus inferred-target within-breast variability during training. The
global symmetry score then adds same-patient context. If both breasts are
abnormal or both are very similar, symmetry can be weak even when profile
evidence remains high.

### PatientModelSetEvaluator

Purpose: evaluate selected models with patient-safe splitting.

Supported modes:

```text
all_on_all:
  train and score same patient table
  optimistic discovery view only

loovm:
  leave one patient out
  pooled left-out predictions

stratified_kfold:
  patient-level StratifiedKFold

repeated_stratified_shuffle:
  repeated patient-level 70/30 split
```

All split-based modes operate at `patientId` level. Measurements from one
patient cannot appear in both train and test inside a split.

## Dataset Routes

Current primary training uses one dataset:

```text
biopsy_patients
```

Reason:

```text
the biopsied breast is the clinically suspicious breast
the biopsied breast has the BENIGN/CANCER endpoint
therefore it is the only current clean target for product training
```

`all_patients` is kept only as an exploratory label-policy and metadata
sensitivity check. It must not be interpreted as the primary v0.1-beta product
training cohort.

Current model-grid YAMLs still contain both primary and exploratory routes:

```text
datasets:
  biopsy_patients  # primary
  all_patients     # exploratory only

models:
  M0
  M1
  M2

validation modes:
  all_on_all
  loovm
  stratified_kfold
```

Historical grid YAMLs were used to compare model families and validation modes.
They are archived on `experiment/aramis-v0.1-research-state`.

Development now keeps product-clean training configs only:

```text
config/training/aramis_v0_1_beta_primary_train.yaml
config/training/aramis_biopsy_patients_m0_m1_m2_v0_1.yaml
config/training/aramis_all_patients_m0_m1_m2_v0_1.yaml
```

The primary model-input cohort is documented in:

```text
docs/modeling/current_model_dataframe_v0_1.md
```

Primary configured training DataFrame:

```text
examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
patients: 180
specimens / breasts: 342
measurement rows: 968
```

Primary train command:

```bash
python -m aramis train --config config/training/aramis_v0_1_beta_primary_train.yaml
```

This primary config trains M0/M1 only. M2 remains an age audit/comparison route
in `config/training/aramis_biopsy_patients_m0_m1_m2_v0_1.yaml`.

## Current Result Pattern

The numerical examples below are the 2026-07-04 sanity run after two controlled
changes:

```text
LR1 aggregation: target-breast logit-average
symmetry features: target/contralateral cosine fields
```

The latest model grid shows the same broad pattern as the exploratory notebooks:

```text
all_on_all:
  high ROC AUC
  useful only as optimistic discovery ceiling

loovm and stratified_kfold:
  lower ROC AUC
  closer to patient-safe generalization behavior

M1:
  current primary candidate
  profile evidence plus SK symmetry block

M2:
  usually strongest ranking
  improvement is partly driven by age
  age branch is audit/comparison, not primary v0.1-beta model

M0:
  profile-only baseline remains weak
```

Current repeated patient-safe 70/30 x50 sanity run.

Primary product-training cohort:

```text
biopsy_patients M0:
  ROC AUC 0.574
  specificity_target_mean 0.359

biopsy_patients M1:
  ROC AUC 0.582
  specificity_target_mean 0.455

biopsy_patients M2:
  ROC AUC 0.623
  specificity_target_mean 0.470
```

Exploratory comparison only:

```text
all_patients M0:
  ROC AUC 0.548
  specificity_target_mean 0.544

all_patients M1:
  ROC AUC 0.532
  specificity_target_mean 0.541

all_patients M2:
  ROC AUC 0.574
  specificity_target_mean 0.559
```

Interpretation: after target-only LR1 aggregation, SK-based M1 improves
specificity over M0 in this sanity run. Age still improves ROC AUC in M2, but
this remains an audit branch because age can act as a clinical shortcut.

## Known Weaknesses

### 1. Age is a strong shortcut

Age-only has shown strong predictive power in prior experiments. This is
clinically plausible, but dangerous for product modeling because it can dominate
XRD information.

Risk:

```text
model may learn clinical population age distribution instead of XRD signal
```

Required control:

```text
always compare M2 against age-only
report whether age is accepted as explicit clinical covariate
do not hide age contribution inside a generic model score
```

### 2. Profile-only XRD signal is weak

M0 is the cleanest XRD profile baseline, but current patient-safe validation is
modest.

Risk:

```text
spectral profile may not yet contain enough stable signal after current
preprocessing/filtering/normalization
```

Required control:

```text
continue tracking M0 separately
do not claim M1/M2 improvement proves spectral biology unless M0 improves
```

### 3. Symmetry features are not yet final

M1 uses cosine features:

```text
profile_p_cancer_logit_average
between_breasts_cosine_distance_mean
target_within_cosine_distance_mean
contralateral_within_cosine_distance_mean
symmetry_cosine_score
symmetry_available
```

Risk:

```text
symmetry gain can mix true biological asymmetry with measurement completeness,
side availability, and preprocessing survival
```

Required control:

```text
compare with symmetry_available-only model
evaluate only patients with complete paired breast context
decide whether availability flags are allowed as model inputs
```

### 4. all_on_all is not validation

`all_on_all` gives high numbers because the same patients are used for training
and scoring.

Risk:

```text
over-optimistic performance
```

Use:

```text
discovery ceiling
debugging
sanity check
not expected prospective performance
```

### 5. LOOVM is strict but high variance

LOOVM leaves one patient out and pools predictions. It is patient-safe, but each
fold trains on almost the whole dataset and tests on one patient.

Risk:

```text
unstable threshold behavior
pooled metrics can hide fold-level instability
```

Required control:

```text
report LOOVM together with stratified_kfold
do not rely on LOOVM alone
```

### 6. KFold estimates are still small-data estimates

StratifiedKFold is patient-safe, but patient count remains small.

Risk:

```text
large uncertainty
specificity at high sensitivity can move substantially by split
```

Required control:

```text
report mean and spread
report TP/FN/TN/FP counts
keep split manifest when formal MLflow route is finalized
```

### 7. Threshold selection is not final

Current threshold policy targets high sensitivity. For split validation,
thresholds are selected on train scores and applied to test scores.

Risk:

```text
specificity depends heavily on threshold policy
```

Required control:

```text
separate ROC AUC ranking from operating-point specificity
always state whether threshold is train-selected or oracle/test-selected
```

### 8. Biopsy cohort is cleaner but smaller

Biopsy-only labels are more reliable, but the dataset is smaller.

Risk:

```text
better labels but wider uncertainty
possible biopsy-workflow selection bias
```

Required control:

```text
compare all_patients vs biopsy_patients
prefer biopsy_patients for endpoint reliability
document that this is a selected clinical subgroup
```

### 9. Preprocessing remains part of the model

The model depends on:

```text
AgBH monochromaticity exclusions
GFRM decoding
sample thickness availability
calibrant thickness
azimuthal integration settings
SNR filtering
normalization window
radial-profile signal gate
label mapping
```

Risk:

```text
changing preprocessing changes the model dataset and model behavior
```

Required control:

```text
store preprocessing YAML in preprocessing joblib
store training YAML in model joblib
store SHA256 for input joblib and configs
use workflow YAML when running preprocess+train together
```

## Current Readiness

Ready for discussion:

```text
preprocessing YAML contract
training YAML contract
workflow YAML contract
sklearn Pipeline wrapper for training
M0/M1/M2 model grid
patient-safe validation modes
traceable joblib artifacts
known weakness list
feature schema and warnings in model joblib
```

Not ready yet:

```text
final product model selection
final target threshold
prediction route
clinical validation claim
FDA/regulatory claim
autonomous diagnosis
```

## Next Review Questions

Questions to resolve before freezing training route:

```text
Should age be included in the product model or only reported as audit/control?
Should symmetry_available be an input feature or only an audit flag?
Main dataset decision: use `biopsy_patients` for primary product training.
Keep `all_patients` only for exploratory sensitivity checks.
Should SK M1 be accepted as the v0.1-beta symmetry route after review?
Should LOOVM or stratified_kfold be the main model-selection view?
What exact specificity target is acceptable at sensitivity near 95%?
```
