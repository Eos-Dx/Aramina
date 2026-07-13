# Current Aramis Model Pipeline And Risks v0.1

Status: research draft. Development behavior is not yet tagged.

This document records how the current candidate works and what must not be
overclaimed.

## Candidate Summary

The tagged v0.1.9 artifact remains a historical smoke-test fixture. Current
development code uses one gated M2Q architecture documented in
`m2q_gated_target_case_model_v0_1.md`.

The model estimates `p_cancer` for decision support. It is not autonomous
diagnosis.

## Data Flow

```text
H5
-> YAML-declared XRD preprocessing
-> measurement-level normalized radial_profile_data
-> patient-safe training/evaluation split
-> LR1 profile model
-> target-breast case p_cancer by logit-average
-> one LR2 with profile + age and gated SK Core4 refinement
-> p_cancer
-> separate reliability from measurement counts
```

Prediction uses the same model feature logic, but `target_side` comes from
predict YAML rather than from labels or biopsy metadata.

## Why Biopsy Patients

The primary training dataset is `biopsy_patients` because:

```text
biopsied breast is the historical suspicious breast
endpoint is the cleanest available BENIGN/CANCER label
contralateral rows are still kept for symmetry
```

Each biopsied breast becomes one training case. A patient with bilateral
biopsies contributes two target-breast cases, but patient-safe splitters keep
both cases in one fold.

`all_patients` is exploratory only.

## Why M2Q

```text
M0:
  target-breast profile score only

M1:
  target-breast profile score + SK target/contralateral symmetry

M1Q:
  same prediction as M1; reliability reported separately

M2/M2Q:
  M2 adds age; M2Q reports reliability separately
```

M2Q is the development candidate because it combines the target-breast profile
score, an optional fixed four-field target/contralateral symmetry block, and age
as an explicit clinical risk prior. Age is clinically meaningful because breast
cancer risk increases with age, but age-only performance is reported separately
to expose shortcut risk.

## Feature Interpretation

Target-breast profile signal:

```text
LR1 scores only target-breast radial_profile_data
measurement probabilities are combined by logit-average
```

Symmetry context:

```text
compares target breast against contralateral breast
uses the fixed SK Core4 feature set as an optional refinement
sets all SK terms to zero when contralateral data is unavailable
symmetry_available controls the gate and is not a learned feature
```

Reliability:

```text
measurement counts and paired-breast sufficiency are report/audit fields
no measurement count is an LR2 input
symmetry_available is a gate and report field, not a learned LR2 input
```

Low reliability does not lower risk. It tells the downstream report to flag the
result as less stable.

## Validation Views

```text
outer repeated stratified patient K-fold:
  main generalization estimate

inner patient K-fold:
  selects LR1/LR2 regularization
  derives one train-only threshold for the final model

repeated patient 80/20 or 70/30 splits:
  robustness check

LOOVM:
  patient-safe but high-variance leave-one-out view

train-all:
  refits the selected architecture after nested validation
  uses inner out-of-fold thresholds
  fitted-cohort metrics remain diagnostics, not validation
```

All split modes are patient-safe.

## Current Development Evidence

See `m2q_gated_target_case_model_v0_1.md`. Its primary evidence is outer
patient-safe nested validation, not train-all fitted-cohort performance.

## Main Risks

### Small Dataset

Specificity at high sensitivity is unstable. Report mean and spread for
patient-safe split modes.

### Age Shortcut

Age is fixed in M2Q because it is a clinically meaningful risk prior, but it
can dominate this small cohort. A0 age-only stays mandatory control evidence.

### Weak Profile-Only Baseline

M0 remains modest in patient-safe validation. Do not claim that symmetry gain
proves spectral biology without continuing to track M0.

### Symmetry Availability

Symmetry can reflect real asymmetry, measurement completeness, or missing
contralateral context. Keep reliability counters and report reliability.

### Train-All Optimism

Train-all scores are fitted-cohort scores. They must not be presented as
prospective validation.

### Threshold Dependence

The operating point targets high sensitivity. Specificity depends strongly on
threshold policy, so every report must state the threshold key and threshold
value.

## Required Product Guards

```text
store preprocessing YAML in preprocessing joblib
store training YAML in model joblib
store prediction preprocessing YAML in model joblib
require patient.target_side in predict YAML
require model_id match between predict YAML and model artifact
require one patient per prediction H5
require H5 schema_version / format match
keep p_cancer and reliability separate
state decision-support-only limitation in report
```
