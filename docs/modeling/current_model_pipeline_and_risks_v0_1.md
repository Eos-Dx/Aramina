# Current Aramis Model Pipeline And Risks v0.1

Status: research draft.

This document records how the current candidate works and what must not be
overclaimed.

## Candidate Summary

```text
model_id: aramis_m2q_t100_train_all_c0p1
dataset: biopsy_patients model-input DataFrame
preprocessing: T100 AgBH quality threshold
selected_model: M2Q
regularization: L2 LogisticRegression, C=0.1
threshold_target: 0.302291
```

The model estimates `p_cancer` for decision support. It is not autonomous
diagnosis.

## Data Flow

```text
H5
-> YAML-declared XRD preprocessing
-> measurement-level normalized radial_profile_data
-> patient-safe training/evaluation split
-> LR1 profile model
-> patient-level target-breast p_cancer by logit-average
-> SK target/contralateral symmetry features
-> reliability counters
-> age + age_available
-> M2Q final LogisticRegression
-> p_cancer
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

`all_patients` is exploratory only.

## Why M2Q

```text
M0:
  target-breast profile score only

M1:
  target-breast profile score + SK target/contralateral symmetry

M1Q:
  M1 + reliability counters

M2/M2Q:
  add age
```

M2Q is the current primary candidate because it keeps the M1Q XRD profile,
same-patient symmetry context, and explicit data-reliability features, and adds
age as an explicit clinical risk prior. Age is clinically meaningful because
breast cancer risk increases with age. In the current M1Q-vs-M2Q comparison,
adding age improved the candidate by roughly 3 percentage points.

## Feature Interpretation

Target-breast profile signal:

```text
LR1 scores only target-breast radial_profile_data
measurement probabilities are combined by logit-average
```

Symmetry context:

```text
compares target breast against contralateral breast
adds target and contralateral within-breast variability
adds SK distance features over q ranges
```

Reliability:

```text
counts valid target measurements
counts valid contralateral measurements
flags whether paired-breast context is sufficient
is reported separately from p_cancer
```

Low reliability does not lower risk. It tells the downstream report to flag the
result as less stable.

## Validation Views

```text
repeated stratified patient K-fold:
  main model-selection signal

repeated patient 80/20 or 70/30 splits:
  robustness check

LOOVM:
  patient-safe but high-variance leave-one-out view

train-all:
  final fitted candidate artifact and operating threshold
  optimistic fitted-cohort view, not validation
```

All split modes are patient-safe.

## Current Evidence Pattern

The selected candidate is documented in:

```text
final_candidate_model_artifact_v0_1.md
```

Key fixed-cohort result:

```text
patients: 164
CANCER patients: 75
BENIGN patients: 89
train-all ROC AUC: 0.889
train-all sensitivity: 0.960
train-all specificity: 0.517
```

This train-all result is used to lock the candidate artifact and threshold. It
is paired with patient-safe validation estimates in the candidate document.

## Main Risks

### Small Dataset

Specificity at high sensitivity is unstable. Report mean and spread for
patient-safe split modes.

### Age Shortcut

Age improves some models but can dominate a small dataset. M2/M2Q remain
comparison models until age use is explicitly accepted.

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
