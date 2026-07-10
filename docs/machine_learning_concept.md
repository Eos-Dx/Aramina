# Aramis Machine Learning Concept v0.1

Status: research draft.

Aramis estimates `p_cancer` for the clinically suspicious breast and returns a
suggested BENIGN/CANCER decision-support class. The output requires review by a
qualified breast-imaging clinician.

## Clinical Scenario

```text
patient has suspicious mammography finding
clinician identifies target breast / suspicious region
left and right breast XRD measurements are collected when possible
Aramis scores the target breast
contralateral breast provides patient-internal symmetry context
```

The first product route assumes `target_side` is supplied by the clinical
caller. Prediction must not infer target side from labels or biopsy metadata.

## Training Cohort Logic

Primary training cohort:

```text
biopsy_patients
```

Reason:

```text
biopsied breast is the historical suspicious breast
biopsy-associated endpoint is the cleanest available BENIGN/CANCER target
contralateral rows are retained to compute symmetry features
```

Current product label grouping:

```text
BENIGN + NORMAL -> BENIGN
CANCER + ATYPICAL + PRE_CANCEROUS -> CANCER
NA -> excluded
```

The all-patients dataset remains exploratory. It is useful for sensitivity
checks but is not the current primary product-training dataset.

## Model Structure

Current primary model family:

```text
M1Q = profile model + SK symmetry block + reliability counters
```

Pipeline:

```text
normalized radial_profile_data
-> LR1 profile LogisticRegression
-> target-breast measurement probabilities
-> logit-average target-breast p_cancer at patient level
-> SK target/contralateral symmetry features
-> reliability counters
-> final LogisticRegression
-> p_cancer
```

Why logit-average:

```text
probability mean treats 0.50 as ordinary average value
logit-average averages evidence before converting back to probability
consistent high-risk measurements remain high-risk even if one measurement is neutral
```

## Compared Model Variants

```text
M0:
  LR1 target-breast profile score only

M0Q:
  M0 plus measurement reliability counters

M1:
  M0 plus SK target/contralateral symmetry block

M1Q:
  M1 plus reliability counters

M2:
  M1 plus age and age_available

M2Q:
  M1Q plus age and age_available
```

Current primary candidate:

```text
M1Q
```

Reason:

```text
adds same-patient symmetry context to target-breast profile evidence
keeps reliability as explicit model features and report metadata
avoids age as a primary shortcut in the first product candidate
```

Age-containing models remain comparison models because age can be clinically
real but can dominate a small dataset.

## Symmetry Features

Symmetry is patient-level context. It does not by itself say which breast is
abnormal. Direction comes from target-breast profile scoring.

Main feature families:

```text
target/contralateral profile mean differences
target within-breast variability
contralateral within-breast variability
SK RMS / weighted RMS blocks in q regions
cosine distance over full q range
fixed Core4 target/contralateral symmetry fields
```

Detailed mathematical definitions are in:

```text
docs/modeling/sk_symmetry_features_v0_1.md
```

Reliability audit fields:

```text
profile_p_cancer_n_measurements
target_measurements
contralateral_measurements
min_measurements_per_breast
target_measurements_ok
contralateral_measurements_ok
paired_measurements_ok
```

Only `profile_p_cancer_n_measurements` enters the M2Q model. The remaining
fields are reported separately from risk. Prediction requires paired breasts,
so `symmetry_available` is not a model feature.

## Current Candidate Choice

```text
model_id: aramis_m2q_t100_core4_c1_0p1_c2_0p1
preprocessing: T100 biopsy-patient model input, strict paired-breast cohort
model: M2Q
regularization: LR1 L2 C=0.1; LR2 L2 C=0.1
target sensitivity: >= 0.95 on fitted development cohort
threshold_target: 0.306342
```

M2Q combines the profile score, fixed SK Core4 symmetry fields, target
measurement count, and age as an explicit clinical risk prior. Age is
clinically relevant because breast cancer risk increases with age.

Why T100:

```text
T70 is stricter but loses more patients
T130 keeps more patients but had weaker M1Q validation behavior
T100 is the current development compromise
```

Why two regularization values:

```text
LR1 and LR2 have independently configurable C values
the current paired-cohort candidate uses C=0.1 in both layers
L2 regularization reduces unstable coefficients in the small cohort
```

Detailed evidence:

```text
docs/modeling/m2q_core4_paired_candidate_v0_1_8.md
```

## Validation Framing

Patient-safe evaluation is required:

```text
measurements from one patient cannot appear in both train and test
```

Views used:

```text
repeated stratified patient K-fold:
  main model-selection signal

repeated 80/20 or 70/30 patient splits:
  robustness check

LOOVM:
  patient-safe leave-one-out view, high variance

train-all:
  final fitted candidate artifact and operating point
  optimistic fitted-cohort metric, not validation
```

The delivered candidate is trained on all available development patients after
model family and regularization are selected. Validation estimates must be read
from patient-safe K-fold / split / LOOVM outputs, not from train-all alone.

## Stop Conditions

Do not present a model as usable if any of these are true:

```text
single-class training data
unknown label mapping
patient leakage
missing preprocessing YAML lineage
missing prediction preprocessing config in model artifact
unstable feature schema
unknown target_side in prediction
unsupported H5 container version
```
