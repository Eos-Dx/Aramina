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

Current model family:

```text
M2Q = profile model + age + gated SK symmetry refinement
```

Pipeline:

```text
normalized radial_profile_data
-> LR1 profile LogisticRegression
-> target-breast measurement probabilities
-> logit-average target-breast p_cancer for one target-breast case
-> age + age_available
-> gated SK target/contralateral symmetry features
-> one final LogisticRegression
-> p_cancer
```

Why logit-average:

```text
probability mean treats 0.50 as ordinary average value
logit-average averages evidence before converting back to probability
consistent high-risk measurements remain high-risk even if one measurement is neutral
```

One biopsied breast creates one historical training case. A patient with two
biopsied breasts creates two cases. Patient-safe splitters use `patientId`, so
both cases always remain in the same fold.

## Compared Model Variants

```text
M0:
  LR1 target-breast profile score only

M0Q:
  same prediction as M0; reliability counters are reported separately

M1:
  M0 plus SK target/contralateral symmetry block

M1Q:
  same prediction as M1; reliability counters are reported separately

M2:
  M1 plus age and age_available

M2Q:
  same prediction as M2; reliability counters are reported separately

A0:
  age and age_available only; shortcut-risk control
```

Current development model:

```text
M2Q with one gated optional-symmetry refinement
```

Reason:

```text
one final model always evaluates target-breast profile and age
SK symmetry terms refine that result only when paired data are available
without a contralateral breast, every SK contribution is zero
reliability remains report metadata and is not a diagnostic feature
```

Age-only A0 remains a mandatory comparison because age is clinically real but
can dominate a small dataset.

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

No reliability field enters the M2Q model. If contralateral data is missing,
`symmetry_available = 0` makes every SK contribution zero. The gate is not a
learned feature and no second fallback model is fitted.

## Current Development Choice

```text
architecture_id: m2q_gated_target_case_v0_1
preprocessing: T100 biopsy-patient model input
model: M2Q
regularization: selected inside patient-safe nested CV
target sensitivity: 0.95 threshold derived from inner out-of-fold predictions
```

M2Q combines profile score, optional SK Core4 symmetry, and age. Measurement
counts affect reliability reporting only. Age is clinically relevant because
breast cancer risk increases with age, but A0 exposes how much performance age
provides without XRD information.

Why T100:

```text
T70 is stricter but loses more patients
T130 keeps more patients but had weaker M1Q validation behavior
T100 is the current development compromise
```

Why two regularization values:

```text
LR1 and LR2 have independently selected C values
selection occurs only inside inner patient-safe folds
L2 regularization reduces unstable coefficients in the small cohort
```

Detailed evidence:

```text
docs/modeling/m2q_gated_target_case_model_v0_1.md
```

## Validation Framing

Patient-safe evaluation is required:

```text
measurements from one patient cannot appear in both train and test
```

Views used:

```text
outer repeated stratified patient K-fold:
  generalization estimate

inner patient K-fold:
  regularization selection and threshold calibration

repeated 80/20 or 70/30 patient splits:
  robustness check

LOOVM:
  patient-safe leave-one-out view, high variance

train-all:
  refitted artifact after nested validation
  fitted-cohort metrics are diagnostics only
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
