# M2Q Core4 Optional-Symmetry Candidate v0.1.9

Status: superseded research record and historical v0.1.9 smoke-test artifact.
It describes decision support only and is not clinical validation. The current
gated target-case M2Q record is `m2q_gated_target_case_model_v0_1.md`.

## Fixed Dataset

```text
preprocessing: T100 biopsy-patient model-input DataFrame
measurements: 893
patients: 164
CANCER patients: 75
BENIGN patients: 89
patients with target/contralateral symmetry data: 150
patients without contralateral symmetry data: 14
```

All 164 patients remain in LR1 and LR2 training. A missing contralateral
breast does not remove a patient or stop prediction. It means only that the
symmetry block cannot refine the result for that patient.

## Missing-Symmetry Policy

```text
symmetry_available = 1: target and contralateral profiles are available
symmetry_available = 0: no contralateral profile is available
```

When `symmetry_available = 0`:

```text
SK Core4 values = 0.0
LR2 still uses target-profile score, target measurement count, age, and age_available
reliability = low
p_cancer is not directly reduced
```

`symmetry_available`, contralateral measurement counts, and reliability flags
are audit/report fields. They are intentionally not LR2 inputs, so the model
cannot learn a missingness shortcut.

## Model Route

```text
target-breast normalized radial profiles
-> LR1 LogisticRegression, L2, C=0.1
-> one p_cancer per target measurement
-> logit-average target-breast profile score
-> available target/contralateral SK Core4 symmetry refinement
-> LR2 LogisticRegression, L2, C=0.1
-> final p_cancer
```

LR2 inputs:

```text
profile_p_cancer_logit_average
sk_wasserstein_distance_full_q2
sk_weightedrms1
sk_weightedrms2
sk_mean_peak_value_abs_delta
profile_p_cancer_n_measurements
age
age_available
```

Core4 definitions are in `sk_symmetry_features_v0_1.md`. The peak field is a
direct target-versus-contralateral difference when both sides are available.

## Regularization

```text
LR1 C=0.1
LR2 C=0.1
penalty=L2
class_weight=balanced
features are standard-scaled before each LogisticRegression
```

LR1 and LR2 retain separate configuration fields because their feature spaces
differ. Both use the conservative C=0.1 selected by the existing patient-safe
regularization grid. L2 shrinks coefficients; it does not select features.

## Current Fitted Artifact

```text
model_id: aramis_m2q_t100_core4_optional_symmetry_c1_0p1_c2_0p1
artifact: examples/prediction_models/aramis_m2q_t100_core4_optional_symmetry_c1_0p1_c2_0p1.joblib
threshold_target: 0.298552
```

The artifact is fitted on all 164 development patients. Its train-all metrics
are optimistic by definition:

| metric | value |
|---|---:|
| ROC AUC | 0.881 |
| sensitivity | 0.960 |
| specificity | 0.472 |
| TP / FN | 72 / 3 |
| TN / FP | 42 / 47 |

## Patient-Safe Validation

```text
repeated stratified patient 5-fold x20
held-out folds: 100
patientId split before LR1 and LR2 fitting
threshold selected in each training fold for target sensitivity 0.95
```

| metric | mean +/- standard deviation |
|---|---:|
| ROC AUC | 0.697 +/- 0.068 |
| sensitivity | 0.829 +/- 0.098 |
| specificity | 0.396 +/- 0.120 |

The held-out sensitivity is lower than the 0.95 training-fold target because
the cohort is small. These values are research evidence, not a claim of
clinical validation. An independent or temporally separated cohort is needed.

## Prediction Contract

Prediction receives one EOS H5 v0.3 container for one patient and requires:

```text
one patient
target_side supplied explicitly in prediction YAML
at least one valid target-breast measurement
the model_id above
```

Contralateral measurements are optional. If present, they provide symmetry
refinement; if absent, the report must state low reliability and require
radiologist review.
