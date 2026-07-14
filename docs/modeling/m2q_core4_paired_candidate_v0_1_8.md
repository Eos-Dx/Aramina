# Historical M2Q Core4 Paired Candidate v0.1.8

Status: superseded research record. This strict paired-breast candidate is
retained for traceability only; its artifact is retained in Git tag
`0.1.8-beta`. The current gated target-case M2Q record is documented in
`m2q_gated_target_case_model_v0_1.md`.

## Fixed Dataset

```text
preprocessing: T100 biopsy-patient model-input DataFrame
paired-breast eligibility: exactly one LEFT and one RIGHT breast side available
measurements before eligibility: 893
measurements after eligibility: 855
patients before eligibility: 164
patients after eligibility: 150
patients excluded: 14
CANCER patients: 70
BENIGN patients: 80
```

The 14 patients without a valid paired-breast context are excluded from the
M2Q training cohort. The product model must not learn an artificial
`symmetry_available` flag or zero-filled symmetry values. Prediction therefore
requires both target and contralateral breast measurements.

## Model Route

```text
target-breast normalized radial profiles
-> LR1 LogisticRegression, L2, C=0.1
-> one p_cancer per target measurement
-> logit-average target-breast profile score
-> LR2 LogisticRegression, L2, C=0.1
-> final p_cancer
```

LR2 inputs are deliberately limited to:

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

`profile_p_cancer_n_measurements` is the only Q input. Contralateral counts,
paired-breast flags, and `symmetry_available` remain report/audit fields and
do not train the risk model.

## Symmetry Core4

The original 15 SK fields were screened in the research branch using LOFO,
held-out permutation importance, coefficient-sign stability, and complete
subset refits. Core4 was the smallest tested subset that preserved or improved
the held-out signal relative to the unrestricted set.

```text
sk_wasserstein_distance_full_q2
  target/contralateral mean-profile shape shift over q=2.0..23.0 nm^-1

sk_weightedrms1
  variance-weighted target/contralateral profile difference over q=7.0..15.0

sk_weightedrms2
  variance-weighted target/contralateral profile difference over q=15.0..23.0

sk_mean_peak_value_abs_delta
  absolute difference of side-specific mean per-measurement peak maxima,
  q=13.0..14.8 nm^-1
```

The peak field is now a true target-versus-contralateral difference. Earlier
implementations averaged peak values across both breasts and therefore did not
represent asymmetry directly.

## Regularization

```text
LR1 C=0.1
LR2 C=0.1
penalty=L2
class_weight=balanced
features are standard-scaled before each LogisticRegression
```

LR1 and LR2 have independent regularization parameters because their feature
spaces differ: LR1 receives the complete 100-point profile, while LR2 receives
eight patient-level features. The current paired-cohort release keeps
`C=0.1` for both layers because the existing repeated patient-safe C grid
selected this conservative value. The parameters remain separate so a future
patient-safe grid can tune a layer without silently changing the other one.
L2 regularization shrinks coefficients; it does not perform feature selection.
Core4 is therefore explicitly fixed before this product fit.

## Current Fitted Artifact

```text
model_id: aramis_m2q_t100_core4_c1_0p1_c2_0p1
artifact: examples/prediction_models/aramis_m2q_t100_core4_c1_0p1_c2_0p1.joblib
threshold_target: 0.306342
```

The delivered artifact is fitted on all 150 eligible development patients.
At its fitted-cohort threshold:

| metric | value |
|---|---:|
| ROC AUC | 0.876 |
| sensitivity | 0.957 |
| specificity | 0.463 |
| TP / FN | 67 / 3 |
| TN / FP | 37 / 43 |

These are train-all metrics and are optimistic by definition.

## Patient-Safe Validation

The same training YAML records repeated stratified patient 5-fold validation:

```text
folds: 5
repeats: 20
held-out folds: 100
patient leakage protection: patientId split before LR1 and LR2 fitting
threshold: selected on each training fold for target sensitivity 0.95
```

| metric | mean +/- standard deviation |
|---|---:|
| ROC AUC | 0.645 +/- 0.077 |
| sensitivity | 0.818 +/- 0.104 |
| specificity | 0.324 +/- 0.119 |

The strict paired-breast rule changes the cohort from the earlier 164-patient
research comparison. Its validation metrics must not be mixed with broader
cohort results. The current result demonstrates a research signal only; an
independent or temporally separated cohort is required for generalization
assessment.

## Prediction Contract

Prediction receives one version-0.3 H5 container for one patient and requires:

```text
one patient
LEFT and RIGHT breast measurements
target_side supplied explicitly in prediction YAML
the model_id above
```

The output remains a decision-support `p_cancer`, suggested class, and a
separate data reliability statement. It requires radiologist review.
