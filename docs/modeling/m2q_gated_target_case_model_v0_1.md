# M2Q Gated Target-Case Model v0.1

Status: current research-draft model record.

This document fixes the current Aramis model architecture and its current
development evidence. It is decision support for radiologist review, not
autonomous diagnosis or a clinical-validation claim.

## Fixed Architecture

```text
target-breast normalized radial_profile_data
-> LR1 LogisticRegression
-> target-breast measurement p_cancer values
-> logit-average to one target-breast profile score
-> one LR2 LogisticRegression
-> final p_cancer
```

LR2 receives:

```text
profile_p_cancer_logit_average
age
age_available
gated SK Core4 symmetry fields:
  sk_wasserstein_distance_full_q2
  sk_weightedrms1
  sk_weightedrms2
  sk_mean_peak_value_abs_delta
```

The profile and age terms always reach LR2. SK terms are standardized from
paired training cases and set to zero when no contralateral breast is available.
`symmetry_available` is a gate and audit field, not a learned predictor. There
is no paired/fallback model route. Measurement counts determine report
reliability only and do not enter LR2.

## Training Unit And Cohort

```text
preprocessing: T100 biopsy-patient model-input DataFrame
measurement rows: 893
patients: 164
specimens / breasts: 314
LR1 biopsy-labelled rows: 496
target-breast cases: 175
  CANCER: 76
  BENIGN: 99
```

One biopsied breast creates one historical target case. A bilateral-biopsy
patient creates two target cases. All measurements and target cases from a
patient remain together in every patient-safe fold.

## Honest Nested Validation

This historical experiment selected the frozen recipe regularization. Current
development training does not repeat hyperparameter selection; it evaluates the
fixed `C1=0.1`, `C2=0.3` recipe with repeated patient-safe stratified k-fold.

Evaluation:

```text
outer evaluation: repeated stratified patient 5-fold x20 = 100 test folds
inner selection: patient-safe 4-fold
LR1 C grid: [0.1, 0.3]
LR2 C grid: [0.1, 0.3]
selection metric: inner out-of-fold operational ROC AUC
evaluation threshold: train-fold threshold applied to held-out fold
deployment threshold: train-all threshold targeting sensitivity >=0.95
```

The full-cohort inner selection chose:

```text
LR1 C = 0.1
LR2 C = 0.3
```

Outer test-fold metrics are not forced to 0.95 sensitivity. They measure how a
train-derived threshold transfers to unseen patients.

| model | inputs | ROC AUC | sensitivity | specificity |
|---|---|---:|---:|---:|
| A0 | age + age_available | 0.703 +/- 0.068 | 0.935 +/- 0.068 | 0.292 +/- 0.109 |
| M0 | target profile | 0.599 +/- 0.079 | 0.954 +/- 0.070 | 0.096 +/- 0.080 |
| M1 | target profile + gated SK Core4 | 0.602 +/- 0.077 | 0.945 +/- 0.078 | 0.139 +/- 0.089 |
| M2Q | target profile + gated SK Core4 + age | 0.657 +/- 0.069 | 0.942 +/- 0.076 | 0.159 +/- 0.085 |

Interpretation:

```text
M1 adds a small specificity gain over M0 in this validation.
M2Q improves ranking over M0/M1, but age-only remains stronger by ROC AUC.
No model establishes stable 0.95 sensitivity on unseen folds.
The architecture is fixed for this research iteration; performance requires a
larger independent cohort before any stronger product claim.
```

## Train-All Diagnostic

The following values use the same `C1=0.1`, `C2=0.3` settings on all 175 target
cases. They describe fitted-cohort separation only and are not validation.

| model | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|
| A0 | 0.700 | 0.961 | 0.242 |
| M0 | 0.832 | 0.961 | 0.374 |
| M1 | 0.854 | 0.961 | 0.485 |
| M2Q | 0.867 | 0.961 | 0.465 |

Train-all is useful for inspecting the fitted artifact and threshold behavior.
It must not replace patient-safe outer-fold metrics.

## Related Documents

```text
current_model_pipeline_and_risks_v0_1.md
  product interpretation and guards

training_pipeline_classes_v0_1.md
  sklearn-like implementation and artifact layout

sk_symmetry_features_v0_1.md
  Core4 definitions
```
