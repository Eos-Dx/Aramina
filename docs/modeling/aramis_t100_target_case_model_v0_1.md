# Aramis T100 Target-Case Model v0.1

Status: frozen research-draft product model definition.

Frozen model version: `0.2.11-beta`.
Frozen model artifact:
`aramis_target_breast_risk_0_2_11-beta_d531ea38c5dc`.

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
paired training cases and set to zero unless both breasts have at least two
valid measurements and all Core4 values are finite. `symmetry_available` is a
gate and audit field, not a learned predictor. There is no paired/fallback
model route. Measurement counts determine report reliability only and do not
enter LR2.

For an unpaired target breast, internal report records
`azimuthal_integration_age`.
The contralateral report block is filled with `unknown`; its absence is never
encoded as zero risk or a BENIGN result.

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

Bilateral-biopsy cases are retained because they match intended product
behaviour: each clinically selected target breast receives an independent
prediction. They form a potentially harder subgroup, however, because both
breasts contribute correlated target cases from one patient. The current
patient-safe split prevents leakage between them; a later validation should
also report bilateral and unilateral target-breast performance separately.

## Frozen Regularization And Evaluation

Historical experiments selected the frozen regularization values. Current
product training does not search hyperparameters. It evaluates the fixed model
and then fits it on all accepted target cases.

Evaluation:

```text
historical regularization search: patient-safe folds over LR1/LR2 C in [0.1, 0.3]
fixed product regularization: LR1 C=0.1, LR2 C=0.3
current evaluation: repeated patient-safe stratified 5-fold x20, seed 42
evaluation threshold: derived on each training fold, applied to its test fold
deployment threshold: derived from final train-all scores at sensitivity >=0.95
```

The historical selection chose:

```text
LR1 C = 0.1
LR2 C = 0.3
```

Held-out fold sensitivity is not forced to 0.95. It measures how a
train-derived threshold transfers to unseen patients. Each run with
`run.evaluation: true` writes the exact 100-fold metrics and held-out
predictions next to the run-specific model artifact. Historical architecture comparisons are
retained in the `experiment` branch rather than duplicated in product docs.

LR2 is trained from LR1 scores fitted on the same training partition rather
than out-of-fold LR1 scores. This is retained deliberately for the current
small research cohort and is documented as a limitation; it must be revisited
before a future independent validation claim.

Current interpretation:

```text
Aramis T100 is a research-draft decision-support model, not a clinical-validation claim.
Age is an important risk signal and must remain visible in model review.
No current result establishes stable 0.95 sensitivity on unseen patients.
A larger independent cohort is required before any stronger product claim.
```

## Frozen 0.2.11-beta Evaluation Record

The frozen `0.2.11-beta` packaged artifact uses the current
`aramis_sk_symmetry_v0_2` feature contract and was evaluated on the T100 biopsy-patient cohort
using the fixed 100 patient-safe folds. The target threshold was derived on the
training patients in each fold and then applied once to that fold's held-out
patients.

| metric | mean across 100 folds | pooled held-out cases (95% bootstrap CI) |
|---|---:|---:|
| ROC AUC | 0.645 +/- 0.069 | 0.656 (0.574 to 0.731) |
| sensitivity | 0.818 +/- 0.099 | 0.829 (0.741 to 0.910) |
| specificity | 0.376 +/- 0.133 | 0.323 (0.228 to 0.420) |

These are research-draft evaluation results, not a clinical performance claim.
Historical candidate artifacts are retained in the experimental branch rather
than as compatibility artifacts in `main`.

## Train On All

`run.train_on_all: true` fits the fixed product model on all accepted cases and derives the deployment
threshold. The model joblib stores executable estimators, frozen threshold,
resolved training YAML, historical preprocessing YAML, prediction
preprocessing YAML, prediction contract, H5 lineage, and a concise frozen
`model_performance` record. The generated `model_description.yaml` provides
the human-readable copy. Full fold metrics and held-out predictions remain in
the `evaluation_metrics.csv` and `evaluation_predictions.csv` files.

Train-all is useful for inspecting fitted-artifact separation and threshold
behavior. It must not replace patient-safe evaluation.

## Related Documents

```text
sk_symmetry_features_v0_1.md
  Core4 definitions

../contracts/training_config_v0_1.md
  fixed training and evaluation contract

../product_development_rules.md
  product controls and known limitations
```
