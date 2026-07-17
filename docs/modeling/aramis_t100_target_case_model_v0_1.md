# Aramis T100 Target-Case Model v0.1

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

For an unpaired target breast, internal report records
`profile_age_with_neutral_symmetry_gate` and `symmetry_refinement: not_applied`.
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

Current interpretation:

```text
Aramis T100 is a research-draft decision-support model, not a clinical-validation claim.
Age is an important risk signal and must remain visible in model review.
No current result establishes stable 0.95 sensitivity on unseen patients.
A larger independent cohort is required before any stronger product claim.
```

## Current 0.2.7 Evaluation Record

The current packaged artifact was evaluated on the T100 biopsy-patient cohort
using the fixed 100 patient-safe folds. The target threshold was derived on the
training patients in each fold and then applied once to that fold's held-out
patients.

| metric | mean across 100 folds | pooled held-out cases (95% bootstrap CI) |
|---|---:|---:|
| ROC AUC | 0.662 +/- 0.069 | 0.673 (0.592 to 0.750) |
| sensitivity | 0.819 +/- 0.099 | 0.855 (0.769 to 0.932) |
| specificity | 0.383 +/- 0.116 | 0.323 (0.231 to 0.422) |

These are research-draft evaluation results, not a clinical performance claim.

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
