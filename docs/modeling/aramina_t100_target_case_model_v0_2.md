# Aramina T100 Target-Case Model v0.2

Status: frozen research-draft product model definition.

Candidate model version: `0.2.13-beta`.
Candidate model artifact:
`aramina_target_breast_risk_0_2_13-beta_f5e4a04cad11`.

The preceding `0.2.12-beta` artifact remains intact under `models/` as the
preserved product baseline. This record describes the separately retrained
`0.2.13-beta` artifact. It is decision support for qualified clinician review,
not autonomous diagnosis or a clinical-validation claim.

## Fixed Architecture

```text
target-breast normalized radial_profile_data
-> LR1 LogisticRegression
-> target-breast measurement p_cancer values
-> logit-average to one target-breast profile score
-> one LR2 LogisticRegression with age and optional gated SK Core4 symmetry
-> final p_cancer and fixed decision threshold
```

LR1 has 100 normalized profile bins and `C=0.1`. LR2 receives the target
profile logit-average, age, age availability, and four gated SK symmetry
fields; it uses `C=0.3`. The profile and age terms always enter LR2. Symmetry
is neutral unless both breasts have at least two valid measurements and all
Core4 values are finite. `symmetry_available` is a gate and audit field, not a
learned predictor.

## Frozen Training Inputs

```text
patients: 164
measurements: 893
target-breast cases: 175
  CANCER: 76
  BENIGN: 99
evaluation: repeated patient-safe stratified 5-fold x20, seed 42
```

The model was retrained from the full historical H5 archive with:

```text
Aramina source: 34f5c4d6d9e615b67ab9982c605bc1085916a846
XRD-preprocessing release: v0.1.9-beta
XRD-preprocessing source: 88dcaa277c5a0d4be2ab637bc5827a14bd106bea
```

All target cases belonging to one patient remain in one patient-safe fold.
One biopsied breast creates one target case; bilateral-biopsy patients create
two target cases.

## Evaluation Record

Patient-safe held-out metrics across 100 folds:

| metric | mean +/- standard deviation |
|---|---:|
| ROC AUC | 0.645 +/- 0.069 |
| sensitivity | 0.818 +/- 0.099 |
| specificity | 0.376 +/- 0.133 |

The deployment threshold is fit on all accepted target cases for target
sensitivity at least `0.95`:

| metric | final train-on-all value |
|---|---:|
| threshold | 0.24666 |
| ROC AUC | 0.86497 |
| sensitivity | 0.96053 (73/76) |
| specificity | 0.49495 (49/99) |
| TP / TN / FN / FP | 73 / 49 / 3 / 50 |

Train-on-all metrics describe the fitted artifact on its training cases. They
are not independent validation. The held-out evaluation remains the evidence
for transfer to new patients.

## Release Boundary

`0.2.13-beta` changes the release lineage to XRD-preprocessing `v0.1.9-beta`.
It does not change the fixed Aramina architecture, feature schema,
regularization, threshold-selection rule, or patient cohort. The retraining
reproduced the prior aggregate metrics on the same accepted cohort.

## Related Records

```text
aramina_t100_target_case_model_v0_1.md
  preserved 0.2.12-beta baseline

current_model_dataframe_v0_1.md
  measurement, breast, target-case, and label rules

../contracts/training_config_v0_1.md
  runnable training contract
```
