# T100 Controlled M0Q/M1Q/M2Q Comparison v0.1

Status: research draft. This is the direct control needed to separate the
contribution of SK symmetry from the contribution of age.

## Fixed Conditions

```text
cohort: T100 biopsy cohort, 164 patients
LR1 C: 0.3
LR2 C: 0.1
validation: repeated patient-safe stratified 5-fold, 20 repeats, 100 folds
threshold: selected on each train fold for target sensitivity 0.95
```

Each candidate uses the identical split manifest. For every fold, LR1 is fit
only on training-patient measurements and LR2 is fit only on the corresponding
training-patient feature rows.

## Model Definitions

```text
M0Q:
  profile risk + target-breast prediction count
  no contralateral or paired-breast availability fields
  no SK symmetry, no age

M1Q all SK:
  M0Q + all 15 Kubitsky SK fields

M1Q screened core 4:
  M0Q + four selected Kubitsky SK fields

M2Q no SK + age:
  M0Q + age / age_available

M2Q screened core 4 + age:
  M0Q + four selected SK fields + age / age_available
```

The selected SK core is:

```text
sk_wasserstein_distance_full_q2
sk_weightedrms1
sk_weightedrms2
sk_mean_peak_value_abs_delta
```

## Honest Held-Out Results

| candidate | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|
| M0Q | 0.619 +/- 0.076 | 0.786 +/- 0.121 | 0.405 +/- 0.114 |
| M1Q all 15 SK | 0.620 +/- 0.078 | 0.752 +/- 0.121 | 0.426 +/- 0.117 |
| M1Q screened core 4 | 0.627 +/- 0.077 | 0.751 +/- 0.122 | 0.431 +/- 0.118 |
| M2Q no SK + age | 0.675 +/- 0.071 | 0.807 +/- 0.113 | 0.440 +/- 0.112 |
| M2Q all 15 SK + age | 0.676 +/- 0.069 | 0.789 +/- 0.113 | 0.453 +/- 0.115 |
| M2Q screened core 4 + age | 0.687 +/- 0.069 | 0.799 +/- 0.103 | 0.460 +/- 0.124 |

## Interpretation

SK symmetry is not equivalent to age:

```text
M0Q -> M1Q screened core 4:
  ROC AUC: +0.007
  sensitivity: -0.035
  specificity: +0.026

M2Q no SK + age -> M2Q screened core 4 + age:
  ROC AUC: +0.012
  sensitivity: -0.008
  specificity: +0.020
```

Therefore, the reduced SK block contributes a small but reproducible internal
improvement in the current corrected T100 experiment. The much larger shift
from M1Q to M2Q is predominantly age information. Age remains a known strong
covariate and must be reported separately; it should not be used to claim that
the XRD symmetry signal alone is strong.

## Fixed-Sensitivity Check

Sensitivity and specificity are a threshold-dependent trade-off. Therefore the
honest fold means above do **not** support the statement that SK raises both
metrics simultaneously: with a threshold selected on each training fold for
training sensitivity 0.95, the SK candidates generally exchange a small amount
of held-out sensitivity for higher held-out specificity.

As a separate descriptive check, repeated held-out predictions were averaged
per patient and the ROC curve was inspected at sensitivity at least 0.95:

| candidate | pooled OOF ROC AUC | sensitivity | maximum specificity at sensitivity >= 0.95 |
|---|---:|---:|---:|
| M0Q | 0.632 | 0.960 | 0.112 |
| M1Q screened core 4 | 0.639 | 0.973 | 0.112 |
| M2Q no SK + age | 0.684 | 0.960 | 0.112 |
| M2Q screened core 4 + age | 0.701 | 0.960 | 0.180 |

This supports the narrower claim that the reduced SK block improves the ROC
operating curve in M2Q: at the same achieved 0.96 sensitivity, M2Q core 4
reaches 0.180 specificity versus 0.112 without SK. It does **not** establish an
independent clinical operating point because the pooled OOF labels were used
to locate that threshold. A fixed product threshold must still be selected
only from training data and then evaluated on a held-out cohort.

Earlier M1/M1Q results with larger gains cannot be compared directly with this
table unless the exact older preprocessing artifact, target-side rule, feature
schema, regularization, and split manifest are reproduced. The current table
is the only valid comparison for the corrected target/contralateral and
peak-delta implementation.

Train-all results are descriptive only and are stored in the CSV. They are not
honest validation estimates.

Results:

```text
docs/modeling/results/t100_m0q_m1q_m2q_controlled_comparison_v0_1.csv
docs/modeling/results/t100_m0q_m1q_m2q_controlled_oof_predictions_v0_1.csv
docs/modeling/results/t100_m0q_m1q_m2q_fixed_sensitivity_oof_v0_1.csv
```
