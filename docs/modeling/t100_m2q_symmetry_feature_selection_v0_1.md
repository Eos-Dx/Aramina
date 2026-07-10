# T100 M2Q Symmetry Feature Selection v0.1

Status: research draft. This is an internal feature-selection experiment, not
clinical validation and not a final product feature schema.

## Scope

Only one model and one preprocessed cohort are evaluated:

```text
model: M2Q
cohort: T100 biopsy cohort
measurements: 893
patients: 164
CANCER patients: 75
BENIGN patients: 89
LR1 regularization: C = 0.3
LR2 regularization: C = 0.1
```

The experiment does not alter `age`, `age_available`,
`profile_p_cancer_n_measurements`, `symmetry_available`, or
`profile_p_cancer_logit_average`. It tests only the 15 `sk_*`
target/contralateral symmetry fields. Q contains only the target-breast
prediction count; contralateral and paired-breast counters are report fields,
not model inputs.

## Why Not SHAP

LR2 is a standard-scaled LogisticRegression. It has no tree structure or
nonlinear interactions for SHAP to reveal. For a linear model, the fitted
standardized coefficient already describes the direction and relative linear
weight of a feature. SHAP would still be unstable when correlated symmetry
fields substitute for one another.

Feature selection therefore uses three complementary tests on the same
patient-safe splits:

1. LOFO: LR2 is refitted after removing one field. The held-out ROC AUC,
   sensitivity, and specificity are compared with full M2Q.
2. Permutation importance: one feature is shuffled only among held-out
   patients; the ROC AUC decrease is recorded. The trained model is unchanged.
3. Coefficient stability: the standardized LR2 coefficient is collected over
   all folds. A stable sign means the direction does not depend on one split.

No individual test is sufficient. Correlated fields can appear unimportant
alone while their group still contributes. Therefore candidate smaller schemas
are retrained and compared as complete models.

## Validation

```text
repeated stratified 5-fold: 20 repeats, 100 patient-safe held-out folds
threshold: selected on each train fold for target sensitivity 0.95
test metrics: calculated only on the corresponding held-out patients
train-all: descriptive fit only; never used for feature selection
```

For every fold, LR1 is trained only on training-patient measurements. Its
profile scores are then calculated separately for train and held-out patients.
LR2 is fitted only on the resulting train feature rows. Thus neither a patient
nor its measurements enter both parts of a fold.

## Individual SK Screen

Positive LOFO delta means that removing a feature reduced held-out ROC AUC.
Positive permutation delta means that shuffling the held-out feature reduced
ROC AUC.

| field | LOFO ROC AUC delta | permutation ROC AUC delta | stable coefficient sign |
|---|---:|---:|---:|
| `sk_wasserstein_distance_full_q2` | +0.0067 | +0.0035 | 100% positive |
| `sk_weightedrms2` | +0.0043 | +0.0089 | 100% negative |
| `sk_mean_peak_value_abs_delta` | +0.0022 | +0.0075 | 100% negative |
| `sk_weightedrms1` | +0.0013 | +0.0069 | 100% negative |
| `sk_cosine_distance_full_q2` | +0.0004 | -0.0019 | 95% positive |
| `sk_wasserstein_distance_mu_tc` | +0.0001 | -0.0019 | 99% positive |
| `sk_meanrms1`, `sk_meanrms2` | <= 0 | <= 0.001 | not useful as individual fields |
| four `sk_sigma_*` fields | < 0 | <= 0.001 | removal is a candidate |
| `sk_mahalanobis1`, `sk_mahalanobis2` | <= 0.0004 | <= 0.0028 | weak independent contribution |
| `sk_peak14_intensity_abs_delta` | -0.0010 | -0.0009 | removal is a candidate |

These values are a screening rank, not a rule for automatic deletion. In
particular, `sk_wasserstein_distance_mu_tc` and
`sk_cosine_distance_full_q2` may be redundant with the full-q Wasserstein
field rather than biologically irrelevant.

Detailed values are saved in:

```text
docs/modeling/results/t100_m2q_sk_feature_selection_summary_v0_1.csv
docs/modeling/results/t100_m2q_sk_feature_selection_per_split_v0_1.csv
docs/modeling/results/t100_m2q_sk_feature_coefficients_v0_1.csv
```

## Complete-Schema Comparison

Every row below uses exactly the same 100 split manifest. `all_sk` has all 15
SK fields. The other rows retain the M2Q core fields and only change the SK
subset.

| candidate | retained SK fields | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|---:|
| `all_sk` | 15 | 0.676 +/- 0.069 | 0.789 +/- 0.113 | 0.453 +/- 0.115 |
| `distance_tail` | 3 | 0.673 +/- 0.067 | 0.811 +/- 0.108 | 0.429 +/- 0.108 |
| `screened_core_3` | 3 | 0.685 +/- 0.069 | 0.809 +/- 0.106 | 0.441 +/- 0.114 |
| `screened_core_4` | 4 | 0.687 +/- 0.069 | 0.799 +/- 0.103 | 0.460 +/- 0.124 |
| `no_sk` | 0 | 0.675 +/- 0.071 | 0.807 +/- 0.113 | 0.440 +/- 0.112 |

`screened_core_4` is the current smaller-schema candidate. Relative to
`all_sk`, it retains four rather than 15 SK fields while giving:

```text
ROC AUC: +0.0107
sensitivity: +0.0100
specificity: +0.0074
```

Its four SK fields are:

```text
sk_wasserstein_distance_full_q2
sk_weightedrms1
sk_weightedrms2
sk_mean_peak_value_abs_delta
```

The resulting M2Q schema has nine fields instead of 20: LR1 risk,
`symmetry_available`, these four SK fields, the target-breast prediction count,
and `age` plus `age_available`.

The train-all row is descriptive and must not be read as an honest result:

| candidate | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|
| `all_sk` | 0.903 | 0.960 | 0.573 |
| `screened_core_4` | 0.903 | 0.960 | 0.573 |
| `screened_core_3` | 0.896 | 0.960 | 0.528 |
| `distance_tail` | 0.890 | 0.973 | 0.449 |
| `no_sk` | 0.880 | 0.960 | 0.528 |

Full comparison:

```text
docs/modeling/results/t100_m2q_sk_feature_subset_comparison_v0_1.csv
```

## Decision Pending Product Freeze

The experimental recommendation is to carry `screened_core_4` forward as the
only candidate SK schema. Do not yet change the product YAML or model contract.
Feature selection and model evaluation used the same internal cohort, so the
smaller schema needs one final frozen-schema confirmation before it is adopted.

The original all-pairs cosine formulation was evaluated separately. It is kept
as an audit and future-cohort candidate, not added to the proposed M2Q schema:

```text
docs/modeling/t100_m2q_pairwise_cosine_experiment_v0_1.md
```

The direct control against M0Q and M1Q is recorded separately. It distinguishes
the limited SK contribution from the stronger age contribution in M2Q:

```text
docs/modeling/t100_m0q_m1q_m2q_controlled_comparison_v0_1.md
```
After that confirmation, update together:

```text
M2Q feature schema
training YAML
predict artifact contract
tests
model documentation
```
