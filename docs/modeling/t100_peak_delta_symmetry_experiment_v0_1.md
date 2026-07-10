# T100 Peak-Delta Symmetry Experiment v0.1

Status: research draft.

Branch: `experiment/aramis-model-selection-v0.1`.

## Purpose

This experiment fixes two SK symmetry features that mixed target and
contralateral breast measurements:

```text
old sk_peak14_intensity
old sk_mean_peak_value
```

The old fields were not pure asymmetry features. They were computed from both
breasts together. The experiment replaces them with target-vs-contralateral
absolute-difference fields:

```text
sk_peak14_intensity_abs_delta
sk_mean_peak_value_abs_delta
```

The goal is to test whether target/contralateral peak differences improve the
patient-level model and to identify SK feature groups that can be removed
without loss of quality.

## Dataset

Input preprocessing artifact:

```text
examples/outputs/model_selection_m1q_v0_1/preprocessing/aramis_t100_biopsy_patients_model_input.joblib
```

Dataset size:

```text
rows: 893
patients: 164
specimens: 314
LR1 rows: 496
CANCER patients: 75
BENIGN patients: 89
```

Only T100 is tested here. T70, T130, LOOVM, and 70/30 are intentionally not
used in this experiment.

## Regularization

Before this experiment, the same `logreg_c` was used for LR1 and LR2. The code
now supports:

```text
lr1_logreg_c
lr2_logreg_c
```

If these fields are absent, both layers fall back to the old `logreg_c`.

This matters because LR1 learns the radial-profile model, while LR2 mixes
patient-level profile probability, symmetry, reliability, and age. These two
tasks do not necessarily need the same regularization.

Regularization grid:

```text
lr1_logreg_c: 0.1, 0.3, 1.0, 3.0
lr2_logreg_c: 0.1, 0.3, 1.0, 3.0
validation: stratified 5-fold
models: M0Q, M1Q, M2Q
```

Result CSV:

```text
docs/modeling/results/t100_peak_delta_regularization_grid_v0_1.csv
```

Best rows in the tested grid:

| model | lr1 C | lr2 C | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|---:|---:|
| M0Q | 0.1 | 0.3 | 0.598 | 0.840 | 0.337 |
| M1Q | 0.1 | 0.3 | 0.612 | 0.773 | 0.373 |
| M2Q | 0.3 | 0.1 | 0.657 | 0.840 | 0.371 |

Conclusion: using the same regularization for both layers is not optimal in
this grid. M2Q performs best with a less constrained LR1 and a more strongly
regularized LR2.

## Fixed-C Comparison

With `lr1_logreg_c=1.0` and `lr2_logreg_c=1.0`, after the peak-delta fix:

| mode | model | ROC AUC | sensitivity | specificity |
|---|---|---:|---:|---:|
| stratified 5-fold | M0Q | 0.572 +/- 0.069 | 0.720 +/- 0.078 | 0.403 +/- 0.110 |
| stratified 5-fold | M1Q | 0.573 +/- 0.074 | 0.667 +/- 0.094 | 0.495 +/- 0.067 |
| stratified 5-fold | M2Q | 0.621 +/- 0.074 | 0.693 +/- 0.090 | 0.494 +/- 0.079 |
| train-all | M0Q | 0.890 | 0.960 | 0.596 |
| train-all | M1Q | 0.915 | 0.960 | 0.640 |
| train-all | M2Q | 0.925 | 0.960 | 0.697 |

The previous T100 M1Q stratified 5-fold result with the old mixed peak fields
was:

```text
ROC AUC 0.577 +/- 0.074
sensitivity 0.613 +/- 0.078
specificity 0.472 +/- 0.082
```

After fixing the peak fields and keeping C=1/C=1, M1Q is not materially better
by ROC AUC. It becomes more specific at the tested threshold but less clearly
separated by ROC. The main improvement appears when M2Q and separate
regularization are used.

## M2Q Feature Ablation

Ablation uses:

```text
model: M2Q
lr1_logreg_c: 0.3
lr2_logreg_c: 0.1
validation: stratified 5-fold
```

Result CSV:

```text
docs/modeling/results/t100_peak_delta_m2q_feature_ablation_v0_1.csv
```

| feature group removed | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|
| none | 0.657 | 0.840 | 0.371 |
| peak delta | 0.656 | 0.827 | 0.405 |
| sigma | 0.671 | 0.813 | 0.416 |
| RMS | 0.657 | 0.827 | 0.382 |
| weighted RMS | 0.656 | 0.827 | 0.383 |
| Mahalanobis | 0.660 | 0.827 | 0.382 |
| distance tail | 0.642 | 0.773 | 0.371 |
| all SK symmetry | 0.664 | 0.800 | 0.405 |

Interpretation:

- The SK block is not yet stable enough to keep all fields unquestioned.
- `distance_tail` fields look most important: removing them gives the clearest
  ROC and sensitivity drop.
- `sigma` fields are candidates for removal: removing them improves ROC and
  specificity in this run, although sensitivity drops.
- `peak_delta` fields are not strongly helpful in this run. They are correct
  semantically but may be redundant with other distance features.
- Removing all SK symmetry slightly increases ROC but lowers sensitivity. This
  is not a reason to remove all symmetry yet; it is a warning that the current
  feature block is too large for the dataset.

## Train-All Coefficients

Coefficient CSV:

```text
docs/modeling/results/t100_peak_delta_m2q_train_all_coefficients_v0_1.csv
```

Top train-all M2Q coefficients with `lr1_logreg_c=0.3` and
`lr2_logreg_c=0.1`:

| feature | coefficient |
|---|---:|
| profile_p_cancer_logit_average | 1.154 |
| age | 0.474 |
| sk_wasserstein_distance_full_q2 | 0.275 |
| sk_weightedrms1 | -0.194 |
| sk_mean_peak_value_abs_delta | -0.162 |
| sk_mahalanobis2 | -0.140 |
| sk_weightedrms2 | -0.136 |
| sk_wasserstein_distance_mu_tc | 0.121 |
| sk_cosine_distance_full_q2 | 0.102 |

Train-all result for this more regularized M2Q:

```text
ROC AUC: 0.904
sensitivity: 0.960
specificity: 0.584
```

This train-all result is lower than the less regularized C=1/C=1 train-all
result, which is expected. Stronger LR2 regularization reduces overfitting and
is therefore more appropriate for a small dataset.

## Current Decision

For the next experiment:

```text
dataset: T100 biopsy cohort
primary model: M2Q
regularization candidate: lr1_logreg_c=0.3, lr2_logreg_c=0.1
validation focus: stratified k-fold
```

Recommended next simplification test:

```text
keep:
profile_p_cancer_logit_average
age / age_available
reliability counters
sk_wasserstein_distance_full_q2
sk_wasserstein_distance_mu_tc
sk_cosine_distance_full_q2

test removing or compressing:
sigma block
peak-delta block
RMS / weighted-RMS / Mahalanobis duplicates
```

This is not a final product claim. It is an experimental result used to reduce
the feature set before product-code cleanup.
