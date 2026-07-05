# Biopsy Primary Model Comparison With SK Symmetry Block v0.1

Clinical framing: research-draft decision support only; requires radiologist
review. These results must not be interpreted as autonomous diagnosis.

## Dataset

Primary cohort: `biopsy_patients`.

Input artifact:

```text
examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
```

Training target side is inferred from biopsy/status metadata for research
training only. Future prediction must receive target side from clinician/config
input.

## Model Definitions

M0:

```text
normalized radial_profile_data
-> LR1 profile LogisticRegression
-> target-breast patient logit-average p_cancer
```

M1:

```text
M0 p_cancer
+ SK symmetry block
-> final LogisticRegression
```

M2:

```text
M1 features
+ age
+ age_available
-> final LogisticRegression
```

SK means Slava Kubitskyi-style symmetry block. It replaces the earlier
current-cosine-only M1 feature set. Current cosine fields remain in the feature
table for audit/comparison, but M1/M2 no longer use them as the primary symmetry
block.

## SK Symmetry Features

```text
sk_meanrms1
sk_weightedrms1
sk_sigma_target1
sk_sigma_contralateral1
sk_mahalanobis1
sk_meanrms2
sk_weightedrms2
sk_sigma_target2
sk_sigma_contralateral2
sk_mahalanobis2
sk_peak14_intensity
sk_mean_peak_value
sk_wasserstein_distance_mu_tc
sk_cosine_distance_full_q2
sk_wasserstein_distance_full_q2
```

## Results

Metrics use the threshold selected from training data to target sensitivity
0.95. In honest split modes, realized test sensitivity can be lower because the
threshold is not optimized on the test set.

| mode | model | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|---:|
| 70/30 x50 | M0 | 0.574 +/- 0.061 | 0.738 +/- 0.122 | 0.359 +/- 0.141 |
| 70/30 x50 | M1 | 0.582 +/- 0.059 | 0.658 +/- 0.103 | 0.455 +/- 0.112 |
| 70/30 x50 | M2 | 0.623 +/- 0.059 | 0.702 +/- 0.108 | 0.470 +/- 0.111 |
| stratified 5-fold | M0 | 0.614 +/- 0.086 | 0.751 +/- 0.154 | 0.366 +/- 0.078 |
| stratified 5-fold | M1 | 0.630 +/- 0.094 | 0.693 +/- 0.144 | 0.470 +/- 0.104 |
| stratified 5-fold | M2 | 0.675 +/- 0.084 | 0.764 +/- 0.110 | 0.449 +/- 0.135 |
| LOOVM | M0 | 0.593 | 0.833 | 0.292 |
| LOOVM | M1 | 0.599 | 0.750 | 0.385 |
| LOOVM | M2 | 0.648 | 0.810 | 0.344 |
| train-all | M0 | 0.851 | 0.952 | 0.490 |
| train-all | M1 | 0.895 | 0.952 | 0.573 |
| train-all | M2 | 0.904 | 0.964 | 0.542 |

## Interpretation

SK M1 improves specificity over M0 in the tested honest modes, especially at
70/30 x50 and stratified 5-fold. This is better than the previous
current-cosine-only M1 result, where M1 did not improve M0 reliably.

M2 remains an age-audit branch. It improves ROC AUC, but age may encode clinical
prior probability rather than spectral information. It should stay separate
until age-only and age-confounding checks are finalized.

Train-all is a discovery ceiling only. It is useful for checking whether the
feature set can fit the cohort, but it is not validation.

## Artifacts

```text
docs/modeling/results/biopsy_primary_model_mode_comparison_sk_symmetry_v0_1.csv
docs/modeling/results/biopsy_target_sk_symmetry_features_v0_1.csv
docs/modeling/results/biopsy_target_sk_vs_current_symmetry_models_v0_1.csv
docs/modeling/results/biopsy_target_sk_vs_current_symmetry_single_features_v0_1.csv
```
