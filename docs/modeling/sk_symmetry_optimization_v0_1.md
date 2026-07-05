# SK Symmetry Optimization v0.1

Clinical framing: research-draft decision support only; requires radiologist
review. This is a feature-discovery experiment, not clinical validation.

## Dataset

```text
input: examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
patients: 162
BENIGN patients: 84
CANCER patients: 78
```

## Feature Families Tested

```text
sk_base
sk_ratios
sk_windows
sk_windows_ratios
sk_windows_reliability
sk_all
```

Windowed SK metrics were computed over:

```text
{'q02_06': (2.0, 6.0), 'q06_10': (6.0, 10.0), 'q10_15': (10.0, 15.0), 'q13_16': (13.0, 16.0), 'q15_23': (15.0, 23.0), 'q02_23': (2.0, 23.0)}
```

## Best Discovery Feature Set

```text
sk_base
```

## Top Single Features

| feature | auc_raw | auc_oriented | direction |
| --- | --- | --- | --- |
| sk_q15_23_peak | 0.647 | 0.647 | 1 |
| sk_ratio_q15_23_to_q10_15_weightedrms | 0.639 | 0.639 | 1 |
| sk_logdiff_q15_23_minus_q10_15_weightedrms | 0.638 | 0.638 | 1 |
| sk_logdiff_q15_23_minus_q10_15_meanrms | 0.617 | 0.617 | 1 |
| sk_q10_15_cosine | 0.393 | 0.607 | -1 |
| sk_q13_16_peak | 0.603 | 0.603 | 1 |
| sk_logdiff_q10_15_minus_q06_10_cosine | 0.401 | 0.599 | -1 |
| sk_q10_15_wasserstein | 0.404 | 0.596 | -1 |
| sk_q10_15_weightedrms | 0.404 | 0.596 | -1 |
| sk_q10_15_meanrms | 0.406 | 0.594 | -1 |
| sk_q13_16_cosine | 0.409 | 0.591 | -1 |
| sk_logdiff_q10_15_minus_q06_10_weightedrms | 0.411 | 0.589 | -1 |
| sk_logdiff_q13_16_minus_q10_15_wasserstein | 0.586 | 0.586 | 1 |
| sk_logdiff_q10_15_minus_q06_10_meanrms | 0.414 | 0.586 | -1 |
| sk_ratio_q10_15_to_q06_10_weightedrms | 0.417 | 0.583 | -1 |

## Symmetry-only Best Rows

| feature_set | model_name | mode | roc_auc_mean | roc_auc_std | pr_auc_mean | sensitivity_mean | sensitivity_std | specificity_mean | specificity_std | threshold_mean | n_rows | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sk_base | LR_L2 | 70/30 x50 | 0.629 | 0.068 | 0.604 | 0.901 | 0.078 | 0.175 | 0.101 | 0.289 | 49 | 50 |
| sk_base | LR_elastic | 70/30 x50 | 0.627 | 0.067 | 0.604 | 0.904 | 0.083 | 0.171 | 0.102 | 0.300 | 49 | 50 |
| sk_windows | SVM_poly2 | 70/30 x50 | 0.627 | 0.068 | 0.636 | 0.782 | 0.098 | 0.318 | 0.123 | -0.143 | 49 | 50 |
| sk_base | LR_L1 | 70/30 x50 | 0.622 | 0.066 | 0.598 | 0.903 | 0.083 | 0.174 | 0.106 | 0.320 | 49 | 50 |
| sk_windows_reliability | SVM_poly2 | 70/30 x50 | 0.617 | 0.064 | 0.626 | 0.757 | 0.098 | 0.330 | 0.135 | -0.116 | 49 | 50 |
| sk_windows | SVM_poly2 | stratified 5-fold | 0.617 | 0.072 | 0.624 | 0.744 | 0.067 | 0.383 | 0.113 | -0.121 | 32 | 5 |
| sk_windows | RF_depth2 | stratified 5-fold | 0.615 | 0.121 | 0.631 | 0.720 | 0.155 | 0.346 | 0.241 | 0.438 | 32 | 5 |
| sk_windows_reliability | SVM_poly2 | stratified 5-fold | 0.613 | 0.065 | 0.617 | 0.732 | 0.058 | 0.371 | 0.078 | -0.094 | 32 | 5 |
| sk_windows | LR_L2 | stratified 5-fold | 0.611 | 0.046 | 0.612 | 0.679 | 0.090 | 0.407 | 0.132 | 0.380 | 32 | 5 |
| sk_base | LR_L2 | stratified 5-fold | 0.607 | 0.062 | 0.620 | 0.872 | 0.093 | 0.250 | 0.155 | 0.304 | 32 | 5 |
| sk_windows | RF_depth3 | train-all | 0.965 | 0.000 | 0.961 | 0.962 | 0.000 | 0.845 | 0.000 | 0.481 | 162 | 1 |
| sk_all | RF_depth3 | train-all | 0.954 | 0.000 | 0.953 | 0.962 | 0.000 | 0.833 | 0.000 | 0.469 | 162 | 1 |
| sk_windows_reliability | RF_depth3 | train-all | 0.954 | 0.000 | 0.945 | 0.962 | 0.000 | 0.821 | 0.000 | 0.448 | 162 | 1 |
| sk_windows_ratios | RF_depth3 | train-all | 0.945 | 0.000 | 0.943 | 0.962 | 0.000 | 0.738 | 0.000 | 0.456 | 162 | 1 |
| sk_all | SVM_poly2 | train-all | 0.919 | 0.000 | 0.897 | 0.962 | 0.000 | 0.548 | 0.000 | 0.065 | 162 | 1 |

## Profile Plus SK Candidates

| feature_set | model_name | mode | roc_auc_mean | roc_auc_std | pr_auc_mean | sensitivity_mean | sensitivity_std | specificity_mean | specificity_std | threshold_mean | n_rows | n_splits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| profile_plus_sk | M0_profile_only | 70/30 x50 | 0.503 | 0.059 | 0.515 | 0.683 | 0.109 | 0.329 | 0.122 | 0.390 | 49 | 50 |
| profile_plus_sk | M1_sk_base | 70/30 x50 | 0.539 | 0.061 | 0.546 | 0.623 | 0.101 | 0.437 | 0.111 | 0.321 | 49 | 50 |
| profile_plus_sk | M2_sk_base_age | 70/30 x50 | 0.577 | 0.056 | 0.587 | 0.669 | 0.093 | 0.438 | 0.105 | 0.293 | 49 | 50 |
| profile_plus_sk | M0_profile_only | stratified 5-fold | 0.516 | 0.079 | 0.531 | 0.717 | 0.098 | 0.312 | 0.123 | 0.390 | 32 | 5 |
| profile_plus_sk | M1_sk_base | stratified 5-fold | 0.530 | 0.058 | 0.541 | 0.667 | 0.048 | 0.394 | 0.078 | 0.335 | 32 | 5 |
| profile_plus_sk | M2_sk_base_age | stratified 5-fold | 0.576 | 0.053 | 0.600 | 0.693 | 0.045 | 0.418 | 0.130 | 0.311 | 32 | 5 |
| profile_plus_sk | M0_profile_only | train-all | 0.840 | 0.000 | 0.796 | 0.962 | 0.000 | 0.429 | 0.000 | 0.368 | 162 | 1 |
| profile_plus_sk | M1_sk_base | train-all | 0.885 | 0.000 | 0.860 | 0.962 | 0.000 | 0.607 | 0.000 | 0.312 | 162 | 1 |
| profile_plus_sk | M2_sk_base_age | train-all | 0.894 | 0.000 | 0.890 | 0.962 | 0.000 | 0.524 | 0.000 | 0.241 | 162 | 1 |

## Artifacts

```text
docs/modeling/results/sk_symmetry_optimization_features_v0_1.csv
docs/modeling/results/sk_symmetry_optimization_single_features_v0_1.csv
docs/modeling/results/sk_symmetry_optimization_symmetry_only_v0_1.csv
docs/modeling/results/sk_symmetry_optimization_profile_plus_sk_v0_1.csv
```
