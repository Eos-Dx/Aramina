# KA vs Current Symmetry Metrics v0.1

Status: research draft. Not clinical validation.

Goal: compare the current Aramis cosine symmetry block with Kubitskii-style
profile symmetry metrics on the current primary biopsy cohort.

Dataset:

```text
examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
```

Endpoint:

```text
target breast CANCER vs BENIGN
inferred target breast = biopsied breast
```

Eligible rows for this comparison:

```text
patients with target and contralateral profiles: 162
BENIGN: 84
CANCER: 78
```

## Current Aramis Symmetry Block

Current training features:

```text
symmetry_available
between_breasts_cosine_distance_mean
target_within_cosine_distance_mean
contralateral_within_cosine_distance_mean
symmetry_cosine_score
target_measurements
contralateral_measurements
```

Main formula:

```text
symmetry_cosine_score =
  between_breasts_cosine_distance_mean
  - mean(target_within_cosine_distance_mean,
         contralateral_within_cosine_distance_mean)
```

## Kubitskii-Style Profile Metrics Tested Here

Profiles are smoothed, normalized near q=6.7, and averaged by side. Metrics are
computed for inferred target breast vs contralateral breast.

Windows:

```text
window 1: q 7..15
window 2: q 15..23
full q2: q 2..23
```

Metrics:

```text
meanrms1 / meanrms2:
  RMS difference between target and contralateral mean profiles

weightedrms1 / weightedrms2:
  RMS difference weighted by target/contralateral replicate variance

sigma_target1/2 and sigma_contralateral1/2:
  replicate variability within each breast

mahalanobis1 / mahalanobis2:
  profile difference scaled by replicate variance

wasserstein_distance_muTC:
  Wasserstein distance between target and contralateral mean profiles

cosine_distance_full_q2:
  cosine distance between target and contralateral mean profiles over q 2..23

peak14_intensity and mean_peak_value_raw:
  peak-region intensity audit features
```

## Result

Symmetry-only model comparison:

| feature set | model | mode | ROC AUC | sensitivity | specificity |
|---|---|---|---:|---:|---:|
| current | LR_L2 | 70/30 x50 | 0.542 +/- 0.066 | 0.958 | 0.126 |
| current | SVM_poly2 | 70/30 x50 | 0.589 +/- 0.062 | 0.958 | 0.143 |
| ka | LR_L2 | 70/30 x50 | 0.610 +/- 0.060 | 0.958 | 0.222 |
| ka | SVM_poly2 | 70/30 x50 | 0.579 +/- 0.075 | 0.958 | 0.194 |
| current+ka | LR_L2 | 70/30 x50 | 0.588 +/- 0.061 | 0.958 | 0.201 |
| current+ka | SVM_poly2 | 70/30 x50 | 0.567 +/- 0.080 | 0.958 | 0.194 |

Train-all ceiling:

| feature set | model | ROC AUC | specificity at sensitivity 0.95 |
|---|---|---:|---:|
| current | LR_L2 | 0.627 | 0.190 |
| current | SVM_poly2 | 0.680 | 0.190 |
| ka | LR_L2 | 0.719 | 0.345 |
| ka | SVM_poly2 | 0.778 | 0.405 |
| current+ka | LR_L2 | 0.731 | 0.357 |
| current+ka | SVM_poly2 | 0.780 | 0.345 |

## Interpretation

The current cosine block is weak for target CANCER vs BENIGN. Kubitskii-style
profile metrics carry more signal on the same biopsy cohort, especially with
LogisticRegression under repeated patient-safe 70/30.

This suggests M1 should not use only the current `symmetry_cosine_score` block.
Next iteration should add a compact KA-style symmetry block to the Aramis
training feature schema and compare:

```text
M1_current_cosine
M1_KA_profile_metrics
M1_current_plus_KA
```

Machine-readable outputs:

```text
docs/modeling/results/biopsy_target_ka_symmetry_features_v0_1.csv
docs/modeling/results/biopsy_target_ka_vs_current_symmetry_single_features_v0_1.csv
docs/modeling/results/biopsy_target_ka_vs_current_symmetry_models_v0_1.csv
```
