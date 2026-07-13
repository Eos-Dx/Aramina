# SK Symmetry Features v0.1

Status: research draft.

This document defines the SK target/contralateral symmetry calculations. The
current product M2Q schema uses only the four fields marked **Core4** below.

Code source:

```text
src/aramis/training.py
_patient_model_feature_columns()
_sk_symmetry_columns()
_sk_target_contralateral_symmetry_features()
```

## Input

For each patient:

```text
q_i: q values, nm^-1
y_i(q): normalized radial_profile_data
side_i: Left or Right
target_side: suspicious breast side
contralateral_side: opposite breast side
```

For training, target side is inferred from biopsy/status metadata. For
prediction, target side must be supplied in prediction YAML.

## Internal Profile Preparation

For target and contralateral breast profiles:

```text
1. restrict q range
2. smooth profile with Savitzky-Golay filter
3. normalize by near-minimum baseline around q = 6.7 +/- 0.25 nm^-1
4. interpolate to common q grid
5. compute side mean and side standard deviation
```

Notation:

```text
T = target breast profile set
C = contralateral breast profile set
mu_T(q) = mean target profile
mu_C(q) = mean contralateral profile
sigma_T(q) = target profile standard deviation across measurements
sigma_C(q) = contralateral profile standard deviation across measurements
```

When both target and contralateral profiles are available, Core4 measures their
asymmetry and refines the one final LR2 score. If the contralateral profile is
absent, `symmetry_available = 0` makes every SK contribution neutral.
`symmetry_available` is a gate and audit field, not a learned model input.

## Q Regions

```text
internal SK ROI: 7.5..23.0 nm^-1
region 1: 7.0..15.0 nm^-1
region 2: 15.0..23.0 nm^-1
full_q2 ROI: 2.0..23.0 nm^-1
```

`full_q2` means full q range starting from `2.0 nm^-1`. It does not mean
q-squared.

## Feature Definitions

### profile_p_cancer_logit_average

Target-breast patient-level profile score from LR1.

```text
logit_i = log(p_i / (1 - p_i))
profile_p_cancer_logit_average = sigmoid(mean(logit_i))
```

### symmetry_available

```text
1 if both target and contralateral breast profiles are present
0 otherwise
```

This field is not a model input for the current product M2Q.

### sk_meanrms1

RMS difference between target and contralateral mean profiles in region 1:

```text
sqrt(mean((mu_T(q) - mu_C(q))^2)), q in 7.0..15.0
```

### sk_weightedrms1 (Core4)

Variance-weighted RMS difference in region 1:

```text
var(q) = sigma_T(q)^2 + sigma_C(q)^2
weight(q) = 1 / max(var(q), percentile_5(var) + 1e-12)
sqrt(sum(weight(q) * (mu_T(q) - mu_C(q))^2) / sum(weight(q)))
```

This downweights q points where replicate variability is high.

### sk_sigma_target1

Target-breast replicate variability in region 1:

```text
sqrt(mean(sigma_T(q)^2)), q in 7.0..15.0
```

### sk_sigma_contralateral1

Contralateral-breast replicate variability in region 1:

```text
sqrt(mean(sigma_C(q)^2)), q in 7.0..15.0
```

### sk_mahalanobis1

Mahalanobis-like distance in region 1:

```text
sqrt(sum((mu_T(q) - mu_C(q))^2 / (sigma_T(q)^2 + sigma_C(q)^2 + 1e-12)))
```

### sk_meanrms2

Same as `sk_meanrms1`, but in region 2:

```text
q in 15.0..23.0 nm^-1
```

### sk_weightedrms2 (Core4)

Same as `sk_weightedrms1`, but in region 2:

```text
q in 15.0..23.0 nm^-1
```

### sk_sigma_target2

Same as `sk_sigma_target1`, but in region 2:

```text
q in 15.0..23.0 nm^-1
```

### sk_sigma_contralateral2

Same as `sk_sigma_contralateral1`, but in region 2:

```text
q in 15.0..23.0 nm^-1
```

### sk_mahalanobis2

Same as `sk_mahalanobis1`, but in region 2:

```text
q in 15.0..23.0 nm^-1
```

### sk_peak14_intensity_abs_delta

Absolute difference of target and contralateral mean-profile maxima around the
main peak:

```text
abs(max(mu_T(q)) - max(mu_C(q))), q in 13.5..14.5 nm^-1
```

### sk_mean_peak_value_abs_delta (Core4)

Absolute difference of side-specific means of per-measurement peak maxima:

```text
for each target measurement y_i:
  target_peak_i = max(y_i(q)), q in 13.0..14.8 nm^-1

for each contralateral measurement y_j:
  contralateral_peak_j = max(y_j(q)), q in 13.0..14.8 nm^-1

sk_mean_peak_value_abs_delta =
  abs(mean(target_peak_i) - mean(contralateral_peak_j))
```

This is a true target/contralateral asymmetry feature.

### sk_wasserstein_distance_mu_tc

Wasserstein-like distance between target and contralateral mean profiles in the
internal SK ROI:

```text
ROI: 7.5..23.0 nm^-1
a(q) = clip(mu_T(q), 0, inf)
b(q) = clip(mu_C(q), 0, inf)
a_norm = a / sum(a)
b_norm = b / sum(b)

sum(abs(cumsum(a_norm) - cumsum(b_norm)) * delta_q)
```

`mu_tc` means:

```text
mu = mean profile
t = target
c = contralateral
```

### sk_cosine_distance_full_q2

Cosine distance between target and contralateral mean profiles on the wider q
range:

```text
ROI: 2.0..23.0 nm^-1
1 - cosine_similarity(mu_T, mu_C)
```

### sk_wasserstein_distance_full_q2 (Core4)

Same Wasserstein-like distance as `sk_wasserstein_distance_mu_tc`, but on the
wider q range:

```text
ROI: 2.0..23.0 nm^-1
```

## Difference Between `mu_tc` And `full_q2`

```text
sk_wasserstein_distance_mu_tc:
  profile: mu_target vs mu_contralateral
  q range: 7.5..23.0 nm^-1
  purpose: target/contralateral shape shift in SK internal ROI

sk_wasserstein_distance_full_q2:
  profile: mu_target vs mu_contralateral
  q range: 2.0..23.0 nm^-1
  purpose: target/contralateral shape shift including low-q region
```

Both are Wasserstein-like distances between side mean profiles. The difference
is the q range.
