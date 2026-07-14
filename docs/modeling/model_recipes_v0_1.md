# Model Recipes v0.1

Code-owned recipe registry: `src/aramis/model_recipes.yaml`.

## m2q_gated_target_case_v0_1

```text
training unit: biopsied target breast
patient split unit: patientId
LR1 input: normalized radial_profile_data
LR1 regularization: LogisticRegression L2, C=0.1
aggregation: target-breast logit-average p_cancer
LR2 input: profile score + age + gated SK Core4 symmetry
LR2 regularization: LogisticRegression L2, C=0.3
selected output model: M2Q
target sensitivity: 0.95
```

SK Core4:

```text
sk_wasserstein_distance_full_q2
sk_weightedrms1
sk_weightedrms2
sk_mean_peak_value_abs_delta
```

When contralateral data are unavailable, symmetry is marked unavailable and its
standardized contribution is forced to zero. Reliability remains report
metadata, not a learned risk feature. Mathematical definitions are in
`docs/modeling/sk_symmetry_features_v0_1.md`.
