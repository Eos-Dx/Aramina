# SK vs Current Symmetry Metrics v0.1

Clinical framing: research-draft decision support only; requires radiologist
review. These results are feature-discovery evidence, not clinical validation.

## Goal

Compare two patient-level symmetry feature families on the current primary
biopsy cohort:

```text
current cosine block
SK symmetry block
current cosine + SK combined block
```

SK means Slava Kubitskyi-style symmetry features. Earlier draft files used the
temporary name `KA`; new Aramis training code and result references should use
`SK`.

## Result Summary

Symmetry-only, CANCER vs BENIGN target breast, patient-safe 70/30 x50:

| feature set | model | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|---:|
| current cosine | LR | 0.542 +/- 0.066 | 0.958 | 0.126 |
| SK | LR | 0.610 +/- 0.060 | 0.958 | 0.222 |
| current + SK | LR | 0.588 +/- 0.061 | 0.958 | 0.201 |
| current cosine | SVM poly2 | 0.589 +/- 0.062 | 0.958 | 0.143 |
| SK | SVM poly2 | 0.579 +/- 0.075 | 0.958 | 0.194 |
| current + SK | SVM poly2 | 0.567 +/- 0.080 | 0.958 | 0.194 |

The compact SK block is more useful than the current cosine-only block for the
symmetry-only logistic model. Simple concatenation of current cosine and SK
features did not improve the honest 70/30 result.

## Product Decision

Use SK symmetry block as the default M1/M2 symmetry block in the current Aramis
research-draft training route.

Keep current cosine fields in the patient feature table for audit and
comparison, but do not use them as the primary M1 feature schema.

## Artifacts

```text
docs/modeling/results/biopsy_target_sk_symmetry_features_v0_1.csv
docs/modeling/results/biopsy_target_sk_vs_current_symmetry_single_features_v0_1.csv
docs/modeling/results/biopsy_target_sk_vs_current_symmetry_models_v0_1.csv
```
