# Aramina T100 Target-Case Model 0.2.15-beta

Status: research-draft full-cohort candidate; not for autonomous diagnosis.

## Purpose

`0.2.15-beta` tests the established radial-profile architecture on all eligible
Human 1.1 data available in `20260827_combined_archive.h5`. The clinical
endpoint, label mapping, preprocessing gates, profile representation, symmetry
logic, and target-sensitivity threshold rule are unchanged from `0.2.14-beta`.

## Identity and lineage

```text
model_id: aramina_target_breast_risk_0_2_15-beta_43b2865632ea
model_sha256: 43b2865632ea4dc2387bd5c23b5cb25083a771a9b4b87afcd235989ed5fbcc1d
input_h5_sha256: e0034c7e2eb59d9a511e3a55e8af3b39e3ea95c0c14b6ddbe3ee06a70c053e9b
input_h5_dvc_md5: 15244bac1dbea063bc6461385773cf76
aramina_git_sha: d1c5d5ad8d193f84f3279a92c7a5ca9237dfef7e
xrd_preprocessing_version: 0.1.10b0
xrd_preprocessing_git_sha: 96c06ea28a88d40ab63ff39845f1748e0bf6a01a
```

XRD-preprocessing `0.1.10b0` adds explicit support for the current Human 1.1 H5
metadata layout. It resolves calibrant thickness from the linked Silver
Behenate acquisition when the archive-level field is absent and derives
P1/P2/P3 only for exactly three ordered `Sample Main` acquisitions.

## Cohort

After the fixed product filters and quality gates:

| Quantity | Value |
| --- | ---: |
| Measurements | 973 |
| Patients | 178 |
| Target-breast cases | 193 |
| CANCER target cases | 82 |
| BENIGN target cases | 111 |

Some patients contribute two target-breast cases. Evaluation splits are always
patient-safe, so both breasts from one patient remain in the same fold.

## Patient-safe evaluation

Repeated stratified patient-level `5-fold x20` evaluation:

| Metric | Mean | Fold SD |
| --- | ---: | ---: |
| ROC AUC | 0.66788 | 0.07502 |
| Sensitivity | 0.82483 | 0.10327 |
| Specificity | 0.37598 | 0.10810 |

The pooled held-out sensitivity is `0.85366`; pooled specificity is `0.33333`.
These values describe internal patient-safe evaluation, not an independent
clinical validation.

## Final train-on-all fit

The final fit uses all 193 target cases. Its threshold is selected to reach the
fixed training sensitivity target of `0.95`.

| Metric | Value |
| --- | ---: |
| Decision threshold | 0.2404104908 |
| Sensitivity | 0.95122 |
| Specificity | 0.47748 |
| ROC AUC | 0.85871 |
| Balanced accuracy | 0.71435 |

These are in-sample fit metrics and must not be interpreted as independent
performance.

## Independent-patient check of the preceding frozen model

Patients absent from the `0.2.14-beta` training manifest were scored with the
frozen `0.2.14-beta` artifact before fitting this candidate. After the current
product gates, the set contained 15 patients and 18 target-breast cases: 6
CANCER and 12 BENIGN. Results were sensitivity `1.0000`, specificity `0.5833`,
and ROC AUC `0.8333`. The corresponding Wilson 95% intervals were wide because
the class counts were small. This is an exploratory independent-patient check,
not definitive validation.

## Limitation

The current archive is not an FDA blind set. `0.2.15-beta` is intended for
controlled uncertainty experiments and internal decision-support research.
