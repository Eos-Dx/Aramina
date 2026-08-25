# Aramina T100 Target-Case Model 0.2.14-beta

Status: research-draft DVC-tracked candidate. Not an independent clinical
validation and not for autonomous diagnosis.

## Purpose

`0.2.14-beta` preserves the `0.2.13-beta` XRD preprocessing and target-breast
model recipe. The change is reproducibility infrastructure: historical input
data are versioned by DVC, standalone training requires DVC lineage, and one
MLflow run records preprocessing, patient-safe evaluation, final train-on-all,
and the executable artifact.

```text
model_id: aramina_target_breast_risk_0_2_14-beta_98526329f40d
model_sha256: 98526329f40dc4fc379d4278bec75005c3c0e598cd65e126c68d6b875d6ac479
Aramina source: f402662f56a7fd2e6215c7067a4fc81448f1c339
XRD-preprocessing: 0.1.9b0, commit 88dcaa277c5a0d4be2ab637bc5827a14bd106bea
MLflow run: d9f7c80bff874f8b812fa35f0e0e3316
```

## Data Identity

```text
DVC contract: aramina_dvc_input_v0_1
dataset_id: aramina_combined_archive_h5
DVC MD5: 46e199e316e95969731d61d8ab4b2c52
H5 SHA256: d2d61e83850b282c3d2479ea436deed821c4488b96983252d294f3d56ee3f1f9
H5 size: 3663436764 bytes
dataset fingerprint: 8ffb7957ca057fe2e3a7b3c149ba118d34bc2d924dbaddaecebb88caadc396e5
```

Accepted cohort: 893 measurements, 164 patients, 314 breasts/specimens, and
175 target-breast cases: 76 CANCER and 99 BENIGN. Eleven patients contribute
two biopsied target breasts. Splits remain patient-safe.

## Fixed Recipe

```text
GFRM -> fixed T100 quality policy
-> 100-point azimuthal integration over 2-23 nm^-1, Poisson errors
-> Poisson SNR >=18 dB
-> median normalization over 6.7-7.1 nm^-1
-> profile logistic regression, C=0.1
-> age and optional gated Core4 symmetry refinement, C=0.3
-> fixed target-sensitivity threshold
```

No preprocessing, feature, regularization, label, target-sensitivity, TRA, or
report decision policy changed relative to `0.2.13-beta`.

## Patient-Safe Evaluation

Repeated stratified 5-fold x20, grouped by patient:

| Metric | Mean | SD |
|---|---:|---:|
| ROC AUC | 0.64541 | 0.06874 |
| Sensitivity | 0.81750 | 0.09902 |
| Specificity | 0.37630 | 0.13254 |

Pooled patient-safe sensitivity is 0.82895 with bootstrap interval
0.74074-0.91026. Pooled specificity is 0.32323 with interval
0.22806-0.42003. These are internal patient-safe estimates, not performance on
an independently collected blind cohort.

## Final Train-On-All Artifact

The executable artifact uses threshold `0.24666`. In-sample values are ROC AUC
0.86497, sensitivity 0.96053, and specificity 0.49495. They describe the fitted
artifact and must not be presented as independent validation.

## Compatibility

Tag `0.2.13-beta` and its legacy YAMLs remain the no-DVC implementation
baseline. Current code can execute frozen `0.2.13-beta` prediction payloads,
but current historical training requires the v0.2 preprocessing, v0.4 training,
and v0.3 combined contracts.
