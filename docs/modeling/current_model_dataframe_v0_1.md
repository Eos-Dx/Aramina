# Current Aramis Model DataFrame v0.1

Status: research draft. This document describes the current DataFrame used by
the Aramis model-development code. It is not clinical validation.

## Current Training Cohort

The current model-grid training YAMLs use the biopsy-patient model-input
DataFrame:

```text
config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
```

Training configs point to this artifact, for example:

```text
config/training/aramis_biopsy_patients_m0_m1_m2_v0_1.yaml
config/training/model_grid_v0_1/18_m2_stratified_kfold_biopsy_patients_model_v0_1.yaml
```

Current counts:

| item | count |
|---|---:|
| measurement rows | 968 |
| patients | 180 |
| specimens / breasts | 342 |
| patients with two breasts | 162 |
| patients with one breast | 18 |
| DataFrame columns | 30 |

## Cohort Rule

The current cohort is biopsy-patient based:

```text
include patient if at least one breast/specimen has biopsy=True
keep contralateral breast rows for that patient
do not require both breasts to be biopsy-positive
do not require two breasts for one-to-many profile modeling
```

This cohort is intended for the first research-draft Aramis model because
biopsy-confirmed patients provide a cleaner endpoint than the broader mixed
clinical metadata cohort. It is still a selected subgroup and may contain
biopsy-workflow selection bias.

## Row, Specimen, And Patient Levels

The DataFrame is measurement-level:

```text
one row = one cleaned XRD measurement position
patientId = patient identifier
specimenId = breast/specimen identifier
side = Left or Right breast
position = measurement position within breast
```

The training code then builds patient-level features:

```text
measurement rows
-> specimen/breast profile scores
-> patient-level profile logit-average
-> inferred target breast from biopsy/status metadata
-> optional target/contralateral symmetry features
-> optional age feature
-> patient-level p_cancer research score
```

Split-based evaluation is patient-safe: rows from one `patientId` must not be
split between train and test.

## Label Mapping

Original specimen statuses in the current DataFrame:

| specimen_status | specimens / breasts |
|---|---:|
| BENIGN | 163 |
| NORMAL | 91 |
| CANCER | 68 |
| PRE_CANCEROUS | 12 |
| ATYPICAL | 8 |

Product label mapping:

```text
BENIGN + NORMAL -> BENIGN
CANCER + ATYPICAL + PRE_CANCEROUS -> CANCER
```

Mapped product groups:

| product_status_group | specimens / breasts | measurement rows |
|---|---:|---:|
| BENIGN | 254 | 714 |
| CANCER | 88 | 254 |

## Biopsy Flags

Biopsy availability in the current model-input DataFrame:

| biopsy | specimens / breasts | measurement rows |
|---|---:|---:|
| True | 193 | 546 |
| False | 149 | 422 |

The `False` rows are mainly contralateral context rows retained because the
patient has at least one biopsy-positive breast/specimen.

## Columns

Current model-input columns:

```text
patientId
specimenId
side
position
started_at
measurementDate
specimen_status
product_status_group
product_diagnosis
patient_product_diagnosis
age
biopsy
sample_biopsy
sample_biopsy_type
sample_height_in
sample_weight_lb
breast_density
birads
sample_thickness_mm
calibrant_thickness_mm
poni_q_max_nm_inv
measurement_data_source
q_range
radial_profile_data
snr_db
specimen_measurement_count
radial_profile_value_at_q
radial_profile_nearest_q_nm_inv
radial_profile_q_delta_nm_inv
radial_profile_value_pass
```

Model-critical columns:

```text
patientId
specimenId
side
product_status_group
biopsy
age
radial_profile_data
snr_db
sample_thickness_mm
```

Other columns are metadata, provenance, or quality-control context.

## Wide-Pool Audit Cohort

For patient-pair inventory and sanity checks, a separate wide cleaned artifact
was used:

```text
examples/outputs/real_h5_yaml_validation/aramis_monochromatic_metadata_pool_max_v0_1.joblib
examples/outputs/patient_diagnosis_pairs_v0_1/README.md
```

The biopsy-patient subset derived from that wide pool has:

| item | count |
|---|---:|
| measurement rows | 893 |
| patients | 164 |
| specimens / breasts | 314 |
| patients with two breasts | 150 |
| patients with one breast | 14 |

This wide-pool subset is useful for auditing breast-pair composition. It is not
yet the artifact used by the current training YAMLs. If we decide that this
wide-derived cohort is the canonical model cohort, preprocessing YAMLs and the
model grid must be updated and rerun.

## Current Modeling Decision

Current model-development default:

```text
primary cohort: aramis_biopsy_patients_model_input_v0_1.joblib
patients: 180
endpoint: BENIGN vs CANCER decision-support p_cancer
row unit: measurement
grouping unit: patientId
decision-support level under discussion: target breast / patient workflow
status: research draft, requires radiologist review
```

Training target-side rule:

```text
primary cohort: biopsy_patients
inferred target breast: biopsied breast
reason: biopsied breast is the clinically suspicious breast and has endpoint
prediction target breast: must be supplied by clinician/config, not inferred
```

LR1 aggregation rule:

```text
measurement p_cancer -> logit(p_cancer) -> mean logit -> sigmoid(mean logit)
model field: profile_p_cancer_logit_average
audit field: profile_p_cancer_probability_mean
```
