# Aramis Data Preprocessing Contract v0.1

Status: research draft.

This document explains how Aramis converts EOS H5 XRD measurements into
model-input DataFrames. Aramis owns product YAMLs and output schemas.
`XRD-preprocessing` owns transformer implementations and YAML pipeline
construction.

## Data Levels

Every filter must have an explicit level:

```text
measurementId:
  one detector measurement / one 2D frame / one radial profile row

specimenId:
  one breast side
  all valid measurements from one specimen share the same product label

patientId:
  one patient
  may contain target and contralateral breast specimens
```

Splits for model validation are always patient-safe: one patient cannot appear
in both train and test.

## Current Product Preprocessing Configs

```text
config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
  primary model-development dataset
  keep patients with at least one biopsy-associated row
  keep contralateral rows for symmetry features
  map NORMAL to BENIGN
  apply T100 AgBH quality exclusions

config/preprocessing/aramis_prediction_patient_model_input_v0_1.yaml
  one incoming prediction patient
  no historical date, diagnosis, biopsy, or AgBH cohort filters
  stored inside trained model joblibs
```

The fixed M2Q development model uses the biopsy-patient preprocessing cohort.
Experimental all-patient cohorts remain outside the development branch.

## Product Label Mapping

Label grouping is specimenId-level:

```text
BENIGN -> BENIGN
NORMAL -> BENIGN
CANCER -> CANCER
ATYPICAL -> CANCER
PRE_CANCEROUS -> CANCER
NA -> excluded
```

Original `specimen_status` is retained for audit. Product labels are written to
`product_status_group`.

## H5-Level Filters

Early filtering happens before heavy raw-frame decoding when possible:

```text
linked_agbh_session_uid not in configured AgBH exclusions
fallback started_at date exclusion only when session UID is absent
PONI q max >= 23 nm^-1
sample_thickness_mm present
calibrant_thickness_mm present and in safety range
position in [P1, P2, P3]
SAMPLE/SAMPLE session and set categories
```

Reasoning:

```text
missing sample thickness makes thickness-corrected azimuthal integration invalid
missing calibrant thickness makes reference correction invalid
insufficient q range cannot produce the required model profile range
bad AgBH monochromaticity can shift/contaminate radial-profile interpretation
```

AgBH exclusion rationale:

```text
docs/agbh_quality_exclusions.md
```

## XRD Pipeline

Ordered YAML-declared route:

```text
H5PoniGeometryCalculatorTransformer
-> H5SessionSelectorTransformer
-> H5ToDataFrameTransformer
-> ProductColumnBuilder
-> q-range / position / thickness filters
-> optional biopsy / patient / specimen filters
-> ProductStatusGroupFilter
-> FaultyPixelDetector
-> ConstantQRangeTransformer
-> AzimuthalIntegration(error_model="poisson", thickness correction)
-> SNRTransformer(snr_method="poisson")
-> SNRFilter(min_snr_db=18.0)
-> PatientSpecimenValidityFilter
-> QRangeValueNormalizer(q_min=6.7, q_max=7.1, statistic="median")
-> RadialProfileValueFilter(q=14 nm^-1, value > 2.0)
-> KeepColumnsTransformer(metadata.output_columns)
```

Shared route file:

```text
config/preprocessing/shared/aramis_pipeline_v0_1.yaml
```

Shared policy file:

```text
config/preprocessing/shared/aramis_policy_v0_1.yaml
```

Its `io.input_h5_path` resolves to `../data/combined_archive.h5` relative to
the Aramis checkout. A standard workspace is therefore:

```text
workspace/
  Aramis/
  data/combined_archive.h5
```

## Fixed Numerical Choices

```text
raw source: gfrm
npt: 100
integration q range: 2..23 nm^-1
required PONI q max: 23 nm^-1
error model: poisson
sample thickness column: sample_thickness_mm
calibrant thickness column: calibrant_thickness_mm
calibrant thickness safety range: 2..40 mm
SNR method: poisson
SNR threshold: 18 dB
normalization: median value in q=6.7..7.1 nm^-1
profile gate: radial profile value near q=14 nm^-1 > 2.0
```

T100 is the current development compromise for AgBH monochromaticity filtering.
It keeps more data than T70 and excludes more questionable calibration days
than T130.

## Output DataFrame

Model-input rows are measurement-level profile rows. Heavy detector payloads
are dropped.

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

Prediction preprocessing additionally may keep:

```text
target_side
target_breast
mammography_suspicious_field
```

## Artifact Contract

`aramis preprocess` writes a joblib with:

```text
kind / version / created_at
dataframe
preprocessing_config_yaml   # fully resolved effective YAML
metadata.input_h5_sha256
metadata.aramis_version
metadata.aramis_git_sha
```

This artifact is the input to training. The same kind of artifact is written
during prediction preprocessing before scoring.

## Why Prediction Preprocessing Differs

Training preprocessing builds historical model-development cohorts and may use:

```text
AgBH quality exclusions
biopsy-patient cohort filters
diagnosis / label filters
```

Prediction preprocessing must not use historical cohort filters. It receives one
new patient H5 and applies only technical validity steps:

```text
thickness requirements
PONI/q-range requirements
faulty pixel detection
azimuthal integration
SNR filtering
normalization
profile gate
output schema
```

The suspicious breast side comes from prediction YAML, not from H5 labels.
