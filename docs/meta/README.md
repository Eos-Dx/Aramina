# Aramis Human-1 Product Metadata

This directory contains canonical JSON/CSV metadata for the Aramis Human-1
research draft product workflow.

These files are product metadata prepared for Aramis by Slava Shcherbakov
(Viacheslav SHCHERBAKOV). Treat them as controlled product inputs, not as
ad-hoc notebook outputs.

Aramis is clinical decision support research draft work. These metadata do not
make the model clinically validated, FDA-cleared, or suitable for autonomous
diagnosis.

## Files

This folder contains reference metadata and audit artifacts. Runtime product
preprocessing YAML files live in:

```text
Aramis/config/preprocessing/
```

Current runnable preprocessing YAMLs:

```text
config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
config/preprocessing/aramis_prediction_patient_model_input_v0_1.yaml
```

They compose smaller fragments:

```text
shared/aramis_policy_v0_1.yaml      GFRM-only product policy
shared/aramis_pipeline_v0_1.yaml    ordered transformer steps
exclusions/agbh_quality_exclusions_t100_v0_1.yaml
outputs/model_input_output_v0_1.yaml
outputs/prediction_model_input_output_v0_1.yaml
```

Runnable root YAMLs extend the shared policy, pipeline, output schema, and
exclusion fragments. Historical training eligibility is declared explicitly in
the product root YAML under `product_filter`.

Short file map:

```text
aramis_product_versioning.json
  controlled Human-1 batch, K-alpha/K-beta, Nova range, and calibrant-thickness metadata

aramis_preprocessing_v0_1_config.json
  machine-readable AgBH monochromaticity exclusion artifact consumed by preprocessing YAMLs

aramis_agbh_kbeta_batch5_6_exclusion_justification_v0_1.py
  marimo evidence notebook for AgBH K-beta shoulder review of batches 5 and 6

aramis_agbh_kbeta_helpers.py
  helper module required by the AgBH K-beta evidence notebook

human1_diagnoses_metadata.json
  canonical Human-1 clinical metadata normalized from the source Excel workbook

human1_diagnoses_metadata_h5_audit.json
  audit summary comparing canonical metadata JSON against combined_archive.h5

human1_diagnoses_metadata_h5_mismatches.csv
  row-level mismatch table for H5-vs-canonical-metadata review

sample_thickness_h5_backfill_2026_07_01.csv
  row-level audit of manual H5 sample-thickness backfill from the 26 Jun 2026 CSV

sample_thickness_h5_backfill_2026_07_01.json
  summary of missing sample thickness before/after H5 backfill and remaining blanks
```

### `aramis_product_versioning.json`

Purpose:

```text
Human-1 versioning
data_batch definitions
K-alpha / K-beta source-line rules
Nova patient ranges
AgBH reference thickness rules
calibrant_thickness_mm field contract
required H5 metadata fields
product filtering policy
```

Use this file when deciding whether a measurement batch is product-usable for a
K-alpha-only Aramis workflow.

### `config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml`

Purpose:

```text
Aramis biopsy-patients model-input preprocessing config
row unit: measurementId
grouping unit: specimenId
decision unit: patientId during model selection
patient-level biopsy policy
keep contralateral rows for symmetry features
NORMAL -> BENIGN product-label policy
same technical XRD preprocessing route as the primary cohort
```

This is the current primary development training dataset. Biopsy selection is
patient-level: keep patients with at least one `biopsy == true` row, then keep
both breast sides when available for symmetry/asymmetry feature generation. At
training time, each biopsied breast becomes one target case; bilateral cases are
never separated across patient-safe folds.

### `config/preprocessing/aramis_prediction_patient_model_input_v0_1.yaml`

Purpose:

```text
Aramis prediction preprocessing config embedded in trained model joblibs
incoming H5 must contain one patient
no historical date/diagnosis/biopsy/AgBH cohort filters
row unit: measurementId
grouping unit: specimenId
raw detector source policy: gfrm only
thickness correction requirements
SNR / normalization / profile-gate parameters
```

Prediction config is normally not run directly. It is stored in trained model
joblibs and used by `python -m aramis predict --config <predict.yaml>`.

Reusable preprocessing YAML template/contract is owned by XRD-preprocessing:

```text
XRD-preprocessing/src/xrd_preprocessing/configs/preprocessing_pipeline_config_template.yaml
```

These files are the concrete Aramis product configs that follow that template.
Each preprocessing YAML owns its own runtime paths:

```text
io.input_h5_path
io.output_joblib_path
```

The product command should receive only the YAML path:

```text
python -m aramis preprocess --config config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
```

Current XRD-preprocessing dependency marker:

```text
version: local
release_tag: v0.1.7-beta
```

Raw-data policy:

```text
Aramis v0.1 product preprocessing uses only GFRM vendor bytes from H5 blobs.
Allowed product source: gfrm.
Allowed product H5 blob candidates: raw_file, artifacts/gfrm.
NPY is allowed only in synthetic tests; TIFF is not used in this product version.
```

### `aramis_preprocessing_v0_1_config.json`

Purpose:

```text
Aramis AgBH monochromaticity product-selection audit artifact
rejected AgBH session IDs
rejected AgBH dates for older-container fallback
AgBH shoulder-metric threshold
detector-distance/q-range eligibility policy
reference AgBH rows
calibrant-thickness policy used by downstream preprocessing notebooks
selection_contract explaining how exclusions were produced and consumed
```

Canonical location:

```text
Aramis/docs/meta/aramis_preprocessing_v0_1_config.json
```

The runtime preprocessing configs are the Aramis product YAML files. Their
`filters.quality_exclusions` blocks hold the controlled exclusion lists. This
JSON explains how those lists were produced.

This config was generated from:

```text
Clinical_trials/Product/Aramis/Aramis_Preprocessing_v0_1.py
```

Initial exported artifact:

```text
Clinical_trials/analysis/aramis_preprocessing_v0_1/aramis_preprocessing_v0_1_config.json
```

The JSON carries its own `purpose`, `provenance`, and `selection_contract`
blocks with notebook path, documentation links, generation summary, rejected
session IDs, rejected-date fallback, and downstream consumers.

Exclusion rationale:

```text
Aramis/docs/agbh_quality_exclusions.md
```

Used by:

```text
Aramis/config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
Aramis/config/preprocessing/aramis_prediction_patient_model_input_v0_1.yaml
```

### `aramis_agbh_kbeta_batch5_6_exclusion_justification_v0_1.py`

Purpose:

```text
marimo evidence notebook for excluded AgBH-linked session dates
focused review of AgBH K-beta shoulder behavior in batches 5 and 6
visual comparison against batch-7 reference-like AgBH profiles
human-readable justification for exclusion policy updates
```

This notebook is a meta/provenance artifact, not a runtime preprocessing step.
Runtime exclusions are still stored in the YAML files under:

```text
Aramis/config/preprocessing/exclusions/agbh_quality_exclusions_t100_v0_1.yaml
```

Use this notebook when explaining why particular AgBH calibration sessions or
fallback dates were excluded from Aramis preprocessing. If the exclusion list is
changed after reviewing the notebook, update:

```text
config/preprocessing/exclusions/agbh_quality_exclusions_t100_v0_1.yaml
docs/agbh_quality_exclusions.md
docs/meta/aramis_preprocessing_v0_1_config.json
MLflow preprocessing artifacts for affected runs
```

### `aramis_agbh_kbeta_helpers.py`

Purpose:

```text
helper functions for AgBH/K-beta notebook
technical H5 calibration discovery
AgBH GFRM integration helpers
K-beta shoulder metric calculations
batch-level plots and exported audit tables
```

This file is kept beside
`aramis_agbh_kbeta_batch5_6_exclusion_justification_v0_1.py` so the marimo
notebook can import it directly.

Regeneration rule:

```text
regenerate with Aramis_Preprocessing_v0_1.py or equivalent scripted export
update the JSON provenance block
rerun Aramis tests and marimo checks
rebuild aramis_docker_training_bundle_0_2_8_beta.zip when full-H5
reproducibility material changes
```

Thickness policy:

```text
filters.thickness.sample.column
  H5/sample attribute used to require specimen thickness before frame loading

filters.thickness.calibrant.column
  H5/session attribute used to require calibrant thickness before frame loading

integration.thickness_correction.sample_thickness_column
integration.thickness_correction.calibrant_thickness_column
  must match the filter columns
  these names are passed directly to AzimuthalIntegration
```

Current AgBH calibrant safety range is `2..40 mm`. Missing sample thickness,
missing calibrant thickness, or calibrant thickness outside this range means the
measurement cannot enter thickness-corrected azimuthal integration.

Current conservative rule recorded in JSON:

```text
include data_batch: 3, 4, 5, 7
exclude data_batch: 1, 2, 6
review required: null / unknown
```

Important: AgBH shoulder-metric review can supersede this draft batch policy.
If batch 5 is confirmed bad for the product workflow, update this JSON first,
then update downstream notebooks, docs, and MLflow artifacts.

Calibrant thickness rule recorded in JSON:

```text
before 2026-04-22: 40 mm
from 2026-04-22: 10 mm
preferred field: calibrant_thickness_mm
```

### `human1_diagnoses_metadata.json`

Purpose:

```text
canonical Human-1 clinical metadata
diagnosis labels
patient/specimen metadata
BI-RADS fields
MRI / biopsy fields
specimen status
source Excel normalization
external_id -> patient/side/position parsing
```

Source workbook:

```text
/Users/sad/Downloads/Human-1 Diagnoses for Matador v4(5).xlsx
```

Do not use the Excel file directly in product notebooks. Product code should
read this JSON. If the Excel source changes, regenerate this JSON and rerun the
H5 audit.

Key contract:

```text
External ID format:
Nova_<patient_number>_<LEFT|RIGHT>_P<point_number>

Comparison key:
patient_external_id | side | position
```

### `human1_diagnoses_metadata_h5_audit.json`

Purpose:

```text
audit of JSON clinical metadata against combined_archive.h5
matched measurement keys
Excel-only keys
H5-only keys
duplicate keys
metadata mismatch summary
```

Use this file before building a product dataset from H5. It shows where H5
metadata differs from the canonical clinical metadata JSON.

Current audit summary:

```text
excel rows: 2086
excel unique measurement keys: 2083
h5 unique measurement keys: 2001
matched keys: 1996
excel-only keys: 87
h5-only keys: 5
mismatch rows: 779
mismatched keys: 558
```

### `human1_diagnoses_metadata_h5_mismatches.csv`

Purpose:

```text
row-level mismatch table for H5 vs canonical JSON
external_id_key
field
Excel value
H5 value
source row
H5 session path
H5 set name
```

Use this file to decide which mismatches are harmless normalization differences
and which require H5 backfill or product-filter changes.

### `sample_thickness_h5_backfill_2026_07_01.csv`

Purpose:

```text
audit table for manual sample-thickness backfill into combined_archive.h5
one row per updated H5 measurement set
records old/new thickness, set_path, patientId, specimenId, side, position
source CSV: 26Jun26_missing_sample_thickness_in_Nova_study(in).csv
```

The H5 update writes `thickness_raw_mm` into the affected measurement set attrs.
`xrd_preprocessing` promotes this to `sample_thickness_mm` when reading the H5.

### `sample_thickness_h5_backfill_2026_07_01.json`

Purpose:

```text
summary of H5 sample-thickness backfill
missing rows before update
CSV rows with numeric thickness
rows updated
remaining rows without sample thickness
```

Current summary:

```text
missing before: 21
numeric CSV thickness rows: 14
backfilled rows: 14
remaining missing after update: 7
```

## Product Rules

Canonical metadata flow:

```text
Excel/source document
-> canonical JSON
-> H5 audit
-> product filter
-> h5_to_df
-> preprocessing
-> MLflow dataset artifact
```

Do not silently change:

```text
data_batch policy
K-alpha / K-beta policy
AgBH thickness rule
diagnosis label mapping
BI-RADS fields
patient/specimen identifiers
H5-vs-JSON mismatch handling
```

Any change must update:

```text
JSON metadata
docs
notebooks/helpers
selected/dropped measurement manifests
MLflow artifacts
```
