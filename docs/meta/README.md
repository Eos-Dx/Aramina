# Controlled Human-1 Metadata And Evidence

This folder preserves controlled inputs and audit evidence for the Aramina
research draft. Runtime product behavior is defined by `config/`, not by files
in this folder.

## File Map

| File | Role |
|---|---|
| `aramina_product_versioning.json` | Human-1 batches, source-line policy, Nova ranges, thickness metadata, and required H5 fields. |
| `aramina_preprocessing_v0_1_config.json` | Machine-readable AgBH monochromaticity decision evidence. |
| `aramina_agbh_kbeta_batch5_6_exclusion_justification_v0_1.py` | Marimo review of AgBH K-beta behavior. |
| `aramina_agbh_kbeta_helpers.py` | Compatibility import used by the notebook. |
| `agbh_kbeta/` | H5 integration, metrics, artifacts, and plotting helpers. |
| `human1_diagnoses_metadata.json` | Canonical normalized Human-1 clinical metadata. |
| `human1_diagnoses_metadata_h5_audit.json` | Summary of canonical-metadata versus H5 comparison. |
| `human1_diagnoses_metadata_h5_mismatches.csv` | Row-level mismatch evidence. |
| `sample_thickness_h5_backfill_2026_07_01.csv` | Row-level sample-thickness backfill audit. |
| `sample_thickness_h5_backfill_2026_07_01.json` | Backfill summary. |

## Runtime Owners

```text
config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml
config/preprocessing/config_preprocessing_prediction_patient_v0_2.yaml
config/preprocessing/exclusions/agbh_quality_exclusions_t100_v0_1.yaml
config/preprocessing/shared/aramina_policy_v0_1.yaml
config/preprocessing/shared/aramina_pipeline_v0_1.yaml
```

Training preprocessing keeps patients with at least one biopsied breast,
retains both sides when available, and maps `NORMAL` to `BENIGN`. Prediction
preprocessing applies technical QC to one patient and does not use historical
diagnosis, biopsy, date, batch, or AgBH cohort filters.

## AgBH Evidence

The JSON, notebook, and helper preserve the reason for excluding selected
calibration sessions or fallback dates. They are evidence, not executable
product configuration. Any exclusion change must update together:

```text
config/preprocessing/exclusions/agbh_quality_exclusions_t100_v0_1.yaml
docs/agbh_quality_exclusions.md
docs/meta/aramina_preprocessing_v0_1_config.json
affected preprocessing and model artifacts
```

Current thickness constraints are:

```text
sample thickness: required and >0 mm
AgBH calibrant thickness: required and 2-40 mm
preferred H5 field: calibrant_thickness_mm
```

The notebook depends on local technical calibration H5 files and is not part of
prediction, training, installation, or packaging.

## Clinical Metadata

`human1_diagnoses_metadata.json` is the normalized clinical source used for
auditing H5 metadata. Product code must not read the source Excel workbook
directly.

```text
external ID: Nova_<patient_number>_<LEFT|RIGHT>_P<point_number>
comparison key: patient_external_id | side | position
```

Recorded H5 audit:

```text
canonical rows: 2086
canonical unique measurement keys: 2083
H5 unique measurement keys: 2001
matched keys: 1996
canonical-only keys: 87
H5-only keys: 5
mismatch rows: 779
mismatched keys: 558
```

These values describe the stored audit. They are not recalculated at runtime.

## Thickness Backfill

The July 2026 audit records 21 initially missing sample-thickness rows, 14
numeric source values, 14 H5 updates, and 7 remaining missing rows. Updated H5
sets store `thickness_raw_mm`; XRD-preprocessing exposes it as
`sample_thickness_mm`.

## Change Rule

Do not silently change batch policy, source-line policy, thickness rules, label
mapping, patient/specimen identifiers, or mismatch handling. Update metadata,
runtime config, decision rationale, tests, and regenerated artifacts together.
