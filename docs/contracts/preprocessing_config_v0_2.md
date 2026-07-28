# Aramina Preprocessing Config Contract v0.2

Status: research draft. Aramina preprocessing YAML is a product policy layered
on the general `xrd-preprocessing` YAML grammar. XRD-preprocessing builds the
declared sklearn transformer route; Aramina validates that a resolved product
YAML retains the approved clinical-research policy before processing starts.

## Product Routes

```text
aramina_biopsy_patients_model_input
  historical training cohort

aramina_prediction_patient_model_input
  one incoming prediction patient
```

Both routes require an explicit marker:

```yaml
aramina_preprocessing:
  contract: aramina_product_preprocessing_v0_2
  name: aramina_biopsy_patients_model_input
  route: training
  version: '0.2'
  clinical_stage: research draft
```

The prediction route uses `route: prediction` and its corresponding product
name. Product CLI runs reject missing markers and route mismatches before H5
processing. `io.input_h5_path` and `io.output_joblib_path` must be non-empty
after YAML composition. `metadata.output_columns` is mandatory, unique, and
must retain patient/specimen/side identifiers, age, q grid, normalized radial
profile, SNR, and measurement count. Prediction output retains session metadata
when the H5 container provides it; absent optional metadata is reported as
`unknown`.

## Fixed Shared Policy

```text
raw source: GFRM only
measurement positions: P1, P2, P3
PONI q max: >=23 nm^-1
sample thickness: required
calibrant thickness: 2..40 mm
integration: 100 q points, q=2..23 nm^-1, Poisson errors
SNR: Poisson, >=18 dB
normalisation: median value at q=6.7..7.1 nm^-1
profile gate: q=14 nm^-1, value >2.0
```

The approved transformer order starts with PONI/session/H5 loading and ends
with faulty-pixel handling, integration, SNR, normalization, profile gate, and
`KeepColumnsTransformer`. A changed order is a product preprocessing change and
must update the model version, documentation, tests, and training artifact.

## Route Differences

Training route enables T100 AgBH exclusions, retains patients with at least one
biopsy row, keeps contralateral measurements, maps `NORMAL` to `BENIGN`, and
keeps model labels `BENIGN` and `CANCER`.

Prediction route disables date, historical AgBH, diagnosis, biopsy, and pairing
cohort filters. It applies only technical quality controls; target side comes
from the prediction request YAML.

The current implementation is
`src/aramina/preprocessing_contract.py`. Product runs reject a resolved YAML
that violates this contract. Generic XRD-preprocessing YAML remains flexible
outside the Aramina product route.

## Artifact And Runtime Identity

New preprocessing outputs use XRD artifact envelope `0.2`. Each joblib stores:

```text
preprocessing_config_yaml
resolved_pipeline_spec
pipeline_fingerprint
metadata.aramina_preprocessing_lineage
```

The lineage records the Aramina route, XRD release tag, XRD package version,
requested revision, full 40-character XRD commit, resolved pipeline semantics,
and SHA-256 pipeline fingerprint. New training accepts only artifact `0.2` with
the training route and exact runtime identity. Prediction by a newly trained
model requires the prediction route and exact model-held lineage.

Legacy artifact `0.1` remains read-only. It is accepted only for prediction by
the frozen `0.2.12-beta` model artifact. New training rejects it because it does
not contain a resolved executable pipeline identity. There is no automatic
migration: reproducing a legacy dataset as `0.2` requires an explicit rerun from
the original H5 and approved YAML.
