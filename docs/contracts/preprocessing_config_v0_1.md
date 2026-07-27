# Aramina Preprocessing Config Contract v0.1

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

Both require non-empty `aramina_preprocessing.name`, scalar `version`,
`clinical_stage`, `io.input_h5_path`, and `io.output_joblib_path` after YAML
composition. `metadata.output_columns` is mandatory, unique, and must retain
patient/specimen/side identifiers, age, q grid, normalized radial profile, SNR,
and measurement count. Prediction output retains session metadata when the H5
container provides it; absent optional metadata is reported as `unknown`.

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
