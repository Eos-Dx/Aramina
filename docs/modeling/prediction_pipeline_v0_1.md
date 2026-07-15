# Prediction Pipeline v0.1

Status: research draft.

This document describes the first Aramis prediction route. It is clinical
decision support only. The external report returns a suggested class and
reliability metadata; the internal report retains `p_cancer` for audit.

## Command

```bash
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
```

## Input Contract

The product route starts from one incoming H5 container:

```text
one-patient H5 container
trained Aramis model joblib
prediction preprocessing YAML stored in model joblib
run.analysis_author
io.input_h5_path
io.input_model_joblib_path
io.output_folder
patient.patient_id
clinician-supplied target_side from predict YAML
```

Expected H5 shape is EOS H5 v0.3:

```text
/
  @format = xrd-session
  @schema_version = 0.3
  session/
    @category = SAMPLE
    @calibrant_thickness_mm
    sample/patient_name
    sets/set_*/
      @patientId
      @specimenId
      @side
      @position
      @specimen_status
      acquisition/sample_thickness
      raw/data
      artifacts/poni
```

For prediction tests, Aramis builds three one-patient v0.3 containers in
`tests/test_prediction.py`. Each file contains one patient, left/right breast
specimens, and three measurement sets per breast. The test then calls
`python -m aramis predict --config <patient_predict.yaml>` for each H5.
Prediction fails before preprocessing when the H5 root `@schema_version` or
`@format` differs from the model-held prediction contract, or when more than one patientId is
present in `/session/sets/set_*`.

Prediction tests also verify that `patient.patient_id` must match the H5
patientId, `patient.target_side` controls which breast is scored by LR1, and
H5 prediction requires the model artifact to carry
`prediction_preprocessing_yaml` and `prediction_contract_yaml`. Predict YAML
cannot override preprocessing, reports, the H5 contract, or decision threshold.

The H5 is preprocessed by the same transformer lineage as training:

```text
H5
-> H5PoniGeometryCalculatorTransformer
-> H5SessionSelectorTransformer
-> h5_to_df
-> ProductColumnBuilder
-> q-range, sample-thickness, calibrant-thickness checks
-> FaultyPixelDetector
-> AzimuthalIntegration(error_model="poisson", thickness correction)
-> SNRTransformer(snr_method="poisson")
-> SNRFilter
-> PatientSpecimenValidityFilter
-> QRangeValueNormalizer
-> RadialProfileValueFilter
-> KeepColumnsTransformer
```

Prediction preprocessing differs from training cohort preprocessing:

```text
no historical date filter
no AgBH quality exclusion list
no diagnosis/status cohort filter
no biopsy filter
```

The `target_side` must come from the clinician-facing predict YAML. It is not
read from H5 metadata and is not inferred from labels, biopsy metadata, or
`specimen_status`. In training, biopsy/status metadata can define the
historical target breast. In prediction, the suspicious breast comes from
clinical input.

Training creates one historical case per biopsied breast. A bilateral-biopsy
patient therefore contributes two cases, while all validation splits remain
strictly patient-safe. Prediction mirrors this behavior: one supplied
`target_side` produces one result; two suspicious breasts require two separate
prediction runs.

The selected joblib is the sole source for model ID, name, version, model entry,
preprocessing contract, report contract, and decision threshold. Predict YAML
does not duplicate these immutable model fields. The fixed development model is
`M2Q`.

## Call Chain

```text
aramis.__main__.main
-> run_prediction_from_config(config_path)
-> load model joblib
-> parse resolved prediction_preprocessing_yaml from model joblib
-> run prediction preprocessing, when io.input_h5_path is present
-> joblib.load(io.input_model_joblib_path)
-> build_patient_prediction_feature_row(...)
-> score artifact-selected M2Q
-> write external report JSON/YAML
-> write internal report JSON/YAML
```

## Feature Construction

For one patient:

```text
select target breast rows
score target radial_profile_data with LR1
logit-average target p_cancer
compute target-vs-contralateral SK symmetry when contralateral breast is present
add age if present
add reliability counters
score contralateral radial_profile_data with LR1 for internal report only
```

Risk and reliability are kept separate:

```text
p_cancer = risk score
reliability = confidence in this result
```

Low reliability does not reduce `p_cancer`. It tells the report that the result
needs more caution, for example when only one valid target-breast measurement is
available or paired-breast symmetry is unavailable.

## Output

Prediction YAML specifies one output folder:

```yaml
io:
  output_folder: ../../examples/outputs/prediction
```

Aramis writes automatic file names:

```text
<patient_id>_<model_id>_<report_id>_prediction_dataframe.joblib
<patient_id>_<model_id>_<report_id>_external_report.json
<patient_id>_<model_id>_<report_id>_external_report.yaml
<patient_id>_<model_id>_<report_id>_internal_report.json
<patient_id>_<model_id>_<report_id>_internal_report.yaml
```

Prediction always writes two report pairs.

External report is minimal:

```text
output_type: aramis_external_report
report_version
report_id
created_at
patient_id
target_side
suggested_class
reliability
reliability_reason
model_version
```

External report does not expose `p_cancer`, threshold, LR1 profile-only scores,
symmetry, age, provenance, raw data, or model internals. `report_id` is
generated automatically and shared with the internal report from the same
prediction operation. `created_at` is a Europe/Paris ISO timestamp with a
numeric UTC offset.

Internal report follows `internal_clinical_report_content_v0_1.md` and contains:

```text
output_type: aramis_internal_clinical_report
report_version
report_id and created_at
model ID/name/version/artifact SHA256
prediction configuration snapshot
scan metadata and measurement summary
target/contralateral LR1 profile evidence
selected SK Core4 values and reliability
final target-side prediction with threshold
```

Report-level naming:

```text
final_prediction.p_cancer
  final target-side risk score

evidence.target_profile.profile_p_cancer
  target-breast LR1 profile-only probability
```

Estimator summaries, model weights, raw feature rows, and training configuration
are intentionally not duplicated in an internal prediction report. They remain
in `model_description.yaml` and the executable model joblib artifact.

Internal report excludes profile statistics, filesystem paths, generic
provenance, and output-file paths.

Contralateral p_cancer is included only in the internal report and only from the
first-layer profile model. The final model remains target-side decision support.

## Current Limitations

- Synthetic unit tests can start from a preprocessed DataFrame joblib only when
  `run.synthetic_test_mode: true`. Product use starts from H5 plus the
  prediction preprocessing config embedded in the model artifact.
- Reports are machine-readable JSON/YAML only. Formatted clinical PDF/report
  rendering remains a separate layer.
- Thresholds come from the training artifact. The model version must therefore
  be reviewed together with its validation mode and intended use.
- `M2/M2Q` use age. Age can carry real clinical signal but can also dominate a
  small dataset, so age-based models require explicit review before product
  fixation.
- Prediction currently supports one selected model per config. Comparing several
  models for review should be done in evaluation notebooks or a future report
  mode.
