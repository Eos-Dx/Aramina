# Prediction Pipeline v0.1

Status: research draft.

This document describes the first Aramis prediction route. It is clinical
decision support only. It returns `p_cancer`, a suggested class, and reliability
metadata for radiologist review. It is not autonomous diagnosis.

## Command

```bash
python -m aramis predict --config config/prediction/aramis_predict_example_v0_1.yaml
```

## Input Contract

The product route starts from one incoming H5 container:

```text
one-patient H5 container
trained Aramis model joblib
prediction preprocessing YAML stored in model joblib
patient_id
clinician-supplied target_side from predict YAML
model_id
selected_model
prediction.author
io.output_folder
container.schema_version
container.format
container.max_patients
reporting.external_report.version
reporting.internal_report.version
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
`@format` differs from the predict YAML, or when more than one patientId is
present in `/session/sets/set_*`.

Prediction tests also verify that `patient.patient_id` must match the H5
patientId, `patient.target_side` controls which breast is scored by LR1, and
H5 prediction requires the model artifact to carry
`prediction_preprocessing_config` unless an explicit development override is
provided in the predict YAML.

The H5 is preprocessed by the same transformer lineage as training:

```text
H5
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

The `model_id` must also come from predict YAML. It is checked against
`training.name` stored inside the model joblib. This prevents accidentally
running a different model artifact than the one requested by the report config.
The `selected_model` field selects the submodel inside the artifact, for example
`M1Q`.

## Call Chain

```text
aramis.__main__.main
-> run_prediction_from_config(config_path)
-> load model joblib
-> read prediction_preprocessing_config from model joblib
-> run prediction preprocessing, when io.input_h5_path is present
-> joblib.load(io.input_model_joblib_path)
-> build_patient_prediction_feature_row(...)
-> score selected model M0/M0Q/M1/M1Q/M2/M2Q
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
<prediction.name>_<patient.patient_id>_prediction_dataframe.joblib
<prediction.name>_<patient.patient_id>_external_report.json
<prediction.name>_<patient.patient_id>_external_report.yaml
<prediction.name>_<patient.patient_id>_internal_report.json
<prediction.name>_<patient.patient_id>_internal_report.yaml
```

Prediction always writes two report pairs.

External report is minimal:

```text
kind: aramis_external_prediction_report
version
patient_id
target_side
model_name
model_id
p_cancer
threshold
suggested_class
risk_level
reliability
reliability_reason
model/data/config SHA256 provenance
decision-support warnings
```

Internal report follows `internal_clinical_report_content_v0_1.md` and contains:

```text
kind: aramis_internal_clinical_report
version
xrd_scan_information
target LR1 profile-only p_cancer
contralateral LR1 profile-only p_cancer
SK symmetry features
age fields
reliability counters
final target-side prediction
intermediate LR1/final model summaries
feature_row
model/data/config SHA256 provenance
decision-support warnings
```

Contralateral p_cancer is included only in the internal report and only from the
first-layer profile model. The final model remains target-side decision support.

## Current Limitations

- Prediction can still start from a preprocessed DataFrame joblib for tests and
  debugging. Product use should start from H5 plus prediction preprocessing
  config.
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
