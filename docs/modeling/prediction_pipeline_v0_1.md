# Prediction Pipeline v0.1

Status: research draft.

This document describes the first Aramis prediction route. It is clinical
decision support only. The external report returns a suggested class,
reliability metadata, and frozen method performance; the internal report
retains `p_cancer` for audit.

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

The repository tracks three one-patient v0.3 integration fixtures under
`examples/prediction_h5/`: benign, cancer, and atypical historical cases.
Each contains one patient, left/right breast specimens, and three measurement
sets per breast. Unit tests additionally create small synthetic H5 containers
to verify invalid-contract cases.
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
no historical diagnosis/status filter
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
`aramis_m2q_t100`.

## Call Chain

```text
aramis.__main__.main
-> run_prediction_from_config(config_path)
-> load model joblib
-> parse resolved prediction_preprocessing_yaml from model joblib
-> run prediction preprocessing, when io.input_h5_path is present
-> build_patient_prediction_feature_row(...)
-> score artifact-selected product model
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
  output_folder: ./examples/outputs/prediction
```

The path is relative to the Aramis project root, not to the prediction YAML.

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
analysis_author
prediction_comment
patient_id
patient_age
target_side
mammography_suspicious_field
scan_date_time
operator_id
hardware_version
eoscan_version
model_name
model_version
method_performance.evaluation_available
method_performance.evaluation_method
method_performance.folds
method_performance.repeats
method_performance.sensitivity
method_performance.sensitivity_std
method_performance.specificity
method_performance.specificity_std
suggested_class
reliability
reliability_reason
```

External report does not expose `p_cancer`, threshold, profile-only scores,
symmetry, provenance, raw data, or model internals. It includes frozen method
sensitivity/specificity with `evaluation_method`, `folds`, and `repeats` so the
numbers are interpretable and traceable to one evaluation route. These are
method-level fields, not patient-specific evidence. `report_id` is
generated automatically and shared with the internal report from the same
prediction operation. `created_at` is a Europe/Paris ISO timestamp with a
numeric UTC offset.

Internal report follows `internal_clinical_report_content_v0_1.md` and contains:

```text
output_type: aramis_internal_clinical_report
report_version
report_id and created_at
model ID/name/version/artifact SHA256
scan metadata and measurement summary
target/contralateral azimuthally integrated profile and final predictions
frozen score percentiles, symmetry availability, and reliability
```

Report-level naming:

```text
breast_predictions.target.final_prediction.p_cancer
  final target-side decision-support score

breast_predictions.target.azimuthal_integration_target_profile.p_cancer
  target-breast LR1 profile-only probability
```

Estimator summaries, model weights, raw feature rows, and training configuration
are intentionally not duplicated in an internal prediction report. They remain
in `model_description.yaml` and the executable model joblib artifact.

Internal report excludes profile statistics, filesystem paths, generic
provenance, and output-file paths.

The internal report scores the contralateral breast with the same final model
artifact for audit only. If no usable contralateral data remain after QC, that
block is explicitly `unknown`; the target still uses the same LR2 with its
gated SK terms neutralized. The external report remains target-side only.

## Current Limitations

- Synthetic unit tests can start from a preprocessed DataFrame joblib only when
  `run.synthetic_test_mode: true`. Product use starts from H5 plus the
  prediction preprocessing config embedded in the model artifact.
- Reports are machine-readable JSON/YAML only. Formatted clinical PDF/report
  rendering remains a separate layer.
- Thresholds come from the training artifact. The model version must therefore
  be reviewed together with its validation mode and intended use.
- The product model uses age. Age can carry real clinical signal but can also dominate a
  small dataset, so age-based models require explicit review before product
  fixation.
- Prediction currently supports one selected model per config. Comparing several
  models for review should be done in evaluation notebooks or a future report
  mode.
