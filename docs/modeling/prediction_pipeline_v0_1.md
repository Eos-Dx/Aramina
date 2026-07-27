# Prediction Pipeline v0.1

Status: research draft.

This document describes the first Aramina prediction route. It is clinical
decision support only. The external report returns the target-side
`biopsy_required` action, reliability metadata, and frozen method performance;
the internal report retains `p_cancer` and TRA for audit.

## Command

```bash
python -m aramina predict --config examples/prediction/configs/config_predict_cancer_example.yaml
```

## Input Contract

The product route starts from one incoming H5 container:

```text
one-patient H5 container
trained Aramina model joblib
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
`aramina_target_breast_risk`.

## Call Chain

```text
aramina.__main__.main
-> run_prediction_from_config(config_path)
-> load model joblib
-> parse resolved prediction_preprocessing_yaml from model joblib
-> run prediction preprocessing, when io.input_h5_path is present
-> patient_features.build_patient_prediction_feature_row(...)
-> prediction_scoring: score target and contralateral sides; neutralize SK symmetry for contralateral
-> prediction_reports: build external and internal reports
-> write external and internal report JSON/YAML
```

## Feature Construction

For one patient:

```text
select target breast rows
score target radial_profile_data with LR1
logit-average target p_cancer
compute target-vs-contralateral SK symmetry only when both breasts have at least
two valid measurements and all Core4 values are finite
add age if present
add reliability counters
score contralateral with full final model while forcing SK symmetry refinement neutral
```

Risk and reliability are kept separate:

```text
p_cancer = risk score
reliability = confidence in this result
```

Low reliability does not reduce `p_cancer`. It tells the report that the result
needs more caution, for example when only one valid target-breast measurement is
available or paired-breast symmetry is unavailable.

Reliability levels are operational data-sufficiency fields:

```text
high:   target >=2 and contralateral >=2 valid measurements; SK refinement applied
medium: target >=2 but paired symmetry is unavailable or cannot be computed
low:    target <2 valid measurements
```

Optional scan metadata is copied only when a single non-empty value is present
among valid target-side rows. Missing or conflicting optional metadata is
reported as `unknown` and never blocks prediction.

## Output

Prediction YAML specifies one output folder:

```yaml
io:
  output_folder: ./examples/outputs/prediction
```

For a YAML under `Aramina/config`, the path is relative to the Aramina project
root. For an external top-level YAML, it is relative to that YAML's directory.

Aramina writes automatic file names:

```text
<patient_id>_<model_id>_<report_id>_prediction_dataframe.joblib
<patient_id>_<model_id>_<report_id>_external_report.yaml
<patient_id>_<model_id>_<report_id>_internal_report.yaml
```

Prediction always writes two report pairs.

External report is minimal:

```text
output_type: aramina_external_report
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
model_metrics.dataset
model_metrics.validation
model_metrics.sensitivity
model_metrics.specificity
risk_probability
decision_threshold
biopsy_required
reliability
reliability_reason
```

External report does not expose profile-only scores, symmetry, TRA, provenance,
raw data, or model internals. `risk_probability` is
the frozen final target-breast score and `decision_threshold` is the associated
fixed model threshold. `biopsy_required` is `true` when the score meets or
exceeds that threshold, otherwise `false`. The report includes sensitivity and
specificity of the selected frozen final model. `model_metrics.dataset` is
`train_on_all_target_breast_cases` and `model_metrics.validation` is
`not_performed`: these figures are final-fit training metrics, not independent
evaluation estimates. `report_id` is
generated automatically and shared with the internal report from the same
prediction operation. `created_at` is a Europe/Paris ISO timestamp with a
numeric UTC offset.

Internal report follows `internal_clinical_report_content_v0_9.md` and contains:

```text
output_type: aramina_internal_clinical_report
report_version
report_id and created_at
model ID/name/version/artifact SHA256
frozen final-model sensitivity/specificity and request comment
one shared decision threshold
scan metadata and measurement summary
target final decision and contralateral full-model evidence
frozen score percentiles, symmetry availability, and reliability
derived TRA level for each available breast
```

TRA is an internal threshold-centred tier. Every completed training run
automatically recalibrates its TRA margins from patient-safe OOF decision
stability and freezes them in the promoted model artifact. It does not change
the target-side `p_cancer` threshold or the `biopsy_required` action.

Report-level naming:

```text
breast_predictions.target.final_prediction.p_cancer
  final target-side decision-support score

breast_predictions.target.azimuthal_integration_profile.p_cancer
  target-breast LR1 profile-only probability
```

Estimator summaries, model weights, raw feature rows, and training configuration
are intentionally not duplicated in an internal prediction report. They remain
in `model_description.yaml` and the executable model joblib artifact.

Internal report excludes profile statistics, filesystem paths, generic
provenance, and output-file paths.

The internal report scores the contralateral breast with the same final model,
but forces its SK symmetry gate to neutral. It exposes LR1 profile evidence,
final `p_cancer` and TRA level with the SK symmetry gate neutralized. It has no
separate target-class risk level or biopsy action. The target remains the
caller-supplied primary decision-support result. If no
usable contralateral data remain after QC,
that block is explicitly `unknown`; the target still uses the same LR2 with its
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
