# Aramis Product API v0.1

Status: research draft.

This document is the developer-facing API contract for Aramis preprocessing,
training, and prediction. Aramis output is decision support only:
`p_cancer`, suggested class, and reliability metadata for clinician review.

## CLI Surface

```bash
python -m aramis preprocess --config <preprocessing.yaml>
python -m aramis train --config <training.yaml>
python -m aramis run --config <workflow.yaml>
python -m aramis predict --config <prediction.yaml>
```

Command-line arguments pass config paths only. Data input/output paths live in
YAML.

## Python Entrypoints

```python
from aramis.pipelines import run_preprocessing_from_config
from aramis.training import run_training_from_config
from aramis.workflows import run_workflow_from_config
from aramis.prediction import run_prediction_from_config
```

These functions return in-memory objects:

```text
run_preprocessing_from_config -> pandas.DataFrame
run_training_from_config      -> training artifact dict
run_workflow_from_config      -> workflow result dict
run_prediction_from_config    -> dict with external_report and internal_report
```

## Product Prediction Input

Prediction starts from one H5 container and one trained model joblib:

```yaml
prediction:
  name: aramis_predict_from_h5
  version: 0.1
  author: OPERATOR_OR_ANALYST

io:
  input_h5_path: /path/to/one_patient.h5
  input_model_joblib_path: /path/to/model.joblib
  output_folder: /path/to/prediction_outputs

patient:
  patient_id: PATIENT_ID
  target_side: Left

model:
  model_id: MODEL_ARTIFACT_ID
  model_name: M2Q
  model_version: "0.2.2-beta"
```

Aramis creates automatic output names inside `io.output_folder`:

```text
<prediction.name>_<patient.patient_id>_<automatic_run_id>_prediction_dataframe.joblib
<prediction.name>_<patient.patient_id>_<automatic_run_id>_external_report.json
<prediction.name>_<patient.patient_id>_<automatic_run_id>_external_report.yaml
<prediction.name>_<patient.patient_id>_<automatic_run_id>_internal_report.json
<prediction.name>_<patient.patient_id>_<automatic_run_id>_internal_report.yaml
```

Training embeds `prediction_preprocessing_config` and an immutable
`prediction_contract` in the model joblib. The contract contains supported H5
schema and format, report versions, report references, and threshold key.
Predict YAML cannot override preprocessing, reporting, container, or decision
settings.

Packaged product model artifact:

```text
examples/prediction_models/aramis_m2q_t100_gated_sk_core4_nested_c1_0p1_c2_0p3.joblib
```

Example one-patient H5 configs in `examples/prediction_h5/*_predict.yaml` point
to a separate synthetic-only smoke-test artifact with embedded raw-array
preprocessing. The fixed gated M2Q architecture is documented in
`docs/modeling/m2q_gated_target_case_model_v0_1.md`.

## H5 Container Contract

Current supported prediction container:

```text
format: xrd-session
schema_version: 0.3
patients per H5: exactly 1
```

Required shape:

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
      raw/data or raw_file/artifacts/gfrm
      artifacts/poni
```

Product preprocessing currently expects GFRM raw data. Synthetic tests may use
Numpy arrays, but product YAML uses:

```text
raw_data.source: gfrm
allowed_sources: [gfrm]
```

`patient.target_side` is supplied by the clinical caller. It is not inferred
from H5 labels, biopsy flags, or `specimen_status`.

Training uses one historical case per biopsied breast. A bilateral-biopsy
patient therefore has two target-breast cases, but both always remain in the
same `patientId` split. Prediction accepts one explicit target breast per run.

## Preprocessing Output Joblib

`aramis preprocess` writes an XRD-preprocessing artifact joblib:

```text
kind: xrd_preprocessing_dataframe
dataframe: pandas.DataFrame
preprocessing_config: resolved YAML dict
preprocessing_config_text: original YAML text
preprocessing_config_sha256
metadata.branch
metadata.input_h5_sha256
metadata.aramis_version
metadata.aramis_git_sha
```

Downstream code should load only the DataFrame with:

```python
from xrd_preprocessing import load_preprocessing_dataframe
df = load_preprocessing_dataframe("preprocessed.joblib")
```

## Model-Input DataFrame Columns

Current model-input preprocessing keeps non-heavy metadata plus normalized
profiles:

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

Heavy frame payloads are not retained in model input joblibs.

## Training Output Joblib

`aramis train` writes:

```text
kind: aramis_training_artifact
model_type: patient_m0_m1_m2_logistic_set
models: fitted selected model objects
model_descriptions
feature_schema
warnings
training_config
training_config_text
training_config_sha256
training_preprocessing_config
training_preprocessing_config_sha256
prediction_preprocessing_config
prediction_preprocessing_config_sha256
input_dataframe_joblib_sha256
dataset_summary
feature_table
metric_summary
split_metrics
split_predictions
metadata.aramis_version
metadata.aramis_git_sha
```

Prediction checks `model.model_id` from YAML against
`training_config.training.name` stored in this artifact.

## Prediction Report Schema

`aramis predict` always writes two JSON/YAML report pairs.

External report is minimal and target-side only:

```text
kind: aramis_external_prediction_report
version
reference_doc
created_at
clinical_stage
intended_use
decision_support_only
requires_radiologist_review
patient_id
target_side
contralateral_side
model_id
model_name
p_cancer
threshold_key
threshold
suggested_class
risk_level
reliability
reliability_reason
provenance
limitations
```

External report does not expose LR1 profile-only fields such as
`profile_p_cancer` or `profile_p_cancer_logit_average`. It reports only the
final model `p_cancer`.

Internal report is audit-oriented and follows
`docs/modeling/internal_clinical_report_content_v0_1.md`:

```text
output_type: aramis_internal_clinical_report
version
reference_doc
created_at
patient_id
target_side
contralateral_side
xrd_scan_information
features.azimuthal_integration.target_profile
features.azimuthal_integration.contralateral_profile
features.symmetry
features.reliability
final_prediction
provenance
outputs
```

Internal report contains human-readable XRD evidence and decision output:

```text
final_prediction.p_cancer
  final M2Q decision-support risk score

features.azimuthal_integration.target_profile.profile_p_cancer
  target-breast LR1 profile-only probability
```

Estimator configuration, feature weights, model schema, thresholds, and raw
model feature rows are stored only in the ML-classifier training output YAML
under `model_registry`; the joblib is the executable model artifact.

Contralateral breast prediction is internal-only. It is produced by the LR1
profile model, not by the final target-side model.

Risk and reliability are separate:

```text
p_cancer:
  model risk score

reliability:
  data sufficiency / measurement context

reliability_reason:
  human-readable reason, for example fewer than 3 valid measurements
```

Low reliability does not reduce `p_cancer`. It tells the report consumer that
the result needs more caution and clinician review.

## Current Development Model

```text
architecture_id: m2q_gated_target_case_v0_1
selected_model: M2Q
preprocessing: T100 biopsy-patient model input
regularization: LR1 L2 C=0.1; LR2 L2 C=0.3
threshold: selected from inner patient-safe out-of-fold predictions
```

See:

```text
docs/modeling/m2q_gated_target_case_model_v0_1.md
```

## Failure Rules

Prediction fails before scoring when:

```text
H5 root @schema_version does not match model-held contract
H5 root @format does not match model-held contract
more than one patientId is present
patient.patient_id does not match H5 patientId
patient.target_side is missing or absent in the preprocessed DataFrame
model.model_id does not match the artifact training.name
model.model_name is not present in the model artifact
model.model_version does not match artifact training.version
model artifact does not contain prediction_preprocessing_config
model artifact does not contain prediction_contract
```

Preprocessing fails or drops rows when:

```text
sample_thickness_mm is missing or invalid
calibrant_thickness_mm is missing or outside configured safety range
PONI q range cannot satisfy required q max
SNR is below configured threshold
profile gate fails
metadata.output_columns cannot be satisfied
```

## Test Coverage

Current product tests cover:

```text
preprocessing YAML route
preprocessing artifact joblib
training artifact joblib
workflow memory/artifact modes
prediction from preprocessed DataFrame for tests
prediction from one-patient H5 v0.3
schema/format/patient/model-id guards
target_side scoring behavior
```
