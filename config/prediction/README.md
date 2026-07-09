# Aramis Prediction YAML

Prediction YAML runs one-patient research-draft decision support. Product use
should start from one H5 container and run prediction preprocessing before
model scoring. The already-preprocessed DataFrame mode is kept for tests and
debugging.

Working H5 smoke-test:

```bash
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
```

H5 template for a new patient:

```bash
config/prediction/aramis_predict_from_h5_template_v0_1.yaml
```

DataFrame-mode debug example:

```bash
config/prediction/aramis_predict_example_v0_1.yaml
```

The example uses the tracked model artifact:

```text
examples/prediction_models/aramis_m2q_t100_train_all_c0p1.joblib
```

Required sections:

```text
prediction.name
prediction.author
io.input_h5_path
io.input_model_joblib_path
io.output_folder
reporting.external_report.version
reporting.external_report.reference_doc
reporting.internal_report.version
reporting.internal_report.reference_doc
container.schema_version
container.format
container.max_patients
patient.patient_id
patient.target_side
model.model_id
model.selected_model
decision.threshold_key
```

`io.output_folder` is enough. Aramis creates automatic output names:

```text
<prediction.name>_<patient.patient_id>_prediction_dataframe.joblib
<prediction.name>_<patient.patient_id>_external_report.json
<prediction.name>_<patient.patient_id>_external_report.yaml
<prediction.name>_<patient.patient_id>_internal_report.json
<prediction.name>_<patient.patient_id>_internal_report.yaml
```

Important rule:

```text
target_side is supplied by clinician/config in predict YAML
target_side is not inferred from labels, biopsy fields, or specimen_status
container.schema_version must match the H5 root @schema_version
container.format must match the H5 root @format
container.max_patients must be 1 for Aramis prediction
model.model_id must match training.name stored inside the model joblib
model.selected_model selects the submodel inside that artifact, for example M2Q
```

The prediction preprocessing YAML is stored inside the trained model joblib as
`prediction_preprocessing_config`. Product `predict.yaml` normally does not
point to a preprocessing YAML.

Product v0.1 route:

```text
one-patient H5
-> clinician-supplied patient.target_side from predict YAML
-> prediction_preprocessing_config from model joblib
-> hot/faulty pixel detection
-> azimuthal integration with sample/calibrant thickness correction
-> SNR calculation and SNR filter
-> q-range normalization
-> model-input DataFrame joblib
-> select one patient
-> select clinician-supplied target breast
-> LR1 scores target-breast radial_profile_data
-> logit-average target-breast p_cancer
-> build SK symmetry features against contralateral breast when available
-> add reliability counters
-> selected M0/M0Q/M1/M1Q/M2/M2Q model
-> external report JSON/YAML
-> internal report JSON/YAML
```

Report language is decision-support only:

```text
p_cancer
suggested_class
risk_level
reliability
reliability_reason
requires_radiologist_review
not for autonomous diagnosis
```

External report is intentionally minimal and target-side only. Internal report
contains audit fields, intermediate model summaries, target LR1
`profile_p_cancer`, and contralateral LR1 profile score for internal review.
`profile_p_cancer_logit_average` is kept only in `feature_row` as an internal
model-audit feature.
