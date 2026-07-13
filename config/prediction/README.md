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

The H5 smoke-test examples use a synthetic-only model artifact:

```text
examples/prediction_models/aramis_m2q_t100_gated_sk_core4_synthetic_h5_example.joblib
```

The general H5 template points to the packaged product artifact, which embeds
the production GFRM preprocessing contract.

Required product sections:

```text
prediction.name
prediction.author
io.input_h5_path
io.input_model_joblib_path
io.output_folder
patient.patient_id
patient.target_side
model.model_id
model.model_name
model.model_version
```

`io.output_folder` is enough. Aramis creates automatic output names:

```text
<prediction.name>_<patient.patient_id>_<automatic_run_id>_prediction_dataframe.joblib
<prediction.name>_<patient.patient_id>_<automatic_run_id>_external_report.json
<prediction.name>_<patient.patient_id>_<automatic_run_id>_external_report.yaml
<prediction.name>_<patient.patient_id>_<automatic_run_id>_internal_report.json
<prediction.name>_<patient.patient_id>_<automatic_run_id>_internal_report.yaml
```

Important rule:

```text
target_side is supplied by clinician/config in predict YAML
target_side is not inferred from labels, biopsy fields, or specimen_status
patient.patient_id must exactly match the single H5 patientId
H5 root @schema_version and @format must match model-held contract
exactly one patient must be present in the H5 container
model.model_id must match training.name stored inside the model joblib
model.model_name must exist inside that artifact, for example M2Q
model.model_version must match training.version stored inside that artifact
```

The training YAML defines `prediction_contract`. Training embeds that contract,
the prediction preprocessing YAML, report versions, and decision threshold in
the model joblib. Product Predict YAML cannot override `preprocessing`,
`reporting`, `container`, or `decision`.

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
-> build paired SK symmetry features when a contralateral breast is available
-> add reliability counters
-> model-selected M2Q decision-support score
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
contains XRD profile evidence, symmetry fields, reliability, and traceability.
It does not duplicate intermediate estimator summaries, feature rows, model
weights, or training configuration; those remain in the ML-classifier training
output YAML under `model_registry`. The joblib is the executable model artifact.
