# Aramis Prediction YAML

Prediction starts one-patient research-draft decision support. The caller supplies only request identity, the one-patient H5, the frozen model artifact, the output folder, and the clinically indicated target side.

```yaml
run:
  analysis_author: OPERATOR_OR_ANALYST
  prediction_comment: "optional free-text comment"
io:
  input_h5_path: ./examples/prediction_h5/cancer_one_patient.h5
  input_model_joblib_path: ./models/aramis_target_breast_risk_<model_id>/model.joblib
  output_folder: ./examples/outputs/prediction_h5_examples
patient:
  patient_id: PATIENT_ID_FROM_H5
  target_side: left
```

Run from the Aramis project root:

```bash
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
```

All relative paths are resolved from the Aramis root. `analysis_author` is the person requesting the report. `prediction_comment` is optional free text and is copied to both reports. `patient_id` must exactly match the only H5 patient. `target_side` is clinical input and must be `left` or `right` in the preprocessed H5 data.

The model joblib supplies everything else: model identity, preprocessing YAML, H5 schema/format contract, report versions, threshold, feature schema, and executable estimator. Predict YAML cannot override any of these fields.

Aramis writes automatic names under `io.output_folder`:

```text
<patient_id>_<model_id>_<report_id>_prediction_dataframe.joblib
<patient_id>_<model_id>_<report_id>_external_report.json
<patient_id>_<model_id>_<report_id>_external_report.yaml
<patient_id>_<model_id>_<report_id>_internal_report.json
<patient_id>_<model_id>_<report_id>_internal_report.yaml
```

External report is target-side only and contains report identity, requesting analyst, optional comment, patient/target identity, model version, frozen method performance, suggested class, and reliability. It intentionally excludes p_cancer and threshold.

Internal report contains two breast blocks. The target block is the formal decision-support result. The contralateral block is internal audit information computed by applying the same final model with that side temporarily treated as target. Each available block contains an azimuthally integrated profile score, final p_cancer, frozen threshold, suggested class, three frozen-training score percentiles, symmetry availability, and reliability. Full contract: `docs/modeling/internal_clinical_report_content_v0_1.md`.

Both reports copy frozen method sensitivity, specificity, and evaluation method from the selected model artifact. They describe the method, not the individual patient. Full evaluation records remain in the artifact and its adjacent evaluation files.

If no usable contralateral breast remains, the contralateral block is explicitly
`unknown`. The target result remains available, with
`model_execution.scoring_path: profile_age_with_neutral_symmetry_gate`; optional
symmetry refinement was not applied.

Prediction stops for unknown YAML fields, schema/format mismatch, zero or multiple H5 patients, patient-ID mismatch, absent target side, or a model missing its embedded prediction preprocessing/contract/reference scores.
