# Aramis Prediction YAML

Prediction starts one-patient research-draft decision support. The caller supplies only request identity, the one-patient H5, the frozen model artifact, the output folder, and the clinically indicated target side.

Formal request contract: `docs/contracts/prediction_config_v0_1.md`.

```yaml
run:
  analysis_author: OPERATOR_OR_ANALYST
  prediction_comment: "optional free-text comment"
io:
  input_h5_path: examples/prediction_h5/cancer_one_patient.h5
  input_model_joblib_path: models/aramis_target_breast_risk_<model_id>/model.joblib
  output_folder: examples/outputs/prediction
patient:
  patient_id: PATIENT_ID_FROM_H5
  target_side: left
```

Run from the Aramis project root:

```bash
python -m aramis predict --config examples/prediction/configs/config_predict_cancer_example.yaml
```

For a YAML under `Aramis/config`, relative paths resolve from the Aramis root.
For an external top-level YAML, they resolve from that YAML's directory.
`analysis_author` is the person requesting the report. `prediction_comment` is
optional free text and is copied to both reports. `patient_id` must exactly
match the only H5 patient. `target_side` is clinical input and must be `left`
or `right` in the preprocessed H5 data. Optional H5 metadata is retained only
when one non-empty target-side value is present; absent or conflicting values
are reported as `unknown` and do not stop prediction.

The model joblib supplies everything else: model identity, preprocessing YAML, H5 schema/format contract, report versions, threshold, feature schema, and executable estimator. Predict YAML cannot override any of these fields.

Aramis writes automatic names under `io.output_folder`:

```text
<patient_id>_<model_id>_<report_id>_prediction_dataframe.joblib
<patient_id>_<model_id>_<report_id>_external_report.json
<patient_id>_<model_id>_<report_id>_external_report.yaml
<patient_id>_<model_id>_<report_id>_internal_report.json
<patient_id>_<model_id>_<report_id>_internal_report.yaml
```

External report is target-side only and contains report identity, requesting analyst, optional comment, patient/target identity, model version, final-model sensitivity/specificity, risk probability, the fixed decision threshold, `target_class_risk_level` (`low` or `high`), and reliability. `high` means `risk_probability >= decision_threshold`; `low` means it is below the threshold. The external report intentionally excludes TRA, suggested class, profile-only scores, and model internals.

Internal report contains one shared threshold policy and two breast blocks. The
target block is the formal decision-support result. The
contralateral block is an internal full-model score: it contains LR1 profile
evidence and final `p_cancer` with SK symmetry refinement neutralized. It
uses the shared threshold to provide `suggested_class`, but its reliability is
always `low`. Full contract:
`docs/modeling/internal_clinical_report_content_v0_6.md`.

Both reports copy final-model sensitivity and specificity from the selected model artifact. `model_metrics.dataset: train_on_all_target_breast_cases` identifies the data used for these final-fit figures; `model_metrics.validation: not_performed` states that they are not an independent validation estimate. Full evaluation records remain in the artifact and its adjacent evaluation files.

If no usable contralateral breast remains, the contralateral block is explicitly
`unknown`. The target result remains available, with
`model_execution.scoring_path: azimuthal_integration_age`; optional
symmetry refinement was not applied.

Prediction stops for unknown YAML fields, schema/format mismatch, zero or multiple H5 patients, patient-ID mismatch, absent target side, or a model missing its embedded prediction preprocessing/contract/reference scores.
