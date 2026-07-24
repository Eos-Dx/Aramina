# Prediction Config Contract v0.1

Status: research draft. `python -m aramis predict --config <yaml>` accepts one
strict request YAML. It cannot alter model architecture, threshold,
preprocessing, report version, or feature schema; those are held by the selected
`model.joblib`.

```yaml
run:
  analysis_author: REQUESTING_ANALYST
  prediction_comment: "optional free text"
io:
  input_h5_path: path/to/one_patient.h5
  input_model_joblib_path: models/<immutable_model_id>/model.joblib
  output_folder: outputs/prediction
patient:
  patient_id: H5_PATIENT_ID
  target_side: left
```

For a YAML stored below `<Aramis>/config/`, relative paths resolve from the
Aramis project root. For an external top-level YAML, they resolve from that
YAML's directory. Required values are non-empty strings. `target_side` is
`left` or `right`. Exactly one input is
required: `input_h5_path` for product prediction, or
`input_dataframe_joblib_path` only with `run.synthetic_test_mode: true` for
tests.

The H5 must match the embedded container contract, currently EOS H5 `0.3` with
format `xrd-session`; it must contain exactly one patient whose identifier
matches `patient.patient_id`. Unknown YAML fields, mismatched schema/format,
missing H5 structure, empty required values, or a missing model-held contract
stop the request before scoring.

The generated output folder receives one preprocessed DataFrame artifact and
both YAML/JSON external and internal reports. Full report fields are defined in
`config/prediction/README.md` and
`docs/modeling/internal_clinical_report_content_v0_9.md`.

The frozen model additionally defines the TRA score contract for internal audit.
Internal reports expose `final_prediction.level` (`TRA 1` through `TRA 5`) for
each available breast. TRA is threshold-centred and automatically recalibrated
from patient-safe OOF evaluation when the model is trained; its level does not
replace the probability decision threshold. External reports expose the target
`risk_probability`, `decision_threshold`, and threshold-derived
`biopsy_required`. Full TRA semantics are defined in
`docs/contracts/tissue_risk_assessment_v0_2.md`.
