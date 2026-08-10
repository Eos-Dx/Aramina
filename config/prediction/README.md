# Prediction YAML

Prediction accepts one request, one patient H5, one frozen model, and one
clinically selected target side.

```yaml
run:
  analysis_author: OPERATOR_OR_ANALYST
  prediction_comment: optional free-text comment
io:
  input_h5_path: examples/prediction_h5/cancer_one_patient.h5
  input_model_joblib_path: models/<model_id>/model.joblib
  output_folder: examples/outputs/prediction
patient:
  patient_id: PATIENT_ID_FROM_H5
  target_side: left
```

```bash
python -m aramina predict \
  --config examples/prediction/configs/config_predict_cancer_example.yaml
```

The patient ID must match the only patient in the H5. `target_side` must be
`left` or `right`. The model supplies preprocessing, feature schema, threshold,
report versions, and executable estimators; request YAML cannot override them.

Output names are generated under `io.output_folder`:

```text
*_prediction_dataframe.joblib
*_external_report.yaml
*_internal_report.yaml
```

The external report contains the target decision-support result. The internal
report adds profile evidence, contralateral scoring, symmetry status, TRA, and
audit metadata. Missing optional H5 metadata is reported as `unknown`.

Canonical contracts:

- [Prediction request](../../docs/contracts/prediction_config_v0_1.md)
- [Prediction route](../../docs/modeling/prediction_pipeline_v0_1.md)
- [Internal report](../../docs/modeling/internal_clinical_report_content_v0_9.md)
