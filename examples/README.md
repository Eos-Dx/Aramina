# Aramis Product Examples

Status: research draft.

Current product examples are one-patient H5 prediction fixtures and packaged
M2Q artifacts:

```text
prediction_h5/
prediction_models/
```

Run a complete one-patient H5 prediction example:

```bash
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
```

Build the current biopsy-patient model input:

```bash
python -m aramis preprocess --config config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
```

Build and train in one workflow:

```bash
python -m aramis preprocess-train --config config/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml
```

Historical Marimo notebooks, all-patient cohorts, and threshold-grid outputs
are retained in `experiment/aramis-model-selection-v0.1`.
