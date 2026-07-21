# Aramis Product Examples

Status: research draft.

Current product examples are one-patient H5 prediction fixtures and packaged
Current product-model artifacts:

```text
prediction_h5/
prediction_models/
```

Run a complete one-patient H5 prediction example:

```bash
python -m aramis predict --config examples/prediction/configs/config_predict_cancer_example.yaml
```

Build the current biopsy-patient model input:

```bash
python -m aramis preprocess --config config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml
```

Build and train in one combined run:

```bash
python -m aramis preprocess-train --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml
```

Historical Marimo notebooks, all-patient cohorts, and threshold-grid outputs
are retained in `experiment/aramis-model-selection-v0.1`.
