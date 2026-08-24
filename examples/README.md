# Aramina Product Examples

Status: research draft.

Validated acquisition record:

```text
acquisition/aramina_acquisition_protocol_v0_1.yaml
```

Current examples are one-patient H5 prediction fixtures and the frozen product
model:

```text
prediction_h5/
../models/
```

Run a complete one-patient H5 prediction example:

```bash
python -m aramina predict --config examples/prediction/configs/config_predict_cancer_example.yaml
```

Build the current biopsy-patient model input:

```bash
python -m aramina preprocess --config config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml
```

Build and train in one combined run:

```bash
python -m aramina preprocess-train --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml
```

Historical notebooks, alternative cohorts, and threshold-grid outputs remain on
experiment branches.
