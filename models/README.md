# Aramis Model Artifacts

Each subdirectory is one self-contained trained model artifact.

```text
models/<model_id>/
  model.joblib
  model_description.yaml
  evaluation.yaml
  evaluation_metrics.csv
  evaluation_predictions.csv
  preprocessing_config.yaml
  prediction_preprocessing_config.yaml
  training_config.yaml
  preprocess_and_train_config.yaml
```

`model.joblib` is executable. `model_description.yaml` is its human-readable
identity and final-fit record. `evaluation.yaml` summarizes patient-safe
validation; the two CSV files hold row-oriented fold metrics and held-out
predictions. All YAML files beside the artifact are immutable training inputs.

Prediction configurations reference a specific `model.joblib`. They never
modify the artifact.

Current product artifact:

```text
aramis_target_breast_risk_0_2_7-beta_0222fcbe16fd/
```

Older artifacts are retained only for internal historical audit.
