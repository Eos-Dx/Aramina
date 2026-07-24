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
predictions. The preprocessing, prediction-preprocessing, training, and optional
preprocess-train YAML files are frozen inputs; the model description and
evaluation YAML are generated output records.

Prediction configurations reference a specific `model.joblib`. They never
modify the artifact.

Tracked frozen product artifact:

```text
aramis_target_breast_risk_0_2_11-beta_d531ea38c5dc/
```

It carries `aramis_sk_symmetry_v0_2`, the threshold-centred
`aramis_tra_v0_2` policy, and the current report contracts. Its TRA policy is
calibrated from the repeated patient-safe OOF evaluation stored beside the
artifact. Only the current frozen product artifact is retained on `main`;
historical candidates belong on the experimental branch.
