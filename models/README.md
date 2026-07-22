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

Tracked legacy compatibility artifact:

```text
aramis_target_breast_risk_0_2_10-beta_ccad65e77adb/
```

It carries `aramis_sk_symmetry_v0_1` and remains executable under the
compatibility path. The current source definition uses
`aramis_sk_symmetry_v0_2` and candidate version `0.2.11-beta`; it requires a
new evaluated train-on-all artifact before it can replace the tracked release
artifact.

Only the frozen reference artifact required by the current Git-tracked examples
is retained on `main`. Historical candidate artifacts belong on the experimental
branch.
