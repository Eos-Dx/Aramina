# Aramina Model Artifacts

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
predictions. The preprocessing, prediction-preprocessing, training, and
preprocess-train YAML files are frozen inputs; the model description and
evaluation YAML are generated output records.

Prediction configurations reference a specific `model.joblib`. They never
modify the artifact.

Tracked product artifacts:

```text
aramina_target_breast_risk_0_2_13-beta_f5e4a04cad11/
aramina_target_breast_risk_0_2_12-beta_9bb911189af6/
aramina_target_breast_risk_0_2_14-beta_98526329f40d/
```

`0.2.12-beta` remains the frozen legacy product artifact used by existing
packaged runtime components. Current source-checkout prediction examples use
the DVC-tracked `0.2.14-beta` candidate.

`0.2.13-beta` is a separately retrained candidate using XRD-preprocessing
`v0.1.9-beta` at commit `88dcaa2`; its model record is
[`docs/modeling/aramina_t100_target_case_model_v0_2.md`](../docs/modeling/aramina_t100_target_case_model_v0_2.md).

`0.2.14-beta` uses the same fixed preprocessing and model recipe as the
no-DVC candidate. Its historical H5 is verified through DVC and its complete
preprocessing/evaluation/final-fit lineage is recorded by MLflow. Its model
record is
[`docs/modeling/aramina_t100_target_case_model_v0_3.md`](../docs/modeling/aramina_t100_target_case_model_v0_3.md).

All artifacts carry `aramina_sk_symmetry_v0_2`, the threshold-centred
`aramina_tra_v0_2` policy, and generated evaluation records. Other historical
candidates belong on experiment branches.
