# Aramina Training Config Contract v0.4

Status: research draft.

```yaml
contract: aramina_training_config_v0_4
model:
  name: aramina_target_breast_risk
  version: 0.2.14-beta
  model_author: Sergey Denisov
  clinical_stage: research draft
  intended_use: Breast cancer decision support; requires radiologist review.
run:
  evaluation: true
  train_on_all: true
input:
  dataframe_joblib_path: examples/outputs/model_input/aramina_biopsy_patients_model_input_v0_2.joblib
output:
  folder: examples/outputs/training
evaluation:
  method: repeated_stratified_kfold
  folds: 5
  repeats: 20
  random_seed: 42
```

Architecture, features, regularization, labels, target sensitivity, threshold
policy, and prediction preprocessing remain code-owned and unchanged from
`0.2.13-beta`. Evaluation is patient-safe repeated stratified 5-fold x20.
Train-on-all fits the executable artifact and freezes its own threshold.

Standalone training accepts only a preprocessing artifact containing the
resolved preprocessing YAML, input-H5 SHA256, and complete verified DVC data
identity. A plain DataFrame or artifact without DVC lineage is rejected.

Canonical tracked training uses the combined v0.3 workflow so one MLflow run
covers source H5 verification, preprocessing, evaluation, and final fit.
