# Training YAML

Training selects one code-owned product architecture. YAML controls run identity,
input/output paths, and evaluation repetition; it cannot change features,
regularization, labels, target sensitivity, or prediction preprocessing.

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
  dataframe_joblib_path: examples/outputs/model_input/dataframe.joblib
output:
  folder: examples/outputs/training
evaluation:
  method: repeated_stratified_kfold
  folds: 5
  repeats: 20
  random_seed: 42
```

```bash
python -m aramina train --list-models
python -m aramina train --describe-model aramina_target_breast_risk
python -m aramina train \
  --config config/training/config_training_target_breast_risk_v0_4.yaml
```

`run.evaluation` writes patient-safe held-out metrics. `run.train_on_all` writes
the executable model and its own frozen threshold. At least one flag must be
true.

Current contract: [Training config v0.4](../../docs/contracts/training_config_v0_4.md).
The legacy `v0_1` YAML remains with tag `0.2.13-beta` and is documented
[separately](../../docs/contracts/training_config_v0_1.md).
