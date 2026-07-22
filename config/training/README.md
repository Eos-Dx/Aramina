# Aramis Training YAML

Public contract: `aramis_training_config_v0_3`.

```yaml
contract: aramis_training_config_v0_3
model:
  name: aramis_target_breast_risk
  version: 0.2.11-beta
  model_author: Sergey Denisov
  clinical_stage: research draft
  intended_use: Breast cancer decision support; requires radiologist review.
run:
  evaluation: true
  train_on_all: true
input:
  dataframe_joblib_path: examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
output:
  folder: examples/outputs/training
evaluation:
  method: repeated_stratified_kfold
  folds: 5
  repeats: 20
  random_seed: 42
```

For a YAML under `Aramis/config`, relative paths resolve from the Aramis root.
For an external top-level YAML, they resolve from that YAML's directory. At
least one `run` flag must be true. `evaluation` writes patient-safe evaluation
artifacts. `train_on_all` writes a frozen executable model; it can be run with
or without an evaluation artifact. The caller may set run flags, model identity,
input/output paths, and supported evaluation folds, repeats, and seed. The
selected model name fixes architecture, feature schema, LR1/LR2 regularization,
label policy, prediction preprocessing, and target sensitivity. `model_author`
identifies the author of the approved model recipe; `run_author` in the combined
preprocess-train YAML identifies the person executing a particular run.

```bash
python -m aramis train --list-models
python -m aramis train --describe-model aramis_target_breast_risk
python -m aramis train --config config/training/config_training_target_breast_risk_v0_1.yaml
```

Full contract: `docs/contracts/training_config_v0_1.md`.
