# Aramis Training YAML

Public contract: `aramis_training_config_v0_1`.

```yaml
contract: aramis_training_config_v0_1
training:
  name: aramis_m2q_t100
  version: 0.2.7-beta
  created_by: Sergey Denisov
  clinical_stage: research draft
  intended_use: Breast cancer decision support; requires radiologist review.
run:
  evaluation: true
  train_on_all: true
input:
  dataframe_joblib_path: ./examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
output:
  folder: ./examples/outputs/training
model:
  recipe: m2q_gated_target_case_v0_1
evaluation:
  method: repeated_stratified_kfold
  folds: 5
  repeats: 20
  random_seed: 42
```

All relative paths resolve from the Aramis project root. At least one `run` flag must be true. `evaluation` writes patient-safe evaluation artifacts. `train_on_all` writes a frozen executable model; it can be run with or without an evaluation artifact. The caller may set run flags, run identity, input/output paths, and supported evaluation folds, repeats, and seed. The recipe fixes architecture, feature schema, LR1/LR2 regularization, label policy, prediction preprocessing, and target sensitivity.

```bash
python -m aramis train --list-recipes
python -m aramis train --describe-recipe m2q_gated_target_case_v0_1
python -m aramis train --config config/training/aramis_m2q_t100_primary_train_v0_1.yaml
```

Full contract: `docs/contracts/training_config_v0_1.md`.
