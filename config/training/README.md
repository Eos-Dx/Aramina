# Aramis training YAML

Public contract: `aramis_training_config_v0_1`.

```yaml
contract: aramis_training_config_v0_1
training:
  name: aramis_m2q_t100
  version: 0.2.3-beta
  created_by: Sergey Denisov
  created_at: "2026-07-14"
  clinical_stage: research draft
  intended_use: Breast cancer decision support; requires radiologist review.
  mode: final_fit
input:
  dataframe_joblib_path: ../../examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
output:
  folder: ../../examples/outputs/training
model:
  recipe: m2q_gated_target_case_v0_1
evaluation:
  method: repeated_stratified_kfold
  folds: 5
  repeats: 20
  random_seed: 42
```

`training.mode`:

| Value | Result |
|---|---|
| `evaluation` | Patient-safe evaluation artifacts only. |
| `final_fit` | Evaluation first, then one train-all model and frozen threshold. |

Development exposes one evaluation method:

```text
repeated_stratified_kfold
```

Patients never cross train/test folds. The recipe, not the run YAML, fixes M2Q,
feature schema, LR1/LR2 regularization, label policy, prediction preprocessing,
and target sensitivity `0.95`.

List or inspect recipes:

```bash
python -m aramis train --list-recipes
python -m aramis train --describe-recipe m2q_gated_target_case_v0_1
```

Run:

```bash
python -m aramis train \
  --config config/training/aramis_m2q_t100_primary_train_v0_1.yaml
```

Every run creates a unique folder. `evaluation` writes:

```text
evaluation.joblib
evaluation.json
evaluation.yaml
evaluation_metrics.csv
evaluation_predictions.csv
```

`final_fit` also writes:

```text
model.joblib
model_description.yaml
```

The model joblib excludes fold predictions and metrics. It contains executable
M2Q, feature schema, frozen train-all threshold, and resolved training,
historical preprocessing, prediction preprocessing, and prediction-contract
YAML snapshots.

Full contract: `docs/contracts/training_config_v0_1.md`.
Recipe details: `docs/modeling/model_recipes_v0_1.md`.
