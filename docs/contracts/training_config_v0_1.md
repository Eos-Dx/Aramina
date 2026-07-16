# Training Config Contract v0.1

Status: research draft.

`python -m aramis train --config <yaml>` accepts a strict YAML contract. It selects a code-owned immutable recipe and evaluation settings; it cannot alter the feature schema, regularization, label policy, target sensitivity, or prediction preprocessing.

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

All relative paths resolve from the Aramis project root. Unknown or missing fields fail immediately.

| Field | Meaning |
|---|---|
| `run.evaluation` | Write patient-safe evaluation artifacts. |
| `run.train_on_all` | Fit one executable model on the complete accepted cohort and write `model.joblib`. |
| `evaluation.*` | Evaluation protocol. Current recipe accepts repeated stratified k-fold only. |
| `model.recipe` | Immutable recipe ID; use `python -m aramis train --list-recipes`. |

At least one `run` flag must be `true`. Both may be `true`. Evaluation chooses a threshold independently in each training fold and scores only held-out patients. Train-on-all freezes its own threshold from complete-cohort scores at the recipe target sensitivity. The latter is an in-sample operating point, not an independent validation claim.

Training requires an Aramis preprocessing artifact with the resolved preprocessing YAML and input-H5 SHA256. The final joblib stores executable estimators, frozen thresholds, score reference distributions for internal report quantiles, all resolved YAML snapshots, source-H5 checksum, code provenance, and runtime package versions. Detailed fold predictions and metrics remain next to the model as `evaluation.*` artifacts.
