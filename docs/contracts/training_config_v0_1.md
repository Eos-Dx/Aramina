# Training Config Contract v0.1

Status: research draft.

`python -m aramis train --config <yaml>` accepts a strict YAML contract. It selects one code-owned fixed product model and evaluation settings; it cannot alter the feature schema, regularization, label policy, target sensitivity, or prediction preprocessing.

```yaml
contract: aramis_training_config_v0_2
model:
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
| `evaluation.*` | Evaluation protocol. The current product model accepts repeated stratified k-fold only. |
| `model.name` | Fixed product model; use `python -m aramis train --list-models`. |

At least one `run` flag must be `true`. Both may be `true`. Evaluation chooses a threshold independently in each training fold and scores only held-out patients. Train-on-all freezes its own threshold from complete-cohort scores at the fixed model target sensitivity. The latter is an in-sample operating point, not an independent validation claim.

The caller may change `training` identity fields, `input`, `output`, the two
`run` flags, and `evaluation.folds`, `evaluation.repeats`, or
`evaluation.random_seed`. `evaluation.method` remains
`repeated_stratified_kfold` in this contract. Model-owned architecture,
feature schema, regularization, label mapping, target sensitivity, and
prediction preprocessing are not public YAML switches.

Training requires an Aramis preprocessing artifact with the resolved preprocessing YAML and input-H5 SHA256. The final joblib stores executable estimators, frozen thresholds, score reference distributions for internal report quantiles, all resolved YAML snapshots, source-H5 checksum, code provenance, and runtime package versions. `model_description.yaml` and `evaluation.yaml` provide the human-readable records; detailed fold metrics and held-out predictions remain in `evaluation_metrics.csv` and `evaluation_predictions.csv`.
