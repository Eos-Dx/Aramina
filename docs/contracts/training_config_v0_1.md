# Training Config Contract v0.1

Status: research draft.

## Purpose

`python -m aramis train --config <yaml>` accepts one strict YAML contract. The
file selects a frozen recipe and evaluation size. It cannot override model
features, regularization, label mapping, target sensitivity, or prediction
preprocessing.

## Fields

| Field | Type | Allowed |
|---|---|---|
| `contract` | string | `aramis_training_config_v0_1` |
| `training.name` | string | non-empty run/model name |
| `training.version` | string | research version |
| `training.created_by` | string | author |
| `training.created_at` | string | ISO date/time |
| `training.clinical_stage` | string | `research draft` for current work |
| `training.intended_use` | string | required decision-support statement |
| `training.mode` | string | `evaluation`, `final_fit` |
| `input.dataframe_joblib_path` | path | Aramis preprocessing artifact joblib |
| `output.folder` | path | parent for unique run folder |
| `model.recipe` | string | ID from code-owned model recipe registry |
| `evaluation.method` | string | `repeated_stratified_kfold` |
| `evaluation.folds` | integer | >=2 |
| `evaluation.repeats` | integer | >=1 |
| `evaluation.random_seed` | integer | deterministic split seed |

The evaluation artifact also reports 95% patient-cluster bootstrap intervals.
Each bootstrap draw retains every target-breast case belonging to a sampled
patient, including bilateral biopsy cases.

Relative paths resolve from the declaring YAML. Unknown and missing fields are
errors.

## Input validation

Training requires an Aramis preprocessing artifact with the resolved
preprocessing YAML and input-H5 SHA256, recipe columns, two labels, finite
fixed-length radial profiles, and patient-safe grouping. Missing age is imputed
from the training-fold median and represented by `age_available=false`.
Technical contract errors stop the run. Weak metrics do not block a `research
draft` final fit.

## Thresholds

Evaluation is fixed to patient-safe repeated stratified `5-fold x20` with seed
`42`. Each fold chooses its threshold from the training patients and reports
metrics only on held-out patients. `final_fit` runs this evaluation first, then
trains on the complete accepted cohort and freezes the threshold from train-all
scores at recipe target sensitivity `>=0.95`. The in-sample result is not an
independent validation claim.
