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

Evaluation method is fixed to patient-safe repeated stratified k-fold. The
current default product YAML uses `5-fold x20` with seed `42`, but `folds`,
`repeats`, and `random_seed` are run parameters. The exact values used for a
run are stored in the evaluation artifact and final model joblib. Each fold
chooses its threshold from the training patients and reports metrics only on
held-out patients. `final_fit` runs this evaluation first, then trains on the
complete accepted cohort and freezes the threshold from train-all scores at
recipe target sensitivity `>=0.95`. The in-sample result is not an independent
validation claim.

The final model joblib stores this immutable evaluation protocol and compact
metric summary. Detailed fold metrics and held-out predictions remain in the
separate `evaluation.*` artifacts beside the model.

## Reproducibility record

Every final model includes `reproducibility`, a portable machine-readable
record. It contains the input-H5 filename and SHA256, resolved historical and
prediction preprocessing YAML, training YAML, the optional preprocess-train
workflow YAML, SHA256 values for each YAML snapshot, Aramis source commit,
installed XRD-preprocessing provenance, Python/platform details, and versions
of the runtime packages used by the model. Absolute local paths are not used as
identity. A direct `train` run records `preprocessed_artifact_train`; a
`preprocess-train` run records `raw_h5_preprocess_train` and preserves the
workflow YAML that connects raw H5 to the model.
