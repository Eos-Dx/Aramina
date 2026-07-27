# Aramina Product API v0.1

Status: research draft. Decision support only; not autonomous diagnosis.

## Commands

```bash
python -m aramina preprocess --config <preprocessing.yaml>
python -m aramina train --config <training.yaml>
python -m aramina preprocess-train --config <preprocessing-and-training.yaml>
python -m aramina predict --config <prediction.yaml>
```

For a YAML under `Aramina/config`, operational paths resolve from the Aramina
project root. External top-level YAML paths resolve from their YAML directory.
Preprocessing `extends` paths resolve under the XRD-preprocessing config loader.
Unknown fields in training, preprocessing-and-training, and prediction contracts
fail immediately.

## Preprocess

Input: EOS H5 container and an Aramina preprocessing YAML.

```text
H5
-> resolved pipeline.steps
-> sklearn Pipeline of XRD-preprocessing transformers
-> measurement-level DataFrame
-> preprocessing joblib artifact
```

The joblib contains the DataFrame, fully resolved effective YAML, input H5
SHA256, Aramina version, and git SHA. Contract:
`docs/data_preprocessing.md` and
`docs/contracts/preprocessing_config_v0_1.md`.

## Train

Input: preprocessing joblib and strict training YAML.

```text
measurement profiles
-> LR1 target-breast profile LogisticRegression
-> logit-average target-breast p_cancer
-> final logistic model with age and gated SK Core4 symmetry
-> frozen train-all model and threshold
```

`run.evaluation` writes patient-safe repeated stratified k-fold artifacts.
`run.train_on_all` fits the fixed product model on all accepted patients and
freezes its train-on-all threshold at target sensitivity `>=0.95`.

SK refinement is applied only when both breasts have at least two valid
measurements and Core4 is finite. Otherwise the same final logistic model uses
neutral SK inputs and retains profile-plus-age evidence.

The model joblib contains executable estimators, feature schema, threshold,
model identity, and resolved YAML snapshots. Fold metrics and predictions stay
in separate evaluation artifacts. Contracts:
`docs/contracts/training_config_v0_1.md` and
`docs/contracts/model_training_results_v0_1.md`.

## Preprocess-train

Input: one preprocessing-and-training YAML referencing preprocessing and training YAMLs.

Preprocessing runs once. Its joblib is saved, while the DataFrame is passed
directly in memory to training. No reload is required between stages.
Contract: `docs/contracts/preprocess_train_config_v0_1.md`.

## Predict

Input YAML:

```yaml
run:
  analysis_author: OPERATOR_OR_ANALYST
  prediction_comment: "optional request comment"
io:
  input_h5_path: /path/to/one_patient.h5
  input_model_joblib_path: /path/to/model.joblib
  output_folder: /path/to/output
patient:
  patient_id: PATIENT_ID
  target_side: left
```

Requirements:

```text
H5 schema_version and format match model-held contract
exactly one patient in H5
patient.patient_id exactly matches H5 patientId
target_side is left or right and comes from clinical caller
prediction preprocessing comes only from model joblib
```

Contract: `docs/contracts/prediction_config_v0_1.md`.

Outputs use one generated report ID:

```text
*_prediction_dataframe.joblib
*_external_report.yaml
*_internal_report.yaml
```

External report contains the target-side final risk probability, decision
threshold, `target_class_risk_level`, `biopsy_required`, reliability, reliability reason, patient/target
identity, report identity, model name/version, and final-model
sensitivity/specificity. `biopsy_required` is the sole target-side action:
`true` when the final risk probability meets or exceeds the frozen threshold,
otherwise `false`. It intentionally excludes profile-only scores, symmetry,
TRA, and model internals.
The internal report repeats `model_metrics` for audit as
`dataset: train_on_all_target_breast_cases` and `validation: not_performed`.

Internal report contains one shared threshold policy, a target block with LR1
profile and final p_cancer, and a contralateral full-model score with SK
symmetry refinement neutralized. Only the caller-selected target receives the
high/low target-class risk level and biopsy action. Both final scores use the
same frozen final-score target-case reference distribution. Report contracts:
`config/prediction/README.md`.

## Stop Conditions

Aramina must fail on:

```text
unknown YAML fields
missing required columns
unknown or single-class labels
inconsistent radial-profile length
patient leakage in evaluation
H5 schema/format mismatch
zero or multiple H5 patients
patient ID mismatch
missing model-held prediction preprocessing or prediction contract
```

Weak research metrics do not silently become a clinical release claim.
