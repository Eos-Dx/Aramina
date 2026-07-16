# Aramis Product API v0.1

Status: research draft. Decision support only. Requires radiologist review.

## Commands

```bash
python -m aramis preprocess --config <preprocessing.yaml>
python -m aramis train --config <training.yaml>
python -m aramis preprocess-train --config <preprocess-train.yaml>
python -m aramis predict --config <prediction.yaml>
```

Operational paths inside public YAML resolve from the Aramis project root.
Preprocessing `extends` paths resolve from their declaring YAML. Unknown fields
in training, preprocess-train, and prediction contracts fail immediately.

## Preprocess

Input: EOS H5 container and an Aramis preprocessing YAML.

```text
H5
-> resolved pipeline.steps
-> sklearn Pipeline of XRD-preprocessing transformers
-> measurement-level DataFrame
-> preprocessing joblib artifact
```

The joblib contains the DataFrame, fully resolved effective YAML, input H5
SHA256, Aramis version, and git SHA. Contract:
`docs/data_preprocessing.md`.

## Train

Input: preprocessing joblib and strict training YAML.

```text
measurement profiles
-> LR1 target-breast profile LogisticRegression
-> logit-average target-breast p_cancer
-> LR2 M2Q with age and gated SK Core4 symmetry
-> frozen train-all model and threshold
```

`run.evaluation` writes patient-safe repeated stratified k-fold artifacts.
`run.train_on_all` fits the recipe on all accepted patients and freezes its
train-on-all threshold at recipe target sensitivity `>=0.95`.

The model joblib contains executable estimators, feature schema, threshold,
model identity, and resolved YAML snapshots. Fold metrics and predictions stay
in separate evaluation artifacts. Contract:
`docs/contracts/training_config_v0_1.md`.

## Preprocess-train

Input: one preprocess-train YAML referencing preprocessing and training YAMLs.

Preprocessing runs once. Its joblib is saved, while the DataFrame is passed
directly in memory to training. No reload is required between stages.

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

Outputs use one generated report ID:

```text
*_prediction_dataframe.joblib
*_external_report.json
*_external_report.yaml
*_internal_report.json
*_internal_report.yaml
```

External report contains suggested class, reliability, reliability reason,
patient/target identity, report identity, and model version. It intentionally
excludes `p_cancer` and threshold.

Internal report contains target and contralateral prediction blocks with
profile-only p_cancer, final p_cancer, decision threshold, class, frozen cohort
quantiles, symmetry availability, reliability, scan metadata, and model
artifact identity. Report contracts:
`config/prediction/README.md`.

## Stop Conditions

Aramis must fail on:

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
