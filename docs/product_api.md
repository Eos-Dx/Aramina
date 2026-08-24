# Aramina Product API

Status: research draft. Decision support only; not autonomous diagnosis.

## Commands

```bash
python -m aramina preprocess --config <preprocessing.yaml>
python -m aramina train --config <training.yaml>
python -m aramina preprocess-train --config <preprocess-train.yaml>
python -m aramina predict --config <prediction.yaml>
python -m aramina promote --run-folder <reviewed-run>
```

Paths in repository configs resolve from the project root. Paths in external
top-level configs resolve from that config's directory.

## Preprocess

```text
EOS H5
-> resolved pipeline.steps
-> XRD-preprocessing transformers
-> measurement DataFrame
-> preprocessing joblib
```

The joblib contains the DataFrame, resolved YAML, input H5 SHA256, package
version, and Git commit. See [preprocessing](data_preprocessing.md).

## Train

```text
normalized measurement profiles
-> profile LogisticRegression
-> logit-average per target breast
-> final LogisticRegression with age and optional gated Core4 symmetry
-> frozen p_cancer threshold
```

Evaluation uses patient-safe repeated stratified folds. Train-on-all creates the
executable artifact and an in-sample operating point; it is not independent
validation. See [training config](contracts/training_config_v0_1.md) and
[training outputs](contracts/model_training_results_v0_1.md).

## Preprocess-Train

Preprocessing runs once. Its artifact is written, while the same DataFrame is
passed directly to patient-safe evaluation and final training. The v0.2 product
contract requires a local or configured MLflow mapping. One tracked run covers
the complete H5-to-model lineage, not a standalone classifier fit. See
[preprocess-train config](contracts/preprocess_train_config_v0_1.md).

The canonical configuration writes local MLflow tracking data to
`examples/outputs/mlflow/aramina_radial_profile.db` under
`aramina_product_radial_profile`. After a completed v0.2
run, inspect it with:

```bash
mlflow ui --backend-store-uri sqlite:///examples/outputs/mlflow/aramina_radial_profile.db --port 5000
```

`python -m aramina train` remains a developer command for an existing
preprocessing artifact. It is not a compliant full product MLflow run and must
not be represented as one. MLflow captures reproducibility metadata and model
outputs; it does not create an independent-validation, clinical-performance, or
regulatory claim.

## Predict

```yaml
run:
  analysis_author: OPERATOR_OR_ANALYST
  prediction_comment: optional request comment
io:
  input_h5_path: path/to/one_patient.h5
  input_model_joblib_path: path/to/model.joblib
  output_folder: path/to/output
patient:
  patient_id: PATIENT_ID
  target_side: left
```

The request must contain one H5 patient whose ID matches `patient_id`.
`target_side` is clinical input. Prediction preprocessing, model identity,
features, threshold, and report versions come from the model artifact.

Outputs:

```text
*_prediction_dataframe.joblib
*_external_report.yaml
*_internal_report.yaml
```

The external report contains target-side probability, threshold-derived
high/low risk, biopsy action, and reliability. The internal report adds profile
evidence, contralateral scoring, symmetry status, TRA, and audit metadata. See
[prediction config](contracts/prediction_config_v0_1.md) and
[prediction route](modeling/prediction_pipeline_v0_1.md).

## Stop Conditions

Execution stops on unknown config fields, missing columns, invalid labels,
profile-length mismatch, patient leakage, H5 contract mismatch, patient-ID
mismatch, absent target side, or incomplete model-held prediction metadata.
