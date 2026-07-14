# MLflow Traceability Plan

Status: planned after product-clean preprocess/train/predict route.

MLflow is not required to run the current CLI, but the product run must later
record preprocessing and model training together. Preprocessing defines the
dataset; a model artifact without preprocessing lineage is not reproducible.

## One Run

One Aramis MLflow run should represent:

```text
one dataset build
one model training/evaluation run
one fixed feature schema
one label mapping
one threshold policy
```

## Required Artifacts

```text
preprocessing_config.yaml/json
training_config.yaml/json
prediction_preprocessing_config.yaml/json
selected_measurement_ids.csv
dropped_measurements.csv
preprocessed_dataset.parquet or csv
feature_schema.json
label_mapping.json
split_manifest.csv
model.joblib
metrics.json
predictions.csv
prediction_report_schema.json
```

## Required Tags / Params

```text
product = Aramis
clinical_stage = research draft
model_id
selected_model
input_h5_checksum
preprocessing_config_sha256
training_config_sha256
prediction_preprocessing_config_sha256
aramis_git_sha
xrd_preprocessing_version
dataset_fingerprint
threshold_key
threshold_value
```

## Current State

Current Aramis joblibs already store the key pieces needed for MLflow:

```text
preprocessing joblib:
  preprocessing YAML text and SHA256
  input H5 SHA256

training joblib:
  training YAML text and SHA256
  training preprocessing YAML
  prediction preprocessing YAML
  feature schema
  metrics
  model objects

prediction report:
  input H5/model/data/config SHA256 provenance
```

MLflow should be connected after the product-clean candidate model and report
schema are stable.

