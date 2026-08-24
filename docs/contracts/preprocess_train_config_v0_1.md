# Preprocess-Train Config Contract v0.2

Status: research draft. `python -m aramina preprocess-train --config <yaml>`
runs product preprocessing once, stores its artifact, and passes the same
in-memory DataFrame directly to evaluation and final training.

This file keeps its historical `v0_1` filename, while the contract declared
inside YAML is v0.2. The implementation rejects v0.1 input and fails closed
when MLflow initialization, lineage generation, or artifact logging fails.

```yaml
contract: aramina_preprocessing_and_training_config_v0_2
preprocessing_and_training:
  name: aramina_target_breast_risk_preprocessing_and_training
  run_author: REQUESTING_ANALYST
  output_folder: examples/outputs/preprocessing_and_training
preprocessing_config_path: config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml
training_config_path: config/training/config_training_target_breast_risk_v0_1.yaml
mlflow:
  enabled: true
  tracking_uri: sqlite:///examples/outputs/mlflow/aramina.db
  experiment_name: aramina_product_fpca30
```

## Strict Fields

All top-level fields above are required. Unknown or missing fields stop
execution. `preprocessing_and_training` accepts only `name`, `run_author`, and
`output_folder`. `mlflow` accepts only `enabled`, `tracking_uri`, and
`experiment_name`.

| Field | Type and rule |
|---|---|
| `mlflow.enabled` | Boolean. `true` is required for a compliant full product run. |
| `mlflow.tracking_uri` | Non-empty MLflow URI. The canonical local store is `sqlite:///examples/outputs/mlflow/aramina.db`. |
| `mlflow.experiment_name` | Non-empty MLflow experiment name. The canonical product experiment is `aramina_product_fpca30`. |

All paths and nested string values are non-empty. For a YAML under
`Aramina/config`, relative paths resolve from the Aramina project root. For an
external top-level YAML, they resolve from its own directory, not from the
working directory. A canonical relative SQLite database path resolves by the
same rule: `sqlite:///examples/outputs/mlflow/aramina.db` becomes an absolute
SQLite URI. Absolute SQLite and remote MLflow URIs remain unchanged.

## Full Product Run

One MLflow run represents exactly one complete product dataset build:

```text
input H5
-> product preprocessing
-> accepted and dropped measurement manifests
-> patient-safe evaluation
-> train-on-all final fit
-> tracked model and reproducibility artifacts
```

The child training config must set both `run.evaluation: true` and
`run.train_on_all: true`. The implementation must reject a v0.2 MLflow run when
either flag is false. Patient-safe evaluation is evidence for the fixed model;
train-on-all metrics describe the fitted cohort and are not independent
validation. MLflow records lineage and results. It does not itself create an
experiment, validation, clinical-performance, or regulatory claim.

`python -m aramina train` can remain a supported developer command, but it is
not a compliant full product run: it starts from a prebuilt DataFrame and does
not itself establish the complete H5-to-model lineage. It must not write a
product MLflow run under this contract.

## Required MLflow Tags

The implementation must write all tags below as strings before logging model
artifacts. Missing values stop the run.

| Tag | Required value source |
|---|---|
| `product` | Fixed product identifier: `aramina`. |
| `intended_use_id` | Code-owned intended-use identifier from the training model. |
| `clinical_stage` | Training configuration model field. |
| `data_contract` | `aramina_preprocessing_and_training_config_v0_2`. |
| `input_h5_id` | Stable input H5 identifier or basename recorded by preprocessing. |
| `input_h5_checksum` | SHA256 calculated from the input H5. |
| `pipeline_version` | Resolved product preprocessing pipeline version. |
| `preprocessing_git_sha` | Git SHA of the XRD-preprocessing implementation used. |
| `model_git_sha` | Git SHA of the Aramina implementation used. |
| `dataset_fingerprint` | Platform-independent SHA256 of accepted values, measurement identities, and feature schema. |

The implementation may add non-clinical search tags, but it must not overwrite
or omit these values.

## Required MLflow Artifacts

The tracked run must contain these artifact paths. JSON is for machine-readable
tracking metadata; the ordinary product output remains YAML and CSV according
to its existing contracts.

| Artifact path | Content |
|---|---|
| `preprocessing_config.json` | Fully resolved preprocessing configuration. |
| `product_filter_rules.json` | Applied product inclusion/exclusion rules. |
| `selected_measurement_ids.csv` | Accepted measurement IDs, patient IDs, and target-side linkage. |
| `dropped_measurements.csv` | Excluded measurements and recorded exclusion reason. |
| `preprocessed_dataset.parquet` | Accepted measurement-level product dataset. |
| `feature_schema.json` | Frozen LR1/LR2 input feature schema. |
| `label_mapping.json` | Product label mapping used for this run. |
| `train_test_split.csv` | Patient-safe fold assignment manifest. |
| `model.joblib` | Executable final train-on-all model artifact. |
| `metrics.json` | Aggregated evaluation and final-fit metrics, including threshold. |
| `predictions.csv` | Held-out patient-case predictions for all evaluation folds. |

The implementation must additionally retain the normal portable run outputs,
including `model_description.yaml`, `evaluation.yaml`, `evaluation_metrics.csv`,
`evaluation_predictions.csv`, and `evaluation_splits.csv`. MLflow artifact names above are fixed so one
run can be checked programmatically without interpreting run-folder layout.

## Local UI

After an implementation-backed v0.2 run completes:

```bash
mlflow ui --backend-store-uri sqlite:///examples/outputs/mlflow/aramina.db --port 5000
```

Open `http://127.0.0.1:5000`, select `aramina_product_fpca30`, and inspect the
run tags, artifacts, and logged metrics. The local UI reads the local tracking
store only; no remote experiment service is implied.

The generated run folder contains `preprocessing/dataframe.joblib`,
`preprocessing/cohort_summary.json`, requested evaluation artifacts, and the
final model when `run.train_on_all: true`. The child YAML contracts remain
authoritative: preprocessing is defined by `docs/data_preprocessing.md`; training
is defined by `training_config_v0_1.md` and
`model_training_results_v0_1.md`.
