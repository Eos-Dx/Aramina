# Preprocess-Train Config Contract v0.1

Status: research draft. `python -m aramina preprocess-train --config <yaml>`
runs product preprocessing once, stores its artifact, and passes the same
in-memory DataFrame directly to training.

```yaml
contract: aramina_preprocessing_and_training_config_v0_1
preprocessing_and_training:
  name: aramina_target_breast_risk_preprocessing_and_training
  run_author: REQUESTING_ANALYST
  output_folder: examples/outputs/preprocessing_and_training
preprocessing_config_path: config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml
training_config_path: config/training/config_training_target_breast_risk_v0_1.yaml
```

All fields are required. All paths and nested values are non-empty strings.
Unknown or missing fields stop execution. For a YAML under `Aramina/config`,
relative paths resolve from the Aramina project root. For an external top-level
YAML, they resolve from its own directory, not from the working directory.

The generated run folder contains `preprocessing/dataframe.joblib`,
`preprocessing/cohort_summary.json`, requested evaluation artifacts, and the
final model when `run.train_on_all: true`. The child YAML contracts remain
authoritative: preprocessing is defined by `docs/data_preprocessing.md`; training
is defined by `training_config_v0_1.md` and
`model_training_results_v0_1.md`.
