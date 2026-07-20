# Preprocess-Train Config Contract v0.1

Status: research draft. `python -m aramis preprocess-train --config <yaml>`
runs product preprocessing once, stores its artifact, and passes the same
in-memory DataFrame directly to training.

```yaml
contract: aramis_preprocessing_and_training_config_v0_1
preprocessing_and_training:
  name: aramis_target_breast_risk_preprocessing_and_training
  run_author: REQUESTING_ANALYST
  output_folder: examples/outputs/preprocessing_and_training
preprocessing_config_path: config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
training_config_path: config/training/aramis_target_breast_risk_primary_train_v0_1.yaml
```

All fields are required. All paths and nested values are non-empty strings.
Unknown or missing fields stop execution. Relative paths resolve from the
Aramis project root, not from the working directory.

The generated run folder contains `preprocessing/dataframe.joblib`,
`preprocessing/cohort_summary.json`, requested evaluation artifacts, and the
final model when `run.train_on_all: true`. The child YAML contracts remain
authoritative: preprocessing is defined by `docs/data_preprocessing.md`; training
is defined by `training_config_v0_1.md` and
`model_training_results_v0_1.md`.
