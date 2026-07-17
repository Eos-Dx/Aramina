# Preprocessing And Training YAML

`preprocess-train` runs approved preprocessing once, stores its traceable DataFrame,
and passes that in-memory DataFrame directly into training.

```yaml
contract: aramis_preprocessing_and_training_config_v0_1
preprocessing_and_training:
  name: aramis_target_breast_risk_preprocessing_and_training
  created_by: Sergey Denisov
  output_folder: ./examples/outputs/preprocessing_and_training
preprocessing_config_path: ./config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
training_config_path: ./config/training/aramis_target_breast_risk_primary_train_v0_1.yaml
```

All relative paths resolve from the Aramis project root. `created_by` identifies the
person starting the run. The resolved preprocessing and training YAML are embedded
in the resulting model artifact.

```bash
python -m aramis preprocess-train \
  --config config/preprocessing_and_training/aramis_target_breast_risk_preprocessing_and_training_v0_1.yaml
```

Output is written under one unique folder with `preprocessing/dataframe.joblib`,
`preprocessing/cohort_summary.json`, evaluation artifacts, and the final model when
`run.train_on_all` is true.
