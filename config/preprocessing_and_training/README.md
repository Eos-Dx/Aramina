# Preprocessing And Training YAML

`preprocess-train` runs approved preprocessing once, stores its traceable DataFrame,
and passes that in-memory DataFrame directly into training.

Formal request contract: `docs/contracts/preprocess_train_config_v0_1.md`.

```yaml
contract: aramina_preprocessing_and_training_config_v0_1
preprocessing_and_training:
  name: aramina_target_breast_risk_preprocessing_and_training
  run_author: Sergey Denisov
  output_folder: examples/outputs/preprocessing_and_training
preprocessing_config_path: config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml
training_config_path: config/training/config_training_target_breast_risk_v0_1.yaml
```

For a YAML under `Aramina/config`, relative paths resolve from the Aramina root.
For an external top-level YAML, they resolve from that YAML's directory.
`run_author` identifies the person starting the run. The resolved preprocessing
and training YAML are embedded in the resulting model artifact.

```bash
python -m aramina preprocess-train \
  --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml
```

Output is written under one unique folder with `preprocessing/dataframe.joblib`,
`preprocessing/cohort_summary.json`, evaluation artifacts, and the final model when
`run.train_on_all` is true.
