# Preprocess-Train YAML

This command runs preprocessing once and passes the same in-memory DataFrame to
training.

```yaml
contract: aramina_preprocessing_and_training_config_v0_1
preprocessing_and_training:
  name: aramina_target_breast_risk_preprocessing_and_training
  run_author: Sergey Denisov
  output_folder: examples/outputs/preprocessing_and_training
preprocessing_config_path: config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml
training_config_path: config/training/config_training_target_breast_risk_v0_1.yaml
```

```bash
python -m aramina preprocess-train \
  --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml
```

The run folder contains the preprocessing artifact, cohort summary, requested
evaluation files, and the final model when `run.train_on_all` is true.

Canonical contract:
[Preprocess-train config](../../docs/contracts/preprocess_train_config_v0_1.md).
