# Preprocess-Train Config Contract v0.1

Status: legacy research draft for tag `0.2.13-beta`.

```yaml
contract: aramina_preprocessing_and_training_config_v0_1
preprocessing_and_training:
  name: aramina_target_breast_risk_preprocessing_and_training
  run_author: Sergey Denisov
  output_folder: examples/outputs/preprocessing_and_training
preprocessing_config_path: config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml
training_config_path: config/training/config_training_target_breast_risk_v0_1.yaml
```

This legacy workflow runs preprocessing and training without DVC or MLflow.
Use it with code checked out at tag `0.2.13-beta`. Current tracked development
uses [preprocess-train contract v0.3](preprocess_train_config_v0_3.md).
