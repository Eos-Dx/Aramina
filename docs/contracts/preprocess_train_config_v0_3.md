# Preprocess-Train Config Contract v0.3

Status: research draft.

```yaml
contract: aramina_preprocessing_and_training_config_v0_3
preprocessing_and_training:
  name: aramina_target_breast_risk_preprocessing_and_training
  run_author: Sergey Denisov
  output_folder: examples/outputs/preprocessing_and_training
preprocessing_config_path: config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml
training_config_path: config/training/config_training_target_breast_risk_v0_4.yaml
mlflow:
  enabled: true
  tracking_uri: sqlite:///examples/outputs/mlflow/aramina_radial_profile.db
  experiment_name: aramina_product_radial_profile
```

One compliant run verifies the DVC-tracked historical H5, runs fixed product
preprocessing, writes accepted and dropped measurement manifests, evaluates on
patient-safe folds, trains on all accepted target cases, and logs the complete
artifact set to MLflow. Missing source provenance, DVC identity, evaluation,
final fit, or required MLflow artifact stops the run.

Prediction is separate. It receives one external patient H5 directly and does
not require DVC.

MLflow artifact contract: `aramina_mlflow_product_run_v0_3`.
