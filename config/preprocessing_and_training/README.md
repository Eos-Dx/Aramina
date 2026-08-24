# Preprocess-Train YAML

This command runs preprocessing once and passes the same in-memory DataFrame to
training.

```yaml
contract: aramina_preprocessing_and_training_config_v0_2
preprocessing_and_training:
  name: aramina_target_breast_risk_preprocessing_and_training
  run_author: Sergey Denisov
  output_folder: examples/outputs/preprocessing_and_training
preprocessing_config_path: config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml
training_config_path: config/training/config_training_target_breast_risk_v0_1.yaml
mlflow:
  enabled: true
  tracking_uri: sqlite:///examples/outputs/mlflow/aramina.db
  experiment_name: aramina_product_fpca30
```

```bash
python -m aramina preprocess-train \
  --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml
```

The run folder contains the preprocessing artifact, cohort summary, requested
evaluation files, and the final model when `run.train_on_all` is true.

`mlflow` is mandatory in the v0.2 product contract. The canonical configuration
writes a local SQLite MLflow database at `examples/outputs/mlflow/aramina.db` and uses the
`aramina_product_fpca30` experiment. One MLflow run represents one complete
product dataset build, patient-safe evaluation, and final train-on-all fit. It
does not create an independent-validation or clinical-performance claim.

The checked-in Python implementation accepts this strict v0.2 contract and
fails closed when tracking or required lineage artifacts are incomplete. Do not
use `python -m aramina train` as a tracked product run: standalone training
lacks the complete H5-to-model lineage.

Inspect completed local runs with:

```bash
mlflow ui --backend-store-uri sqlite:///examples/outputs/mlflow/aramina.db --port 5000
```

Open `http://127.0.0.1:5000` in a browser.

Canonical contract:
[Preprocess-train config](../../docs/contracts/preprocess_train_config_v0_1.md).
