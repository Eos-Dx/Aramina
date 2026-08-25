# Legacy 0.2.13-beta Contracts

Tag `0.2.13-beta` remains the implementation baseline without DVC.

Legacy runnable YAMLs retained in the repository:

- `config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml`
- `config/preprocessing/config_preprocessing_prediction_patient_v0_1.yaml`
- `config/training/config_training_target_breast_risk_v0_1.yaml`
- `config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml`

Use these files with code checked out at tag `0.2.13-beta`. Current code accepts
only the new schema contracts and intentionally does not add aliases. This keeps
the implementation team on the frozen no-DVC release while new development can
use DVC without silently changing the legacy runtime.
