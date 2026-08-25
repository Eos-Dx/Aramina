# Legacy 0.2.13-beta Contracts

Tag `0.2.13-beta` remains the implementation baseline without DVC.

Legacy runnable YAMLs retained in the repository:

- `config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml`
- `config/preprocessing/config_preprocessing_prediction_patient_v0_1.yaml`
- `config/training/config_training_target_breast_risk_v0_1.yaml`
- `config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml`

Use these files with code checked out at tag `0.2.13-beta`. Current historical
training accepts only the new DVC schemas and intentionally does not add
training aliases. Current prediction retains a narrow compatibility path for
the v0.1 payload embedded in frozen `0.2.13-beta` joblibs. This keeps the
implementation team on the no-DVC release while new development can use DVC
without silently changing the legacy training runtime.
