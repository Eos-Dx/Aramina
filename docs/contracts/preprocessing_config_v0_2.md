# Aramina Preprocessing Config Contract v0.2

Status: research draft.

Current product YAMLs declare:

```yaml
aramina_preprocessing:
  contract: aramina_product_preprocessing_v0_2
  name: aramina_biopsy_patients_model_input
  version: 0.2
  clinical_stage: research draft
```

The historical-training route additionally requires `data_version` contract
`aramina_dvc_input_v0_1`. Aramina verifies the materialized H5 path, byte size,
DVC MD5, and independently calculated SHA256 before preprocessing. The verified
identity is stored in preprocessing, training, and MLflow artifacts.

The one-patient prediction route uses the same technical preprocessing policy
but accepts the request H5 directly and does not use DVC.
Current code also recognizes the embedded v0.1 prediction payload in frozen
`0.2.13-beta` joblibs; this compatibility path cannot be used for training.

Numerical preprocessing is unchanged from v0.1: GFRM source; P1/P2/P3;
100-point integration over 2-23 nm^-1 with Poisson errors; Poisson SNR >=18 dB;
median normalization over 6.7-7.1 nm^-1; and the same ordered XRD transformers,
T100 exclusions, label mapping, and retained columns.

Runnable YAMLs:

- `config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml`
- `config/preprocessing/config_preprocessing_prediction_patient_v0_2.yaml`

Changing numerical preprocessing remains a model change and requires updated
code, documentation, tests, model version, and tracked artifacts.
