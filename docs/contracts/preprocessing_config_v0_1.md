# Aramina Preprocessing Config Contract v0.1

Status: legacy research draft for tag `0.2.13-beta`.

The two v0.1 YAMLs define historical-training and one-patient prediction routes:

- `config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml`
- `config/preprocessing/config_preprocessing_prediction_patient_v0_1.yaml`

Both use the fixed GFRM, P1/P2/P3, 100-point integration, Poisson SNR >=18 dB,
6.7-7.1 nm^-1 median normalization, and ordered preprocessing policy. Historical
training uses T100 exclusions and biopsy-patient cohort selection. Prediction
disables historical cohort filters.

This contract predates DVC and does not contain `data_version`. Use it only with
code checked out at tag `0.2.13-beta`. Current development uses
[preprocessing contract v0.2](preprocessing_config_v0_2.md).
