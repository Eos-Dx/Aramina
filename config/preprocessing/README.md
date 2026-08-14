# Preprocessing YAML

Two fixed product routes are tracked:

| Config | Purpose |
|---|---|
| `config_preprocessing_biopsy_patients_v0_2.yaml` | Build historical training input. |
| `config_preprocessing_prediction_patient_v0_2.yaml` | Prepare one incoming patient. |

Both use readable `extends` fragments. XRD-preprocessing resolves them into one
effective mapping before execution.

```text
shared/       fixed XRD policy and ordered pipeline
exclusions/   controlled historical quality exclusions
schema/       retained DataFrame columns
```

Frozen numerical policy:

```text
source: GFRM
positions: P1, P2, P3
PONI q coverage: >=23 nm^-1
sample thickness: >0 mm
calibrant thickness: 2-40 mm
integration: 256 points, 2-23 nm^-1, Poisson error model
SNR: Poisson, >=18 dB
normalization: median over 6.7-7.1 nm^-1
```

```bash
python -m aramina preprocess \
  --config config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml
```

The historical route expects `data/combined_archive.h5`. Prediction replaces
that path with the request H5 and applies no historical cohort filters.

Canonical contracts:

- [Preprocessing config](../../docs/contracts/preprocessing_config_v0_2.md)
- [Data and artifact route](../../docs/data_preprocessing.md)
