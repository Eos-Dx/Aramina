# Aramina preprocessing YAML

Formal product contract: `docs/contracts/preprocessing_config_v0_1.md`.

Current development configs:

```text
config_preprocessing_biopsy_patients_v0_1.yaml
config_preprocessing_prediction_patient_v0_1.yaml
```

The first config builds the T100 historical training cohort. It keeps patients
with at least one biopsied breast, retains contralateral measurements, and maps
NORMAL to BENIGN. The prediction config performs only measurement QC and signal
preparation for one incoming patient. It has no historical diagnosis, biopsy,
date, or AgBH exclusion filter.

Both top-level files use `extends` only for repository readability. At runtime,
`xrd_preprocessing.load_preprocessing_config` resolves them into one mapping.
`pipeline.steps` is the executable order. XRD-preprocessing permits flexible
routes, but the two Aramina product routes have a fixed validated order under the
Aramina preprocessing contract. XRD-preprocessing does not impose an Aramina
branch concept.

For a YAML stored beneath `Aramina/config`, operational `io` paths resolve from
the Aramina project root. An external top-level YAML resolves public paths from
its own directory. Repository `extends` are resolved by the XRD-preprocessing
loader. Repository configs use root-level paths such as
`./config/preprocessing/shared/aramina_pipeline_v0_1.yaml`.

```text
shared/       common XRD settings and ordered transformer pipeline
exclusions/   T100 AgBH quality exclusions and evidence reference
schema/       explicit retained DataFrame columns
```

Canonical product constraints:

```text
raw source: GFRM only
positions: P1, P2, P3
PONI coverage: q_max >= 23 nm^-1
sample thickness: required and >0 mm
calibrant thickness: required and 2-40 mm
integration: Poisson error model, 100 q points, 2-23 nm^-1
SNR: Poisson, >=18 dB
normalization: median value in 6.7-7.1 nm^-1
```

Run:

```bash
python -m aramina preprocess \
  --config config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml
```

Before using the historical-training config, supply the approved full archive
as `data/combined_archive.h5` under the Aramina project root. This large input
is deliberately excluded from Git. The prediction preprocessing config receives
its one-patient H5 path from the prediction request and does not use the
historical archive.

Output joblib contains:

```text
kind
version
created_at
dataframe
preprocessing_config_yaml   # fully resolved effective YAML
metadata                    # input H5 SHA256, Aramina version, git SHA
```

Contract details: `docs/data_preprocessing.md`.
