# Aramis preprocessing YAML

Current development configs:

```text
aramis_biopsy_patients_model_input_v0_1.yaml
aramis_prediction_patient_model_input_v0_1.yaml
```

The first config builds the T100 historical training cohort. It keeps patients
with at least one biopsied breast, retains contralateral measurements, and maps
NORMAL to BENIGN. The prediction config performs only measurement QC and signal
preparation for one incoming patient. It has no historical diagnosis, biopsy,
date, or AgBH exclusion filter.

Both top-level files use `extends` only for repository readability. At runtime,
`xrd_preprocessing.load_preprocessing_config` resolves them into one mapping.
`pipeline.steps` is the executable order. Steps may be added, removed, disabled,
or reordered when the transformer contract allows it. XRD-preprocessing does
not impose an Aramis branch concept.

Operational `io` paths and `extends` paths are relative to the Aramis project
root. Repository configs use root-level paths such as
`./config/preprocessing/shared/aramis_pipeline_v0_1.yaml`.

```text
shared/       common XRD settings and ordered transformer pipeline
exclusions/   T100 AgBH quality exclusions and evidence reference
outputs/      explicit retained DataFrame columns
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
python -m aramis preprocess \
  --config config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
```

Output joblib contains:

```text
kind
version
created_at
dataframe
preprocessing_config_yaml   # fully resolved effective YAML
metadata                    # input H5 SHA256, Aramis version, git SHA
```

Contract details: `docs/data_preprocessing.md`.
