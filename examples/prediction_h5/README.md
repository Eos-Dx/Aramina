# Aramis Prediction H5 Examples

These files are small one-patient H5 v0.3 containers for `aramis predict`
smoke tests.

They are synthetic fixtures. They test API behavior, preprocessing, reporting,
and one-patient container validation. They are not clinical examples.

Files:

```text
px01_one_patient.h5
px02_one_patient.h5
px03_one_patient.h5
cancer_one_patient.h5
atypical_one_patient.h5
benign_one_patient.h5
```

Each H5 contains:

```text
one patientId
left breast measurements
right breast measurements
3 measurements per breast
raw/data arrays
sample_thickness_mm
calibrant_thickness_mm
PONI text
schema_version = 0.3
format = xrd-session
```

Run:

```bash
cd Aramis
conda activate eosproduct

python -m aramis predict --config examples/prediction_h5/px01_predict.yaml
python -m aramis predict --config examples/prediction_h5/px02_predict.yaml
python -m aramis predict --config examples/prediction_h5/px03_predict.yaml

python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
python -m aramis predict --config examples/prediction_h5/atypical_predict.yaml
python -m aramis predict --config examples/prediction_h5/benign_predict.yaml
```

The YAML files use the tracked example model:

```text
examples/prediction_models/aramis_m2q_t100_gated_sk_core4_synthetic_h5_example.joblib
```

The H5 files are synthetic API fixtures. Their `p_cancer` values only prove that
the prediction route runs; they are not clinical example scores.

Reports are written to:

```text
examples/outputs/prediction_h5_examples/
```

Note:

```text
aramis_prediction_patient_raw_h5_example_v0_1.yaml
```

is example-only. It uses embedded `raw/data` arrays. Product H5 preprocessing
uses the prediction preprocessing config embedded in the trained model artifact.
