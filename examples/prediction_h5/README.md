# Aramis Prediction H5 Examples

These files are small one-patient H5 v0.3 containers for `aramis predict`
examples. They are extracted from the larger study archive without modifying
the source archive. Each contains real embedded GFRM frames, PONI geometry,
sample thicknesses, calibrant thickness and both breast sides for one patient.

They are integration examples, not clinical validation cases. Historical
specimen status is included only to show the fixture type; prediction output is
decision support and must not be interpreted as a retrospective validation.

Files:

```text
benign_one_patient.h5
cancer_one_patient.h5
atypical_one_patient.h5
```

Each H5 contains:

```text
one patientId
two specimenIds: left and right breast
three measurements per breast
embedded GFRM measurement payloads
sample_thickness_mm and calibrant_thickness_mm
PONI text for every measurement
schema_version = 0.3
format = xrd-session
```

Run:

```bash
cd Aramis
conda activate eosproduct

python -m aramis predict --config examples/prediction_h5/benign_predict.yaml
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
python -m aramis predict --config examples/prediction_h5/atypical_predict.yaml
```

All three YAML files use the tracked final M2Q model artifact:

```text
examples/prediction_models/aramis_m2q_t100_0_2_6_beta.joblib
```

The fixture-builder script is provided for reproducibility. It requires local
access to the large source archive and is not part of normal installation:

```bash
python examples/prediction_h5/create_gfrm_prediction_fixtures.py \
  /path/to/combined_archive.h5
```

Reports are written to:

```text
examples/outputs/prediction_h5_examples/
```

Product preprocessing is read from the model artifact; it is not overridden by
these example YAML files.
