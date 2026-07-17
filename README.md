# Aramis

Status: research draft.

Aramis is an EOS breast-XRD decision-support prototype. It produces `p_cancer`,
a suggested BENIGN/CANCER support class, and reliability metadata for review by
a qualified breast-imaging clinician. It is not autonomous diagnosis, not a
biopsy replacement, and not a radiologist replacement.

## Intended Use

```text
target population:
  women with suspicious mammography findings, currently BI-RADS 3 / BI-RADS 4

clinical question:
  does the clinically suspicious breast likely need biopsy?

clinical user:
  radiologist / qualified breast-imaging clinician

input:
  one patient H5 container with left/right breast XRD measurements
  clinician-supplied target_side
  trained Aramis model joblib

output:
  p_cancer
  suggested BENIGN/CANCER decision-support class
  reliability level and reason
  JSON/YAML report payload
```

## Product Route

```text
one-patient H5
-> prediction preprocessing from model joblib
-> normalized radial_profile_data
-> LR1 profile model on target breast
-> one LR2: profile + age + optional gated SK Core4 refinement
-> p_cancer + suggested class + reliability
```

Training uses the same preprocessing family and fixed product model definition:

```text
model: aramis_m2q_t100
preprocessing: T100 biopsy-patient model-input DataFrame
regularization: LR1 L2 C=0.1; LR2 L2 C=0.3
default evaluation: repeated patient-safe stratified 5-fold x20
deployment threshold: train-all scores at target sensitivity >=0.95
training unit: one biopsied target breast
```

The model combines the target-breast XRD profile score, four fixed SK
target/contralateral symmetry fields and age in one final model. Profile and
age are always evaluated. When no contralateral breast is available, the gated
SK contribution is exactly zero. Age-only performance is reported as a
shortcut-risk control. Measurement counts remain reliability fields and
symmetry availability is a gate, not a learned risk feature.

The fixed architecture and its current evaluation are recorded in
[`docs/modeling/aramis_t100_target_case_model_v0_1.md`](docs/modeling/aramis_t100_target_case_model_v0_1.md).
The packaged artifact embeds its prediction preprocessing and immutable
prediction contract. Predict YAML supplies identity and paths only.

## Commands

Install from a fresh clone:

```bash
git clone https://github.com/Eos-Dx/Aramis.git
cd Aramis
./install.sh
```

Windows:

```bat
git clone https://github.com/Eos-Dx/Aramis.git
cd Aramis
install.bat
```

If `conda` is missing, the installer asks to install Miniforge. See:

```text
INSTALL.md
```

```bash
python -m aramis preprocess --config config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
python -m aramis train --config config/training/aramis_m2q_t100_primary_train_v0_1.yaml
python -m aramis preprocess-train --config config/preprocessing_and_training/aramis_biopsy_patients_primary_preprocessing_and_training_v0_1.yaml
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
```

`preprocess` and `train` build the development model artifacts. Product
prediction should start from H5 with `predict`. DataFrame prediction input is
kept only for tests and debugging.

Prediction template for a new patient H5:

```bash
config/prediction/aramis_predict_from_h5_template_v0_1.yaml
```

`run.train_on_all: true` creates a unique run folder with `model.joblib`,
`model_description.yaml`, `evaluation.yaml`, and separate detailed CSV evaluation
artifacts. Use that generated
model path in prediction YAML.

The packaged model also records full raw-H5 training reproducibility: H5
SHA256, YAML snapshots and checksums, code/dependency provenance, runtime
versions, and evaluation summary. The full-H5 reproduction bundle is built with
`packaging/reproducible_training_bundle/make_bundle.sh`; its Windows and
macOS/Linux launchers reuse existing environments, refresh pinned code commits,
and write a stage-by-stage training log.

## Documentation Map

Product API and developer contract:

```text
docs/product_api.md
```

Preprocessing contract:

```text
docs/data_preprocessing.md
config/preprocessing/README.md
```

Model and evidence:

```text
docs/modeling/aramis_t100_target_case_model_v0_1.md
docs/modeling/current_model_dataframe_v0_1.md
```

Prediction route and report schema:

```text
docs/modeling/prediction_pipeline_v0_1.md
config/prediction/README.md
```

Training contract:

```text
docs/contracts/training_config_v0_1.md
config/training/README.md
```

Combined preprocessing and training contract:

```text
config/preprocessing_and_training/README.md
```

Evidence for choices:

```text
docs/agbh_quality_exclusions.md
docs/modeling/aramis_t100_target_case_model_v0_1.md
docs/modeling/current_model_dataframe_v0_1.md
docs/meta/README.md
```

## Core Files

```text
src/aramis/pipelines.py
  YAML-governed preprocessing wrapper around XRD-preprocessing transformers

src/aramis/training.py
  patient-safe target-breast training artifact builder

src/aramis/prediction.py
  one-patient prediction route and report writer

src/aramis/workflows.py
  preprocess-train runner
```

## H5 Contract Summary

Prediction supports EOS H5 v0.3 in the current code:

```text
root @format = xrd-session
root @schema_version = 0.3
exactly one patientId per prediction H5
left/right breast specimen sets when available
sample thickness for every retained measurement
calibrant_thickness_mm in H5 metadata
PONI artifact for azimuthal integration
raw GFRM data for product preprocessing
```

The suspicious breast is not inferred from H5 labels. It is supplied in predict
YAML:

```yaml
patient:
  patient_id: PATIENT_ID
  target_side: Left
```

## Verification

Current product-code tests cover preprocessing, training, preprocess-train, and
prediction routes.

```bash
conda run --no-capture-output -n eosproduct ruff check .
conda run --no-capture-output -n eosproduct pytest -q
conda run --no-capture-output -n eosproduct pytest --cov=aramis --cov-report=term-missing -q
```
