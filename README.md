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

Training uses the same preprocessing family but a historical model-development
cohort. Current untagged development evaluation is:

```text
architecture_id: m2q_gated_target_case_v0_1
evaluation_run: aramis_t100_honest_nested_operational_experiment
preprocessing: T100 biopsy-patient model-input DataFrame
selected_model: M2Q
regularization: LR1 L2 C=0.1; LR2 L2 C=0.3
threshold: derived from inner patient-safe out-of-fold predictions at target sensitivity 0.95
training unit: one biopsied target breast
```

M2Q combines the target-breast XRD profile score, four fixed SK
target/contralateral symmetry fields and age in one final model. Profile and
age are always evaluated. When no contralateral breast is available, the gated
SK contribution is exactly zero. Age-only performance is reported as a
shortcut-risk control. Measurement counts remain reliability fields and
symmetry availability is a gate, not a learned risk feature.

The fixed architecture and its current evaluation are recorded in
[`docs/modeling/m2q_gated_target_case_model_v0_1.md`](docs/modeling/m2q_gated_target_case_model_v0_1.md).
The packaged M2Q artifact embeds its prediction preprocessing and immutable
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
python -m aramis run --config config/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
```

`preprocess` and `train` build the development model artifacts. Product
prediction should start from H5 with `predict`. DataFrame prediction input is
kept only for tests and debugging.

Prediction template for a new patient H5:

```bash
config/prediction/aramis_predict_from_h5_template_v0_1.yaml
```

Working one-patient prediction examples use:

```text
examples/prediction_h5/cancer_one_patient.h5
examples/prediction_models/aramis_m2q_t100_gated_sk_core4_synthetic_h5_example.joblib
```

The synthetic artifact is for repository smoke tests only. The product artifact
is `aramis_m2q_t100_gated_sk_core4_nested_c1_0p1_c2_0p3.joblib` and embeds the
production GFRM preprocessing contract.

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

Model concept and candidate choice:

```text
docs/machine_learning_concept.md
docs/modeling/m2q_gated_target_case_model_v0_1.md
docs/modeling/current_model_pipeline_and_risks_v0_1.md
docs/documentation_audit_v0_1.md
```

Prediction route and report schema:

```text
docs/modeling/prediction_pipeline_v0_1.md
config/prediction/README.md
```

Training route:

```text
docs/modeling/training_pipeline_classes_v0_1.md
config/training/README.md
```

Evidence for choices:

```text
docs/agbh_quality_exclusions.md
docs/modeling/m1q_regularization_experiment_v0_1.md
docs/modeling/m1q_threshold_mode_comparison_v0_1.md
docs/modeling/current_model_dataframe_v0_1.md
docs/meta/README.md
```

## Core Files

```text
src/aramis/pipelines.py
  YAML-governed preprocessing wrapper around XRD-preprocessing transformers

src/aramis/training.py
  patient-safe target-breast M0/M0Q/M1/M1Q/M2/M2Q training artifact builder

src/aramis/prediction.py
  one-patient prediction route and report writer

src/aramis/workflows.py
  preprocess+train workflow runner
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

Current product-code tests cover preprocessing, training, workflow, and
prediction routes.

```bash
conda run --no-capture-output -n eosproduct ruff check .
conda run --no-capture-output -n eosproduct pytest -q
conda run --no-capture-output -n eosproduct pytest --cov=aramis --cov-report=term-missing -q
```
