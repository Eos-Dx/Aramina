# Aramis

Aramis is the EOS research draft product for breast XRD decision support.

The assembly-plan source is:

```text
/Users/sad/Downloads/aramis_assembly_plan_v1.docx
```

## Product Description

Aramis is planned as an ML classifier deployed on the EOS platform and connected
to client-facing software. For XRD scans from patients referred to biopsy, it
provides a breast-level malignancy risk output for decision support.

Clinical problem:

```text
reduce unnecessary biopsies for patients with suspicious tumors after mammography
support tumor malignancy risk assessment
provide supplementary decision support
require qualified clinician / radiologist review
```

Draft target population:

```text
patients with suspicious breast findings after mammography analysis
patients referred to biopsy
Nova-study Human-1 patients 101-438 for model-development data
```

Draft classification task:

```text
malignant vs benign
low/high p_cancer class
breast/sample-level output for suspicious side
```

This repository must not present Aramis as autonomous diagnosis, biopsy
replacement, radiologist replacement, FDA-cleared, or clinically validated unless
separate validation and regulatory evidence is added.

## Planned Product Deliverables

Assembly-plan deliverables:

```text
ML classifier in joblib format
standardized documented code
reproducible training pipeline
training QC criteria
public clinical report YAML/template
internal clinical report YAML/template
platform integration plan
```

Draft classifier use:

```text
input:
  XRD scan of both left and right breast sides
  at least one measurement per side
  H5 container input
  model-training config file

output:
  p_cancer / low-high malignancy-risk class
  YAML with information for public/internal reports
  classifier training QC criteria
```

Assembly-plan target QC criteria are planning targets, not validated product
performance:

```text
sensitivity target: >95%
specificity target: maximize, target >50%
```

## Repository Split

`XRD-preprocessing` is the common preprocessing core for Aramis and Bremen:

```text
H5 raw data
-> normalized azimuthally integrated curves
-> intensity vs q
```

This Aramis repository owns Aramis-specific processing and modeling:

```text
azimuthally integrated curves
-> Aramis features
-> ML classifier training
-> report/QC artifacts
```

Planned Aramis feature families from the assembly plan:

```text
complete azimuthal integration, components approach
cosine asymmetry distance, symmetry approach
```

Current draft focus:

```text
H5 container
-> product split/filter
-> XRD preprocessing
-> model-ready dataset
-> classifier
-> MLflow lineage
```

Planned command-level product interface:

```text
python -m aramis preprocess --config /path/to/preprocess.yaml
python -m aramis training --config /path/to/training.yaml
python -m aramis predict --config /path/to/predict.yaml
```

`preprocess` config owns input H5 path, output DataFrame/joblib path, raw-data
source, H5 quality exclusions, branch rules, and XRD preprocessing parameters.
`training` config will own dataset paths, split logic, model family, MLflow
tracking, and trained model output. `predict` config will own one-patient H5
input, fixed preprocessing/model versions, and JSON/YAML report output.

Prediction input contract for the first draft:

```text
one H5 container
one patient
two breast-side specimen groups when available
output: p_cancer / suggested class for decision support
requires radiologist review
```

MLflow is part of the product run because preprocessing defines the dataset.

Product-development rules:

```text
docs/product_development_rules.md
```

Machine-learning concept:

```text
docs/machine_learning_concept.md
```

Data-preprocessing contract:

```text
docs/data_preprocessing.md
```

Preprocessing code split:

```text
src/aramis/pipelines.py
  thin Aramis wrappers:
    load_preprocessing_config(...)
    build_pipeline_from_config(...)
    AramisOneToOnePreprocessingPipeline(...).fit_transform(h5_path)
    AramisOneToManyPreprocessingPipeline(...).fit_transform(h5_path)
  run_one_to_one_preprocessing_pipeline(...)
  run_one_to_many_preprocessing_pipeline(...)
  preprocessing artifact joblib export when requested
```

Aramis does not hardcode preprocessing transformer order in Python. Runnable
branch YAMLs extend a shared route:

```text
config/preprocessing/shared/aramis_pipeline_v0_1.yaml
```

That route declares `pipeline.steps`. XRD-preprocessing owns the transformer
registry, `$ref` / `$concat` resolution, and sklearn Pipeline construction.
Aramis resolves YAML input/output paths, requires `metadata.output_columns`,
runs the declared pipeline, and writes a preprocessing artifact joblib.

Runtime call chain:

```text
python -m aramis preprocess --config <yaml>
-> aramis.__main__.main
-> aramis.pipelines.run_preprocessing_from_config
-> xrd_preprocessing.load_preprocessing_config
-> xrd_preprocessing.build_pipeline_from_config
-> YAML-declared XRD transformers
-> KeepColumnsTransformer(metadata.output_columns + branch_settings.output_columns)
-> xrd_preprocessing.save_preprocessing_artifact(final_df, io.output_joblib_path)
```

The joblib contains `dataframe`, resolved `preprocessing_config`, original
`preprocessing_config_text`, config path, config SHA256, and run metadata.
Downstream code should use `xrd_preprocessing.load_preprocessing_dataframe`
when it needs only the DataFrame.

Synthetic regression tests:

```text
tests/synthetic_aramis_h5.py
  one known H5 fixture with raw/data and processed/data 2D arrays

tests/test_aramis_preprocessing_one_to_one.py
  checks one-to-one DataFrame fields and joblib roundtrip

tests/test_aramis_preprocessing_one_to_many.py
  checks one-to-many DataFrame fields and joblib roundtrip
```

Run real-H5 DataFrame examples:

```bash
conda activate eosproduct
cd /Users/sad/dev/Aramis

python -m aramis preprocess --config \
  config/preprocessing/aramis_one_to_one_max_v0_1.yaml

python -m aramis preprocess --config \
  config/preprocessing/aramis_one_to_many_max_v0_1.yaml
```

Interactive edit mode:

```bash
python -m marimo edit examples/aramis_dataframe_one_to_one_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_one_to_one_max_v0_1.yaml

python -m marimo edit examples/aramis_dataframe_one_to_many_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_one_to_many_max_v0_1.yaml

python -m marimo edit examples/aramis_one_to_many_product_model_v0_1.py -- \
  --standard-dataframe-joblib-path examples/outputs/aramis_one_to_many_benign_cancer_dataframe.joblib \
  --biopsy-dataframe-joblib-path examples/outputs/aramis_one_to_many_benign_cancer_biopsy_dataframe.joblib
```

Notebook behavior:

```text
default settings run automatically
changed settings are frozen until Validate settings is clicked
visualizations stay inside the notebook
joblib DataFrame export is the only default file output
```

Default output:

```text
examples/outputs/aramis_one_to_one_dataframe.joblib
examples/outputs/aramis_one_to_one_biopsy_dataframe.joblib
examples/outputs/aramis_one_to_many_benign_cancer_dataframe.joblib
examples/outputs/aramis_one_to_many_benign_cancer_biopsy_dataframe.joblib
```

Biopsy-only outputs use different cohort rules:

```text
one-to-many biopsy:
  row-level biopsy filter
  keep only biopsy=True specimen rows

one-to-one biopsy:
  patient-level biopsy filter
  keep patients with any biopsy=True row, then keep both breast sides
```

Input H5 and output joblib paths are owned by each preprocessing YAML under
`io.input_h5_path` and `io.output_joblib_path`.

Data-quality and monochromaticity limitations are tracked in:

```text
docs/machine_learning_concept.md#data-quality-and-monochromaticity
```

Product versioning/config:

```text
docs/meta/aramis_product_versioning.json
  Human-1 batch/source-line/calibrant-thickness product versioning

docs/meta/aramis_preprocessing_v0_1_config.json
  AgBH monochromaticity QC audit artifact
  contains purpose/provenance/selection_contract
  YAML filters.quality_exclusions drives H5-level filtering before GFRM loading

config/preprocessing/aramis_main_max_v0_1.yaml
config/preprocessing/aramis_main_min_v0_1.yaml
  shared non-runnable preprocessing bases
  compose shared policy, pipeline order, quality exclusions, and output schema

config/preprocessing/shared/aramis_policy_v0_1.yaml
config/preprocessing/shared/aramis_pipeline_v0_1.yaml
config/preprocessing/exclusions/agbh_quality_exclusions_v0_1.yaml
config/preprocessing/outputs/max_output_v0_1.yaml
config/preprocessing/outputs/min_output_v0_1.yaml
config/preprocessing/branches/*.yaml
  readable YAML fragments used by runnable branch configs

config/preprocessing/aramis_one_to_one_max_v0_1.yaml
config/preprocessing/aramis_one_to_one_biopsy_max_v0_1.yaml
  one-to-one branch preprocessing configs
  decision unit: patientId

config/preprocessing/aramis_one_to_many_max_v0_1.yaml
config/preprocessing/aramis_one_to_many_biopsy_max_v0_1.yaml
  one-to-many BENIGN/CANCER branch preprocessing configs
  decision unit: specimenId

config/preprocessing/*_min_v0_1.yaml
  minimal-output versions of the runnable branches
```

Reusable preprocessing YAML template/contract lives in:

```text
XRD-preprocessing/src/xrd_preprocessing/configs/preprocessing_branch_config_template.yaml
  commented reusable branch YAML template
```

Product metadata README:

```text
docs/meta/README.md
```

Run draft notebook:

```bash
conda env update -f environment.yml
conda activate eosproduct
marimo run examples/aramis_mlflow_draft.py
```

Local MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open:

```text
http://127.0.0.1:5000
```
