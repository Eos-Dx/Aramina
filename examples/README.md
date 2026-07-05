# Aramis Examples

These marimo notebooks are package examples. They should run from this folder
without importing helper code from `Clinical_trials/Product/Aramis`.

Files:

```text
aramis_dataframe_one_to_one_v0_1.py
aramis_dataframe_one_to_many_v0_1.py
aramis_dataframe_all_patients_v0_1.py
aramis_dataframe_biopsy_patients_v0_1.py
aramis_one_to_many_logistic_baseline_v0_1.py
aramis_one_to_many_product_model_v0_1.py
aramis_loocv_main_models_experiment.py
aramis_patient_cv_main_models_experiment.py
aramis_ka_symmetry_only_experiment.py
aramis_lr1_profile_plus_ka_symmetry_svm_poly2_train_all.py
aramis_final_experimental_model_v0_1.py
aramis_product_notebook_helpers.py
preprocess_one_to_one.sh
preprocess_one_to_one_biopsy.sh
preprocess_one_to_many.sh
preprocess_one_to_many_biopsy.sh
preprocess_one_to_one_minimal.sh
preprocess_one_to_one_biopsy_minimal.sh
preprocess_one_to_many_minimal.sh
preprocess_one_to_many_biopsy_minimal.sh
preprocess_all.sh
```

The helper file intentionally lives beside the notebooks because marimo examples
import it directly:

```python
import aramis_product_notebook_helpers as helpers
```

Run:

```bash
cd ~/dev/eosproduct/Aramis
conda activate eosproduct
```

For a test install that uses `ENV_NAME=eosproduct1`, use:

```bash
cd ~/dev/eosproduct1/Aramis
conda activate eosproduct1
```

Preprocess DataFrames directly from YAML:

```bash
python -m aramis preprocess --config config/preprocessing/aramis_one_to_one_max_v0_1.yaml
python -m aramis preprocess --config config/preprocessing/aramis_one_to_one_biopsy_max_v0_1.yaml
python -m aramis preprocess --config config/preprocessing/aramis_one_to_many_max_v0_1.yaml
python -m aramis preprocess --config config/preprocessing/aramis_one_to_many_biopsy_max_v0_1.yaml
```

Build current model-input DataFrames:

```bash
python -m aramis preprocess --config config/preprocessing/aramis_all_patients_model_input_v0_1.yaml
python -m aramis preprocess --config config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
```

Minimal joblib exports:

```bash
python -m aramis preprocess --config config/preprocessing/aramis_one_to_one_min_v0_1.yaml
python -m aramis preprocess --config config/preprocessing/aramis_one_to_one_biopsy_min_v0_1.yaml
python -m aramis preprocess --config config/preprocessing/aramis_one_to_many_min_v0_1.yaml
python -m aramis preprocess --config config/preprocessing/aramis_one_to_many_biopsy_min_v0_1.yaml
```

Equivalent example scripts:

```bash
./examples/preprocess_one_to_one.sh
./examples/preprocess_one_to_one_biopsy.sh
./examples/preprocess_one_to_many.sh
./examples/preprocess_one_to_many_biopsy.sh
./examples/preprocess_one_to_one_minimal.sh
./examples/preprocess_one_to_one_biopsy_minimal.sh
./examples/preprocess_one_to_many_minimal.sh
./examples/preprocess_one_to_many_biopsy_minimal.sh
./examples/preprocess_all.sh
```

Each branch YAML owns both input and output paths:

```yaml
io:
  input_h5_path: ../../../data/combined_archive.h5
  output_joblib_path: ../../examples/outputs/aramis_one_to_one_dataframe.joblib
```

The preprocessing route is stored once in:

```text
config/preprocessing/shared/aramis_pipeline_v0_1.yaml
```

That file defines ordered XRD-preprocessing transformer steps:

```yaml
pipeline:
  steps:
    - name: h5_to_df
      transformer: H5ToDataFrameTransformer
      params:
        data_preference:
          $ref: raw_data.source
    - name: keep_columns
      transformer: KeepColumnsTransformer
      params:
        columns:
          $concat:
            - $ref: metadata.output_columns
            - $ref: branch_settings.output_columns
```

The `transformer` names are XRD-preprocessing transformer registry entries.
Runnable root YAMLs extend this shared route plus policy, output schema, and
branch cohort fragments. Aramis reads the YAML, asks XRD-preprocessing to build
the sklearn Pipeline, and writes a preprocessing artifact joblib to
`io.output_joblib_path`. The artifact stores the final DataFrame plus
preprocessing YAML text, config SHA256, H5 SHA256, Aramis version/git SHA, and
branch.

Train model grid entries:

```bash
python -m aramis train --config config/training/model_grid_v0_1/18_m2_stratified_kfold_biopsy_patients_model_v0_1.yaml
```

Run preprocess+train workflow:

```bash
python -m aramis run --config config/workflows/aramis_biopsy_patients_m2_kfold_v0_1.yaml
```

The default workflow mode is `memory`: preprocessing writes the joblib footprint
and passes the fresh DataFrame directly into training. Use `mode: artifact` in
the workflow YAML to force training to reload the saved preprocessing joblib.

Run marimo notebooks:

```bash

python -m marimo run examples/aramis_dataframe_one_to_one_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_one_to_one_max_v0_1.yaml

python -m marimo run examples/aramis_dataframe_one_to_many_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_one_to_many_max_v0_1.yaml

python -m marimo run examples/aramis_dataframe_all_patients_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_all_patients_model_input_v0_1.yaml

python -m marimo run examples/aramis_dataframe_biopsy_patients_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml

python -m marimo run examples/aramis_one_to_many_logistic_baseline_v0_1.py -- \
  --dataframe-joblib-path examples/outputs/aramis_one_to_many_benign_cancer_dataframe.joblib

python -m marimo run examples/aramis_one_to_many_product_model_v0_1.py -- \
  --standard-dataframe-joblib-path examples/outputs/aramis_one_to_many_benign_cancer_dataframe.joblib \
  --biopsy-dataframe-joblib-path examples/outputs/aramis_one_to_many_benign_cancer_biopsy_dataframe.joblib

python -m marimo run examples/aramis_final_experimental_model_v0_1.py -- \
  --one-to-many-joblib-path examples/outputs/aramis_one_to_many_benign_cancer_biopsy_dataframe.joblib \
  --one-to-one-joblib-path examples/outputs/aramis_one_to_one_dataframe.joblib
```

Run patient-level leave-one-out M0-M3 comparison:

```bash
cd /Users/sad/dev/Aramis
conda activate eosproduct
python examples/aramis_loocv_main_models_experiment.py
```

Run repeated patient-level CV M0-M3 comparison:

```bash
cd /Users/sad/dev/Aramis
conda activate eosproduct
python examples/aramis_patient_cv_main_models_experiment.py
```

Run Kubitskii-style symmetry-only experiment:

```bash
cd /Users/sad/dev/Aramis
conda activate eosproduct
python examples/aramis_ka_symmetry_only_experiment.py
```

Run train-on-all LR1 profile mean plus Kubitskii symmetry SVM-poly2 experiment:

```bash
cd /Users/sad/dev/Aramis
conda activate eosproduct
python examples/aramis_lr1_profile_plus_ka_symmetry_svm_poly2_train_all.py
```

Default Aramis product config:

```text
docs/meta/aramis_preprocessing_v0_1_config.json
```

This JSON stores provenance: source preprocessing notebook, generation summary,
documentation links, downstream notebook consumers, and the `selection_contract`
for AgBH monochromaticity exclusions. Runtime input/output paths are stored in
the branch YAML, not passed as command-line paths.

Default branch preprocessing YAMLs:

```text
config/preprocessing/aramis_one_to_one_max_v0_1.yaml
config/preprocessing/aramis_one_to_one_biopsy_max_v0_1.yaml
config/preprocessing/aramis_one_to_many_max_v0_1.yaml
config/preprocessing/aramis_one_to_many_biopsy_max_v0_1.yaml
```

Biopsy cohort meaning:

```text
one-to-many biopsy:
  row-level biopsy filter
  keep only biopsy=True specimen rows

one-to-one biopsy:
  patient-level biopsy filter
  keep patients with any biopsy=True row, then keep both breast sides
```

Each notebook reads its own root branch YAML by default. Override only when
testing a controlled replacement:

```bash
python -m marimo run examples/aramis_dataframe_one_to_one_v0_1.py -- \
  --aramis-preprocessing-config-path /path/to/aramis_one_to_one_max_v0_1.yaml
```

Default output:

```text
examples/outputs/aramis_one_to_one_dataframe.joblib
examples/outputs/aramis_one_to_many_benign_cancer_dataframe.joblib
```

To keep more columns in preprocessing joblib, edit the output schema YAML:

```yaml
metadata:
  output_columns:
    - patientId
    - specimenId
    - q_range
    - radial_profile_data_raw
    - radial_profile_data

normalization:
  save_initial_data: true
```

`metadata.output_columns` is mandatory. The final joblib contains exactly those
columns, in that order. To keep `radial_profile_data_raw`, set
`normalization.save_initial_data: true` and list `radial_profile_data_raw` in
`metadata.output_columns`.

`radial_profile_data` is always the final normalized profile. With
`save_initial_data: true`, `radial_profile_data_raw` stores the profile before
normalization.

Biopsy-only one-to-many output:

```bash
python -m marimo run examples/aramis_dataframe_one_to_many_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_one_to_many_biopsy_max_v0_1.yaml
```

The first model notebook starts from the one-to-many joblib and does not reopen
the H5 container. It trains `LogisticRegression` on the full normalized
`radial_profile_data` profile over 20 repeated patient-safe 70/30 splits and
plots ROC curves for BENIGN vs CANCER.

The first product-model notebook compares the standard one-to-many joblib with
the biopsy-only one-to-many joblib. For each DataFrame it trains
`LogisticRegression` on measurement profiles, aggregates measurement
probabilities to specimen/breast rows, selects thresholds on train OOF specimen
scores, and evaluates ROC/threshold metrics on held-out test patients over
repeated patient-safe 70/30 splits.

The final experimental-model notebook starts from biopsy-only one-to-many
targets and the one-to-one paired DataFrame. It compares M0-M3 fusion concepts:
one-to-many only, one-to-many plus symmetry, plus quality, and plus age/BMI. It
also includes control ablations for age-only, BMI-only, availability-only,
single availability flags, M2+age, and M2+BMI. All splits remain patient-safe.
Missing symmetry is encoded with an explicit availability flag and a zero value,
not as a biological zero.

Run the current LR1/LR2 patient-safe symmetry experiment from the MAX
monochromatic metadata pool:

```bash
python examples/aramis_lr1_lr2_symmetry_patient_split_experiment.py
```

This experiment excludes `NORMAL` from BENIGN-vs-CANCER target/training rows,
keeps `NORMAL` in the source pool for contralateral symmetry distances, and
splits strictly by `patientId`.
