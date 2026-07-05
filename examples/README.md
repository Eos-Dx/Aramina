# Aramis Examples

Status: research draft.

These examples are thin product-facing notebooks/scripts. Reusable preprocessing,
training, and prediction logic lives in `src/aramis` and `xrd_preprocessing`.

Current notebooks:

```text
aramis_dataframe_one_to_one_v0_1.py
aramis_dataframe_one_to_many_v0_1.py
aramis_dataframe_all_patients_v0_1.py
aramis_dataframe_biopsy_patients_v0_1.py
aramis_one_to_many_logistic_baseline_v0_1.py
aramis_one_to_many_product_model_v0_1.py
```

Helper file:

```text
aramis_product_notebook_helpers.py
```

Direct preprocessing:

```bash
python -m aramis preprocess --config config/preprocessing/aramis_all_patients_model_input_v0_1.yaml
python -m aramis preprocess --config config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
```

Direct training:

```bash
python -m aramis train --config config/training/aramis_v0_1_beta_primary_train.yaml
python -m aramis train --config config/training/aramis_biopsy_patients_m0_m1_m2_v0_1.yaml
```

Preprocess + train workflow:

```bash
python -m aramis run --config config/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml
```

Prediction from a preprocessed DataFrame artifact:

```bash
python -m aramis predict --config config/prediction/aramis_predict_example_v0_1.yaml
```

Prediction from patient H5:

```bash
python -m aramis predict --config config/prediction/aramis_predict_from_h5_template_v0_1.yaml
```

Run current marimo DataFrame notebooks:

```bash
python -m marimo run examples/aramis_dataframe_all_patients_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_all_patients_model_input_v0_1.yaml

python -m marimo run examples/aramis_dataframe_biopsy_patients_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
```

Run legacy branch-inspection notebooks:

```bash
python -m marimo run examples/aramis_dataframe_one_to_one_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_one_to_one_max_v0_1.yaml

python -m marimo run examples/aramis_dataframe_one_to_many_v0_1.py -- \
  --aramis-preprocessing-config-path config/preprocessing/aramis_one_to_many_max_v0_1.yaml
```

Historical exploratory notebooks and threshold-grid scripts are archived on:

```text
experiment/aramis-v0.1-research-state
```
