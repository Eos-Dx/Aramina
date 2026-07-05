# MLflow In Aramis

MLflow stores one complete product run.

That means:

```text
input H5
product split/filter
preprocessing
model-ready dataset
model
metrics
predictions
```

## Why Preprocessing Goes Into MLflow

The same H5 container can feed several products.

Example:

```text
EOS H5 v0.3
├── Aramis dataset
│   ├── label / status filter
│   ├── BENIGN vs CANCER labels
│   └── Aramis classifier
└── Bremen dataset
    ├── different product filter
    ├── different labels or target
    └── Bremen model
```

So MLflow must record:

```text
which H5 was used
which rows were selected
which rows were dropped
which preprocessing parameters were used
which model was trained
which metrics came out
```

Without preprocessing lineage, the model is not reproducible.

## Local Draft

Run local MLflow UI:

```bash
cd /Users/sad/dev/Aramis
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Notebook:

```bash
marimo run examples/aramis_mlflow_draft.py
```

Default mode is `dry_run`.

It writes the exact artifacts locally but does not call MLflow.

Uncheck `dry run` after MLflow is installed.

## Logged Artifacts

```text
preprocessed_dataset.parquet
preprocessed_dataset.csv, fallback when parquet engine is not installed
preprocessing_config.json
product_filter_rules.json
feature_schema.json
params.json
metrics.json
model.joblib
```

## Logged Metrics

Draft:

```text
roc_auc
balanced_accuracy
```

Later:

```text
sensitivity
specificity
PPV
NPV
threshold
calibration metrics
confidence intervals
```

## Aramis First Classifier

Output:

```text
p_cancer
suggested_class = BENIGN if p_cancer < threshold
suggested_class = CANCER if p_cancer >= threshold
requires radiologist review
```

Later output:

```text
p_cancer
suggested_class
decision_threshold
confidence interval
model version
preprocessing version
```
