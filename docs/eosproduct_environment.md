# eosproduct Environment

Aramis product preprocessing depends on `xrd-preprocessing`.

Canonical development environment:

```text
conda env update -n eosproduct -f environment.yml
conda activate eosproduct
```

Required package groups:

```text
H5/container:
  h5py
  eosdx-container from /Users/sad/dev/container

RAW GFRM / XRD physics:
  xrd-preprocessing
  pyFAI
  fabio
  scipy
  scikit-learn
  joblib

DataFrames / storage:
  numpy
  pandas
  pyarrow
  PyYAML

Product notebooks:
  marimo
  matplotlib

ML tracking and baseline models:
  mlflow
  lightgbm

Development / validation:
  pytest
  ruff
```

Current Aramis product config references:

```text
xrd_preprocessing.release_tag = v0.1.6-beta
```

For local development, `environment.yml` installs:

```text
/Users/sad/dev/container
/Users/sad/dev/XRD-preprocessing[dev]
/Users/sad/dev/Aramis[dev]
```

For reproducible package metadata, `pyproject.toml` points Aramis to:

```text
xrd-preprocessing @ git+https://github.com/Eos-Dx/XRD-preprocessing.git@v0.1.6-beta
```

Validation commands:

```text
python -m ruff check .
pytest -q
python -m marimo check examples/aramis_dataframe_all_patients_v0_1.py examples/aramis_dataframe_biopsy_patients_v0_1.py
```
