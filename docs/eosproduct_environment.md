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

Development / validation:
  pytest
  ruff
```

Current Aramis product config references:

```text
xrd_preprocessing.release_tag = v0.1.7-beta
```

For local development, create or update the Conda environment, then install
the checked-out Aramis package. Its `pyproject.toml` installs the pinned
`xrd-preprocessing` dependency:

```text
conda env update -n eosproduct -f environment.yml
python -m pip install -e ".[dev]"
```

For reproducible package metadata, `pyproject.toml` points Aramis to:

```text
xrd-preprocessing @ git+https://github.com/Eos-Dx/XRD-preprocessing.git@v0.1.7-beta
```

Validation commands:

```text
python -m ruff check .
pytest -q
python -m aramis predict --config config/prediction/prediction_examples/cancer_predict.yaml
```
