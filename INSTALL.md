# Aramis Install

Status: research draft decision-support prototype.

This repository can be cloned and used without the EOS Product bundle. The git
repository contains small one-patient H5 examples and a tracked M2Q model
artifact:

```text
examples/prediction_h5/
examples/prediction_models/aramis_m2q_t100_train_all_c0p1.joblib
```

## macOS / Linux

```bash
git clone https://github.com/Eos-Dx/Aramis.git
cd Aramis
./install.sh
```

If `conda` is missing, `install.sh` asks to install Miniforge into
`~/miniforge3`.

## Windows

```bat
git clone https://github.com/Eos-Dx/Aramis.git
cd Aramis
install.bat
```

If `conda` is missing, `install.bat` asks to install Miniforge into
`%USERPROFILE%\miniforge3`.

## Manual Commands

```bash
conda env create -n eosproduct -f environment.yml
conda activate eosproduct
python -m pip install -e ".[dev]"
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
```

## Prediction Examples

```bash
python -m aramis predict --config examples/prediction_h5/px01_predict.yaml
python -m aramis predict --config examples/prediction_h5/px02_predict.yaml
python -m aramis predict --config examples/prediction_h5/px03_predict.yaml
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
python -m aramis predict --config examples/prediction_h5/atypical_predict.yaml
python -m aramis predict --config examples/prediction_h5/benign_predict.yaml
```

Reports are written to:

```text
examples/outputs/prediction_h5_examples/
```

These H5 files are synthetic smoke-test fixtures. They prove installation,
preprocessing, prediction, and report writing. They are not clinical validation
examples.
