# Aramis Install

Status: research draft decision-support prototype.

This repository can be cloned and used without the EOS Product bundle. The git
repository contains small one-patient H5 examples and a tracked product model
artifact:

```text
examples/prediction_h5/
models/aramis_target_breast_risk_<model_id>/model.joblib
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
python -m aramis predict --config config/prediction/prediction_examples/cancer_predict.yaml
```

## Prediction Examples

```bash
python -m aramis predict --config config/prediction/prediction_examples/benign_predict.yaml
python -m aramis predict --config config/prediction/prediction_examples/cancer_predict.yaml
python -m aramis predict --config config/prediction/prediction_examples/atypical_predict.yaml
```

Reports are written under:

```text
examples/outputs/prediction_examples/
```

These are real one-patient GFRM fixtures extracted from the larger archive.
They prove installation, preprocessing, prediction and report writing. They
are not clinical validation examples.

## Full Training Reproduction

`packaging/reproducible_training_bundle/make_bundle.sh` creates a separate ZIP
with the full historical H5. The bundle uses Docker rather than this Conda
environment. Run `install_and_train.bat` on Windows or `./install_and_train.sh`
on macOS/Linux; the bundle README describes training, prediction examples, and
external H5 prediction. Every run writes logs and artifacts under the bundle
`outputs/` directory.
