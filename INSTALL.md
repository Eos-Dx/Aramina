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
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
```

## Prediction Examples

```bash
python -m aramis predict --config examples/prediction_h5/benign_predict.yaml
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
python -m aramis predict --config examples/prediction_h5/atypical_predict.yaml
```

Reports are written to:

```text
examples/outputs/prediction_h5_examples/
```

These are real one-patient GFRM fixtures extracted from the larger archive.
They prove installation, preprocessing, prediction and report writing. They
are not clinical validation examples.

## Full Training Reproduction

`packaging/reproducible_training_bundle/make_bundle.sh` creates a separate ZIP
with the full historical H5. Run `install_and_train.bat` on Windows or
`./install_and_train.sh` on macOS/Linux. Repeated runs reuse the environment and
workspace, refresh the exact commits declared in `bundle_manifest.json`, and
write a timestamped log under `workspace/logs/`.
