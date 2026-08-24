# Aramina

Research-draft breast-XRD decision support. Aramina scores one clinically
selected target breast as `BENIGN` or `CANCER` and reports whether biopsy is
required under the frozen model threshold. It is not an autonomous diagnosis.

## Current Product

```text
one-patient EOS H5 v0.3
-> frozen prediction preprocessing
-> normalized 256-bin XRD profiles
-> fold-local FPCA30 and profile LogisticRegression
-> target-breast profile score
-> final LogisticRegression with age and optional gated symmetry
-> p_cancer, risk level, biopsy_required, reliability
```

```text
model: aramina_target_breast_risk
source model definition: 0.3.1-beta
preserved executable artifacts: 0.2.12-beta and 0.2.13-beta
cohort: T100 biopsy-patient target-breast cases
evaluation: repeated patient-safe stratified 5-fold x20
regularization: profile C=0.1; final model C=0.3
```

The current source architecture is recorded in the
[`0.3.1-beta` model record](docs/modeling/aramina_fpca30_target_case_model_v0_3.md).
Preserved executable `0.2.x` artifacts and their historical records remain
under [`models/`](models/README.md). No `0.3.1-beta` joblib is tracked in Git;
it is reproducibly created by the preprocessing-and-training command.

## Install

```bash
git clone https://github.com/Eos-Dx/Aramina.git
cd Aramina
./install.sh
```

Windows:

```bat
git clone https://github.com/Eos-Dx/Aramina.git
cd Aramina
install.bat
```

See [INSTALL.md](INSTALL.md) for Conda, Docker, examples, and full-training
requirements.

## Commands

```bash
python -m aramina preprocess --config config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml
python -m aramina train --config config/training/config_training_target_breast_risk_v0_1.yaml
python -m aramina preprocess-train --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml
python -m aramina predict --config examples/prediction/configs/config_predict_cancer_example.yaml
```

The full historical archive is required only for preprocessing and training.
Prediction examples use the tracked one-patient H5 fixtures.

The canonical `preprocess-train` config creates one fail-closed MLflow product
run covering preprocessing, patient-safe evaluation, and final train-on-all.
Inspect local runs with:

```bash
mlflow ui --backend-store-uri examples/outputs/mlflow --port 5000
```

## Repository Map

| Path | Content |
|---|---|
| [`src/aramina/`](src/aramina/) | Product code. |
| [`config/`](config/README.md) | Runnable input YAML. |
| [`contracts/`](contracts/README.md) | Filled output examples. |
| [`docs/`](docs/README.md) | Canonical technical documentation. |
| [`examples/`](examples/README.md) | Runnable prediction examples. |
| [`models/`](models/README.md) | Frozen model artifact and training outputs. |
| [`packaging/`](packaging/) | Docker and offline bundles. |
| [`tests/`](tests/data/README.md) | Contract, unit, integration, and H5 tests. |

## Verify

```bash
conda run --no-capture-output -n eosproduct ruff check .
conda run --no-capture-output -n eosproduct pytest -q
```
