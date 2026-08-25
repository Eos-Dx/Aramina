# Aramina

Research-draft breast-XRD decision support. Aramina scores one clinically
selected target breast as `BENIGN` or `CANCER` and reports whether biopsy is
required under the frozen model threshold. It is not an autonomous diagnosis.

## Current Product

```text
one-patient EOS H5 v0.3
-> frozen prediction preprocessing
-> normalized 100-point XRD profiles
-> profile LogisticRegression
-> target-breast profile score
-> final LogisticRegression with age and optional gated symmetry
-> p_cancer, risk level, biopsy_required, reliability
```

```text
model: aramina_target_breast_risk
legacy implementation release: 0.2.13-beta (no DVC requirement)
current data-versioned release: 0.2.14-beta
cohort: T100 biopsy-patient target-breast cases
evaluation: repeated patient-safe stratified 5-fold x20
regularization: profile C=0.1; final model C=0.3
```

The released artifact is under [`models/`](models/README.md). Architecture,
cohort, metrics, threshold, and limitations are recorded in the
[current product-model record](docs/modeling/aramina_t100_target_case_model_v0_1.md).
The separately retrained XRD `v0.1.9-beta` candidate is recorded in
[`docs/modeling/aramina_t100_target_case_model_v0_2.md`](docs/modeling/aramina_t100_target_case_model_v0_2.md).
The same fixed recipe retrained with mandatory DVC/MLflow lineage as
`0.2.14-beta` is recorded in
[`docs/modeling/aramina_t100_target_case_model_v0_3.md`](docs/modeling/aramina_t100_target_case_model_v0_3.md).

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
python -m aramina preprocess --config config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml
python -m aramina train --config config/training/config_training_target_breast_risk_v0_4.yaml
python -m aramina preprocess-train --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_3.yaml
python -m aramina predict --config examples/prediction/configs/config_predict_cancer_example.yaml
```

The full historical archive is required only for preprocessing and training.
Its exact internal revision is tracked by DVC; see
[Data versioning](docs/data_versioning.md). Prediction examples use the tracked
one-patient H5 fixtures and do not require DVC.

Tag `0.2.13-beta` and its `v0_1` YAML files remain the implementation baseline
without DVC. Release `0.2.14-beta` preserves the same XRD preprocessing and
model recipe, but requires verified DVC lineage for historical training.

The canonical `preprocess-train` config creates one fail-closed MLflow product
run covering preprocessing, patient-safe evaluation, and final train-on-all.
Inspect local runs with:

```bash
mlflow ui --backend-store-uri sqlite:///examples/outputs/mlflow/aramina_radial_profile.db --port 5000
```

## Repository Map

| Path | Content |
|---|---|
| [`src/aramina/`](src/aramina/) | Product code. |
| [`data/`](data/README.md) | DVC pointer and internal-storage instructions. |
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
