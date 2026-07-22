# Aramis

Status: research-draft breast-XRD decision-support prototype.

Aramis accepts one EOS H5 `0.3` container for one patient, preprocesses its
left/right XRD measurements, and returns a target-breast BENIGN/CANCER
decision-support class with reliability metadata. It is for review by a
qualified breast-imaging clinician; it is not autonomous diagnosis or a
replacement for biopsy or radiologist review.

## Product Route

```text
one-patient H5
-> model-held prediction preprocessing
-> normalized radial_profile_data
-> LR1 target-breast profile score
-> LR2: profile + age + optional gated SK symmetry refinement
-> p_cancer, suggested class, reliability
```

Current product definition:

```text
model: aramis_target_breast_risk
training cohort: T100 biopsy-patient target-breast cases
regularization: LR1 L2 C=0.1; LR2 L2 C=0.3
default evaluation: repeated patient-safe stratified 5-fold x20
deployment threshold: train-all score at target sensitivity >=0.95
```

## Installation

Two supported routes:

```text
Git clone + Conda: development, inspection, direct local runs.
Docker bundle: reproducible full-H5 training and demonstration.
```

For a clone and Conda environment:

```bash
git clone https://github.com/Eos-Dx/Aramis.git
cd Aramis
./install.sh
```

Windows:

```bat
git clone https://github.com/Eos-Dx/Aramis.git
cd Aramis
install.bat
```

Detailed instructions: [INSTALL.md](INSTALL.md).

Local `preprocess`, `train`, and `preprocess-train` require the full historical
H5 archive, which is intentionally not stored in Git. Place the approved
archive at `data/combined_archive.h5` under the Aramis project root before
running those commands. The one-patient prediction examples are self-contained
and do not require this archive. The Docker bundle includes its own verified
copy under `data/`.

## Main Commands

```bash
python -m aramis preprocess --config config/preprocessing/config_preprocessing_biopsy_patients_v0_1.yaml
python -m aramis train --config config/training/config_training_target_breast_risk_v0_1.yaml
python -m aramis preprocess-train --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml
python -m aramis predict --config examples/prediction/configs/config_predict_cancer_example.yaml
```

`preprocess` and `train` are development routes. Production-style H5 scoring
uses `predict`. The packaged model holds preprocessing, threshold, feature
schema, and report contract.

## Documentation

```text
INSTALL.md                                      clone/Conda route
docs/product_api.md                             H5 input and report API
config/preprocessing/README.md                  preprocessing config
config/training/README.md                       training config
config/prediction/README.md                     prediction config
config/preprocessing_and_training/README.md     combined route
docs/modeling/aramis_t100_target_case_model_v0_1.md
                                                model rationale and limits
docs/modeling/prediction_pipeline_v0_1.md       prediction route
contracts/                                      filled output-contract examples
docs/contracts/                                 canonical YAML and artifact contracts
docs/meta/README.md                             decision evidence
```

## Verification

```bash
conda run --no-capture-output -n eosproduct ruff check .
conda run --no-capture-output -n eosproduct pytest -q
```
