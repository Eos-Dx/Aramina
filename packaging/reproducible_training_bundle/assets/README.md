# Aramis Docker Reproducible Training Bundle

Status: research-draft decision-support prototype.

## Windows

1. Extract the entire ZIP to a local drive with sufficient free space.
2. Double-click `install_and_train.bat`, or run:

   ```powershell
   .\install_and_train.ps1
   ```

On its first use, the script downloads and installs Docker Desktop with its
WSL 2 Linux backend, then starts the Docker engine. It requires internet
access and accepts the Docker license. If Windows needs a restart after WSL 2
is enabled, restart and run the same file again. Later runs reuse Docker.

The script verifies the bundled H5 checksum, loads the bundled native
`linux/amd64` runtime image on first use, and runs preprocessing plus training.
It does not install Conda, Git, Python, pyFAI, Aramis, or XRD-preprocessing on
Windows.

The H5 archive is mounted read-only from `data/combined_archive.h5`; it is
never copied into the Docker image. Generated artifacts and logs are written
to `outputs/` beside this README.

If Docker Desktop already exists but its Linux engine is stopped, the script
starts it and waits up to five minutes for it to become ready.

## Prediction examples

After training completes, run all three one-patient H5 examples with the model
just created under `outputs/preprocessing_and_training/`.

Windows:

```powershell
.\predict_examples.ps1
```

macOS/Linux:

```bash
./predict_examples.sh
```

The scripts run the same Linux Docker image. They automatically select the
newest `model.joblib` under `outputs/preprocessing_and_training/`, run cancer, benign,
and atypical H5 fixtures, then write reports to:

```text
outputs/prediction_examples/cancer/
outputs/prediction_examples/benign/
outputs/prediction_examples/atypical/
```

Pass a particular model produced by this bundle when needed:

```powershell
.\predict_examples.ps1 -ModelPath .\outputs\preprocessing_and_training\...\model.joblib
```

```bash
./predict_examples.sh --model ./outputs/preprocessing_and_training/.../model.joblib
```

The supplied prediction YAML templates are external under
`config/prediction/prediction_examples/`. The launcher writes resolved copies containing
the selected model path to `outputs/prediction_examples/resolved_configs/`.

## Select a configuration

The bundle includes only operational YAML files: one preprocessing-and-training config,
its preprocessing and training configs, the prediction preprocessing config,
three bundled prediction templates, and required `extends` fragments. Select one preprocessing-and-training YAML; it names
the preprocessing and training YAMLs to use. All operational paths resolve
from the mounted Aramis project root.

Windows:

```powershell
.\install_and_train.ps1 -PreprocessTrainConfig config/preprocessing_and_training/aramis_target_breast_risk_preprocessing_and_training_v0_1.yaml
```

macOS/Linux:

```bash
./install_and_train.sh --preprocess-train-config config/preprocessing_and_training/aramis_target_breast_risk_preprocessing_and_training_v0_1.yaml
```

To use a different evaluation, copy the standard training YAML, change only
its `evaluation` section, then create a preprocessing-and-training YAML under
`config/preprocessing_and_training/` that points to it and the desired preprocessing
YAML. The product model is fixed inside the Docker image. A custom
preprocessing-and-training run is saved in the generated model artifacts and deliberately
skips comparison to the fixed reference model.

## macOS/Linux

Run:

```bash
./install_and_train.sh
```

Docker is required on these systems as well.

The macOS/Linux launcher automatically loads the native `linux/arm64` image on
Apple Silicon and `linux/amd64` image on Intel/AMD systems.
