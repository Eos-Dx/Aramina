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

The runtime and bundled frozen reference artifact are `0.2.12-beta`:
`aramis_target_breast_risk_0_2_12-beta_f8af641a2e49`. Training produces a
traceable `0.2.12-beta` candidate and compares it with this reference when the
model contracts match.

The H5 archive is mounted read-only from `data/combined_archive.h5`; it is
never copied into the Docker image. Generated artifacts and logs are written
to `outputs/` beside this README.

If Docker Desktop already exists but its Linux engine is stopped, the script
starts it and waits up to five minutes for it to become ready.

## Predict a new patient H5

`predict_examples` only verifies that the bundle, Docker image, and bundled
fixtures work. For a new patient, prepare a request YAML using the prediction
contract and run `predict`. The launcher deliberately receives the H5, model,
and output folder as command-line paths so Docker can mount only those host
folders. It writes `prediction_request_resolved.yaml` beside the reports; this
copy records the exact paths used inside the Linux runtime.

Windows:

```powershell
.\predict.ps1 `
  -Config D:\aramis_requests\patient_001.yaml `
  -InputH5 D:\aramis_requests\patient_001.h5 `
  -ModelPath .\outputs\preprocessing_and_training\<run_id>\training\model.joblib `
  -OutputFolder D:\aramis_results\patient_001
```

macOS/Linux:

```bash
./predict.sh \
  --config /data/aramis_requests/patient_001.yaml \
  --input-h5 /data/aramis_requests/patient_001.h5 \
  --model ./outputs/preprocessing_and_training/<run_id>/training/model.joblib \
  --output-folder /data/aramis_results/patient_001
```

The YAML supplies `analysis_author`, `prediction_comment`, `patient_id`, and
`target_side`. Its `io` values remain required by the Aramis prediction
contract, but the Docker launcher replaces them only in the resolved runtime
copy with the selected H5, model, and output folder. The original YAML is never
modified. The H5 must satisfy the one-patient EOS H5 `0.3` contract.

The output folder receives a preprocessed DataFrame joblib, external and
internal reports in YAML, and the resolved request YAML.
The external report is target-breast decision support. The internal report also
contains the contralateral score and quality/reliability information.
The launcher also prints the external YAML report to the terminal after a
successful prediction.

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

Each fixture's external YAML report is also printed to the terminal.

Pass a particular model produced by this bundle when needed:

```powershell
.\predict_examples.ps1 -ModelPath .\outputs\preprocessing_and_training\...\model.joblib
```

```bash
./predict_examples.sh --model ./outputs/preprocessing_and_training/.../model.joblib
```

The supplied prediction YAML templates are external under
`examples/prediction/configs/`. The launcher writes resolved copies containing
the selected model path to `outputs/prediction_examples/resolved_configs/`.

## Select a configuration

The bundle includes only operational YAML files: one preprocessing-and-training config,
its preprocessing and training configs, the prediction preprocessing config,
three bundled prediction templates, and required `extends` fragments. Select one preprocessing-and-training YAML; it names
the preprocessing and training YAMLs to use. All operational paths resolve
from the mounted Aramis project root.

Windows:

```powershell
.\install_and_train.ps1 -PreprocessTrainConfig config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml
```

macOS/Linux:

```bash
./install_and_train.sh --preprocess-train-config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml
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
