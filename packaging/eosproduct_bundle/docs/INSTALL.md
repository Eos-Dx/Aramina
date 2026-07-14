# EOS Product Bundle

Purpose:

```text
install XRD-preprocessing
install Aramis
install container
create empty Bremen folder
copy bundled H5 data
create/update conda env eosproduct
install Miniforge automatically if conda is missing
install container, XRD-preprocessing, and Aramis as editable local packages
run tests
run Aramis one-patient prediction examples
```

Run:

```bash
tar -xzf eosproduct_onboarding_bundle.tar.gz
cd eosproduct_onboarding_bundle
./install.sh
```

Windows PowerShell:

```powershell
tar -xzf eosproduct_onboarding_bundle.tar.gz
cd eosproduct_onboarding_bundle
.\install.ps1
```

The installer asks before running tests or prediction examples. If tests are
accepted, XRD-preprocessing and Aramis tests run together in a separate Terminal
window on macOS/Linux or a separate PowerShell window on Windows.

If git clone/update succeeds, installer uses pinned product refs:

```text
XRD-preprocessing: v0.1.7-beta
Aramis: 0.2.3-beta
container: feat/v0_3-eoscan-session-container
```

If git is unavailable or clone fails, installer uses the bundled repository
fallback.

If `conda` is not installed, the installer asks to install Miniforge into:

```text
~/miniforge3
```

If `git` is not installed, the installer asks to install it. On macOS/Linux,
`install.sh` tries Homebrew, Apple Command Line Tools, or apt where available.
On Windows, `install.ps1` tries `winget install Git.Git`.

Default target:

```text
~/dev/eosproduct
```

Manual test commands:

```bash
./run_tests.sh ~/dev/eosproduct xrd
./run_tests.sh ~/dev/eosproduct aramis
./run_tests.sh ~/dev/eosproduct all
```

Windows:

```powershell
.\run_tests.ps1 -TargetRoot "$HOME\dev\eosproduct" -Mode all
```

Manual prediction example after install:

```bash
./run_aramis_prediction_examples.sh ~/dev/eosproduct
```

Windows:

```powershell
.\run_aramis_prediction_examples.ps1 -TargetRoot "$HOME\dev\eosproduct"
```

Single prediction example:

```bash
cd ~/dev/eosproduct/Aramis
conda activate eosproduct
python -m aramis predict --config examples/prediction_h5/cancer_predict.yaml
```

Build full-data bundle:

```bash
DATA_H5=/path/to/combined_archive.h5 ./make_bundle.sh
```
