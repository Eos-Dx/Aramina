# Aramis Browser Demonstrator Bundle

Local research-draft browser demonstrator for the frozen
`aramis_target_breast_risk` model version `0.2.11-beta`. The selected model,
its prediction preprocessing contract, threshold, feature schema and report
generation code are embedded in the Docker image.

This bundle is for an interactive visual demonstration. For programmatic
one-patient prediction, use the separate Aramis Prediction API Bundle.

## Contents

```text
aramis_demo_linux_amd64_0_2_11_beta.tar    Windows x86-64 / Intel macOS image
aramis_demo_linux_arm64_0_2_11_beta.tar    Apple Silicon macOS image
start_demo.sh / start_demo.ps1             Start the local browser demonstrator
stop_demo.sh                               Stop the demonstrator
fixtures/                                  Three one-patient H5 reference fixtures
bundle_manifest.yaml                       Image and model identity / checksums
```

The image does not contain the clinical archive. Provide a local EOS H5 v0.3
archive at start-up; it is mounted read-only. The archive must have
`format = xrd-session-archive` and contain sample sessions with GFRM payloads,
PONI geometry, sample thicknesses and AgBH calibrant thicknesses.

## Start

Install and start Docker Desktop first.

macOS/Linux:

```bash
cd aramis_demo_bundle_0_2_11_beta
bash ./start_demo.sh \
  --source-h5 /absolute/path/to/combined_archive.h5 \
  --output-folder ./outputs
```

Windows PowerShell:

```powershell
Set-Location aramis_demo_bundle_0_2_11_beta
.\start_demo.ps1 `
  -SourceH5 C:\data\combined_archive.h5 `
  -OutputFolder .\outputs
```

Open `http://localhost:8501`. The source archive is read-only. Each selected
case is exported temporarily as a one-patient H5, predicted by the frozen
model, then removed. YAML, JSON and PDF reports are retained under the supplied
host `output-folder`.

The `INTERPRETATION GUIDELINES` section of external PDFs explains the fixed
threshold interpretation: `high` supports `Biopsy required`; `low` supports
`Biopsy not required`. This is research-draft decision support; final clinical
decisions remain with the qualified clinician.

## Stop

macOS/Linux:

```bash
bash ./stop_demo.sh
```

Windows PowerShell:

```powershell
docker rm --force aramis-demo
```
