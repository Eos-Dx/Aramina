# Aramisvisor Demonstrator Bundle

Local research-draft demonstrator for frozen product artifact
`aramis_target_breast_risk_0_2_12-beta_f8af641a2e49`.

The bundle contains two immutable Docker images:

```text
Aramis prediction API -> owns model.joblib and preprocessing contract
Aramisvisor Streamlit platform -> selects a patient, sends one H5 to API, renders reports
```

The browser platform never loads a model artifact. It sends an H5 v0.3 and the
minimal request fields to `POST /predict`. The API returns external and
internal report payloads. The platform writes YAML and PDF reports to the host
output folder.

## Start

Install and start Docker Desktop.

macOS/Linux:

```bash
cd aramis_demo_bundle_0_2_12_beta
bash ./start_demo.sh \
  --source-h5 /absolute/path/to/combined_archive.h5 \
  --output-folder ./outputs
```

Windows PowerShell:

```powershell
Set-Location aramis_demo_bundle_0_2_12_beta
.\start_demo.ps1 `
  -SourceH5 C:\data\combined_archive.h5 `
  -OutputFolder .\outputs
```

Open `http://localhost:8501`. API documentation is at
`http://localhost:8000/docs`. The archive is mounted read-only. Each selected
case is temporarily exported as one H5 v0.3, sent to the API, then removed.

Output files:

```text
outputs/<report_id>/
  external_report.yaml
  internal_report.yaml
  external_report.pdf
  internal_report.pdf
```

The patient selector labels training-cohort patients red and patients not used
for model training blue. This is demonstration provenance only, not model input
or independent validation.

The bundled Model test tab uses a separate T130 quality-controlled,
patient-disjoint same-source cohort. It is exploratory evidence, not an
independent external-validation claim.

## Stop

macOS/Linux:

```bash
bash ./stop_demo.sh
```

Windows PowerShell:

```powershell
.\stop_demo.ps1
```
