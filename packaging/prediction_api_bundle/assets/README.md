# Aramis Prediction API Bundle

Local research-draft API bundle for one-patient Aramis prediction. It contains
the immutable `aramis_target_breast_risk` model version `0.2.11-beta`, its
prediction preprocessing contract, and two Docker images:

| Platform | Image archive |
|---|---|
| Windows x86-64 / Intel macOS | `aramis_prediction_api_linux_amd64_0_2_11_beta.tar` |
| Apple Silicon macOS | `aramis_prediction_api_linux_arm64_0_2_11_beta.tar` |

The API is the same local service used by the Aramisvisor Streamlit
demonstrator. It receives one EOS H5 v0.3 container and a small request JSON;
the selected model, preprocessing, threshold, feature schema and report
contracts cannot be overridden by the caller.

## Contents

```text
contracts/                 API, H5, direct-CLI and report contracts
examples/h5/               cancer, benign and atypical one-patient H5 fixtures
examples/requests/         matching HTTP request JSON files
examples/direct_cli_config/ reference YAML for direct `python -m aramis predict`
start_api.sh / .ps1        load correct image and start API on localhost:8000
predict.sh / .ps1          send an H5 + request JSON and save response JSON
stop_api.sh                stop the local API container
bundle_manifest.yaml       model/image identity and SHA-256 checksums
```

## Start API

Install and start Docker Desktop first. Keep the service local; this bundle has
no authentication or TLS and must not be exposed directly to an untrusted
network.

macOS/Linux:

```bash
cd aramis_prediction_api_bundle_0_2_11_beta
bash ./start_api.sh
curl http://127.0.0.1:8000/health
```

Windows PowerShell:

```powershell
Set-Location aramis_prediction_api_bundle_0_2_11_beta
.\start_api.ps1
curl.exe http://127.0.0.1:8000/health
```

Open interactive API documentation at `http://127.0.0.1:8000/docs`.

## Run included example

macOS/Linux:

```bash
mkdir -p outputs
bash ./predict.sh \
  --input-h5 examples/h5/cancer_one_patient.h5 \
  --request-json examples/requests/cancer_request.json \
  --output-json outputs/cancer_response.json
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
.\predict.ps1 `
  -InputH5 examples\h5\cancer_one_patient.h5 `
  -RequestJson examples\requests\cancer_request.json `
  -OutputJson outputs\cancer_response.json
```

The returned JSON contains `external_report` and `internal_report`. The API
does not persist report files; the caller must save the response. Aramisvisor
converts these payloads into host-side YAML, JSON and PDF reports.

## HTTP request

`POST /predict` uses `multipart/form-data`:

| Form field | Required | Value |
|---|---:|---|
| `input_h5` | yes | one EOS H5 v0.3 file |
| `request_json` | yes | JSON object shown below |

```json
{
  "analysis_author": "Dr Example",
  "prediction_comment": "optional free text",
  "patient_id": "Nova_214",
  "target_side": "left"
}
```

`analysis_author`, `patient_id`, and `target_side` are required.
`prediction_comment` is optional. `target_side` is `left` or `right`; it is
the breast selected by the qualified clinician as suspicious. `patient_id`
must match the only patient identifier inside the H5.

See `contracts/API_CONTRACT.md` before integrating a client.

## H5 contract

The request H5 must be a single-patient EOS H5 container with:

```text
format = xrd-session
schema_version = 0.3
exactly one patientId
measurement GFRM payloads and PONI geometry
sample and AgBH calibrant thicknesses
```

The input may contain only the target breast. A contralateral breast is
optional; when absent, prediction still runs but the internal report records
that symmetry refinement is unavailable. The bundled fixtures contain both
breasts and three measurements per breast.

## Stop API

```bash
bash ./stop_api.sh
```

On Windows:

```powershell
docker rm --force aramis-prediction-api
```
