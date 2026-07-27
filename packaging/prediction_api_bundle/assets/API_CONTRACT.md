# Aramina Prediction HTTP API Contract v0.1

Status: local research-draft decision-support API. It serves one frozen model
artifact only: `aramina_target_breast_risk` version `0.2.12-beta`.

## Base URL

```text
http://127.0.0.1:8000
```

The default service binds to the local host through Docker port mapping. No
authentication, access control, or TLS is supplied by this demonstration
bundle. Put a separately designed authenticated gateway in front of it before
any networked use.

## `GET /health`

Returns readiness only. It does not score an H5 or expose model parameters.

```json
{"status": "ready", "model_artifact": "model.joblib"}
```

`503` means the immutable model artifact is unavailable inside the container.

## `POST /predict`

Content type: `multipart/form-data`.

| Field | Type | Required | Rules |
|---|---|---:|---|
| `input_h5` | file | yes | filename ends in `.h5`; one EOS H5 v0.3 patient container |
| `request_json` | text | yes | JSON object defined below |

`request_json` schema:

```json
{
  "analysis_author": "non-empty text",
  "prediction_comment": "optional text",
  "patient_id": "must match the only H5 patientId",
  "target_side": "left or right"
}
```

No additional fields are accepted. The caller cannot provide a model path,
preprocessing YAML, threshold, feature list, report version, or output folder.
Those are frozen inside the image/model artifact.

### H5 requirements

The service applies the model-held prediction preprocessing. Before scoring,
the H5 is required to have:

```text
root @format = xrd-session
root @schema_version = 0.3
exactly one patientId matching request_json.patient_id
valid measurement payloads, PONI geometry, sample thickness and AgBH calibrant thickness
```

At least target-breast quality-passing measurements are needed. A
contralateral breast is optional. When unavailable or insufficient after QC,
the model applies its neutral symmetry gate and the internal report records
`symmetry.available: false` and lower reliability for that breast.

### Success response: `200`

```json
{
  "external_report": {"...": "external report v0.6 payload"},
  "internal_report": {"...": "internal clinical report v0.9 payload"}
}
```

The target external report carries `risk_probability`, fixed
`decision_threshold`, threshold-derived `target_class_risk_level` and
`biopsy_required`, reliability, model version, and final-fit model
sensitivity/specificity. The internal report contains target and contralateral
profile/final predictions, target-side high/low risk level and biopsy action,
TRA level, audit metadata, and symmetry availability. The contralateral block
is evidence only and has no biopsy action.

The service deliberately returns payloads in memory. It deletes the uploaded
H5 and temporary prediction files after the response. The client is responsible
for saving JSON/YAML/PDF outputs. Araminavisor is one reference client that
writes reports on its host volume and renders PDFs.

### Errors

| Status | Meaning |
|---:|---|
| `400` | invalid request JSON, unsupported field, malformed/mismatched H5, patient-ID mismatch, or prediction preprocessing failure |
| `503` | model artifact unavailable in container |

The response body has FastAPI `detail` text intended for integration logs.

## Reference client

```bash
curl --fail-with-body --request POST http://127.0.0.1:8000/predict \
  --form 'input_h5=@examples/h5/cancer_one_patient.h5;type=application/x-hdf5' \
  --form 'request_json=<examples/requests/cancer_request.json' \
  --output outputs/cancer_response.json
```

The full OpenAPI definition is `openapi.yaml`; FastAPI also serves the same
interactive schema at `/docs` while the container is running.
