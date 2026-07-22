param(
    [Parameter(Mandatory = $true)][string]$InputH5,
    [Parameter(Mandatory = $true)][string]$RequestJson,
    [Parameter(Mandatory = $true)][string]$OutputJson,
    [string]$ApiUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
if (!(Test-Path -LiteralPath $InputH5 -PathType Leaf)) { throw "Missing H5: $InputH5" }
if (!(Test-Path -LiteralPath $RequestJson -PathType Leaf)) { throw "Missing request JSON: $RequestJson" }
New-Item -ItemType Directory -Force (Split-Path -Parent $OutputJson) | Out-Null

& curl.exe --fail-with-body --silent --show-error `
    --request POST "$ApiUrl/predict" `
    --form "input_h5=@$InputH5;type=application/x-hdf5" `
    --form "request_json=<$RequestJson" `
    --output $OutputJson
if ($LASTEXITCODE -ne 0) { throw "Prediction request failed." }

Write-Host "Prediction response: $OutputJson"
