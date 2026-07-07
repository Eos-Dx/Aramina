param(
    [string]$TargetRoot = "$HOME\dev\eosproduct",
    [string]$EnvName = $env:ENV_NAME
)

if (-not $EnvName) {
    $EnvName = "eosproduct"
}

$ErrorActionPreference = "Stop"

$AramisRoot = Join-Path $TargetRoot "Aramis"
$AllPatientsConfig = Join-Path $AramisRoot "config\preprocessing\aramis_all_patients_model_input_v0_1.yaml"
$BiopsyPatientsConfig = Join-Path $AramisRoot "config\preprocessing\aramis_biopsy_patients_model_input_v0_1.yaml"

$AllPatientsCmd = "cd `"$AramisRoot`"; conda run --no-capture-output -n `"$EnvName`" python -m marimo run --host 127.0.0.1 --port 27181 --no-token examples/aramis_dataframe_all_patients_v0_1.py -- --aramis-preprocessing-config-path `"$AllPatientsConfig`""
$BiopsyPatientsCmd = "cd `"$AramisRoot`"; conda run --no-capture-output -n `"$EnvName`" python -m marimo run --host 127.0.0.1 --port 27182 --no-token examples/aramis_dataframe_biopsy_patients_v0_1.py -- --aramis-preprocessing-config-path `"$BiopsyPatientsConfig`""

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Aramis all-patients model input: http://127.0.0.1:27181'; $AllPatientsCmd"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Aramis biopsy-patients model input: http://127.0.0.1:27182'; $BiopsyPatientsCmd"
