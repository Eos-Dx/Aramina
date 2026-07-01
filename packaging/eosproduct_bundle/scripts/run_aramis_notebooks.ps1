param(
    [string]$TargetRoot = "$HOME\dev\eosproduct",
    [string]$EnvName = $env:ENV_NAME
)

if (-not $EnvName) {
    $EnvName = "eosproduct"
}

$ErrorActionPreference = "Stop"

$AramisRoot = Join-Path $TargetRoot "Aramis"
$OneToOneConfig = Join-Path $AramisRoot "config\preprocessing\aramis_one_to_one_max_v0_1.yaml"
$OneToManyConfig = Join-Path $AramisRoot "config\preprocessing\aramis_one_to_many_max_v0_1.yaml"

$OneToOneCmd = "cd `"$AramisRoot`"; conda run --no-capture-output -n `"$EnvName`" python -m marimo run --host 127.0.0.1 --port 27181 --no-token examples/aramis_dataframe_one_to_one_v0_1.py -- --aramis-preprocessing-config-path `"$OneToOneConfig`""
$OneToManyCmd = "cd `"$AramisRoot`"; conda run --no-capture-output -n `"$EnvName`" python -m marimo run --host 127.0.0.1 --port 27182 --no-token examples/aramis_dataframe_one_to_many_v0_1.py -- --aramis-preprocessing-config-path `"$OneToManyConfig`""

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Aramis one-to-one: http://127.0.0.1:27181'; $OneToOneCmd"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Aramis one-to-many: http://127.0.0.1:27182'; $OneToManyCmd"
