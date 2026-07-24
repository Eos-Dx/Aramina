param(
    [Parameter(Mandatory = $true)]
    [string]$SourceH5,
    [string]$OutputFolder = ".\outputs",
    [int]$Port = 8501,
    [int]$ApiPort = 8000
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path -LiteralPath $SourceH5 -PathType Leaf)) {
    throw "Missing source H5: $SourceH5"
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { throw "Docker Desktop is not running." }

$Architecture = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
if ($Architecture -eq [System.Runtime.InteropServices.Architecture]::Arm64) {
    $ApiImage = "eosdx/aramis-prediction-api:0.2.12-beta-arm64"
    $ApiArchive = Join-Path $ScriptDir "aramis_prediction_api_linux_arm64_0_2_12_beta.tar"
    $PlatformImage = "eosdx/aramisvisor-demo:0.2.12-beta-arm64"
    $PlatformArchive = Join-Path $ScriptDir "aramisvisor_demo_linux_arm64_0_2_12_beta.tar"
} else {
    $ApiImage = "eosdx/aramis-prediction-api:0.2.12-beta-amd64"
    $ApiArchive = Join-Path $ScriptDir "aramis_prediction_api_linux_amd64_0_2_12_beta.tar"
    $PlatformImage = "eosdx/aramisvisor-demo:0.2.12-beta-amd64"
    $PlatformArchive = Join-Path $ScriptDir "aramisvisor_demo_linux_amd64_0_2_12_beta.tar"
}
foreach ($Archive in @($ApiArchive, $PlatformArchive)) {
    if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw "Missing image archive: $Archive" }
}
docker image inspect $ApiImage *> $null
if ($LASTEXITCODE -ne 0) { docker load --input $ApiArchive }
docker image inspect $PlatformImage *> $null
if ($LASTEXITCODE -ne 0) { docker load --input $PlatformArchive }

$ResolvedH5 = (Resolve-Path -LiteralPath $SourceH5).Path
New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null
$ResolvedOutput = (Resolve-Path -LiteralPath $OutputFolder).Path

docker network inspect aramis-demo-network *> $null
if ($LASTEXITCODE -ne 0) { docker network create aramis-demo-network | Out-Null }
docker rm --force aramis-demo aramis-demo-api *> $null
docker run --detach --name aramis-demo-api --network aramis-demo-network --publish "${ApiPort}:8000" $ApiImage | Out-Null
docker run --detach `
    --name aramis-demo `
    --network aramis-demo-network `
    --publish "${Port}:8501" `
    --env "ARAMIS_PREDICTION_API_URL=http://aramis-demo-api:8000" `
    --env "ARAMIS_DEMO_OUTPUT_ROOT=/opt/aramisvisor/app/static/reports" `
    --env "ARAMIS_MODEL_TEST_ARTIFACT_PATH=/opt/aramisvisor/static/model_test/aramis_mri_or_biopsy_held_out_t130.joblib" `
    --env "STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true" `
    --volume "${ResolvedH5}:/data/source_archive.h5:ro" `
    --volume "${ResolvedOutput}:/opt/aramisvisor/app/static/reports" `
    $PlatformImage | Out-Null

Write-Host "Aramis browser demonstrator: http://127.0.0.1:${Port}"
Write-Host "Aramis prediction API: http://127.0.0.1:${ApiPort}/docs"
Write-Host "Host report folder: $ResolvedOutput"
