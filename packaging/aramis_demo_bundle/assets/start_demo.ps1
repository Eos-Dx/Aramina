param(
    [Parameter(Mandatory = $true)]
    [string]$SourceH5,
    [string]$OutputFolder = ".\\outputs",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path -LiteralPath $SourceH5 -PathType Leaf)) {
    throw "Missing source H5: $SourceH5"
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not running."
}

$Architecture = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
if ($Architecture -eq [System.Runtime.InteropServices.Architecture]::Arm64) {
    $ImageTag = "eosdx/aramis-demo:0.2.11-beta-arm64"
    $ImageArchive = Join-Path $ScriptDir "aramis_demo_linux_arm64_0_2_11_beta.tar"
} else {
    $ImageTag = "eosdx/aramis-demo:0.2.11-beta-amd64"
    $ImageArchive = Join-Path $ScriptDir "aramis_demo_linux_amd64_0_2_11_beta.tar"
}

docker image inspect $ImageTag *> $null
if ($LASTEXITCODE -ne 0) {
    docker load --input $ImageArchive
}

$ResolvedH5 = (Resolve-Path -LiteralPath $SourceH5).Path
New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null
$ResolvedOutput = (Resolve-Path -LiteralPath $OutputFolder).Path

docker rm --force aramis-demo *> $null
docker run --detach `
    --name aramis-demo `
    --publish "${Port}:8501" `
    --env "ARAMIS_DEMO_OUTPUT_ROOT=/opt/aramis-demo/static/reports" `
    --volume "${ResolvedH5}:/data/source_archive.h5:ro" `
    --volume "${ResolvedOutput}:/opt/aramis-demo/static/reports" `
    $ImageTag | Out-Null

Write-Host "Aramis browser demonstrator: http://127.0.0.1:${Port}"
Write-Host "Host report folder: $ResolvedOutput"
