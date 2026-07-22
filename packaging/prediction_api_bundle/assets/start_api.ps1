param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ImageTag = "eosdx/aramis-prediction-api:0.2.10-beta-amd64"
$ImageArchive = Join-Path $BundleRoot "aramis_prediction_api_linux_amd64_0_2_10_beta.tar"

docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop must be installed and running."
}

docker image inspect $ImageTag *> $null
if ($LASTEXITCODE -ne 0) {
    docker load --input $ImageArchive
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to load the API Docker image."
    }
}

docker rm --force aramis-prediction-api *> $null
docker run --detach --name aramis-prediction-api --publish "${Port}:8000" $ImageTag | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Unable to start the Aramis prediction API."
}

Write-Host "Aramis prediction API: http://127.0.0.1:$Port"
Write-Host "OpenAPI documentation: http://127.0.0.1:$Port/docs"
