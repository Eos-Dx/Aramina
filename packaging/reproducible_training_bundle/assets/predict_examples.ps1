param(
    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"
$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Get-Content (Join-Path $BundleDir "bundle_manifest.json") -Raw | ConvertFrom-Json
$ConfigDir = Join-Path $BundleDir "config"
$ExampleH5Dir = Join-Path $BundleDir "examples\prediction_h5"
$OutputDir = Join-Path $BundleDir "outputs"
$LogDir = Join-Path $OutputDir "logs"
$ImageTag = $Manifest.image_amd64_tag
$ImagePlatform = $Manifest.image_amd64_platform
$ImageArchive = Join-Path $BundleDir $Manifest.image_amd64_archive

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("predict_examples_{0}.log" -f (Get-Date -Format "yyyyMMddTHHmmss"))
Start-Transcript -Path $LogPath | Out-Null

function Write-Stage {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ==="
}

function Invoke-Docker {
    param([string]$Description, [string[]]$Arguments)
    Write-Stage $Description
    & docker.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

try {
    Get-Command docker.exe -ErrorAction Stop | Out-Null
    & docker.exe version | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Linux engine is not running. Start Docker Desktop, then rerun this script."
    }

    if ([string]::IsNullOrWhiteSpace($ModelPath)) {
        $model = Get-ChildItem -Path (Join-Path $OutputDir "preprocess_train") -Filter "model.joblib" -File -Recurse -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if (-not $model) {
            throw "No trained model found under outputs\preprocess_train\. Run install_and_train first or pass -ModelPath."
        }
        $ModelPath = $model.FullName
    }

    $resolvedModel = (Resolve-Path -LiteralPath $ModelPath -ErrorAction Stop).Path
    $resolvedOutput = (Resolve-Path -LiteralPath $OutputDir).Path.TrimEnd([char[]]@('\', '/'))
    $prefix = $resolvedOutput + '\'
    if (-not $resolvedModel.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "ModelPath must point inside bundle outputs\: $resolvedModel"
    }
    $containerModel = "/opt/Aramis/examples/outputs/" + $resolvedModel.Substring($prefix.Length).Replace('\', '/')

    $imageInspect = Start-Process `
        -FilePath "docker.exe" `
        -ArgumentList @("image", "inspect", $ImageTag) `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($imageInspect.ExitCode -ne 0) {
        if (-not (Test-Path $ImageArchive)) {
            throw "Missing Docker image archive: $ImageArchive"
        }
        $actualImage = (Get-FileHash -LiteralPath $ImageArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualImage -ne $Manifest.image_amd64_archive_sha256.ToLowerInvariant()) {
            throw "Docker image SHA256 mismatch. Extract a fresh bundle."
        }
        Invoke-Docker "Load validated Linux runtime image" @("load", "--input", $ImageArchive)
    }

    Invoke-Docker "Run three prediction fixtures" @(
        "run", "--rm", "--platform", $ImagePlatform,
        "--mount", "type=bind,src=$ConfigDir,dst=/opt/Aramis/config,readonly",
        "--mount", "type=bind,src=$ExampleH5Dir,dst=/opt/Aramis/examples/prediction_h5,readonly",
        "--mount", "type=bind,src=$OutputDir,dst=/opt/Aramis/examples/outputs",
        $ImageTag,
        "bash", "/opt/aramis-bundle/run_prediction_examples_docker.sh", "--model", $containerModel
    )

    Write-Stage "Prediction examples completed"
    Write-Host "Reports: $(Join-Path $OutputDir 'prediction_examples')"
    Write-Host "Log saved to: $LogPath"
} finally {
    Stop-Transcript | Out-Null
}
