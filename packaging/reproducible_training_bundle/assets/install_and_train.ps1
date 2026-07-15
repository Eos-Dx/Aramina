param()

$ErrorActionPreference = "Stop"
$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Get-Content (Join-Path $BundleDir "bundle_manifest.json") -Raw | ConvertFrom-Json
$DataDir = Join-Path $BundleDir "data"
$OutputDir = Join-Path $BundleDir "outputs"
$LogDir = Join-Path $OutputDir "logs"
$ImageArchive = Join-Path $BundleDir $Manifest.image_archive

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("install_and_train_{0}.log" -f (Get-Date -Format "yyyyMMddTHHmmss"))
Start-Transcript -Path $LogPath | Out-Null

function Write-Stage {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ==="
}

function Invoke-Docker {
    param([string]$Description, [string[]]$Arguments)
    Write-Stage $Description
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

try {
    Write-Stage "Aramis Docker reproducible training bundle"
    Write-Host "Log: $LogPath"

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker Desktop is required. Install it, enable the WSL 2 Linux engine, start Docker Desktop, then rerun this script."
    }
    & docker version | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop is installed but its Linux engine is not running. Start Docker Desktop, wait until it reports Engine running, then rerun this script."
    }

    $H5Path = Join-Path $DataDir "combined_archive.h5"
    if (-not (Test-Path $H5Path)) {
        throw "Missing bundled H5 input: $H5Path"
    }
    $ActualSha256 = (Get-FileHash -LiteralPath $H5Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $Manifest.h5_sha256.ToLowerInvariant()) {
        throw "Bundled H5 SHA256 mismatch. Expected $($Manifest.h5_sha256), got $ActualSha256. Extract a fresh bundle."
    }
    Write-Host "Bundled H5 SHA256 verified: $ActualSha256"

    & docker image inspect $Manifest.image_tag *> $null
    if ($LASTEXITCODE -ne 0) {
        if (-not (Test-Path $ImageArchive)) {
            throw "Missing Docker image archive: $ImageArchive"
        }
        $ArchiveSha256 = (Get-FileHash -LiteralPath $ImageArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ArchiveSha256 -ne $Manifest.image_archive_sha256.ToLowerInvariant()) {
            throw "Docker image SHA256 mismatch. Expected $($Manifest.image_archive_sha256), got $ArchiveSha256. Extract a fresh bundle."
        }
        Invoke-Docker "Load validated Linux runtime image" @("load", "--input", $ImageArchive)
    } else {
        Write-Stage "Reuse loaded Linux runtime image"
        Write-Host $Manifest.image_tag
    }

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Invoke-Docker "Run Linux preprocessing and training" @(
        "run", "--rm",
        "--mount", "type=bind,src=$DataDir,dst=/opt/data,readonly",
        "--mount", "type=bind,src=$OutputDir,dst=/opt/Aramis/examples/outputs",
        $Manifest.image_tag,
        "bash", "/opt/aramis-bundle/run_training_docker.sh"
    )

    Write-Stage "Bundle completed"
    Write-Host "Outputs: $OutputDir"
    Write-Host "Log saved to: $LogPath"
} finally {
    Stop-Transcript | Out-Null
}
