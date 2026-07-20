param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$InputH5,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [Parameter(Mandatory = $true)][string]$OutputFolder
)

$ErrorActionPreference = "Stop"
$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Get-Content (Join-Path $BundleDir "bundle_manifest.json") -Raw | ConvertFrom-Json
$LogDir = Join-Path $BundleDir "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("predict_{0}.log" -f (Get-Date -Format "yyyyMMddTHHmmss"))
Start-Transcript -Path $LogPath | Out-Null

function Resolve-RequiredFile {
    param([string]$Value, [string]$Description)
    $item = Get-Item -LiteralPath $Value -ErrorAction Stop
    if ($item.PSIsContainer) { throw "$Description must be a file: $Value" }
    return $item.FullName
}

function Write-Stage { param([string]$Message) Write-Host ""; Write-Host "=== $Message ===" }

function Invoke-Docker {
    param([string]$Description, [string[]]$Arguments)
    Write-Stage $Description
    & docker.exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE." }
}

try {
    $resolvedConfig = Resolve-RequiredFile $Config "Config"
    $resolvedH5 = Resolve-RequiredFile $InputH5 "InputH5"
    $resolvedModel = Resolve-RequiredFile $ModelPath "ModelPath"
    New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null
    $resolvedOutput = (Resolve-Path -LiteralPath $OutputFolder).Path.TrimEnd([char[]]@('\', '/'))

    Get-Command docker.exe -ErrorAction Stop | Out-Null
    & docker.exe version | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Docker Linux engine is not running. Start Docker Desktop, then rerun this script." }

    $imageTag = $Manifest.image_amd64_tag
    $imagePlatform = $Manifest.image_amd64_platform
    $imageArchive = Join-Path $BundleDir $Manifest.image_amd64_archive
    $imageInspect = Start-Process -FilePath "docker.exe" -ArgumentList @("image", "inspect", $imageTag) -WindowStyle Hidden -Wait -PassThru
    if ($imageInspect.ExitCode -ne 0) {
        if (-not (Test-Path $imageArchive)) { throw "Missing Docker image archive: $imageArchive" }
        $actualImage = (Get-FileHash -LiteralPath $imageArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualImage -ne $Manifest.image_amd64_archive_sha256.ToLowerInvariant()) { throw "Docker image SHA256 mismatch. Extract a fresh bundle." }
        Invoke-Docker "Load validated Linux runtime image" @("load", "--input", $imageArchive)
    }

    Invoke-Docker "Run external H5 prediction" @(
        "run", "--rm", "--platform", $imagePlatform,
        "--mount", "type=bind,src=$(Split-Path -Parent $resolvedConfig),dst=/opt/aramis-user-config,readonly",
        "--mount", "type=bind,src=$(Split-Path -Parent $resolvedH5),dst=/opt/aramis-user-input,readonly",
        "--mount", "type=bind,src=$(Split-Path -Parent $resolvedModel),dst=/opt/aramis-user-model,readonly",
        "--mount", "type=bind,src=$resolvedOutput,dst=/opt/aramis-user-output",
        $imageTag,
        "bash", "/opt/aramis-bundle/run_prediction_docker.sh",
        "--config", "/opt/aramis-user-config/$(Split-Path -Leaf $resolvedConfig)",
        "--input-h5", "/opt/aramis-user-input/$(Split-Path -Leaf $resolvedH5)",
        "--model", "/opt/aramis-user-model/$(Split-Path -Leaf $resolvedModel)",
        "--output-folder", "/opt/aramis-user-output"
    )

    Write-Stage "Prediction completed"
    Write-Host "Reports: $resolvedOutput"
    Write-Host "Log saved to: $LogPath"
} finally {
    Stop-Transcript | Out-Null
}
