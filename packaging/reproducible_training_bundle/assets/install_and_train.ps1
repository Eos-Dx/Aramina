param(
    [string]$Workspace = ""
)

$ErrorActionPreference = "Stop"
$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Get-Content (Join-Path $BundleDir "bundle_manifest.json") -Raw | ConvertFrom-Json

if ([string]::IsNullOrWhiteSpace($Workspace)) {
    $Workspace = Join-Path $BundleDir "workspace"
}

function Find-Conda {
    $Command = Get-Command conda -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    foreach ($Candidate in @(
        "$HOME\miniforge3\Scripts\conda.exe",
        "$HOME\miniconda3\Scripts\conda.exe",
        "$HOME\anaconda3\Scripts\conda.exe"
    )) {
        if (Test-Path $Candidate) { return $Candidate }
    }
    return $null
}

function Install-Miniforge {
    $Prefix = "$HOME\miniforge3"
    $Installer = Join-Path $env:TEMP "Miniforge3-Windows-x86_64.exe"
    Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile $Installer
    Start-Process -FilePath $Installer -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$Prefix" -Wait
    return "$Prefix\Scripts\conda.exe"
}

function Ensure-Conda {
    $Conda = Find-Conda
    if ($Conda) { return $Conda }
    Write-Host "conda not found; installing Miniforge for the current user."
    return Install-Miniforge
}

function Find-Git {
    $Command = Get-Command git -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    foreach ($Candidate in @(
        (Join-Path $env:ProgramFiles "Git\cmd\git.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Git\cmd\git.exe")
    )) {
        if (Test-Path $Candidate) { return $Candidate }
    }
    return $null
}

function Ensure-Git {
    $Git = Find-Git
    if ($Git) { return $Git }
    $Winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "Git is required. Install Git for Windows, then rerun install_and_train.bat."
    }
    Write-Host "Git not found; installing Git for Windows with winget."
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    $Git = Find-Git
    if (-not $Git) {
        throw "Git installation completed but git.exe is not available. Open a new terminal and rerun install_and_train.bat."
    }
    return $Git
}

$Conda = Ensure-Conda
$Git = Ensure-Git
$AramisRepo = Join-Path $Workspace "Aramis"
$XrdRepo = Join-Path $Workspace "XRD-preprocessing"
$ExpectedH5 = Join-Path $Workspace "eos_play\jupyter_notebooks\Clinical_trials\data\product-aramis-data\combined_archive.h5"
$BundledH5 = Join-Path $BundleDir "data\combined_archive.h5"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ExpectedH5) | Out-Null
Copy-Item -Force $BundledH5 $ExpectedH5

if (Test-Path $AramisRepo) { Remove-Item -Recurse -Force $AramisRepo }
if (Test-Path $XrdRepo) { Remove-Item -Recurse -Force $XrdRepo }

& $Git clone $Manifest.aramis_repository $AramisRepo
& $Git -C $AramisRepo checkout $Manifest.aramis_commit
& $Git clone $Manifest.xrd_preprocessing_repository $XrdRepo
& $Git -C $XrdRepo checkout $Manifest.xrd_preprocessing_commit

$EnvironmentFile = Join-Path $BundleDir "environment.yml"
$EnvNames = & $Conda env list
if ($EnvNames -match "(?m)^$($Manifest.environment_name)\s") {
    & $Conda env update -n $Manifest.environment_name -f $EnvironmentFile
} else {
    & $Conda env create -n $Manifest.environment_name -f $EnvironmentFile
}

$XrdInstall = "$XrdRepo" + "[dev]"
$AramisInstall = "$AramisRepo" + "[dev]"
& $Conda run --no-capture-output -n $Manifest.environment_name python -m pip install -e $XrdInstall
& $Conda run --no-capture-output -n $Manifest.environment_name python -m pip install --no-deps -e $AramisInstall

Push-Location $AramisRepo
try {
    & $Conda run --no-capture-output -n $Manifest.environment_name python -m aramis preprocess-train --config $Manifest.workflow_config
    $LatestTraining = Get-ChildItem -Path "examples\outputs\workflows" -Filter "model.joblib" -Recurse |
        Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if (-not $LatestTraining) { throw "No generated model.joblib was found." }
    $ReferenceModel = Join-Path $AramisRepo $Manifest.reference_model_relative_path
    & $Conda run --no-capture-output -n $Manifest.environment_name python scripts\compare_model_artifacts.py --reference $ReferenceModel --candidate $LatestTraining.FullName
    Write-Host "Generated model: $($LatestTraining.FullName)"
} finally {
    Pop-Location
}
