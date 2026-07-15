param(
    [string]$Workspace = ""
)

$ErrorActionPreference = "Stop"
$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Get-Content (Join-Path $BundleDir "bundle_manifest.json") -Raw | ConvertFrom-Json

if ([string]::IsNullOrWhiteSpace($Workspace)) {
    $Workspace = Join-Path $BundleDir "workspace"
}

$LogDir = Join-Path $Workspace "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("install_and_train_{0}.log" -f (Get-Date -Format "yyyyMMddTHHmmss"))
Start-Transcript -Path $LogPath | Out-Null

function Write-Stage {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ==="
}

function Invoke-External {
    param(
        [string]$Description,
        [string]$FilePath,
        [string[]]$Arguments
    )
    Write-Stage $Description
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
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
    Write-Stage "Installing Miniforge"
    Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile $Installer
    $Process = Start-Process -FilePath $Installer -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$Prefix" -Wait -PassThru
    if ($Process.ExitCode -ne 0) {
        throw "Miniforge installation failed with exit code $($Process.ExitCode)."
    }
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
    Write-Stage "Installing Git for Windows"
    & $Winget.Source install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Git for Windows installation failed with exit code $LASTEXITCODE."
    }
    $Git = Find-Git
    if (-not $Git) {
        throw "Git installation completed but git.exe is not available. Open a new terminal and rerun install_and_train.bat."
    }
    return $Git
}

function Sync-Repository {
    param(
        [string]$Repository,
        [string]$Path,
        [string]$Commit,
        [string]$Name,
        [string]$Git
    )
    if (Test-Path $Path) {
        if (-not (Test-Path (Join-Path $Path ".git"))) {
            throw "$Name path exists but is not a Git checkout: $Path"
        }
        Invoke-External "Fetch $Name" $Git @("-C", $Path, "fetch", "--tags", "--prune", "origin")
    } else {
        Invoke-External "Clone $Name" $Git @("clone", $Repository, $Path)
    }
    Invoke-External "Checkout $Name commit $Commit" $Git @("-C", $Path, "checkout", "--detach", $Commit)
    Invoke-External "Reset $Name checkout" $Git @("-C", $Path, "reset", "--hard", $Commit)
}

function Link-BundleData {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (-not (Test-Path (Join-Path $Source "combined_archive.h5"))) {
        throw "Missing bundled H5 input: $(Join-Path $Source 'combined_archive.h5')"
    }
    if (Test-Path $Destination) {
        $ResolvedDestination = (Resolve-Path $Destination).Path
        $ResolvedSource = (Resolve-Path $Source).Path
        if ($ResolvedDestination -eq $ResolvedSource) {
            Write-Stage "Reuse bundle data link"
            return
        }
        throw "Workspace data path already exists and does not reference bundle data: $Destination"
    }
    Write-Stage "Link bundled H5 input"
    try {
        New-Item -ItemType Junction -Path $Destination -Target $Source | Out-Null
    } catch {
        throw "Unable to create a workspace data junction. Use a local NTFS workspace: $($_.Exception.Message)"
    }
}

function Verify-BundledH5 {
    param(
        [string]$Path,
        [string]$ExpectedSha256
    )
    Write-Stage "Verify bundled H5 checksum"
    $ActualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "Bundled H5 SHA256 mismatch. Expected $ExpectedSha256, got $ActualSha256. Extract a fresh bundle."
    }
    Write-Host "Bundled H5 SHA256 verified: $ActualSha256"
}

try {
    Write-Stage "Aramis reproducible training bundle"
    Write-Host "Workspace: $Workspace"
    Write-Host "Log: $LogPath"
    Write-Host "Expected input H5 SHA256: $($Manifest.h5_sha256)"

    $Conda = Ensure-Conda
    $Git = Ensure-Git
    $AramisRepo = Join-Path $Workspace "Aramis"
    $XrdRepo = Join-Path $Workspace "XRD-preprocessing"
    $WorkspaceData = Join-Path $Workspace "data"
    $BundledData = Join-Path $BundleDir "data"

    Verify-BundledH5 (Join-Path $BundledData "combined_archive.h5") $Manifest.h5_sha256
    Link-BundleData $BundledData $WorkspaceData
    Sync-Repository $Manifest.aramis_repository $AramisRepo $Manifest.aramis_commit "Aramis" $Git
    Sync-Repository $Manifest.xrd_preprocessing_repository $XrdRepo $Manifest.xrd_preprocessing_commit "XRD-preprocessing" $Git

    $EnvNames = & $Conda env list
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to list conda environments."
    }
    $EnvironmentFile = Join-Path $BundleDir "environment.yml"
    if ($EnvNames -match "(?m)^$($Manifest.environment_name)\s") {
        Write-Stage "Reuse conda environment $($Manifest.environment_name)"
    } else {
        Invoke-External "Create conda environment $($Manifest.environment_name)" $Conda @("env", "create", "-n", $Manifest.environment_name, "-f", $EnvironmentFile)
    }

    $XrdInstall = "${XrdRepo}[dev]"
    $AramisInstall = "${AramisRepo}[dev]"
    Invoke-External "Install XRD-preprocessing from selected commit" $Conda @("run", "--no-capture-output", "-n", $Manifest.environment_name, "python", "-m", "pip", "install", "-e", $XrdInstall)
    Invoke-External "Install Aramis from selected commit" $Conda @("run", "--no-capture-output", "-n", $Manifest.environment_name, "python", "-m", "pip", "install", "--no-deps", "-e", $AramisInstall)
    Invoke-External "Verify Python imports" $Conda @("run", "--no-capture-output", "-n", $Manifest.environment_name, "python", "-c", "import aramis, xrd_preprocessing; print('aramis', aramis.__file__); print('xrd_preprocessing', xrd_preprocessing.__file__)")

    $WorkflowOutput = Join-Path $AramisRepo "examples\outputs\workflows"
    if (Test-Path $WorkflowOutput) {
        Write-Stage "Remove prior generated workflow outputs"
        Remove-Item -Recurse -Force $WorkflowOutput
    }

    Push-Location $AramisRepo
    try {
        Invoke-External "Run preprocessing and training" $Conda @("run", "--no-capture-output", "-n", $Manifest.environment_name, "python", "-m", "aramis", "preprocess-train", "--config", $Manifest.workflow_config, "--verbose")
        $LatestTraining = Get-ChildItem -Path "examples\outputs\workflows" -Filter "model.joblib" -Recurse |
            Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
        if (-not $LatestTraining) { throw "No generated model.joblib was found." }
        $ReferenceModel = Join-Path $AramisRepo $Manifest.reference_model_relative_path
        Invoke-External "Compare generated model with reference" $Conda @("run", "--no-capture-output", "-n", $Manifest.environment_name, "python", "scripts\compare_model_artifacts.py", "--reference", $ReferenceModel, "--candidate", $LatestTraining.FullName)
        Write-Host "Generated model: $($LatestTraining.FullName)"
    } finally {
        Pop-Location
    }

    Write-Stage "Bundle completed"
    Write-Host "Log saved to: $LogPath"
} finally {
    Stop-Transcript | Out-Null
}
