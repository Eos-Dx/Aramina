param(
    [string]$EnvName = "eosproduct",
    [switch]$AutoRun
)

$ErrorActionPreference = "Stop"
$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DefaultTarget = Join-Path $HOME "dev\$EnvName"

function Ask-YesNo($Prompt, $Default = "y") {
    if ($AutoRun) {
        return $Default.ToLower().StartsWith("y")
    }
    $Suffix = if ($Default.ToLower().StartsWith("y")) { "Y/n" } else { "y/N" }
    $Answer = Read-Host "$Prompt [$Suffix]"
    if ([string]::IsNullOrWhiteSpace($Answer)) {
        $Answer = $Default
    }
    return $Answer.ToLower().StartsWith("y")
}

function Ask-Value($Prompt, $Default) {
    if ($AutoRun) {
        return $Default
    }
    $Answer = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrWhiteSpace($Answer) -or $Answer -match "^(y|yes)$") {
        return $Default
    }
    return $Answer
}

function Find-Conda {
    $Command = Get-Command conda -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }
    $Candidates = @(
        "$HOME\miniforge3\Scripts\conda.exe",
        "$HOME\miniconda3\Scripts\conda.exe",
        "$HOME\anaconda3\Scripts\conda.exe"
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            return $Candidate
        }
    }
    return $null
}

function Install-Miniforge {
    $Prefix = "$HOME\miniforge3"
    $Installer = Join-Path $env:TEMP "Miniforge3-Windows-x86_64.exe"
    $Url = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
    Write-Host "Downloading Miniforge: $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Installer
    Start-Process -FilePath $Installer -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$Prefix" -Wait
    return "$Prefix\Scripts\conda.exe"
}

function Ensure-Conda {
    $Conda = Find-Conda
    if ($Conda) {
        return $Conda
    }
    if (Ask-YesNo "conda not found. Install Miniforge to ~/miniforge3?" "y") {
        return Install-Miniforge
    }
    throw "conda is required. Install Miniforge/Miniconda and rerun install.ps1."
}

function Ensure-Git {
    $Command = Get-Command git -ErrorAction SilentlyContinue
    if ($Command) {
        return $true
    }
    if (-not (Ask-YesNo "git not found. Install Git if possible?" "y")) {
        return $false
    }
    $Winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($Winget) {
        winget install --id Git.Git -e --source winget
        return [bool](Get-Command git -ErrorAction SilentlyContinue)
    }
    Write-Host "winget not found. Install Git manually from https://git-scm.com/download/win or use bundled fallback."
    return $false
}

function Copy-OrUpdateRepo($Name, $Url, $Target, $RequiredPath, $Ref = "") {
    $Fallback = Join-Path $BundleDir "repos\$Name"
    if ($script:UseGit -eq "yes") {
        if (Test-Path (Join-Path $Target ".git")) {
            git -C $Target fetch --tags origin
            if ($Ref) {
                git -C $Target checkout $Ref
            } else {
                git -C $Target pull --ff-only
            }
        } else {
            Remove-Item -Recurse -Force $Target -ErrorAction SilentlyContinue
            if ($Ref) {
                git clone --branch $Ref $Url $Target
            } else {
                git clone $Url $Target
            }
        }
        if ((Test-Path (Join-Path $Target ".git")) -and ((-not $RequiredPath) -or (Test-Path (Join-Path $Target $RequiredPath)))) {
            return
        }
        Write-Host "Git checkout for $Name is not usable; using bundled fallback."
    }
    Remove-Item -Recurse -Force $Target -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Target) | Out-Null
    Copy-Item -Recurse -Force $Fallback $Target
}

$TargetRoot = Ask-Value "Target root" $DefaultTarget
New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

$script:UseGit = "no"
if ((Ensure-Git) -and (Ask-YesNo "Use git to clone/update repos?" "y")) {
    $script:UseGit = "yes"
}

Copy-OrUpdateRepo "XRD-preprocessing" "https://github.com/Eos-Dx/XRD-preprocessing.git" (Join-Path $TargetRoot "XRD-preprocessing") "src\xrd_preprocessing\configs\preprocessing_branch_config_template.yaml" "v0.1.6-beta"
Copy-OrUpdateRepo "Aramis" "https://github.com/Eos-Dx/Aramis.git" (Join-Path $TargetRoot "Aramis") "examples\prediction_models\aramis_m2q_t100_core4_optional_symmetry_c1_0p1_c2_0p1.joblib" "0.1.9-beta"
Copy-OrUpdateRepo "container" "https://github.com/Eos-Dx/container.git" (Join-Path $TargetRoot "container") "pyproject.toml" "feat/v0_3-eoscan-session-container"
New-Item -ItemType Directory -Force -Path (Join-Path $TargetRoot "Bremen") | Out-Null

$DataDir = Join-Path $TargetRoot "data"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$BundledH5 = Join-Path $BundleDir "data\combined_archive.h5"
if (Test-Path $BundledH5) {
    Copy-Item -Force $BundledH5 (Join-Path $DataDir "combined_archive.h5")
}

if (Ask-YesNo "Create/update conda env $EnvName?" "y") {
    $Conda = Ensure-Conda
    $EnvList = & $Conda env list
    if ($EnvList -match "(?m)^$EnvName\s") {
        & $Conda env update -n $EnvName -f (Join-Path $BundleDir "environment.yml")
    } else {
        & $Conda env create -n $EnvName -f (Join-Path $BundleDir "environment.yml")
    }
    & $Conda run -n $EnvName python -m pip install -e (Join-Path $TargetRoot "container")
    $XrdDevPath = (Join-Path $TargetRoot "XRD-preprocessing") + "[dev]"
    $AramisDevPath = (Join-Path $TargetRoot "Aramis") + "[dev]"
    & $Conda run -n $EnvName python -m pip install -e $XrdDevPath
    & $Conda run -n $EnvName python -m pip install --no-deps -e $AramisDevPath
    & $Conda run -n $EnvName python -c "import aramis, xrd_preprocessing; print('imports ok'); print('xrd_preprocessing', xrd_preprocessing.__file__); print('aramis', aramis.__file__)"
}

Write-Host "Ready: $TargetRoot"
Write-Host "Run tests: .\run_tests.ps1 -TargetRoot `"$TargetRoot`" -Mode all -EnvName $EnvName"
Write-Host "Run notebooks: .\run_aramis_notebooks.ps1 -TargetRoot `"$TargetRoot`" -EnvName $EnvName"
Write-Host "Run prediction examples: .\run_aramis_prediction_examples.ps1 -TargetRoot `"$TargetRoot`" -EnvName $EnvName"

if ($AutoRun) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:ENV_NAME='$EnvName'; & '$BundleDir\run_tests.ps1' -TargetRoot '$TargetRoot' -Mode all"
    & "$BundleDir\run_aramis_notebooks.ps1" -TargetRoot $TargetRoot -EnvName $EnvName
    exit 0
}

if (Ask-YesNo "Run XRD-preprocessing and Aramis tests now?" "n") {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:ENV_NAME='$EnvName'; & '$BundleDir\run_tests.ps1' -TargetRoot '$TargetRoot' -Mode all"
}
if (Ask-YesNo "Launch Aramis marimo notebooks now?" "n") {
    & "$BundleDir\run_aramis_notebooks.ps1" -TargetRoot $TargetRoot -EnvName $EnvName
}
if (Ask-YesNo "Run Aramis prediction examples now?" "n") {
    & "$BundleDir\run_aramis_prediction_examples.ps1" -TargetRoot $TargetRoot -EnvName $EnvName
}
