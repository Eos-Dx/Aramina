param(
    [string]$EnvName = "eosproduct",
    [switch]$SkipExample
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path

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
    $Answer = Read-Host "conda not found. Install Miniforge to ~/miniforge3? [Y/n]"
    if ([string]::IsNullOrWhiteSpace($Answer) -or $Answer.ToLower().StartsWith("y")) {
        return Install-Miniforge
    }
    throw "conda is required. Install Miniforge/Miniconda and rerun install.bat."
}

$Conda = Ensure-Conda
$EnvironmentYml = Join-Path $RepoDir "environment.yml"
$EnvList = & $Conda env list

if ($EnvList -match "(?m)^$EnvName\s") {
    & $Conda env update -n $EnvName -f $EnvironmentYml
} else {
    & $Conda env create -n $EnvName -f $EnvironmentYml
}

$AraminaDevPath = "${RepoDir}[dev]"
& $Conda run --no-capture-output -n $EnvName python -m pip install -e $AraminaDevPath
& $Conda run --no-capture-output -n $EnvName python -c "import aramina, xrd_preprocessing; print('imports ok'); print('aramina', aramina.__file__); print('xrd_preprocessing', xrd_preprocessing.__file__)"

if (-not $SkipExample) {
    $PredictYaml = Join-Path $RepoDir "examples\prediction\configs\config_predict_cancer_example.yaml"
    & $Conda run --no-capture-output -n $EnvName python -m aramina predict --config $PredictYaml
}

Write-Host ""
Write-Host "Ready."
Write-Host ""
Write-Host "Activate:"
Write-Host "  conda activate $EnvName"
Write-Host ""
Write-Host "Run prediction example:"
Write-Host "  cd `"$RepoDir`""
Write-Host "  python -m aramina predict --config examples\prediction\configs\config_predict_cancer_example.yaml"
