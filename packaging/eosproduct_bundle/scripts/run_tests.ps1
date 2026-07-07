param(
    [string]$TargetRoot = "$HOME\dev\eosproduct",
    [string]$Mode = "all",
    [string]$EnvName = $env:ENV_NAME
)

if (-not $EnvName) {
    $EnvName = "eosproduct"
}

$ErrorActionPreference = "Stop"

function Run-Step($Name, $ScriptBlock) {
    Write-Host ""
    Write-Host $Name
    & $ScriptBlock
}

if ($Mode -eq "xrd" -or $Mode -eq "all") {
    Run-Step "Testing XRD-preprocessing" {
        Push-Location "$TargetRoot\XRD-preprocessing"
        conda run -n $EnvName python -m ruff check .
        conda run -n $EnvName pytest -q
        Pop-Location
    }
}

if ($Mode -eq "aramis" -or $Mode -eq "all") {
    Run-Step "Testing Aramis" {
        Push-Location "$TargetRoot\Aramis"
        conda run -n $EnvName python -m ruff check .
        conda run -n $EnvName pytest -q
        conda run -n $EnvName python -m marimo check examples/aramis_dataframe_all_patients_v0_1.py
        conda run -n $EnvName python -m marimo check examples/aramis_dataframe_biopsy_patients_v0_1.py
        Pop-Location
    }
}
