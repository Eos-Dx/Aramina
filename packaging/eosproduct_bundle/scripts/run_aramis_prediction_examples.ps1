param(
    [string]$TargetRoot = "$HOME\dev\eosproduct",
    [string]$EnvName = "eosproduct"
)

$ErrorActionPreference = "Stop"
$AramisRoot = Join-Path $TargetRoot "Aramis"
Push-Location $AramisRoot
try {
    $Configs = @(
        "examples\prediction_h5\px01_predict.yaml",
        "examples\prediction_h5\px02_predict.yaml",
        "examples\prediction_h5\px03_predict.yaml",
        "examples\prediction_h5\cancer_predict.yaml",
        "examples\prediction_h5\atypical_predict.yaml",
        "examples\prediction_h5\benign_predict.yaml"
    )
    foreach ($Config in $Configs) {
        Write-Host "--- $Config"
        conda run --no-capture-output -n $EnvName python -m aramis predict --config $Config
    }
    Write-Host "Prediction reports: $(Join-Path $AramisRoot 'examples\outputs\prediction_h5_examples')"
}
finally {
    Pop-Location
}
