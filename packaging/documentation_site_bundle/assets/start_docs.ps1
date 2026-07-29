$ErrorActionPreference = "Stop"
$SitePath = Join-Path $PSScriptRoot "site\index.html"
if (-not (Test-Path -LiteralPath $SitePath)) {
    throw "Missing documentation site: $SitePath"
}
Start-Process $SitePath
