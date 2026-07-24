$ErrorActionPreference = "Stop"
docker rm --force aramis-demo aramis-demo-api *> $null
Write-Host "Aramis browser demonstrator and local API stopped."
