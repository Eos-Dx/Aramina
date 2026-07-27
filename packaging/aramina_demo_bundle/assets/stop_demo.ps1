$ErrorActionPreference = "Stop"
docker rm --force aramina-demo aramina-demo-api *> $null
Write-Host "Aramina browser demonstrator and local API stopped."
