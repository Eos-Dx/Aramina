param(
    [string]$WorkflowConfig = "config/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml"
)

$ErrorActionPreference = "Stop"
$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Get-Content (Join-Path $BundleDir "bundle_manifest.json") -Raw | ConvertFrom-Json
$DataDir = Join-Path $BundleDir "data"
$OutputDir = Join-Path $BundleDir "outputs"
$LogDir = Join-Path $OutputDir "logs"
$ConfigDir = Join-Path $BundleDir "config"
$ImageTag = $Manifest.image_amd64_tag
$ImagePlatform = $Manifest.image_amd64_platform
$ImageArchive = Join-Path $BundleDir $Manifest.image_amd64_archive
$DockerDesktopInstallerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("install_and_train_{0}.log" -f (Get-Date -Format "yyyyMMddTHHmmss"))
Start-Transcript -Path $LogPath | Out-Null

function Write-Stage {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ==="
}

function Invoke-Docker {
    param([string]$Description, [string[]]$Arguments)
    Write-Stage $Description
    & $script:DockerExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Find-DockerExe {
    $command = Get-Command docker.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"),
        (Join-Path $env:ProgramFiles "Docker\Docker\resources\bin\docker.exe")
    )
    return $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

function Enable-Wsl2IfNeeded {
    & wsl.exe --version *> $null
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Stage "Enable WSL 2 required by Docker Desktop"
    $process = Start-Process `
        -FilePath "wsl.exe" `
        -ArgumentList @("--install", "--no-distribution") `
        -Verb RunAs `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "WSL 2 installation failed with exit code $($process.ExitCode). Enable WSL 2 and rerun this script."
    }
    throw "WSL 2 was installed. Restart Windows, then rerun install_and_train.bat."
}

function Ensure-DockerDesktop {
    $dockerExe = Find-DockerExe
    if (-not $dockerExe) {
        Enable-Wsl2IfNeeded
        Write-Stage "Download Docker Desktop"
        $installer = Join-Path $env:TEMP "Docker Desktop Installer.exe"
        Invoke-WebRequest -Uri $DockerDesktopInstallerUrl -OutFile $installer

        try {
            Write-Stage "Install Docker Desktop with the WSL 2 backend"
            $process = Start-Process `
                -FilePath $installer `
                -ArgumentList @("install", "--quiet", "--accept-license", "--backend=wsl-2", "--user") `
                -Wait `
                -PassThru
            if ($process.ExitCode -ne 0) {
                throw "Docker Desktop installation failed with exit code $($process.ExitCode)."
            }
        } finally {
            Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
        }

        $dockerExe = Find-DockerExe
        if (-not $dockerExe) {
            throw "Docker Desktop was installed but docker.exe was not found. Restart Windows, then rerun this script."
        }
    }

    $desktopCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe"),
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe")
    )
    $desktopExe = $desktopCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($desktopExe) {
        Start-Process -FilePath $desktopExe
    }

    Write-Stage "Wait for Docker Linux engine"
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        $process = Start-Process `
            -FilePath $dockerExe `
            -ArgumentList @("version") `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        if ($process.ExitCode -eq 0) {
            return $dockerExe
        }
        Start-Sleep -Seconds 5
    }
    throw "Docker Desktop is installed but its Linux engine did not start within five minutes. Open Docker Desktop, resolve its shown error, then rerun this script."
}

function Resolve-WorkflowConfig {
    param([string]$Value)
    $configRoot = (Resolve-Path -LiteralPath $ConfigDir).Path
    $candidate = Join-Path $BundleDir $Value
    $workflowPath = (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
    $configRootWithSeparator = $configRoot.TrimEnd([char[]]@('\', '/')) + '\'
    if (-not $workflowPath.StartsWith($configRootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Workflow config must be inside bundled config/: $Value"
    }
    $relative = $workflowPath.Substring($configRootWithSeparator.Length).Replace('\', '/')
    if (-not $relative.StartsWith("workflows/")) {
        throw "Workflow config must be inside bundled config/workflows/: $Value"
    }
    return "/opt/aramis-bundle-config/$relative"
}

try {
    Write-Stage "Aramis Docker reproducible training bundle"
    Write-Host "Log: $LogPath"

    $script:DockerExe = Ensure-DockerDesktop
    & $script:DockerExe version | Out-Host
    $ResolvedWorkflowConfig = Resolve-WorkflowConfig $WorkflowConfig
    Write-Host "Workflow config: $ResolvedWorkflowConfig"

    $H5Path = Join-Path $DataDir "combined_archive.h5"
    if (-not (Test-Path $H5Path)) {
        throw "Missing bundled H5 input: $H5Path"
    }
    $ActualSha256 = (Get-FileHash -LiteralPath $H5Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $Manifest.h5_sha256.ToLowerInvariant()) {
        throw "Bundled H5 SHA256 mismatch. Expected $($Manifest.h5_sha256), got $ActualSha256. Extract a fresh bundle."
    }
    Write-Host "Bundled H5 SHA256 verified: $ActualSha256"

    $imageInspect = Start-Process `
        -FilePath $script:DockerExe `
        -ArgumentList @("image", "inspect", $ImageTag) `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($imageInspect.ExitCode -ne 0) {
        if (-not (Test-Path $ImageArchive)) {
            throw "Missing Docker image archive: $ImageArchive"
        }
        $ArchiveSha256 = (Get-FileHash -LiteralPath $ImageArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ArchiveSha256 -ne $Manifest.image_amd64_archive_sha256.ToLowerInvariant()) {
            throw "Docker image SHA256 mismatch. Expected $($Manifest.image_amd64_archive_sha256), got $ArchiveSha256. Extract a fresh bundle."
        }
        Invoke-Docker "Load validated Linux runtime image" @("load", "--input", $ImageArchive)
    } else {
        Write-Stage "Reuse loaded Linux runtime image"
        Write-Host $ImageTag
    }

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Invoke-Docker "Run Linux preprocessing and training" @(
        "run", "--rm", "--platform", $ImagePlatform,
        "--mount", "type=bind,src=$DataDir,dst=/opt/data,readonly",
        "--mount", "type=bind,src=$ConfigDir,dst=/opt/aramis-bundle-config,readonly",
        "--mount", "type=bind,src=$OutputDir,dst=/opt/Aramis/examples/outputs",
        $ImageTag,
        "bash", "/opt/aramis-bundle/run_training_docker.sh", "--workflow-config", $ResolvedWorkflowConfig
    )

    Write-Stage "Bundle completed"
    Write-Host "Outputs: $OutputDir"
    Write-Host "Log saved to: $LogPath"
} finally {
    Stop-Transcript | Out-Null
}
