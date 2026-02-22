[CmdletBinding()]
param(
    [string]$Server = "https://api.hack-europe.fjws.p3.openshiftapps.com:443",
    [string]$Token = "sha256~Wsedgej25Xtx9aZnxfiBLOK6fSRSj4RgqgpZpn64Aus",
    [string]$Namespace = "hack-europe-team-i",
    [string]$SchedulerBuildConfigName = "metatune-scheduler",
    [string]$TrainerBuildConfigName = "metatune-trainer",
    [switch]$SkipLogin,
    [switch]$DeployAfterBuild,
    [switch]$KeepBuildContext
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-OcCommand {
    param([Parameter(Mandatory = $true)][string]$RepoRoot)
    $globalOc = Get-Command "oc" -ErrorAction SilentlyContinue
    if ($globalOc) {
        return $globalOc.Source
    }
    $fallback = Join-Path $RepoRoot "simulations\oc.exe"
    if (Test-Path -Path $fallback -PathType Leaf) {
        return (Resolve-Path $fallback).Path
    }
    throw "Required OpenShift CLI not found. Install 'oc' in PATH or place oc.exe at $fallback"
}

function Invoke-Oc {
    param(
        [Parameter(Mandatory = $true)][string[]]$Args,
        [switch]$IgnoreExitCode
    )
    $tmpOut = [System.IO.Path]::GetTempFileName()
    $tmpErr = [System.IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process `
            -FilePath $script:OcCmd `
            -ArgumentList $Args `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $tmpOut `
            -RedirectStandardError $tmpErr

        $stdout = Get-Content -Path $tmpOut -Raw -ErrorAction SilentlyContinue
        $stderr = Get-Content -Path $tmpErr -Raw -ErrorAction SilentlyContinue
        $outputLines = @()
        if ($stdout) {
            $outputLines += ($stdout -split "`r?`n" | Where-Object { $_ -ne "" })
        }
        if ($stderr) {
            $outputLines += ($stderr -split "`r?`n" | Where-Object { $_ -ne "" })
        }

        if (-not $IgnoreExitCode -and $proc.ExitCode -ne 0) {
            $cmd = "oc " + ($Args -join " ")
            $msg = ($outputLines -join [Environment]::NewLine)
            throw "Command failed (exit=$($proc.ExitCode)): $cmd`n$msg"
        }
        return $outputLines
    }
    finally {
        Remove-Item -Path $tmpOut -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $tmpErr -Force -ErrorAction SilentlyContinue
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot
$script:OcCmd = Resolve-OcCommand -RepoRoot $repoRoot

if (-not $SkipLogin) {
    Write-Host "Logging in to OpenShift..."
    Invoke-Oc -Args @("login", "--server=$Server", "--token=$Token") | Out-Host
}
Invoke-Oc -Args @("project", $Namespace) | Out-Host

Write-Host "Applying ImageStreams + BuildConfigs..."
$buildYaml = @"
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  name: $SchedulerBuildConfigName
  namespace: $Namespace
---
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: $SchedulerBuildConfigName
  namespace: $Namespace
spec:
  runPolicy: Serial
  source:
    type: Binary
  strategy:
    type: Docker
    dockerStrategy:
      dockerfilePath: simulations/openshift-pod/Dockerfile.scheduler
  output:
    to:
      kind: ImageStreamTag
      name: "${SchedulerBuildConfigName}:latest"
---
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  name: $TrainerBuildConfigName
  namespace: $Namespace
---
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: $TrainerBuildConfigName
  namespace: $Namespace
spec:
  runPolicy: Serial
  source:
    type: Binary
  strategy:
    type: Docker
    dockerStrategy:
      dockerfilePath: simulations/openshift-pod/Dockerfile.trainer
  output:
    to:
      kind: ImageStreamTag
      name: "${TrainerBuildConfigName}:latest"
"@

$tmpBuildSpec = [System.IO.Path]::GetTempFileName() + ".yaml"
try {
    Set-Content -Path $tmpBuildSpec -Value $buildYaml -Encoding UTF8
    Invoke-Oc -Args @("apply", "-f", $tmpBuildSpec) | Out-Host
}
finally {
    if (Test-Path $tmpBuildSpec) {
        Remove-Item $tmpBuildSpec -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Verifying BuildConfigs exist..."
Invoke-Oc -Args @("get", "bc", $SchedulerBuildConfigName, "-n", $Namespace) | Out-Host
Invoke-Oc -Args @("get", "bc", $TrainerBuildConfigName, "-n", $Namespace) | Out-Host

$tmpContextRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("metatune-oc-build-" + [Guid]::NewGuid().ToString("N"))
Write-Host "Preparing slim build context: $tmpContextRoot"
New-Item -ItemType Directory -Path (Join-Path $tmpContextRoot "simulations") -Force | Out-Null
Copy-Item -Path (Join-Path $repoRoot "simulations\openshift-pod") -Destination (Join-Path $tmpContextRoot "simulations\openshift-pod") -Recurse -Force
Copy-Item -Path (Join-Path $repoRoot "simulations\nanoGPT-LoRA-master") -Destination (Join-Path $tmpContextRoot "simulations\nanoGPT-LoRA-master") -Recurse -Force

Write-Host "Starting scheduler image build in OpenShift..."
try {
    Invoke-Oc -Args @(
        "start-build",
        $SchedulerBuildConfigName,
        "--from-dir=$tmpContextRoot",
        "--follow",
        "--wait"
    ) | Out-Host

    Write-Host "Starting trainer image build in OpenShift..."
    Invoke-Oc -Args @(
        "start-build",
        $TrainerBuildConfigName,
        "--from-dir=$tmpContextRoot",
        "--follow",
        "--wait"
    ) | Out-Host
}
finally {
    if (-not $KeepBuildContext -and (Test-Path $tmpContextRoot)) {
        Remove-Item -Path $tmpContextRoot -Recurse -Force -ErrorAction SilentlyContinue
    } elseif (Test-Path $tmpContextRoot) {
        Write-Host "Keeping build context at: $tmpContextRoot"
    }
}

$schedulerImage = "image-registry.openshift-image-registry.svc:5000/$Namespace/$SchedulerBuildConfigName:latest"
$trainerImage = "image-registry.openshift-image-registry.svc:5000/$Namespace/$TrainerBuildConfigName:latest"

Write-Host ""
Write-Host "Build complete."
Write-Host "Scheduler image: $schedulerImage"
Write-Host "Trainer image:   $trainerImage"
Write-Host ""
Write-Host "Deploy command:"
Write-Host ".\simulations\openshift-pod\scripts\deploy_and_run.ps1 -SchedulerImage `"$schedulerImage`" -TrainerImage `"$trainerImage`" -SkipBuild -SkipPush -SkipDatasetUpload -SkipSubmit -SkipWait"

if ($DeployAfterBuild) {
    Write-Host "Deploying scheduler with built images..."
    & "$repoRoot\simulations\openshift-pod\scripts\deploy_and_run.ps1" `
      -SchedulerImage $schedulerImage `
      -TrainerImage $trainerImage `
      -SkipBuild `
      -SkipPush `
      -SkipDatasetUpload `
      -SkipSubmit `
      -SkipWait `
      -SkipLogin
}
