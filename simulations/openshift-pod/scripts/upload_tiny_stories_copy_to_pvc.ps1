[CmdletBinding()]
param(
    [string]$Server = "https://api.hack-europe.fjws.p3.openshiftapps.com:443",
    [string]$Token = "sha256~Wsedgej25Xtx9aZnxfiBLOK6fSRSj4RgqgpZpn64Aus",
    [string]$Namespace = "hack-europe-team-i",
    [string]$SchedulerPod = "",
    [string]$SchedulerLabel = "app=metatune-scheduler",
    [string]$SharedPvcName = "train-cache-pvc",
    [string]$HelperPodName = "metatune-pvc-uploader",
    [bool]$CreateHelperPodIfMissing = $true,
    [int]$HelperPodWaitSeconds = 600,
    [switch]$KeepHelperPod,
    [string]$SourceFolder = "tiny_stories_copy",
    [string]$DatasetName = "tinystories",
    [string]$DatasetsRoot = "/data/datasets",
    [switch]$SkipLogin,
    [switch]$ClearDestination,
    [switch]$DisableRsync,
    [switch]$RequireTrainingBins
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
            -FilePath $OcCmd `
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
$OcCmd = Resolve-OcCommand -RepoRoot $repoRoot

if (-not $SkipLogin) {
    Write-Host "Logging in to OpenShift..."
    Invoke-Oc -Args @("login", "--server=$Server", "--token=$Token") | Out-Host
}
Invoke-Oc -Args @("project", $Namespace) | Out-Host

$sourcePath = Resolve-Path $SourceFolder -ErrorAction Stop
if (-not (Test-Path -Path $sourcePath -PathType Container)) {
    throw "Source folder not found: $SourceFolder"
}

if (-not $SchedulerPod) {
    $podNamesRaw = & $OcCmd get pod -n $Namespace -l $SchedulerLabel --field-selector=status.phase=Running -o name 2>$null
    if ($podNamesRaw) {
        $podNames = ($podNamesRaw -split "`r?`n" | Where-Object { $_ -ne "" })
        if ($podNames.Count -gt 0) {
            $SchedulerPod = ($podNames[0] -replace "^pod/", "")
        }
    }
}

$usingHelperPod = $false
if (-not $SchedulerPod -and $CreateHelperPodIfMissing) {
    Write-Host "No running scheduler pod found. Creating helper pod '$HelperPodName' with PVC '$SharedPvcName'..."
    Invoke-Oc -Args @("delete", "pod", $HelperPodName, "-n", $Namespace, "--ignore-not-found=true") | Out-Host

    $helperPodYaml = @"
apiVersion: v1
kind: Pod
metadata:
  name: $HelperPodName
  namespace: $Namespace
  labels:
    app: metatune-pvc-uploader
spec:
  restartPolicy: Never
  containers:
    - name: uploader
      image: registry.access.redhat.com/ubi9/ubi-minimal:latest
      command: ["sleep", "36000"]
      volumeMounts:
        - name: shared-data
          mountPath: /data
  volumes:
    - name: shared-data
      persistentVolumeClaim:
        claimName: $SharedPvcName
"@

    $helperPodYaml | & $OcCmd apply -f - | Out-Host

    $waitOut = Invoke-Oc -Args @(
        "wait",
        "--for=condition=Ready",
        "pod/$HelperPodName",
        "-n",
        $Namespace,
        "--timeout=${HelperPodWaitSeconds}s"
    ) -IgnoreExitCode
    if ($LASTEXITCODE -ne 0) {
        $phase = (Invoke-Oc -Args @("get", "pod", $HelperPodName, "-n", $Namespace, "-o", "jsonpath={.status.phase}") -IgnoreExitCode | Out-String).Trim()
        $podDescribe = Invoke-Oc -Args @("describe", "pod", $HelperPodName, "-n", $Namespace) -IgnoreExitCode
        $pvcDescribe = Invoke-Oc -Args @("describe", "pvc", $SharedPvcName, "-n", $Namespace) -IgnoreExitCode
        $waitMsg = ($waitOut | Out-String).Trim()
        $podMsg = ($podDescribe | Out-String).Trim()
        $pvcMsg = ($pvcDescribe | Out-String).Trim()
        throw @"
Helper pod did not become Ready within ${HelperPodWaitSeconds}s.
Phase: $phase
Wait output:
$waitMsg

Pod describe:
$podMsg

PVC describe:
$pvcMsg
"@
    }
    $waitOut | Out-Host
    $SchedulerPod = $HelperPodName
    $usingHelperPod = $true
}

if (-not $SchedulerPod) {
    throw "Could not resolve scheduler pod. Pass -SchedulerPod, deploy scheduler, or set -CreateHelperPodIfMissing `$true."
}

$destDir = "$DatasetsRoot/$DatasetName"
Write-Host "Using scheduler pod: $SchedulerPod"
Write-Host "Copy source: $sourcePath"
Write-Host "PVC target: $destDir"

try {
    Invoke-Oc -Args @("exec", "-n", $Namespace, $SchedulerPod, "--", "mkdir", "-p", $destDir) | Out-Host
    if ($ClearDestination) {
        Write-Host "Clearing destination directory..."
        Invoke-Oc -Args @("exec", "-n", $Namespace, $SchedulerPod, "--", "sh", "-c", "rm -rf '$destDir'/*") | Out-Host
    }

    $files = Get-ChildItem -Path $sourcePath -File
    if (-not $files) {
        throw "No files found in source folder: $sourcePath"
    }

    $didRsync = $false
    if (-not $DisableRsync) {
        Push-Location $sourcePath
        try {
            Write-Host "Syncing source folder with oc rsync (strategy=tar)..."
            $rsyncOut = Invoke-Oc -Args @(
                "rsync",
                ".",
                "${Namespace}/${SchedulerPod}:$destDir",
                "--strategy=tar"
            ) -IgnoreExitCode
            if ($LASTEXITCODE -eq 0) {
                $didRsync = $true
                $rsyncOut | Out-Host
            } else {
                Write-Warning "oc rsync failed; falling back to per-file oc cp."
                ($rsyncOut | Out-String).Trim() | Out-Host
            }
        }
        finally {
            Pop-Location
        }
    }

    if (-not $didRsync) {
        Push-Location $sourcePath
        try {
            foreach ($file in $files) {
                $remotePath = "${Namespace}/${SchedulerPod}:$destDir/$($file.Name)"
                # Use relative path on Windows to avoid drive-letter ':' being parsed as remote spec.
                $localPath = ".\$($file.Name)"
                Write-Host "Copying $($file.Name)..."
                Invoke-Oc -Args @("cp", $localPath, $remotePath) | Out-Host
            }
        }
        finally {
            Pop-Location
        }
    }

    Write-Host "Verifying copied files..."
    Invoke-Oc -Args @("exec", "-n", $Namespace, $SchedulerPod, "--", "sh", "-c", "ls -lah '$destDir'") | Out-Host
}
finally {
    if ($usingHelperPod -and -not $KeepHelperPod) {
        Write-Host "Deleting helper pod '$HelperPodName'..."
        Invoke-Oc -Args @("delete", "pod", $HelperPodName, "-n", $Namespace, "--ignore-not-found=true") -IgnoreExitCode | Out-Host
    }
}

$localTrainBin = Join-Path $sourcePath "train.bin"
$localValBin = Join-Path $sourcePath "val.bin"
$hasLocalBins = (Test-Path $localTrainBin -PathType Leaf) -and (Test-Path $localValBin -PathType Leaf)

if ($RequireTrainingBins -and -not $hasLocalBins) {
    throw "train.bin/val.bin not found in source folder, but -RequireTrainingBins was set."
}

if (-not $hasLocalBins) {
    Write-Warning "Source folder does not contain train.bin and val.bin."
    Write-Warning "Current trainer expects /data/datasets/<dataset>/train.bin and val.bin."
    Write-Warning "Copied files are available, but run submission will fail until bin files are present."
} else {
    Write-Host "train.bin and val.bin detected in source folder."
}

Write-Host "Upload complete."
