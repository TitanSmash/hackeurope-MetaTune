[CmdletBinding()]
param(
    [string]$Server = "https://api.hack-europe.fjws.p3.openshiftapps.com:443",

    [string]$Token = "sha256~Wsedgej25Xtx9aZnxfiBLOK6fSRSj4RgqgpZpn64Aus",

    [string]$Namespace = "hack-europe-team-i",

    [Parameter(Mandatory = $true)]
    [string]$SchedulerImage,

    [Parameter(Mandatory = $true)]
    [string]$TrainerImage,

    [string]$DatasetName = "tinystories",

    [Parameter(Mandatory = $true)]
    [string]$LocalTrainBin,

    [Parameter(Mandatory = $true)]
    [string]$LocalValBin,

    [ValidateSet("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")]
    [string]$Model = "gpt2",

    [int]$LoraR = 16,
    [int]$LoraAlpha = 32,
    [double]$LoraDropout = 0.05,
    [double]$LearningRate = 0.0003,
    [int]$BatchSize = 8,
    [int]$TokenBudget = 25000000,
    [int]$SeqLen = 1024,
    [int]$Seed = 1,
    [string]$JobName = "",

    [int]$PollIntervalSeconds = 10,
    [int]$PollTimeoutMinutes = 180,

    [switch]$SkipBuild,
    [switch]$SkipPush,
    [switch]$SkipDatasetUpload,
    [switch]$SkipSubmit,
    [switch]$SkipWait,
    [switch]$SkipLogin,
    [switch]$InsecureSkipTlsVerify
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

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

function Get-OcJsonPath {
    param(
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$JsonPath
    )
    $result = & $script:OcCmd get $Kind $Name -n $Namespace -o "jsonpath=$JsonPath"
    if (-not $result) {
        throw "Failed to read $Kind/$Name jsonpath $JsonPath"
    }
    return $result
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

function Enable-InsecureTlsIfRequested {
    if (-not $InsecureSkipTlsVerify) {
        return
    }
    $irm = Get-Command Invoke-RestMethod
    if (-not $irm.Parameters.ContainsKey("SkipCertificateCheck")) {
        [System.Net.ServicePointManager]::ServerCertificateValidationCallback = { $true }
    }
}

function Invoke-RouteRest {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("Get", "Post")][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [string]$BodyJson = ""
    )
    $params = @{
        Method = $Method
        Uri = $Uri
        ContentType = "application/json"
        TimeoutSec = 120
    }
    if ($BodyJson) {
        $params["Body"] = $BodyJson
    }
    $irm = Get-Command Invoke-RestMethod
    if ($InsecureSkipTlsVerify -and $irm.Parameters.ContainsKey("SkipCertificateCheck")) {
        $params["SkipCertificateCheck"] = $true
    }
    return Invoke-RestMethod @params
}

function Apply-ManifestTemplate {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][hashtable]$Replacements
    )
    $content = Get-Content -Raw -Path $Path
    foreach ($key in $Replacements.Keys) {
        $content = $content.Replace($key, [string]$Replacements[$key])
    }
    $tmpSpec = [System.IO.Path]::GetTempFileName() + ".yaml"
    try {
        Set-Content -Path $tmpSpec -Value $content -Encoding UTF8
        Invoke-Oc -Args @("apply", "-f", $tmpSpec) | Out-Host
    }
    finally {
        if (Test-Path $tmpSpec) {
            Remove-Item $tmpSpec -Force -ErrorAction SilentlyContinue
        }
    }
}

Assert-Command -Name "docker"
Enable-InsecureTlsIfRequested

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..\..")).Path
Set-Location $repoRoot
$script:OcCmd = Resolve-OcCommand -RepoRoot $repoRoot

$manifestsDir = Join-Path $repoRoot "simulations\openshift-pod\manifests"

if (-not $SkipDatasetUpload) {
    if (-not (Test-Path -Path $LocalTrainBin -PathType Leaf)) {
        throw "Local train.bin not found: $LocalTrainBin"
    }
    if (-not (Test-Path -Path $LocalValBin -PathType Leaf)) {
        throw "Local val.bin not found: $LocalValBin"
    }
}

if (-not $SkipLogin) {
    Write-Host "Logging in to OpenShift..."
    Invoke-Oc -Args @("login", "--server=$Server", "--token=$Token") | Out-Host
}
Invoke-Oc -Args @("project", $Namespace) | Out-Host

if (-not $SkipBuild) {
    Write-Host "Building scheduler image..."
    & docker build -f "simulations/openshift-pod/Dockerfile.scheduler" -t $SchedulerImage .
    Write-Host "Building trainer image..."
    & docker build -f "simulations/openshift-pod/Dockerfile.trainer" -t $TrainerImage .
}

if (-not $SkipPush) {
    Write-Host "Pushing scheduler image..."
    & docker push $SchedulerImage
    Write-Host "Pushing trainer image..."
    & docker push $TrainerImage
}

Write-Host "Applying RBAC and PVC manifests..."
Invoke-Oc -Args @("apply", "-f", (Join-Path $manifestsDir "scheduler-serviceaccount-role-rolebinding.yaml")) | Out-Host
Invoke-Oc -Args @("apply", "-f", (Join-Path $manifestsDir "shared-pvc.yaml")) | Out-Host

Write-Host "Applying config and deployment with image substitution..."
Apply-ManifestTemplate -Path (Join-Path $manifestsDir "scheduler-configmap.yaml") -Replacements @{
    '${TRAINER_IMAGE}' = $TrainerImage
}
Apply-ManifestTemplate -Path (Join-Path $manifestsDir "scheduler-deployment.yaml") -Replacements @{
    '${SCHEDULER_IMAGE}' = $SchedulerImage
}

Invoke-Oc -Args @("apply", "-f", (Join-Path $manifestsDir "scheduler-service.yaml")) | Out-Host
Invoke-Oc -Args @("apply", "-f", (Join-Path $manifestsDir "scheduler-route.yaml")) | Out-Host

Write-Host "Waiting for scheduler rollout..."
Invoke-Oc -Args @("rollout", "status", "deploy/metatune-scheduler", "-n", $Namespace) | Out-Host

$pod = (Invoke-Oc -Args @("get", "pod", "-n", $Namespace, "-l", "app=metatune-scheduler", "-o", "jsonpath={.items[0].metadata.name}") | Out-String).Trim()
if (-not $pod) {
    throw "Failed to locate scheduler pod"
}
Write-Host "Scheduler pod: $pod"

if (-not $SkipDatasetUpload) {
    Write-Host "Uploading dataset bins to shared PVC path /data/datasets/$DatasetName ..."
    Invoke-Oc -Args @("exec", "-n", $Namespace, $pod, "--", "mkdir", "-p", "/data/datasets/$DatasetName") | Out-Host
    Invoke-Oc -Args @("cp", $LocalTrainBin, "${Namespace}/${pod}:/data/datasets/$DatasetName/train.bin") | Out-Host
    Invoke-Oc -Args @("cp", $LocalValBin, "${Namespace}/${pod}:/data/datasets/$DatasetName/val.bin") | Out-Host
}

$routeHost = (Invoke-Oc -Args @("get", "route", "metatune-scheduler", "-n", $Namespace, "-o", "jsonpath={.spec.host}") | Out-String).Trim()
if (-not $routeHost) {
    throw "Failed to resolve route host for metatune-scheduler"
}
Write-Host "Scheduler route: https://$routeHost"

if ($SkipSubmit) {
    Write-Host "Skipping run submission as requested."
    return
}

$request = @{
    model = $Model
    dataset_rel_path = $DatasetName
    hp = @{
        lora_r = $LoraR
        lora_alpha = $LoraAlpha
        lora_dropout = $LoraDropout
        learning_rate = $LearningRate
        batch_size = $BatchSize
    }
    token_budget = $TokenBudget
    seq_len = $SeqLen
    seed = $Seed
}
if ($JobName -ne "") {
    $request["job_name"] = $JobName
}

$requestJson = $request | ConvertTo-Json -Depth 6
Write-Host "Submitting training run..."
$createResp = Invoke-RouteRest -Method Post -Uri "https://$routeHost/api/v1/runs" -BodyJson $requestJson
$runId = [string]$createResp.run_id
if (-not $runId) {
    throw "Run submission did not return run_id"
}
Write-Host "Run submitted: $runId"

if ($SkipWait) {
    Write-Host "Skipping status wait as requested."
    return
}

$deadline = (Get-Date).AddMinutes($PollTimeoutMinutes)
while ($true) {
    if ((Get-Date) -gt $deadline) {
        throw "Timed out waiting for run $runId"
    }
    $statusResp = Invoke-RouteRest -Method Get -Uri "https://$routeHost/api/v1/runs/$runId"
    $status = [string]$statusResp.status
    Write-Host ("[{0}] run {1} status={2}" -f (Get-Date -Format "s"), $runId, $status)
    if ($status -in @("SUCCEEDED", "FAILED")) {
        $statusResp | ConvertTo-Json -Depth 8
        break
    }
    Start-Sleep -Seconds $PollIntervalSeconds
}
