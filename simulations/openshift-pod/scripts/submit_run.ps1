[CmdletBinding()]
param(
    [string]$Server = "https://api.hack-europe.fjws.p3.openshiftapps.com:443",
    [string]$Token = "sha256~Wsedgej25Xtx9aZnxfiBLOK6fSRSj4RgqgpZpn64Aus",
    [string]$Namespace = "hack-europe-team-i",
    [string]$RouteHost = "",
    [string]$RunId = "",

    [ValidateSet("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")]
    [string]$Model = "gpt2",
    [string]$DatasetName = "tinystories",
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
    [switch]$SkipLogin,
    [switch]$SkipSubmit,
    [switch]$SkipWait,
    [switch]$InsecureSkipTlsVerify
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot
$OcCmd = Resolve-OcCommand -RepoRoot $repoRoot
Enable-InsecureTlsIfRequested

if (-not $SkipLogin) {
    Write-Host "Logging in to OpenShift..."
    Invoke-Oc -Args @("login", "--server=$Server", "--token=$Token") | Out-Host
}
Invoke-Oc -Args @("project", $Namespace) | Out-Host

if (-not $RouteHost) {
    $RouteHost = (Invoke-Oc -Args @("get", "route", "metatune-scheduler", "-n", $Namespace, "-o", "jsonpath={.spec.host}") | Out-String).Trim()
}
if (-not $RouteHost) {
    throw "Route host is empty. Pass -RouteHost or ensure route metatune-scheduler exists."
}
Write-Host "Using scheduler route: https://$RouteHost"

if (-not $SkipSubmit) {
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
    $createResp = Invoke-RouteRest -Method Post -Uri "https://$RouteHost/api/v1/runs" -BodyJson $requestJson
    $RunId = [string]$createResp.run_id
    if (-not $RunId) {
        throw "Run submission did not return run_id"
    }
    Write-Host "Run submitted: $RunId"
} elseif (-not $RunId) {
    throw "When -SkipSubmit is used you must pass -RunId"
}

if ($SkipWait) {
    return
}

$deadline = (Get-Date).AddMinutes($PollTimeoutMinutes)
while ($true) {
    if ((Get-Date) -gt $deadline) {
        throw "Timed out waiting for run $RunId"
    }
    $statusResp = Invoke-RouteRest -Method Get -Uri "https://$RouteHost/api/v1/runs/$RunId"
    $status = [string]$statusResp.status
    Write-Host ("[{0}] run {1} status={2}" -f (Get-Date -Format "s"), $RunId, $status)
    if ($status -in @("SUCCEEDED", "FAILED")) {
        $statusResp | ConvertTo-Json -Depth 8
        break
    }
    Start-Sleep -Seconds $PollIntervalSeconds
}
