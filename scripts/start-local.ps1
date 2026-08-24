param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$localUrl = "http://localhost:3000/kernel-lab?op=muladd"

function Test-VisualLab {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200 -and $response.Content -match "MoE Mul"
    }
    catch {
        return $false
    }
}

if (Test-VisualLab -Url $localUrl) {
    Write-Host "Triton Ascend Visual Lab is already running locally." -ForegroundColor Green
    if (-not $NoBrowser) {
        Start-Process $localUrl
    }
    exit 0
}

$npmCommand = Get-Command npm.cmd -ErrorAction Stop
$nodeCommand = Get-Command node.exe -ErrorAction Stop
$npmCli = Join-Path (Split-Path -Parent $npmCommand.Source) "node_modules\npm\bin\npm-cli.js"
if (-not (Test-Path -LiteralPath $npmCli)) {
    throw "Cannot find npm-cli.js next to npm.cmd. Reinstall Node.js/npm and try again."
}
$encodedUrl = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($localUrl))
$browserOpener = @"
`$url = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('$encodedUrl'))
for (`$attempt = 0; `$attempt -lt 90; `$attempt++) {
    try {
        `$response = Invoke-WebRequest -Uri `$url -UseBasicParsing -TimeoutSec 2
        if (`$response.StatusCode -eq 200 -and `$response.Content -match 'MoE Mul') {
            Start-Process `$url
            exit 0
        }
    }
    catch {
    }
    Start-Sleep -Seconds 1
}
"@
$encodedOpener = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($browserOpener))

if (-not $NoBrowser) {
    Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -ArgumentList @(
        "-NoLogo",
        "-NoProfile",
        "-EncodedCommand",
        $encodedOpener
    )
}

Set-Location -LiteralPath $projectRoot
Write-Host "Starting Triton Ascend Visual Lab..." -ForegroundColor Cyan
Write-Host "URL: $localUrl" -ForegroundColor DarkGray
Write-Host "Press Ctrl+C in this window to stop the server." -ForegroundColor Yellow
Write-Host ""

& $nodeCommand.Source $npmCli run dev -- --host 127.0.0.1 --port 3000
