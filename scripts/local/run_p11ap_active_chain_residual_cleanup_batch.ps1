param(
    [string]$BuildDir = "build\p11ap"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$StepName,
        [string]$Command,
        [string[]]$StepArgs
    )

    & $Command @StepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "step failed ($StepName), exit=$LASTEXITCODE"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    Invoke-Step -StepName "run_p11ap_active_chain_residual_cleanup" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ap_active_chain_residual_cleanup.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11ap_main")
    )

    $summaryPath = Join-Path $BuildDir "batch_summary.txt"
    @(
        "status: PASS",
        "scope: local-only",
        "posture: Catapult-facing progress only",
        "closure: not Catapult closure; not SCVerify closure",
        "PASS: run_p11ap_active_chain_residual_cleanup_batch"
    ) | Set-Content -Path $summaryPath -Encoding UTF8

    Write-Host "PASS: run_p11ap_active_chain_residual_cleanup_batch"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
