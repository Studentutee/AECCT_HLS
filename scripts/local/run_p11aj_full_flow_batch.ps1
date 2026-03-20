param(
    [string]$BuildDir = "build\p11aj"
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

    Invoke-Step -StepName "run_p11aj_top_managed_sram_provenance" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11aj_top_managed_sram_provenance.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11aj")
    )

    Invoke-Step -StepName "run_p11ah_full_loop_batch" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ah_full_loop_batch.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11ah_regression"),
        "-SkipCatapultProgress"
    )

    $summaryPath = Join-Path $BuildDir "batch_summary.txt"
    @(
        "status: PASS",
        "scope: local-only",
        "catapult_progress: not executed",
        "closure: not Catapult closure; not SCVerify closure",
        "PASS: run_p11aj_full_flow_batch"
    ) | Set-Content -Path $summaryPath -Encoding UTF8

    Write-Host "PASS: run_p11aj_full_flow_batch"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}

