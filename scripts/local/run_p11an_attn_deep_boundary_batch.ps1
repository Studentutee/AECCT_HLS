param(
    [string]$BuildDir = "build\p11an"
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

    Invoke-Step -StepName "run_p11an_attn_deep_boundary" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11an_attn_deep_boundary.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11an_main")
    )

    Invoke-Step -StepName "run_p11al_catapult_top_wrapper" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11al_catapult_top_wrapper.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11al_regression")
    )

    Invoke-Step -StepName "run_p11ak_attnout_finalx_bridge" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ak_attnout_finalx_bridge.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11ak_regression")
    )

    Invoke-Step -StepName "run_p11ah_full_loop_local_e2e" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ah_full_loop_local_e2e.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11ah_regression")
    )

    $summaryPath = Join-Path $BuildDir "batch_summary.txt"
    @(
        "status: PASS",
        "scope: local-only",
        "posture: Catapult-facing progress only",
        "closure: not Catapult closure; not SCVerify closure",
        "PASS: run_p11an_attn_deep_boundary_batch"
    ) | Set-Content -Path $summaryPath -Encoding UTF8

    Write-Host "PASS: run_p11an_attn_deep_boundary_batch"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
