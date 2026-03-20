param(
    [string]$BuildDir = "build\p11ah",
    [switch]$SkipCatapultProgress
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

    Invoke-Step -StepName "run_p11ah_full_loop_local_e2e" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ah_full_loop_local_e2e.ps1",
        "-BuildDir", (Join-Path $BuildDir "full_loop")
    )

    $p11agArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ag_attention_chain_batch.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11ag_regression")
    )
    if ($SkipCatapultProgress) {
        $p11agArgs += "-SkipCatapultProgress"
    }
    Invoke-Step -StepName "run_p11ag_attention_chain_batch" -Command "powershell" -StepArgs $p11agArgs

    $summaryPath = Join-Path $BuildDir "batch_summary.txt"
    $catapultStatus = "executed"
    if ($SkipCatapultProgress) {
        $catapultStatus = "not executed"
    }
    @(
        "status: PASS",
        "scope: local-only",
        ("catapult_progress: {0}" -f $catapultStatus),
        "closure: not Catapult closure; not SCVerify closure",
        "PASS: run_p11ah_full_loop_batch"
    ) | Set-Content -Path $summaryPath -Encoding UTF8

    Write-Host "PASS: run_p11ah_full_loop_batch"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
