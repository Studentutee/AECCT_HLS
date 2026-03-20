param(
    [string]$BuildDir = "build\p11ag",
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
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "step failed ($StepName), exit=$exitCode"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    Invoke-Step -StepName "run_p11ac_phasea_top_managed" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ac_phasea_top_managed.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11ac")
    )
    Invoke-Step -StepName "run_p11ad_impl_q_path" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ad_impl_q_path.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11ad")
    )
    Invoke-Step -StepName "run_p11ae_impl_qk_score" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ae_impl_qk_score.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11ae")
    )
    Invoke-Step -StepName "run_p11af_impl_softmax_out" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11af_impl_softmax_out.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11af")
    )
    Invoke-Step -StepName "run_p11aeaf_e2e_smoke" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11aeaf_e2e_smoke.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11aeaf_e2e")
    )
    Invoke-Step -StepName "run_p11ag_attention_chain_correction" -Command "powershell" -StepArgs @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11ag_attention_chain_correction.ps1",
        "-BuildDir", (Join-Path $BuildDir "p11ag_validator")
    )

    if (-not $SkipCatapultProgress) {
        Invoke-Step -StepName "run_p11aeaf_catapult_progress" -Command "powershell" -StepArgs @(
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/local/run_p11aeaf_catapult_progress.ps1",
            "-BuildDir", (Join-Path $BuildDir "catapult_progress")
        )
    }

    $summaryPath = Join-Path $BuildDir "batch_summary.txt"
    $catapultStatus = "not executed"
    if (-not $SkipCatapultProgress) {
        $catapultStatus = "executed"
    }
    @(
        "status: PASS",
        "scope: local-only",
        ("catapult_progress: {0}" -f $catapultStatus),
        "closure: not Catapult closure; not SCVerify closure",
        "PASS: run_p11ag_attention_chain_batch"
    ) | Set-Content -Path $summaryPath -Encoding UTF8

    Write-Host "PASS: run_p11ag_attention_chain_batch"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
