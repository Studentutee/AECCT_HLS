param(
    [string]$BuildDir = "build\\target_layer_attention_multilayer_acceptance"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-File {
    param([string]$Path)
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "required file missing: $Path"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$verdictPath = $null
$manifestPath = $null
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $runnerLog = Join-Path $BuildDir "runner.log"
    $nestedBuild = Join-Path $BuildDir "p11aj"
    $mainlineCheckerOutDir = Join-Path $BuildDir "checker_mainline"
    $multilayerCheckerOutDir = Join-Path $BuildDir "checker_multilayer"
    $targetLayerCheckerOutDir = Join-Path $BuildDir "checker_target_layer"
    $verdictPath = Join-Path $BuildDir "verdict.txt"
    $manifestPath = Join-Path $BuildDir "file_manifest.txt"

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir $nestedBuild *> $runnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "run_p11aj_top_managed_sram_provenance failed, exit=$LASTEXITCODE"
    }

    $runLog = Join-Path $nestedBuild "run.log"
    Require-File -Path $runLog

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_lid0_attention_mainline_acceptance.ps1 -RepoRoot $repoRoot -RunLog $runLog -OutDir $mainlineCheckerOutDir
    if ($LASTEXITCODE -ne 0) {
        throw "check_lid0_attention_mainline_acceptance failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_lid0_attention_multilayer_acceptance.ps1 -RepoRoot $repoRoot -RunLog $runLog -OutDir $multilayerCheckerOutDir
    if ($LASTEXITCODE -ne 0) {
        throw "check_lid0_attention_multilayer_acceptance failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_target_layer_attention_multilayer_acceptance.ps1 -RepoRoot $repoRoot -RunLog $runLog -OutDir $targetLayerCheckerOutDir
    if ($LASTEXITCODE -ne 0) {
        throw "check_target_layer_attention_multilayer_acceptance failed, exit=$LASTEXITCODE"
    }

    $mainlineSummary = Join-Path $mainlineCheckerOutDir "check_lid0_attention_mainline_acceptance_summary.txt"
    $multilayerSummary = Join-Path $multilayerCheckerOutDir "check_lid0_attention_multilayer_acceptance_summary.txt"
    $targetLayerSummary = Join-Path $targetLayerCheckerOutDir "check_target_layer_attention_multilayer_acceptance_summary.txt"
    Require-File -Path $mainlineSummary
    Require-File -Path $multilayerSummary
    Require-File -Path $targetLayerSummary

    @(
        "task: target_layer_attention_multilayer_acceptance",
        "status: PASS",
        "banner: PASS: run_target_layer_attention_multilayer_acceptance",
        "scope: local-only",
        "closure: not Catapult closure; not SCVerify closure"
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("runner_log={0}" -f $runnerLog),
        ("run_log={0}" -f $runLog),
        ("mainline_checker_summary={0}" -f $mainlineSummary),
        ("multilayer_checker_summary={0}" -f $multilayerSummary),
        ("target_layer_checker_summary={0}" -f $targetLayerSummary),
        ("verdict={0}" -f $verdictPath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host "PASS: run_target_layer_attention_multilayer_acceptance"
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            "task: target_layer_attention_multilayer_acceptance",
            "status: FAIL",
            ("message: {0}" -f $_.Exception.Message),
            "scope: local-only"
        ) | Set-Content -Path $verdictPath -Encoding UTF8
    }
    if ($manifestPath -and -not (Test-Path $manifestPath -PathType Leaf)) {
        @(
            ("build_dir={0}" -f $BuildDir),
            "status=FAIL"
        ) | Set-Content -Path $manifestPath -Encoding UTF8
    }
    exit 1
}
finally {
    Pop-Location
}
