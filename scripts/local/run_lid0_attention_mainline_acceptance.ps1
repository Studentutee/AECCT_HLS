param(
    [string]$BuildDir = "build\\lid0_attention_mainline_acceptance"
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
    $checkerOutDir = Join-Path $BuildDir "checker"
    $verdictPath = Join-Path $BuildDir "verdict.txt"
    $manifestPath = Join-Path $BuildDir "file_manifest.txt"

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir $nestedBuild *> $runnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "run_p11aj_top_managed_sram_provenance failed, exit=$LASTEXITCODE"
    }

    $runLog = Join-Path $nestedBuild "run.log"
    Require-File -Path $runLog

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_lid0_attention_mainline_acceptance.ps1 -RepoRoot $repoRoot -RunLog $runLog -OutDir $checkerOutDir
    if ($LASTEXITCODE -ne 0) {
        throw "check_lid0_attention_mainline_acceptance failed, exit=$LASTEXITCODE"
    }

    $checkerSummary = Join-Path $checkerOutDir "check_lid0_attention_mainline_acceptance_summary.txt"
    Require-File -Path $checkerSummary

    @(
        "task: lid0_attention_mainline_acceptance",
        "status: PASS",
        "banner: PASS: run_lid0_attention_mainline_acceptance",
        "scope: local-only",
        "closure: not Catapult closure; not SCVerify closure"
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("runner_log={0}" -f $runnerLog),
        ("run_log={0}" -f $runLog),
        ("checker_summary={0}" -f $checkerSummary),
        ("verdict={0}" -f $verdictPath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host "PASS: run_lid0_attention_mainline_acceptance"
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            "task: lid0_attention_mainline_acceptance",
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
