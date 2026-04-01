param(
    [string]$BuildDir = "build\\p11w4pack\\softmaxout_writeback_regression_pack",
    [switch]$SkipC7
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$verdictPath = $null
$summaryPath = $null

Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null
    $summaryPath = Join-Path $BuildDir "summary.log"
    $verdictPath = Join-Path $BuildDir "verdict.txt"
    $manifestPath = Join-Path $BuildDir "file_manifest.txt"

    $runs = @(
        @{
            Name = "W4C4"
            Script = "scripts/local/run_p11w4c4_softmaxout_writeback_single_selected_probe.ps1"
            BuildSubdir = "c4"
            Banner = "PASS: run_p11w4c4_softmaxout_writeback_single_selected_probe"
        },
        @{
            Name = "W4C5"
            Script = "scripts/local/run_p11w4c5_softmaxout_writeback_single_selected_consume_bridge.ps1"
            BuildSubdir = "c5"
            Banner = "PASS: run_p11w4c5_softmaxout_writeback_single_selected_consume_bridge"
        },
        @{
            Name = "W4C6"
            Script = "scripts/local/run_p11w4c6_softmaxout_writeback_small_family_consume_bridge.ps1"
            BuildSubdir = "c6"
            Banner = "PASS: run_p11w4c6_softmaxout_writeback_small_family_consume_bridge"
        },
        @{
            Name = "P11AF"
            Script = "scripts/local/run_p11af_impl_softmax_out.ps1"
            BuildSubdir = "p11af"
            Banner = "PASS: run_p11af_impl_softmax_out"
        }
    )

    if (-not $SkipC7) {
        $runs += @{
            Name = "W4C7"
            Script = "scripts/local/run_p11w4c7_softmaxout_writeback_family_bound_hardening.ps1"
            BuildSubdir = "c7"
            Banner = "PASS: run_p11w4c7_softmaxout_writeback_family_bound_hardening"
        }
    }

    "task: P00-W4-SOFTMAXOUT-WRITEBACK-REGRESSION-PACK" | Set-Content -Path $summaryPath -Encoding UTF8
    "scope: local-only" | Add-Content -Path $summaryPath -Encoding UTF8
    "closure: not Catapult closure; not SCVerify closure" | Add-Content -Path $summaryPath -Encoding UTF8

    foreach ($run in $runs) {
        $subBuild = Join-Path $BuildDir $run.BuildSubdir
        $cmd = @(
            "-ExecutionPolicy", "Bypass",
            "-File", $run.Script,
            "-BuildDir", $subBuild
        )
        & powershell @cmd
        if ($LASTEXITCODE -ne 0) {
            throw ("runner failed: {0}" -f $run.Script)
        }
        ("[{0}] {1}" -f $run.Name, $run.Banner) | Add-Content -Path $summaryPath -Encoding UTF8
    }

    @(
        "task: P00-W4-SOFTMAXOUT-WRITEBACK-REGRESSION-PACK",
        "status: PASS",
        "banner: PASS: run_p11w4_softmaxout_writeback_regression_pack",
        "scope: local-only",
        "closure: not Catapult closure; not SCVerify closure"
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("summary={0}" -f $summaryPath),
        ("verdict={0}" -f $verdictPath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host "PASS: run_p11w4_softmaxout_writeback_regression_pack"
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            "task: P00-W4-SOFTMAXOUT-WRITEBACK-REGRESSION-PACK",
            "status: FAIL",
            ("message: {0}" -f $_.Exception.Message)
        ) | Set-Content -Path $verdictPath -Encoding UTF8
    }
    exit 1
}
finally {
    Pop-Location
}
