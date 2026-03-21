param(
    [string]$BuildDir = "build\p11as\catapult_launch",
    [string]$CatapultCmd = "",
    [switch]$SkipPreflight
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-CatapultCommand {
    param([string]$UserCmd)

    if ($UserCmd -ne "") {
        $explicit = Get-Command $UserCmd -ErrorAction SilentlyContinue
        if ($explicit) { return $explicit.Source }
        if (Test-Path $UserCmd) { return (Resolve-Path $UserCmd).Path }
        return $null
    }

    $fromPath = Get-Command "catapult" -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }

    $candidates = @()
    if ($env:CATAPULT_HOME) {
        $candidates += (Join-Path $env:CATAPULT_HOME "bin\catapult.exe")
    }
    if ($env:MGC_HOME) {
        $candidates += (Join-Path $env:MGC_HOME "bin\catapult.exe")
    }
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            return (Resolve-Path $c).Path
        }
    }
    return $null
}

function Require-Contains {
    param(
        [string]$Path,
        [string]$Needle
    )
    if (-not (Select-String -Path $Path -SimpleMatch -Quiet $Needle)) {
        throw "required text missing in $Path : $Needle"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildAbs = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
$preflightDir = Join-Path $buildAbs "preflight"
$envProbeLog = Join-Path $buildAbs "env_probe.log"
$runLog = Join-Path $buildAbs "run.log"
$catapultLog = Join-Path $buildAbs "catapult.log"
$verdictPath = Join-Path $buildAbs "verdict.txt"
$manifestPath = Join-Path $buildAbs "file_manifest.txt"
$preflightLog = Join-Path $buildAbs "preflight.log"

Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $buildAbs > $null

    if (-not $SkipPreflight) {
        & powershell -NoProfile -ExecutionPolicy Bypass `
            -File scripts/check_p11as_corrected_chain_launch_pack.ps1 `
            -OutDir $preflightDir `
            -Phase pre *> $preflightLog
        if ($LASTEXITCODE -ne 0) {
            throw "p11as preflight checker failed, exit=$LASTEXITCODE"
        }
    } else {
        "SKIP: preflight requested by -SkipPreflight" | Set-Content -Path $preflightLog -Encoding UTF8
    }

    $resolvedCatapult = Resolve-CatapultCommand -UserCmd $CatapultCmd
    $probeLines = @(
        ("timestamp={0}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")),
        ("MGC_HOME={0}" -f $env:MGC_HOME),
        ("CATAPULT_HOME={0}" -f $env:CATAPULT_HOME),
        ("PATH={0}" -f $env:PATH),
        ("resolved_catapult={0}" -f $(if ($resolvedCatapult) { $resolvedCatapult } else { "<not-found>" }))
    )
    $probeLines | Set-Content -Path $envProbeLog -Encoding UTF8

    if (-not $resolvedCatapult) {
        @(
            "P11AS_STATUS NOT_EXECUTED",
            "P11AS_REASON catapult command not found (PATH/CATAPULT_HOME/MGC_HOME)",
            ("P11AS_PREFLIGHT {0}" -f $(if ($SkipPreflight) { "SKIPPED" } else { "PASS" })),
            "P11AS_TRUE_CATAPULT executed=false"
        ) | Set-Content -Path $runLog -Encoding UTF8

        @(
            "task: P00-011AS",
            "status: NOT_EXECUTED",
            "reason: catapult command not found",
            "scope: tool-ready launch-prep only",
            "closure: not Catapult closure; not SCVerify closure"
        ) | Set-Content -Path $verdictPath -Encoding UTF8

        @(
            ("run_log={0}" -f $runLog),
            ("env_probe_log={0}" -f $envProbeLog),
            ("preflight_log={0}" -f $preflightLog),
            ("preflight_dir={0}" -f $preflightDir),
            ("project_tcl={0}" -f (Join-Path $repoRoot "scripts/catapult/p11as_corrected_chain_project.tcl")),
            ("filelist={0}" -f (Join-Path $repoRoot "scripts/catapult/p11as_corrected_chain_filelist.f")),
            ("verdict={0}" -f $verdictPath)
        ) | Set-Content -Path $manifestPath -Encoding UTF8

        Write-Host "P11AS fail-fast: catapult command not found (PATH/CATAPULT_HOME/MGC_HOME)."
        exit 2
    }

    $env:AECCT_P11AS_REPO_ROOT = $repoRoot
    $env:AECCT_P11AS_CATAPULT_OUTDIR = (Join-Path $buildAbs "project")
    $projectTcl = Join-Path $repoRoot "scripts/catapult/p11as_corrected_chain_project.tcl"

    & $resolvedCatapult -shell -file $projectTcl 2>&1 | Tee-Object -FilePath $catapultLog
    if ($LASTEXITCODE -ne 0) {
        throw "catapult execution failed, exit=$LASTEXITCODE"
    }

    Require-Contains -Path $catapultLog -Needle "P11AS_CANONICAL_SYNTH_ENTRY TopManagedAttentionChainCatapultTop::run"
    Require-Contains -Path $catapultLog -Needle "P11AS_STAGE analyze DONE"
    Require-Contains -Path $catapultLog -Needle "P11AS_STAGE compile DONE"
    Require-Contains -Path $catapultLog -Needle "P11AS_STAGE elaborate DONE"

    @(
        "P11AS_STATUS EXECUTED",
        ("P11AS_CATAPULT_CMD {0}" -f $resolvedCatapult),
        "P11AS_TRUE_CATAPULT executed=true",
        "P11AS_STAGE analyze DONE",
        "P11AS_STAGE compile DONE",
        "P11AS_STAGE elaborate DONE",
        "PASS: run_p11as_corrected_chain_catapult_launch"
    ) | Set-Content -Path $runLog -Encoding UTF8

    @(
        "task: P00-011AS",
        "status: PASS",
        "banner: PASS: run_p11as_corrected_chain_catapult_launch",
        "scope: corrected-chain Catapult launch-prep + tool execution",
        "closure: not Catapult closure; not SCVerify closure"
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("run_log={0}" -f $runLog),
        ("catapult_log={0}" -f $catapultLog),
        ("env_probe_log={0}" -f $envProbeLog),
        ("preflight_log={0}" -f $preflightLog),
        ("preflight_dir={0}" -f $preflightDir),
        ("project_tcl={0}" -f $projectTcl),
        ("filelist={0}" -f (Join-Path $repoRoot "scripts/catapult/p11as_corrected_chain_filelist.f")),
        ("verdict={0}" -f $verdictPath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host "PASS: run_p11as_corrected_chain_catapult_launch"
    exit 0
}
catch {
    Write-Error $_
    if (-not (Test-Path $verdictPath)) {
        @(
            "task: P00-011AS",
            "status: FAIL",
            ("message: {0}" -f $_.Exception.Message)
        ) | Set-Content -Path $verdictPath -Encoding UTF8
    }
    if (-not (Test-Path $manifestPath)) {
        @(
            ("build_dir={0}" -f $buildAbs),
            "status=FAIL"
        ) | Set-Content -Path $manifestPath -Encoding UTF8
    }
    exit 1
}
finally {
    Pop-Location
}
