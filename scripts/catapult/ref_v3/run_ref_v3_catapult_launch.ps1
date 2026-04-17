param(
    [string]$BuildDir = "build\ref_v3\catapult_launch",
    [string]$CatapultCmd = ""
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

$repoRoot = (Resolve-Path (Join-Path (Join-Path $PSScriptRoot "..\..") "..")).Path
$buildAbs = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
$runLog = Join-Path $buildAbs "run.log"
$catapultLog = Join-Path $buildAbs "catapult.log"
$envProbeLog = Join-Path $buildAbs "env_probe.log"
$verdictPath = Join-Path $buildAbs "verdict.txt"

Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $buildAbs > $null

    $resolvedCatapult = Resolve-CatapultCommand -UserCmd $CatapultCmd
    @(
        ("timestamp={0}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")),
        ("MGC_HOME={0}" -f $env:MGC_HOME),
        ("CATAPULT_HOME={0}" -f $env:CATAPULT_HOME),
        ("PATH={0}" -f $env:PATH),
        ("resolved_catapult={0}" -f $(if ($resolvedCatapult) { $resolvedCatapult } else { "<not-found>" }))
    ) | Set-Content -Path $envProbeLog -Encoding UTF8

    if (-not $resolvedCatapult) {
        @(
            "REFV3_STATUS NOT_EXECUTED",
            "REFV3_REASON catapult command not found (PATH/CATAPULT_HOME/MGC_HOME)",
            "REFV3_TRUE_CATAPULT executed=false"
        ) | Set-Content -Path $runLog -Encoding UTF8

        @(
            "task: REF_V3_CATAPULT_EXPERIMENT_TRACK",
            "status: NOT_EXECUTED",
            "reason: catapult command not found",
            "posture: not Catapult closure; not SCVerify closure; not synth closure"
        ) | Set-Content -Path $verdictPath -Encoding UTF8
        exit 2
    }

    $env:AECCT_REFV3_REPO_ROOT = $repoRoot
    $env:AECCT_REFV3_CATAPULT_OUTDIR = (Join-Path $buildAbs "project")
    $projectTcl = Join-Path $repoRoot "scripts/catapult/ref_v3/project.tcl"

    & $resolvedCatapult -shell -file $projectTcl 2>&1 | Tee-Object -FilePath $catapultLog
    if ($LASTEXITCODE -ne 0) {
        throw "catapult execution failed, exit=$LASTEXITCODE"
    }
    if (-not (Select-String -Path $catapultLog -SimpleMatch -Quiet "REFV3_STAGE compile DONE")) {
        throw "go compile did not reach REFV3_STAGE compile DONE"
    }

    @(
        "REFV3_STATUS EXECUTED",
        ("REFV3_CATAPULT_CMD {0}" -f $resolvedCatapult),
        "REFV3_TRUE_CATAPULT executed=true",
        "REFV3_STAGE compile DONE"
    ) | Set-Content -Path $runLog -Encoding UTF8

    @(
        "task: REF_V3_CATAPULT_EXPERIMENT_TRACK",
        "status: PASS",
        "scope: Catapult compile-first",
        "posture: not Catapult closure; not SCVerify closure; not synth closure"
    ) | Set-Content -Path $verdictPath -Encoding UTF8
    exit 0
}
catch {
    Write-Error $_
    @(
        "task: REF_V3_CATAPULT_EXPERIMENT_TRACK",
        "status: FAIL",
        ("message: {0}" -f $_.Exception.Message)
    ) | Set-Content -Path $verdictPath -Encoding UTF8
    exit 1
}
finally {
    Pop-Location
}
