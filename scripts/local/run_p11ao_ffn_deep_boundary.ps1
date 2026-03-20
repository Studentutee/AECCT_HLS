param(
    [string]$BuildDir = "build\p11ao\p11ao"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-PassString {
    param(
        [string]$LogPath,
        [string]$Needle
    )
    if (-not (Select-String -Path $LogPath -SimpleMatch -Quiet $Needle)) {
        throw "required PASS string missing in $LogPath : $Needle"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$verdictPath = $null
$manifestPath = $null
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $checkPreLog = Join-Path $BuildDir 'check_p11ao_pre.log'
    $checkPostLog = Join-Path $BuildDir 'check_p11ao_post.log'
    $amRunnerLog = Join-Path $BuildDir 'run_p11am_surface.log'
    $anCheckerLog = Join-Path $BuildDir 'check_p11an_regression.log'
    $alRunnerLog = Join-Path $BuildDir 'run_p11al_regression.log'
    $akRunnerLog = Join-Path $BuildDir 'run_p11ak_regression.log'
    $verdictPath = Join-Path $BuildDir 'verdict.txt'
    $manifestPath = Join-Path $BuildDir 'file_manifest.txt'

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ao_ffn_deep_boundary.ps1 -OutDir $BuildDir -Phase pre *> $checkPreLog
    if ($LASTEXITCODE -ne 0) {
        throw "AO pre checker failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11am_catapult_compile_surface.ps1 -BuildDir (Join-Path $BuildDir 'p11am_surface') *> $amRunnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "embedded AM compile-surface runner failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11an_attn_deep_boundary.ps1 -OutDir $BuildDir -Phase post *> $anCheckerLog
    if ($LASTEXITCODE -ne 0) {
        throw "AN regression checker failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11al_catapult_top_wrapper.ps1 -BuildDir (Join-Path $BuildDir 'p11al_regression') *> $alRunnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "AL regression runner failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ak_attnout_finalx_bridge.ps1 -BuildDir (Join-Path $BuildDir 'p11ak_regression') *> $akRunnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "AK regression runner failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ao_ffn_deep_boundary.ps1 -OutDir $BuildDir -Phase post *> $checkPostLog
    if ($LASTEXITCODE -ne 0) {
        throw "AO post checker failed, exit=$LASTEXITCODE"
    }

    $amRunLog = Join-Path (Join-Path $BuildDir 'p11am_surface') 'run.log'
    $amSynthLog = Join-Path (Join-Path $BuildDir 'p11am_surface') 'build_synth_surface.log'
    $amVerdict = Join-Path (Join-Path $BuildDir 'p11am_surface') 'verdict.txt'
    $alRunLog = Join-Path (Join-Path $BuildDir 'p11al_regression') 'run.log'
    $akRunLog = Join-Path (Join-Path $BuildDir 'p11ak_regression') 'run.log'

    Require-PassString -LogPath $checkPreLog -Needle 'PASS: check_p11ao_ffn_deep_boundary'
    Require-PassString -LogPath $checkPostLog -Needle 'PASS: check_p11ao_ffn_deep_boundary'
    Require-PassString -LogPath $amRunLog -Needle 'PASS: tb_top_managed_catapult_compile_prep_p11am'
    Require-PassString -LogPath $amRunLog -Needle 'P11AM_MAINLINE_PATH_TAKEN PASS'
    Require-PassString -LogPath $amRunLog -Needle 'P11AM_FALLBACK_NOT_TAKEN PASS'
    Require-PassString -LogPath $amSynthLog -Needle 'PASS: p11am_synth_surface_compile'
    Require-PassString -LogPath $amVerdict -Needle 'status: PASS'
    Require-PassString -LogPath $anCheckerLog -Needle 'PASS: check_p11an_attn_deep_boundary'
    Require-PassString -LogPath $alRunLog -Needle 'PASS: tb_top_managed_catapult_wrapper_p11al'
    Require-PassString -LogPath $alRunLog -Needle 'TOP_WRAPPER_FALLBACK_NOT_TAKEN PASS'
    Require-PassString -LogPath $alRunLog -Needle 'PASS: run_p11al_catapult_top_wrapper'
    Require-PassString -LogPath $akRunLog -Needle 'PASS: tb_attnout_finalx_bridge_p11ak'
    Require-PassString -LogPath $akRunLog -Needle 'BRIDGE_ATTNOUT_TO_FINALX_DIRECT_CONSUMPTION PASS'
    Require-PassString -LogPath $akRunLog -Needle 'PASS: run_p11ak_attnout_finalx_bridge'

    @(
        'task: P00-011AO',
        'status: PASS',
        'banner: PASS: run_p11ao_ffn_deep_boundary',
        'scope: local-only',
        'posture: Catapult-facing progress only',
        'closure: not Catapult closure; not SCVerify closure'
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("ao_pre_check_log={0}" -f $checkPreLog),
        ("ao_post_check_log={0}" -f $checkPostLog),
        ("am_runner_log={0}" -f $amRunnerLog),
        ("am_run_log={0}" -f $amRunLog),
        ("am_synth_log={0}" -f $amSynthLog),
        ("an_checker_log={0}" -f $anCheckerLog),
        ("al_runner_log={0}" -f $alRunnerLog),
        ("al_run_log={0}" -f $alRunLog),
        ("ak_runner_log={0}" -f $akRunnerLog),
        ("ak_run_log={0}" -f $akRunLog),
        ("verdict={0}" -f $verdictPath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host 'PASS: run_p11ao_ffn_deep_boundary'
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            'task: P00-011AO',
            'status: FAIL',
            ("message: {0}" -f $_.Exception.Message)
        ) | Set-Content -Path $verdictPath -Encoding UTF8
    }
    if ($manifestPath -and (Test-Path $manifestPath -PathType Leaf) -eq $false) {
        @(
            ("build_dir={0}" -f $BuildDir),
            'status=FAIL'
        ) | Set-Content -Path $manifestPath -Encoding UTF8
    }
    exit 1
}
finally {
    Pop-Location
}
