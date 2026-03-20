param(
    [string]$BuildDir = "build\p11ap\p11ap"
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

    $checkPreLog = Join-Path $BuildDir 'check_p11ap_pre.log'
    $checkPostLog = Join-Path $BuildDir 'check_p11ap_post.log'
    $amRunnerLog = Join-Path $BuildDir 'run_p11am_surface.log'
    $anRunnerLog = Join-Path $BuildDir 'run_p11an_regression.log'
    $aoRunnerLog = Join-Path $BuildDir 'run_p11ao_regression.log'
    $ahRunnerLog = Join-Path $BuildDir 'run_p11ah_regression.log'
    $verdictPath = Join-Path $BuildDir 'verdict.txt'
    $manifestPath = Join-Path $BuildDir 'file_manifest.txt'

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ap_active_chain_residual_rawptr.ps1 -OutDir $BuildDir -Phase pre *> $checkPreLog
    if ($LASTEXITCODE -ne 0) {
        throw "AP pre checker failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11am_catapult_compile_surface.ps1 -BuildDir (Join-Path $BuildDir 'p11am_surface') *> $amRunnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "AM compile-surface runner failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11an_attn_deep_boundary.ps1 -BuildDir (Join-Path $BuildDir 'p11an_regression') *> $anRunnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "AN regression runner failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ao_ffn_deep_boundary.ps1 -BuildDir (Join-Path $BuildDir 'p11ao_regression') *> $aoRunnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "AO regression runner failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir (Join-Path $BuildDir 'p11ah_regression') *> $ahRunnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "AH full-loop regression runner failed, exit=$LASTEXITCODE"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ap_active_chain_residual_rawptr.ps1 -OutDir $BuildDir -Phase post *> $checkPostLog
    if ($LASTEXITCODE -ne 0) {
        throw "AP post checker failed, exit=$LASTEXITCODE"
    }

    $amRunLog = Join-Path (Join-Path $BuildDir 'p11am_surface') 'run.log'
    $amSynthLog = Join-Path (Join-Path $BuildDir 'p11am_surface') 'build_synth_surface.log'
    $anVerdict = Join-Path (Join-Path $BuildDir 'p11an_regression') 'verdict.txt'
    $aoVerdict = Join-Path (Join-Path $BuildDir 'p11ao_regression') 'verdict.txt'
    $alRunLog = Join-Path (Join-Path (Join-Path $BuildDir 'p11ao_regression') 'p11al_regression') 'run.log'
    $akRunLog = Join-Path (Join-Path (Join-Path $BuildDir 'p11ao_regression') 'p11ak_regression') 'run.log'
    $ahRunLog = Join-Path (Join-Path $BuildDir 'p11ah_regression') 'run.log'

    Require-PassString -LogPath $checkPreLog -Needle 'PASS: check_p11ap_active_chain_residual_rawptr'
    Require-PassString -LogPath $checkPostLog -Needle 'PASS: check_p11ap_active_chain_residual_rawptr'
    Require-PassString -LogPath $checkPostLog -Needle 'active_chain_remaining_raw_pointer_sites: none detected on targeted Attn+FFN synth-facing chain'
    Require-PassString -LogPath $amRunLog -Needle 'PASS: tb_top_managed_catapult_compile_prep_p11am'
    Require-PassString -LogPath $amRunLog -Needle 'P11AM_MAINLINE_PATH_TAKEN PASS'
    Require-PassString -LogPath $amRunLog -Needle 'P11AM_FALLBACK_NOT_TAKEN PASS'
    Require-PassString -LogPath $amSynthLog -Needle 'PASS: p11am_synth_surface_compile'
    Require-PassString -LogPath $anRunnerLog -Needle 'PASS: run_p11an_attn_deep_boundary'
    Require-PassString -LogPath $anVerdict -Needle 'status: PASS'
    Require-PassString -LogPath $aoRunnerLog -Needle 'PASS: run_p11ao_ffn_deep_boundary'
    Require-PassString -LogPath $aoVerdict -Needle 'status: PASS'
    Require-PassString -LogPath $alRunLog -Needle 'PASS: tb_top_managed_catapult_wrapper_p11al'
    Require-PassString -LogPath $akRunLog -Needle 'PASS: tb_attnout_finalx_bridge_p11ak'
    Require-PassString -LogPath $ahRunLog -Needle 'PASS: tb_full_loop_local_e2e_p11ah'
    Require-PassString -LogPath $ahRunLog -Needle 'FULL_LOOP_FALLBACK_NOT_TAKEN PASS'
    Require-PassString -LogPath $ahRunLog -Needle 'PASS: run_p11ah_full_loop_local_e2e'

    @(
        'task: P00-011AP',
        'status: PASS',
        'banner: PASS: run_p11ap_active_chain_residual_cleanup',
        'scope: local-only',
        'posture: Catapult-facing progress only',
        'closure: not Catapult closure; not SCVerify closure'
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("ap_pre_check_log={0}" -f $checkPreLog),
        ("ap_post_check_log={0}" -f $checkPostLog),
        ("am_runner_log={0}" -f $amRunnerLog),
        ("an_runner_log={0}" -f $anRunnerLog),
        ("ao_runner_log={0}" -f $aoRunnerLog),
        ("ah_runner_log={0}" -f $ahRunnerLog),
        ("am_run_log={0}" -f $amRunLog),
        ("am_synth_log={0}" -f $amSynthLog),
        ("an_verdict={0}" -f $anVerdict),
        ("ao_verdict={0}" -f $aoVerdict),
        ("al_run_log={0}" -f $alRunLog),
        ("ak_run_log={0}" -f $akRunLog),
        ("ah_run_log={0}" -f $ahRunLog),
        ("verdict={0}" -f $verdictPath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host 'PASS: run_p11ap_active_chain_residual_cleanup'
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            'task: P00-011AP',
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
