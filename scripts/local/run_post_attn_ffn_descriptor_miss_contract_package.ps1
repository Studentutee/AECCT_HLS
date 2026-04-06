param(
    [string]$BuildDir = "build\\post_attn\\ffn_descriptor_miss_contract_package"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-LocalRunner {
    param(
        [string]$ScriptPath,
        [string]$RunnerBuildDir,
        [string]$RunnerLog
    )

    & powershell -ExecutionPolicy Bypass -File $ScriptPath -BuildDir $RunnerBuildDir *> $RunnerLog
    if ($LASTEXITCODE -ne 0) {
        throw "runner failed ($ScriptPath), exit=$LASTEXITCODE"
    }
}

function Require-PassString {
    param(
        [string]$LogPath,
        [string]$Needle
    )

    if (-not (Select-String -Path $LogPath -SimpleMatch -Quiet $Needle)) {
        throw "required PASS string missing in $LogPath : $Needle"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$verdictPath = $null
$manifestPath = $null

Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $auBuildDir = Join-Path $BuildDir "p11au_transformerlayer"
    $avBuildDir = Join-Path $BuildDir "p11av_top_pipeline"
    $g6BuildDir = Join-Path $BuildDir "p11g6_ffn_core"

    $auRunnerLog = Join-Path $BuildDir "run_p11au_wrapper.log"
    $avRunnerLog = Join-Path $BuildDir "run_p11av_wrapper.log"
    $g6RunnerLog = Join-Path $BuildDir "run_p11g6_wrapper.log"

    $packageLog = Join-Path $BuildDir "run.log"
    $verdictPath = Join-Path $BuildDir "verdict.txt"
    $manifestPath = Join-Path $BuildDir "file_manifest.txt"

    Invoke-LocalRunner `
        -ScriptPath "scripts\\local\\run_p11au_transformerlayer_ffn_higher_level_ownership_seam.ps1" `
        -RunnerBuildDir $auBuildDir `
        -RunnerLog $auRunnerLog
    Invoke-LocalRunner `
        -ScriptPath "scripts\\local\\run_p11av_top_ffn_handoff_assembly_smoke.ps1" `
        -RunnerBuildDir $avBuildDir `
        -RunnerLog $avRunnerLog
    Invoke-LocalRunner `
        -ScriptPath "scripts\\local\\run_p11g6_ffn_fallback_observability.ps1" `
        -RunnerBuildDir $g6BuildDir `
        -RunnerLog $g6RunnerLog

    $auRunLog = Join-Path $auBuildDir "run.log"
    $avRunLog = Join-Path $avBuildDir "run.log"
    $g6RunLog = Join-Path $g6BuildDir "run.log"

    $requiredAu = @(
        "W1_INPUT_MAINLINE_TAKEN_POINTER_PATH PASS",
        "W1_INPUT_PRELOAD_FALLBACK_EXPECTED_POINTER_PATH PASS",
        "W1_WEIGHT_MAINLINE_TAKEN_POINTER_PATH PASS",
        "W1_WEIGHT_PRELOAD_FALLBACK_EXPECTED_POINTER_PATH PASS",
        "W2_WEIGHT_MAINLINE_TAKEN_POINTER_PATH PASS",
        "W2_PRELOAD_FALLBACK_EXPECTED_POINTER_PATH PASS",
        "PASS: tb_transformerlayer_ffn_higher_level_ownership_seam"
    )
    foreach ($line in $requiredAu) {
        Require-PassString -LogPath $auRunLog -Needle $line
    }

    $requiredAv = @(
        "TOP_PIPELINE_LID0_FFN_HANDOFF_POINTER_PATH PASS",
        "TOP_PIPELINE_LID0_FFN_HANDOFF_MONOTONICITY_POINTER PASS",
        "TOP_PIPELINE_LID0_FFN_HANDOFF_EXPECTED_COMPARE PASS",
        "PASS: tb_top_ffn_handoff_assembly_smoke_p11av"
    )
    foreach ($line in $requiredAv) {
        Require-PassString -LogPath $avRunLog -Needle $line
    }

    $requiredG6 = @(
        "G6FFN_SUBWAVE_B_REJECT_STAGE_W1 PASS",
        "G6FFN_SUBWAVE_B_REJECT_STAGE_W2 PASS",
        "G6FFN_SUBWAVE_B_NONSTRICT_FALLBACK_OBS PASS",
        "PASS: tb_g6_ffn_fallback_observability_p11g6b"
    )
    foreach ($line in $requiredG6) {
        Require-PassString -LogPath $g6RunLog -Needle $line
    }

    @(
        "POST_ATTN_FFN_DESCRIPTOR_MISS_CONTRACT_PACKAGE START",
        "PRESENT -> TOPFED_CONSUME PASS",
        "MISS -> FALLBACK_PRELOAD PASS",
        "NO_FAKE_TOPFED_ACCEPT PASS",
        "COUNTER / MARKER CONSISTENCY PASS",
        "REJECT_STAGE_OBSERVABILITY PASS",
        "closure posture: not Catapult closure; not SCVerify closure",
        "PASS: run_post_attn_ffn_descriptor_miss_contract_package"
    ) | Set-Content -Path $packageLog -Encoding UTF8

    @(
        "task: POST_ATTN_FFN_DESCRIPTOR_MISS_CONTRACT_PACKAGE",
        "status: PASS",
        "banner: PASS: run_post_attn_ffn_descriptor_miss_contract_package",
        "scope: local-only",
        "closure: not Catapult closure; not SCVerify closure"
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("package_log={0}" -f $packageLog),
        ("package_verdict={0}" -f $verdictPath),
        ("runner_p11au_log={0}" -f $auRunnerLog),
        ("runner_p11av_log={0}" -f $avRunnerLog),
        ("runner_p11g6_log={0}" -f $g6RunnerLog),
        ("p11au_run_log={0}" -f $auRunLog),
        ("p11av_run_log={0}" -f $avRunLog),
        ("p11g6_run_log={0}" -f $g6RunLog)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host "PASS: run_post_attn_ffn_descriptor_miss_contract_package"
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            "task: POST_ATTN_FFN_DESCRIPTOR_MISS_CONTRACT_PACKAGE",
            "status: FAIL",
            ("message: {0}" -f $_.Exception.Message)
        ) | Set-Content -Path $verdictPath -Encoding UTF8
    }
    if ($manifestPath -and (Test-Path $manifestPath -PathType Leaf) -eq $false) {
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
