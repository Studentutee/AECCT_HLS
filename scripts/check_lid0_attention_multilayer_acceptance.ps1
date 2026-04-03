param(
    [string]$RepoRoot = ".",
    [string]$RunLog = "build\\lid0_attention_multilayer_acceptance\\p11aj\\run.log",
    [string]$OutDir = "build\\lid0_attention_multilayer_acceptance\\checker"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Join-RepoPath {
    param(
        [string]$RepoRootPath,
        [string]$Path
    )
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRootPath $Path))
}

function Get-RepoRelativePath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )
    $baseUri = New-Object System.Uri(([System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'))
    $targetUri = New-Object System.Uri([System.IO.Path]::GetFullPath($TargetPath))
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('\', '/')
}

function Require-Line {
    param(
        [string]$Path,
        [string]$Needle
    )
    if (-not (Select-String -Path $Path -SimpleMatch -Quiet $Needle)) {
        throw "required line missing in $Path : $Needle"
    }
}

function Get-FirstLineNumber {
    param(
        [string]$Path,
        [string]$Needle
    )
    $m = Select-String -Path $Path -SimpleMatch $Needle | Select-Object -First 1
    if ($null -eq $m) {
        throw "line missing for ordering check in $Path : $Needle"
    }
    return [int]$m.LineNumber
}

$repo = [System.IO.Path]::GetFullPath((Resolve-Path $RepoRoot).Path)
$runLogAbs = Join-RepoPath -RepoRootPath $repo -Path $RunLog
$outDirAbs = Join-RepoPath -RepoRootPath $repo -Path $OutDir
New-Item -ItemType Directory -Force -Path $outDirAbs > $null

$logPath = Join-Path $outDirAbs "check_lid0_attention_multilayer_acceptance.log"
$summaryPath = Join-Path $outDirAbs "check_lid0_attention_multilayer_acceptance_summary.txt"
Set-Content -Path $logPath -Value "===== check_lid0_attention_multilayer_acceptance =====" -Encoding UTF8

if (-not (Test-Path $runLogAbs -PathType Leaf)) {
    throw "run log not found: $runLogAbs"
}

$requiredLines = @(
    "CASE_BASELINE_N1_LID0_ATTN_MAINLINE_FLAGS p11ad_mainline_q_path_taken=1 p11ac_mainline_path_taken=1 p11ae_mainline_score_path_taken=1 p11af_mainline_softmax_output_path_taken=1",
    "CASE_BASELINE_N1_LID0_ATTN_FALLBACK_FLAGS p11ad_q_fallback_taken=0 p11ac_fallback_taken=0 p11ae_score_fallback_taken=0 p11af_softmax_output_fallback_taken=0",
    "CASE_BASELINE_N1_ATTN_COMPAT_SHELL_OBSERVABILITY target_layer_id=0 n_layers=1 shell_enabled_count=0 shell_disabled_count=1 shell_enabled_last_layer_id=4294967295 shell_disabled_last_layer_id=0 target_shell_disabled_count=1 non_target_shell_enabled_count=0",
    "CASE_BASELINE_N1_ATTN_COMPAT_SHELL_DISABLED PASS",
    "CASE_BASELINE_N1_ATTN_COMPAT_SHELL_NO_NON_TARGET_LAYERS PASS",
    "CASE_BASELINE_N1_HANDOFF_COUNTER_CONSERVATION PASS",
    "FULL_LOOP_MAINLINE_PATH_TAKEN PASS",
    "FULL_LOOP_FALLBACK_NOT_TAKEN PASS",
    "CASE_MIXED_N3_LID0_ATTN_MAINLINE_FLAGS p11ad_mainline_q_path_taken=1 p11ac_mainline_path_taken=1 p11ae_mainline_score_path_taken=1 p11af_mainline_softmax_output_path_taken=1",
    "CASE_MIXED_N3_LID0_ATTN_FALLBACK_FLAGS p11ad_q_fallback_taken=0 p11ac_fallback_taken=0 p11ae_score_fallback_taken=0 p11af_softmax_output_fallback_taken=0",
    "CASE_TARGET0_N3_ATTN_COMPAT_SHELL_OBSERVABILITY target_layer_id=0 n_layers=3 shell_enabled_count=2 shell_disabled_count=1 shell_enabled_last_layer_id=2 shell_disabled_last_layer_id=0 target_shell_disabled_count=1 non_target_shell_enabled_count=2",
    "CASE_TARGET0_N3_ATTN_COMPAT_SHELL_DISABLED PASS",
    "CASE_TARGET0_N3_NON_TARGET_ATTN_COMPAT_SHELL_RETAINS_LEGACY PASS",
    "CASE_MIXED_N3_HANDOFF_COUNTER_CONSERVATION PASS",
    "CASE_MIXED_N3_LID0_MARKERS_STABLE_VS_BASELINE PASS",
    "CASE_MIXED_N3_MARKER_REPEATABILITY PASS",
    "CASE_MIXED_N3_FINAL_X_DETERMINISTIC PASS",
    "CASE_MIXED_N3_FINAL_X_NONFINITE_SCAN PASS",
    "CASE_MIXED_N3_NONZERO_LAYER_EFFECT_OBSERVED PASS",
    "CASE_MIXED_N3_ACCEPTANCE PASS",
    "PASS: tb_top_managed_sram_provenance_p11aj"
)

foreach ($line in $requiredLines) {
    Require-Line -Path $runLogAbs -Needle $line
    Add-Content -Path $logPath -Value ("verified: {0}" -f $line) -Encoding UTF8
}

$baselineLine = Get-FirstLineNumber -Path $runLogAbs -Needle "CASE_BASELINE_N1_LID0_ATTN_MAINLINE_FLAGS"
$mixedLine = Get-FirstLineNumber -Path $runLogAbs -Needle "CASE_MIXED_N3_LID0_ATTN_MAINLINE_FLAGS"
if ($baselineLine -ge $mixedLine) {
    throw "baseline markers do not appear before mixed-layer markers"
}
Add-Content -Path $logPath -Value ("ordering: baseline_line={0} mixed_line={1}" -f $baselineLine, $mixedLine) -Encoding UTF8

@(
    "status: PASS",
    ("repo_root: {0}" -f $repo),
    ("run_log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $runLogAbs)),
    ("checker_log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $logPath)),
    "scope: local-only",
    "closure: not Catapult closure; not SCVerify closure",
    "banner: PASS: check_lid0_attention_multilayer_acceptance"
) | Set-Content -Path $summaryPath -Encoding UTF8

Add-Content -Path $logPath -Value "PASS: check_lid0_attention_multilayer_acceptance" -Encoding UTF8
Write-Host "PASS: check_lid0_attention_multilayer_acceptance"
