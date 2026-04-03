param(
    [string]$RepoRoot = ".",
    [string]$RunLog = "build\\target_layer_attention_multilayer_acceptance\\p11aj\\run.log",
    [string]$OutDir = "build\\target_layer_attention_multilayer_acceptance\\checker_target_layer"
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

$logPath = Join-Path $outDirAbs "check_target_layer_attention_multilayer_acceptance.log"
$summaryPath = Join-Path $outDirAbs "check_target_layer_attention_multilayer_acceptance_summary.txt"
Set-Content -Path $logPath -Value "===== check_target_layer_attention_multilayer_acceptance =====" -Encoding UTF8

if (-not (Test-Path $runLogAbs -PathType Leaf)) {
    throw "run log not found: $runLogAbs"
}

$requiredLines = @(
    "CASE_TARGET0_N3_TARGET_ATTN_MAINLINE_FLAGS target_layer_id=0 p11ad_mainline_q_path_taken=1 p11ac_mainline_path_taken=1 p11ae_mainline_score_path_taken=1 p11af_mainline_softmax_output_path_taken=1",
    "CASE_TARGET0_N3_TARGET_ATTN_FALLBACK_FLAGS target_layer_id=0 p11ad_q_fallback_taken=0 p11ac_fallback_taken=0 p11ae_score_fallback_taken=0 p11af_softmax_output_fallback_taken=0",
    "CASE_TARGET0_N3_MANAGED_TARGET_SELECTION target_layer_id=0 gate_taken_count=1 last_layer_id=0",
    "CASE_TARGET0_N3_MANAGED_TARGET_SELECTION PASS",
    "CASE_TARGET0_N3_NON_TARGET_LAYERS_NOT_MANAGED PASS",
    "CASE_TARGET1_N3_TARGET_ATTN_MAINLINE_FLAGS target_layer_id=1 p11ad_mainline_q_path_taken=1 p11ac_mainline_path_taken=1 p11ae_mainline_score_path_taken=1 p11af_mainline_softmax_output_path_taken=1",
    "CASE_TARGET1_N3_TARGET_ATTN_FALLBACK_FLAGS target_layer_id=1 p11ad_q_fallback_taken=0 p11ac_fallback_taken=0 p11ae_score_fallback_taken=0 p11af_softmax_output_fallback_taken=0",
    "CASE_TARGET1_N3_MANAGED_TARGET_SELECTION target_layer_id=1 gate_taken_count=1 last_layer_id=1",
    "CASE_TARGET1_N3_MANAGED_TARGET_SELECTION PASS",
    "CASE_TARGET1_N3_NON_TARGET_LAYERS_NOT_MANAGED PASS",
    "CASE_TARGET1_N3_HANDOFF_COUNTER_CONSERVATION PASS",
    "CASE_TARGET1_N3_MARKER_REPEATABILITY PASS",
    "CASE_TARGET1_N3_FINAL_X_DETERMINISTIC PASS",
    "CASE_TARGET1_N3_FINAL_X_NONFINITE_SCAN PASS",
    "CASE_TARGET1_N3_ACCEPTANCE PASS",
    "PASS: tb_top_managed_sram_provenance_p11aj"
)

foreach ($line in $requiredLines) {
    Require-Line -Path $runLogAbs -Needle $line
    Add-Content -Path $logPath -Value ("verified: {0}" -f $line) -Encoding UTF8
}

$target0Line = Get-FirstLineNumber -Path $runLogAbs -Needle "CASE_TARGET0_N3_TARGET_ATTN_MAINLINE_FLAGS"
$target1Line = Get-FirstLineNumber -Path $runLogAbs -Needle "CASE_TARGET1_N3_TARGET_ATTN_MAINLINE_FLAGS"
if ($target0Line -ge $target1Line) {
    throw "target0 baseline markers do not appear before target1 markers"
}
Add-Content -Path $logPath -Value ("ordering: target0_line={0} target1_line={1}" -f $target0Line, $target1Line) -Encoding UTF8

@(
    "status: PASS",
    ("repo_root: {0}" -f $repo),
    ("run_log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $runLogAbs)),
    ("checker_log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $logPath)),
    "scope: local-only",
    "closure: not Catapult closure; not SCVerify closure",
    "banner: PASS: check_target_layer_attention_multilayer_acceptance"
) | Set-Content -Path $summaryPath -Encoding UTF8

Add-Content -Path $logPath -Value "PASS: check_target_layer_attention_multilayer_acceptance" -Encoding UTF8
Write-Host "PASS: check_target_layer_attention_multilayer_acceptance"
