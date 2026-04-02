param(
    [string]$RepoRoot = ".",
    [string]$RunLog = "build\\lid0_attention_mainline_acceptance\\p11aj\\run.log",
    [string]$OutDir = "build\\lid0_attention_mainline_acceptance\\checker"
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

$repo = [System.IO.Path]::GetFullPath((Resolve-Path $RepoRoot).Path)
$runLogAbs = Join-RepoPath -RepoRootPath $repo -Path $RunLog
$outDirAbs = Join-RepoPath -RepoRootPath $repo -Path $OutDir
New-Item -ItemType Directory -Force -Path $outDirAbs > $null

$logPath = Join-Path $outDirAbs "check_lid0_attention_mainline_acceptance.log"
$summaryPath = Join-Path $outDirAbs "check_lid0_attention_mainline_acceptance_summary.txt"
Set-Content -Path $logPath -Value "===== check_lid0_attention_mainline_acceptance =====" -Encoding UTF8

if (-not (Test-Path $runLogAbs -PathType Leaf)) {
    throw "run log not found: $runLogAbs"
}

$requiredLines = @(
    "LID0_ATTN_MAINLINE_FLAGS p11ad_mainline_q_path_taken=1 p11ac_mainline_path_taken=1 p11ae_mainline_score_path_taken=1 p11af_mainline_softmax_output_path_taken=1",
    "LID0_ATTN_FALLBACK_FLAGS p11ad_q_fallback_taken=0 p11ac_fallback_taken=0 p11ae_score_fallback_taken=0 p11af_softmax_output_fallback_taken=0",
    "LID0_ATTN_STAGE_AD_MAINLINE_TAKEN PASS",
    "LID0_ATTN_STAGE_AC_MAINLINE_TAKEN PASS",
    "LID0_ATTN_STAGE_AE_MAINLINE_TAKEN PASS",
    "LID0_ATTN_STAGE_AF_MAINLINE_TAKEN PASS",
    "LID0_ATTN_STAGE_AD_FALLBACK_NOT_TAKEN PASS",
    "LID0_ATTN_STAGE_AC_FALLBACK_NOT_TAKEN PASS",
    "LID0_ATTN_STAGE_AE_FALLBACK_NOT_TAKEN PASS",
    "LID0_ATTN_STAGE_AF_FALLBACK_NOT_TAKEN PASS",
    "LID0_ATTN_DIRECT_SRAM_FALLBACK_NOT_TAKEN PASS",
    "FULL_LOOP_MAINLINE_PATH_TAKEN PASS",
    "FULL_LOOP_FALLBACK_NOT_TAKEN PASS",
    "PASS: tb_top_managed_sram_provenance_p11aj"
)

foreach ($line in $requiredLines) {
    Require-Line -Path $runLogAbs -Needle $line
    Add-Content -Path $logPath -Value ("verified: {0}" -f $line) -Encoding UTF8
}

@(
    "status: PASS",
    ("repo_root: {0}" -f $repo),
    ("run_log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $runLogAbs)),
    ("checker_log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $logPath)),
    "scope: local-only",
    "closure: not Catapult closure; not SCVerify closure",
    "banner: PASS: check_lid0_attention_mainline_acceptance"
) | Set-Content -Path $summaryPath -Encoding UTF8

Add-Content -Path $logPath -Value "PASS: check_lid0_attention_mainline_acceptance" -Encoding UTF8
Write-Host "PASS: check_lid0_attention_mainline_acceptance"
