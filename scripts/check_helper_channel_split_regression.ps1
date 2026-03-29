param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\helper_channel_guard"
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

$repo = [System.IO.Path]::GetFullPath((Resolve-Path $RepoRoot).Path)
$outDirAbs = Join-RepoPath -RepoRootPath $repo -Path $OutDir
New-Item -ItemType Directory -Force -Path $outDirAbs > $null

$logPath = Join-Path $outDirAbs "check_helper_channel_split_regression.log"
$summaryPath = Join-Path $outDirAbs "check_helper_channel_split_regression_summary.txt"
Set-Content -Path $logPath -Value "===== check_helper_channel_split_regression =====" -Encoding UTF8

function Write-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $logPath -Value $Message -Encoding UTF8
}

function Write-Summary {
    param(
        [string]$Status,
        [string]$Detail
    )
    @(
        ("status: {0}" -f $Status),
        ("repo_root: {0}" -f $repo),
        ("out_dir: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $outDirAbs)),
        ("log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $logPath)),
        ("detail: {0}" -f $Detail)
    ) | Set-Content -Path $summaryPath -Encoding UTF8
}

function Fail-Check {
    param([string]$Reason)
    Write-Log "FAIL: check_helper_channel_split_regression"
    Write-Log $Reason
    Write-Summary -Status "FAIL" -Detail $Reason
    exit 1
}

function Require-True {
    param(
        [bool]$Condition,
        [string]$Reason
    )
    if (-not $Condition) {
        Fail-Check $Reason
    }
}

function Require-Regex {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Reason
    )
    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($Text, $Pattern)) {
        Fail-Check $Reason
    }
}

function Forbid-Regex {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Reason
    )
    if ([System.Text.RegularExpressions.Regex]::IsMatch($Text, $Pattern)) {
        Fail-Check $Reason
    }
}

$targets = @(
    "src/blocks/AttnPhaseBTopManagedSoftmaxOut.h",
    "src/blocks/AttnPhaseBTopManagedQkScore.h",
    "src/blocks/AttnPhaseATopManagedQ.h",
    "src/blocks/AttnPhaseATopManagedKv.h",
    "tb/tb_q_path_impl_p11ad.cpp",
    "tb/tb_kv_build_stream_stage_p11ac.cpp"
)
foreach ($rel in $targets) {
    Require-True -Condition (Test-Path (Join-Path $repo $rel)) -Reason ("required file missing: {0}" -f $rel)
}

$afPath = Join-Path $repo "src/blocks/AttnPhaseBTopManagedSoftmaxOut.h"
$aePath = Join-Path $repo "src/blocks/AttnPhaseBTopManagedQkScore.h"
$adPath = Join-Path $repo "src/blocks/AttnPhaseATopManagedQ.h"
$acPath = Join-Path $repo "src/blocks/AttnPhaseATopManagedKv.h"
$tbAdPath = Join-Path $repo "tb/tb_q_path_impl_p11ad.cpp"
$tbAcPath = Join-Path $repo "tb/tb_kv_build_stream_stage_p11ac.cpp"

$afText = Get-Content -Path $afPath -Raw
$aeText = Get-Content -Path $aePath -Raw
$adText = Get-Content -Path $adPath -Raw
$acText = Get-Content -Path $acPath -Raw
$tbAdText = Get-Content -Path $tbAdPath -Raw
$tbAcText = Get-Content -Path $tbAcPath -Raw

# AF guard: score + v must not collapse back into a shared helper input channel.
Require-Regex -Text $afText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedWorkPacket>\s+attn_phaseb_softmax_score_ch_t\s*;' -Reason "AF split typedef (score channel) missing"
Require-Regex -Text $afText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedWorkPacket>\s+attn_phaseb_softmax_v_ch_t\s*;' -Reason "AF split typedef (v channel) missing"
Require-Regex -Text $afText -Pattern 'attn_phaseb_block_softmax_out_consume_emit\s*\(\s*attn_phaseb_softmax_score_ch_t&\s*score_ch\s*,\s*attn_phaseb_softmax_v_ch_t&\s*v_ch\s*,' -Reason "AF consume signature must keep score/v split channels"
Require-Regex -Text $afText -Pattern '\bscore_ch\.nb_read\s*\(\s*score_pkt\s*\)' -Reason "AF consume path must read score from score_ch"
Require-Regex -Text $afText -Pattern '\bv_ch\.nb_read\s*\(\s*v_pkt\s*\)' -Reason "AF consume path must read V from v_ch"
Forbid-Regex -Text $afText -Pattern 'attn_phaseb_block_softmax_out_consume_emit\s*\(\s*attn_phaseb_softmax_pkt_ch_t&\s*in_ch' -Reason "AF regression detected: shared in_ch consume signature restored"

# AE guard: q + k must not collapse back into a shared helper input channel.
Require-Regex -Text $aeText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedWorkPacket>\s+attn_phaseb_q_pkt_ch_t\s*;' -Reason "AE split typedef (q channel) missing"
Require-Regex -Text $aeText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedWorkPacket>\s+attn_phaseb_k_pkt_ch_t\s*;' -Reason "AE split typedef (k channel) missing"
Require-Regex -Text $aeText -Pattern 'attn_phaseb_block_qk_dot_consume_emit\s*\(\s*attn_phaseb_q_pkt_ch_t&\s*q_ch\s*,\s*attn_phaseb_k_pkt_ch_t&\s*k_ch\s*,' -Reason "AE consume signature must keep q/k split channels"
Require-Regex -Text $aeText -Pattern '\bq_ch\.nb_read\s*\(\s*q_pkt\s*\)' -Reason "AE consume path must read Q from q_ch"
Require-Regex -Text $aeText -Pattern '\bk_ch\.nb_read\s*\(\s*k_pkt\s*\)' -Reason "AE consume path must read K from k_ch"
Forbid-Regex -Text $aeText -Pattern 'attn_phaseb_block_qk_dot_consume_emit\s*\(\s*attn_phaseb_qk_pkt_ch_t&\s*in_ch' -Reason "AE regression detected: shared in_ch consume signature restored"

# AD guard: x + wq must not collapse back into a shared helper input channel.
Require-Regex -Text $adText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedPacket>\s+attn_q_x_pkt_ch_t\s*;' -Reason "AD legacy split typedef (x packet channel) missing"
Require-Regex -Text $adText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedPacket>\s+attn_q_wq_pkt_ch_t\s*;' -Reason "AD legacy split typedef (wq packet channel) missing"
Require-Regex -Text $adText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedWorkPacket>\s+attn_q_x_work_pkt_ch_t\s*;' -Reason "AD split typedef (x channel) missing"
Require-Regex -Text $adText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedWorkPacket>\s+attn_q_wq_work_pkt_ch_t\s*;' -Reason "AD split typedef (wq channel) missing"
Require-Regex -Text $adText -Pattern 'attn_top_emit_phasea_q_work_unit\s*\(\s*const\s+SramView&\s+sram\s*,\s*u32_t\s+x_row_base_word\s*,\s*u32_t\s+token_idx\s*,\s*u32_t\s+d_tile_idx\s*,\s*attn_q_x_pkt_ch_t&\s*x_ch\s*,\s*attn_q_wq_pkt_ch_t&\s*wq_ch' -Reason "AD legacy emit signature must keep x/wq split packet channels"
Require-Regex -Text $adText -Pattern 'attn_block_phasea_q_consume_emit\s*\(\s*attn_q_x_pkt_ch_t&\s*x_ch\s*,\s*attn_q_wq_pkt_ch_t&\s*wq_ch\s*,\s*attn_q_pkt_ch_t&\s*out_ch' -Reason "AD legacy consume signature must keep x/wq split packet channels"
Require-Regex -Text $adText -Pattern 'attn_block_phasea_q_consume_emit_token_work_tiles\s*\(\s*attn_q_x_work_pkt_ch_t&\s*x_ch\s*,\s*attn_q_wq_work_pkt_ch_t&\s*wq_ch\s*,' -Reason "AD consume signature must keep x/wq split channels"
Require-Regex -Text $adText -Pattern '\bx_ch\.nb_read\s*\(\s*x_pkt\s*\)' -Reason "AD consume path must read X from x_ch"
Require-Regex -Text $adText -Pattern '\bwq_ch\.nb_read\s*\(\s*wq_pkt\s*\)' -Reason "AD consume path must read WQ from wq_ch"
Forbid-Regex -Text $adText -Pattern 'attn_top_emit_phasea_q_work_unit\s*\([^\)]*attn_q_pkt_ch_t&\s*in_ch' -Reason "AD legacy regression detected: single in_ch emit signature restored"
Forbid-Regex -Text $adText -Pattern 'attn_block_phasea_q_consume_emit\s*\(\s*attn_q_pkt_ch_t&\s*in_ch\s*,\s*attn_q_pkt_ch_t&\s*out_ch' -Reason "AD legacy regression detected: shared in_ch consume signature restored"
Forbid-Regex -Text $adText -Pattern 'attn_block_phasea_q_consume_emit_token_work_tiles\s*\(\s*attn_q_work_pkt_ch_t&\s*in_ch' -Reason "AD regression detected: shared in_ch consume signature restored"

Require-Regex -Text $tbAdText -Pattern '\baecct::attn_q_x_pkt_ch_t\s+x_ch\s*;' -Reason "AD TB must instantiate legacy split x packet channel"
Require-Regex -Text $tbAdText -Pattern '\baecct::attn_q_wq_pkt_ch_t\s+wq_ch\s*;' -Reason "AD TB must instantiate legacy split wq packet channel"
Require-Regex -Text $tbAdText -Pattern 'LEGACY_WORK_UNIT_SPLIT_PATH PASS' -Reason "AD TB must report legacy split path PASS banner"

# AC guard: x + wk + wv must not collapse back into a shared helper input channel.
Require-Regex -Text $acText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedWorkPacket>\s+attn_x_work_pkt_ch_t\s*;' -Reason "AC split typedef (x channel) missing"
Require-Regex -Text $acText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedWorkPacket>\s+attn_wk_work_pkt_ch_t\s*;' -Reason "AC split typedef (wk channel) missing"
Require-Regex -Text $acText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedWorkPacket>\s+attn_wv_work_pkt_ch_t\s*;' -Reason "AC split typedef (wv channel) missing"
Require-Regex -Text $acText -Pattern 'attn_block_phasea_kv_consume_emit_token_work_tiles\s*\(\s*attn_x_work_pkt_ch_t&\s*x_ch\s*,\s*attn_wk_work_pkt_ch_t&\s*wk_ch\s*,\s*attn_wv_work_pkt_ch_t&\s*wv_ch\s*,' -Reason "AC consume signature must keep x/wk/wv split channels"
Require-Regex -Text $acText -Pattern '\bx_ch\.nb_read\s*\(\s*x_pkt\s*\)' -Reason "AC consume path must read X from x_ch"
Require-Regex -Text $acText -Pattern '\bwk_ch\.nb_read\s*\(\s*wk_pkt\s*\)' -Reason "AC consume path must read WK from wk_ch"
Require-Regex -Text $acText -Pattern '\bwv_ch\.nb_read\s*\(\s*wv_pkt\s*\)' -Reason "AC consume path must read WV from wv_ch"
Forbid-Regex -Text $acText -Pattern 'attn_block_phasea_kv_consume_emit_token_work_tiles\s*\(\s*attn_work_pkt_ch_t&\s*in_ch' -Reason "AC regression detected: shared in_ch consume signature restored"

# AC legacy work-unit guard: split helper-only channel classes to avoid mixed-payload HOL risk.
Require-Regex -Text $acText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedPacket>\s+attn_x_pkt_ch_t\s*;' -Reason "AC legacy split typedef (x channel) missing"
Require-Regex -Text $acText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedPacket>\s+attn_wk_pkt_ch_t\s*;' -Reason "AC legacy split typedef (wk channel) missing"
Require-Regex -Text $acText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedPacket>\s+attn_wv_pkt_ch_t\s*;' -Reason "AC legacy split typedef (wv channel) missing"
Require-Regex -Text $acText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedPacket>\s+attn_k_pkt_ch_t\s*;' -Reason "AC legacy split typedef (k channel) missing"
Require-Regex -Text $acText -Pattern '(?m)^\s*typedef\s+ac_channel<AttnTopManagedPacket>\s+attn_v_pkt_ch_t\s*;' -Reason "AC legacy split typedef (v channel) missing"
Require-Regex -Text $acText -Pattern 'attn_top_emit_phasea_kv_work_unit\s*\(\s*const\s+SramView&\s+sram\s*,\s*u32_t\s+x_row_base_word\s*,\s*u32_t\s+token_idx\s*,\s*u32_t\s+d_tile_idx\s*,\s*attn_x_pkt_ch_t&\s*x_ch\s*,\s*attn_wk_pkt_ch_t&\s*wk_ch\s*,\s*attn_wv_pkt_ch_t&\s*wv_ch' -Reason "AC legacy emit signature must keep x/wk/wv split channels"
Require-Regex -Text $acText -Pattern 'attn_block_phasea_kv_consume_emit\s*\(\s*attn_x_pkt_ch_t&\s*x_ch\s*,\s*attn_wk_pkt_ch_t&\s*wk_ch\s*,\s*attn_wv_pkt_ch_t&\s*wv_ch\s*,\s*attn_k_pkt_ch_t&\s*k_ch\s*,\s*attn_v_pkt_ch_t&\s*v_ch' -Reason "AC legacy consume signature must keep split in/out channels"
Require-Regex -Text $acText -Pattern 'attn_top_writeback_phasea_kv_work_unit\s*\(\s*SramView&\s+sram\s*,\s*u32_t\s+scr_k_row_base_word\s*,\s*u32_t\s+scr_v_row_base_word\s*,\s*u32_t\s+token_idx\s*,\s*u32_t\s+d_tile_idx\s*,\s*attn_k_pkt_ch_t&\s*k_ch\s*,\s*attn_v_pkt_ch_t&\s*v_ch' -Reason "AC legacy writeback signature must keep k/v split channels"
Forbid-Regex -Text $acText -Pattern 'attn_top_emit_phasea_kv_work_unit\s*\([^\)]*attn_pkt_ch_t&\s*in_ch' -Reason "AC legacy regression detected: single in_ch emit signature restored"
Forbid-Regex -Text $acText -Pattern 'attn_block_phasea_kv_consume_emit\s*\(\s*attn_pkt_ch_t&\s*in_ch\s*,\s*attn_pkt_ch_t&\s*out_ch' -Reason "AC legacy regression detected: shared in_ch/out_ch consume signature restored"
Forbid-Regex -Text $acText -Pattern 'attn_top_writeback_phasea_kv_work_unit\s*\([^\)]*attn_pkt_ch_t&\s*out_ch' -Reason "AC legacy regression detected: shared out_ch writeback signature restored"

Require-Regex -Text $tbAcText -Pattern '\baecct::attn_x_pkt_ch_t\s+x_ch_' -Reason "AC TB must declare split x channel"
Require-Regex -Text $tbAcText -Pattern '\baecct::attn_wk_pkt_ch_t\s+wk_ch_' -Reason "AC TB must declare split wk channel"
Require-Regex -Text $tbAcText -Pattern '\baecct::attn_wv_pkt_ch_t\s+wv_ch_' -Reason "AC TB must declare split wv channel"
Require-Regex -Text $tbAcText -Pattern '\baecct::attn_k_pkt_ch_t\s+k_ch_' -Reason "AC TB must declare split k channel"
Require-Regex -Text $tbAcText -Pattern '\baecct::attn_v_pkt_ch_t\s+v_ch_' -Reason "AC TB must declare split v channel"

Write-Log "guard: AF score/v split anchors OK"
Write-Log "guard: AE q/k split anchors OK"
Write-Log "guard: AD x/wq split anchors OK"
Write-Log "guard: AD legacy work-unit split anchors OK"
Write-Log "guard: AC x/wk/wv split anchors OK"
Write-Log "guard: AC legacy work-unit split anchors OK"
Write-Log "PASS: check_helper_channel_split_regression"
Write-Summary -Status "PASS" -Detail "all helper split regression guards passed"
exit 0
