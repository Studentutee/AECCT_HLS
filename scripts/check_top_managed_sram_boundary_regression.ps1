param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\\top_managed_sram_guard"
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

$logPath = Join-Path $outDirAbs "check_top_managed_sram_boundary_regression.log"
$summaryPath = Join-Path $outDirAbs "check_top_managed_sram_boundary_regression_summary.txt"
Set-Content -Path $logPath -Value "===== check_top_managed_sram_boundary_regression =====" -Encoding UTF8

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
    Write-Log "FAIL: check_top_managed_sram_boundary_regression"
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

$topPath = Join-Path $repo "src/Top.h"
$layerPath = Join-Path $repo "src/blocks/TransformerLayer.h"
$phaseAKVPath = Join-Path $repo "src/blocks/AttnPhaseATopManagedKv.h"
$phaseAQPath = Join-Path $repo "src/blocks/AttnPhaseATopManagedQ.h"
$qkScorePath = Join-Path $repo "src/blocks/AttnPhaseBTopManagedQkScore.h"
$softmaxOutPath = Join-Path $repo "src/blocks/AttnPhaseBTopManagedSoftmaxOut.h"
Require-True -Condition (Test-Path $topPath) -Reason "required file missing: src/Top.h"
Require-True -Condition (Test-Path $layerPath) -Reason "required file missing: src/blocks/TransformerLayer.h"
Require-True -Condition (Test-Path $phaseAKVPath) -Reason "required file missing: src/blocks/AttnPhaseATopManagedKv.h"
Require-True -Condition (Test-Path $phaseAQPath) -Reason "required file missing: src/blocks/AttnPhaseATopManagedQ.h"
Require-True -Condition (Test-Path $qkScorePath) -Reason "required file missing: src/blocks/AttnPhaseBTopManagedQkScore.h"
Require-True -Condition (Test-Path $softmaxOutPath) -Reason "required file missing: src/blocks/AttnPhaseBTopManagedSoftmaxOut.h"

$topText = Get-Content -Path $topPath -Raw
$layerText = Get-Content -Path $layerPath -Raw
$phaseAKVText = Get-Content -Path $phaseAKVPath -Raw
$phaseAQText = Get-Content -Path $phaseAQPath -Raw
$qkScoreText = Get-Content -Path $qkScorePath -Raw
$softmaxOutText = Get-Content -Path $softmaxOutPath -Raw

# G4 ingest/base-shadow ownership anchors: infer ingest should be explicit Top contract dispatch.
Require-Regex -Text $topText -Pattern '(?ms)struct\s+InferIngestContract\s*\{[\s\S]*?in_base_word\s*;[\s\S]*?len_words_expected\s*;[\s\S]*?len_words_valid\s*;[\s\S]*?token_range\s*;[\s\S]*?tile_range\s*;[\s\S]*?phase_id\s*;' -Reason "InferIngestContract fields missing"
Require-Regex -Text $topText -Pattern '(?ms)struct\s+IngestMetadataSurface\s*\{[\s\S]*?owner_opcode\s*;[\s\S]*?base_word\s*;[\s\S]*?len_words_expected\s*;[\s\S]*?len_words_valid\s*;[\s\S]*?active\s*;' -Reason "cross-command IngestMetadataSurface fields missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+IngestMetadataSurface\s+cfg_metadata_surface\s*\(\s*const\s+TopRegs&\s+regs\s*\)' -Reason "cfg_metadata_surface helper missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+IngestMetadataSurface\s+param_metadata_surface\s*\(\s*const\s+TopRegs&\s+regs\s*\)' -Reason "param_metadata_surface helper missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+IngestMetadataSurface\s+infer_metadata_surface\s*\(\s*const\s+TopRegs&\s+regs\s*\)' -Reason "infer_metadata_surface helper missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+bool\s+ingest_meta_span_in_sram\s*\(\s*const\s+IngestMetadataSurface&\s+m\s*,\s*uint32_t\s+fallback_words\s*\)' -Reason "ingest_meta_span_in_sram helper missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+bool\s+ingest_meta_owner_matches_rx\s*\(\s*const\s+IngestMetadataSurface&\s+m\s*,\s*ReceiverState\s+rx\s*\)' -Reason "ingest_meta_owner_matches_rx helper missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+bool\s+ingest_meta_len_exact\s*\(\s*const\s+IngestMetadataSurface&\s+m\s*,\s*uint32_t\s+fallback_words\s*\)' -Reason "ingest_meta_len_exact helper missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+uint8_t\s+ingest_commit_diag_error\s*\(\s*const\s+IngestMetadataSurface&\s+m\s*,\s*ReceiverState\s+rx\s*,\s*uint32_t\s+fallback_words\s*,\s*uint8_t\s+len_mismatch_err\s*,\s*bool\s+require_span_check\s*\)' -Reason "ingest_commit_diag_error helper missing"
Require-Regex -Text $topText -Pattern '(?ms)struct\s+AcceptedCommitMetadataRecord\s*\{[\s\S]*?owner_opcode\s*;[\s\S]*?base_word\s*;[\s\S]*?len_words_expected\s*;[\s\S]*?len_words_valid\s*;[\s\S]*?rx_state\s*;[\s\S]*?phase_id\s*;[\s\S]*?phase_valid\s*;[\s\S]*?valid\s*;' -Reason "AcceptedCommitMetadataRecord fields missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+AcceptedCommitMetadataRecord\s+make_invalid_accepted_commit_metadata_record\s*\(\s*\)' -Reason "make_invalid_accepted_commit_metadata_record helper missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+void\s+record_accepted_commit_metadata\s*\(\s*AcceptedCommitMetadataRecord&\s+r\s*,\s*const\s+IngestMetadataSurface&\s+m\s*,\s*ReceiverState\s+rx\s*,\s*u32_t\s+phase_id\s*,\s*bool\s+phase_valid\s*\)' -Reason "record_accepted_commit_metadata helper missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+uint8_t\s+ingest_commit_diag_and_record\s*\(\s*AcceptedCommitMetadataRecord&\s+record\s*,\s*const\s+IngestMetadataSurface&\s+m\s*,\s*ReceiverState\s+rx\s*,\s*uint32_t\s+fallback_words\s*,\s*uint8_t\s+len_mismatch_err\s*,\s*bool\s+require_span_check\s*,\s*u32_t\s+phase_id\s*,\s*bool\s+phase_valid\s*\)' -Reason "ingest_commit_diag_and_record helper missing"
Require-Regex -Text $topText -Pattern '(?ms)infer_refresh_preproc_ranges\s*\(\s*InferIngestContract&\s+c\s*,\s*uint32_t\s+x_out_words\s*\)' -Reason "infer_refresh_preproc_ranges helper missing"
Require-Regex -Text $topText -Pattern '(?ms)clear_infer_ingest_contract\s*\(\s*regs\.infer_ingest_contract\s*\)\s*;' -Reason "infer_session_clear must reset infer ingest contract"
Require-Regex -Text $topText -Pattern '(?ms)run_preproc_block[\s\S]*?regs\.infer_ingest_contract\.len_words_valid[\s\S]*?regs\.infer_ingest_contract\.in_base_word[\s\S]*?contract\.phase_id\s*=\s*regs\.infer_ingest_contract\.phase_id[\s\S]*?contract\.token_range\s*=\s*regs\.infer_ingest_contract\.token_range[\s\S]*?contract\.tile_range\s*=\s*regs\.infer_ingest_contract\.tile_range' -Reason "run_preproc_block must consume infer ingest contract len/base/phase/range"
Require-Regex -Text $topText -Pattern '(?ms)run_preproc_block[\s\S]*?u32_t\s+topfed_in_payload\s*\[\s*PREPROC_IN_WORDS_EXPECTED\s*\][\s\S]*?PREPROC_TOPFED_INPUT_PRELOAD_LOOP[\s\S]*?PreprocEmbedSPECoreWindow<\s*u32_t\*\s*>\s*\([\s\S]*?topfed_in_payload\s*\)' -Reason "run_preproc_block must preload and pass top-fed infer payload window"
Require-Regex -Text $topText -Pattern '(?ms)run_infer_pipeline[\s\S]*?infer_label_words_view\s*\(\s*regs\s*,\s*sram\s*\)' -Reason "run_infer_pipeline must source FinalHead labels from Top-managed SRAM ingest view"
Require-Regex -Text $topText -Pattern '(?ms)infer_ingest_one_word[\s\S]*?infer_metadata_surface\s*\(\s*regs\s*\)[\s\S]*?ingest_meta_expected_words\s*\(\s*meta\s*,\s*\(uint32_t\)INFER_IN_WORDS_EXPECTED\s*\)[\s\S]*?infer_store_one_word\s*\(\s*regs\s*,\s*sram\s*,\s*idx\s*,\s*w\s*\)' -Reason "infer_ingest_one_word must dispatch via infer metadata surface helpers"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+bool\s+infer_contract_span_in_sram\s*\(\s*const\s+InferIngestContract&\s+c\s*\)' -Reason "infer_contract_span_in_sram helper missing"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+void\s+infer_contract_arm_for_op_infer\s*\(\s*TopRegs&\s+regs\s*\)' -Reason "infer_contract_arm_for_op_infer helper missing"
Require-Regex -Text $topText -Pattern '(?ms)else\s+if\s*\(\s*op\s*==\s*\(uint8_t\)OP_INFER\s*\)[\s\S]*?infer_contract_arm_for_op_infer\s*\(\s*regs\s*\)\s*;[\s\S]*?infer_metadata_surface\s*\(\s*regs\s*\)[\s\S]*?ingest_meta_span_in_sram\s*\(\s*infer_meta\s*,\s*\(uint32_t\)INFER_IN_WORDS_EXPECTED\s*\)' -Reason "OP_INFER entry must validate infer ingest via metadata surface helper"
Require-Regex -Text $topText -Pattern '(?ms)else\s+if\s*\(\s*op\s*==\s*\(uint8_t\)OP_INFER\s*\)[\s\S]*?if\s*\(\s*!ingest_meta_span_in_sram\s*\(\s*infer_meta\s*,\s*\(uint32_t\)INFER_IN_WORDS_EXPECTED\s*\)\s*\)\s*\{\s*ctrl_rsp\.write\s*\(\s*pack_ctrl_rsp_err\s*\(\s*\(uint8_t\)ERR_MEM_RANGE\s*\)\s*\)\s*;[\s\S]*?\}\s*else\s*\{\s*regs\.state\s*=\s*ST_INFER_RX\s*;\s*ctrl_rsp\.write\s*\(\s*pack_ctrl_rsp_ok\s*\(\s*\(uint8_t\)OP_INFER\s*\)\s*\)\s*;' -Reason "OP_INFER preflight reject/accept response contract missing"
Require-Regex -Text $topText -Pattern '(?ms)else\s+if\s*\(\s*op\s*==\s*\(uint8_t\)OP_LOAD_W\s*\)[\s\S]*?param_ingest_span_legal\s*\(\s*regs\s*\)' -Reason "OP_LOAD_W must validate span via metadata harmonized helper"
Require-Regex -Text $topText -Pattern '(?ms)if\s*\(\s*op\s*==\s*\(uint8_t\)OP_CFG_COMMIT\s*\)\s*\{[\s\S]*?ingest_commit_diag_error\s*\([\s\S]*?RX_CFG[\s\S]*?ERR_CFG_LEN_MISMATCH' -Reason "CFG commit-time diagnostics must use harmonized helper and mismatch mapping"
Require-Regex -Text $topText -Pattern '(?ms)param_ingest_one_word[\s\S]*?ingest_commit_diag_and_record\s*\(\s*regs\.accepted_commit_record\s*,[\s\S]*?RX_PARAM[\s\S]*?ERR_PARAM_LEN_MISMATCH' -Reason "PARAM commit-time diagnostics + accepted record harmonization missing"
Require-Regex -Text $topText -Pattern '(?ms)infer_ingest_one_word[\s\S]*?ingest_commit_diag_and_record\s*\(\s*regs\.accepted_commit_record\s*,[\s\S]*?RX_INFER' -Reason "INFER commit-time diagnostics + accepted record harmonization missing"
Require-Regex -Text $topText -Pattern '(?ms)else\s+if\s*\(\s*op\s*==\s*\(uint8_t\)OP_CFG_COMMIT\s*\)[\s\S]*?record_accepted_commit_metadata\s*\(\s*regs\.accepted_commit_record\s*,\s*cfg_meta\s*,\s*RX_CFG\s*,' -Reason "CFG accept path must update accepted commit metadata record"
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+bool\s+top_peek_accepted_commit_record_valid\s*\(\s*\)' -Reason "accepted commit record peek helper missing"
Forbid-Regex -Text $topText -Pattern '(?ms)infer_ingest_one_word[\s\S]{0,900}\bsram\s*\[\s*IN_BASE_WORD\s*\+\s*idx\s*\]\s*=\s*w\s*;' -Reason "infer_ingest_one_word regressed to hardcoded IN_BASE_WORD direct write"
Forbid-Regex -Text $topText -Pattern '(?ms)run_infer_pipeline[\s\S]{0,1800}regs\.infer_input_shadow' -Reason "run_infer_pipeline regressed to shadow-array label source"

# Top-owned contract dispatch anchors: preproc/layernorm/final head.
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+void\s+run_preproc_block\s*\(\s*TopRegs&\s*regs\s*,\s*u32_t\*\s*sram\s*\)' -Reason "Top preproc dispatch must take TopRegs& for contract ownership"
Require-Regex -Text $topText -Pattern '(?ms)PreprocBlockContract&\s+contract\s*=\s*regs\.preproc_contract\s*;' -Reason "Top preproc contract ownership anchor missing"
Require-Regex -Text $topText -Pattern '(?ms)PreprocEmbedSPECoreWindow<\s*u32_t\*\s*>\s*\([\s\S]*?contract' -Reason "Top preproc path must dispatch PreprocEmbedSPECoreWindow with Top-owned contract"
Forbid-Regex -Text $topText -Pattern '(?ms)run_preproc_block\s*\(\s*TopRegs&\s*regs\s*,\s*u32_t\*\s*sram\s*\)\s*\{[\s\S]{0,2600}\bPreprocEmbedSPE\s*\(' -Reason "Top preproc dispatch regressed to wrapper-owned contract path"

Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+void\s+run_layernorm_block\s*\(\s*TopRegs&\s*regs\s*,\s*u32_t\*\s*sram\s*\)' -Reason "Top layernorm dispatch must take TopRegs& for contract ownership"
Require-Regex -Text $topText -Pattern '(?ms)LayerNormBlockContract&\s+contract\s*=\s*regs\.layernorm_contract\s*;' -Reason "Top layernorm contract ownership anchor missing"
Require-Regex -Text $topText -Pattern '(?ms)LayerNormBlockCoreWindow<\s*u32_t\*\s*>\s*\([\s\S]*?contract' -Reason "Top layernorm path must dispatch LayerNormBlockCoreWindow with Top-owned contract"
Forbid-Regex -Text $topText -Pattern '(?ms)run_layernorm_block\s*\(\s*TopRegs&\s*regs\s*,\s*u32_t\*\s*sram\s*\)\s*\{[\s\S]{0,2800}\bLayerNormBlock\s*\(' -Reason "Top layernorm dispatch regressed to wrapper-owned contract path"

Require-Regex -Text $topText -Pattern '(?ms)run_infer_pipeline[\s\S]*?run_preproc_block\s*\(\s*regs\s*,\s*sram\s*\)\s*;' -Reason "run_infer_pipeline must dispatch Top-owned preproc contract path"
Require-Regex -Text $topText -Pattern '(?ms)run_infer_pipeline[\s\S]*?run_layernorm_block\s*\(\s*regs\s*,\s*sram\s*\)\s*;' -Reason "run_infer_pipeline must dispatch Top-owned layernorm contract path"
Require-Regex -Text $topText -Pattern '(?ms)FinalHeadContract&\s+contract\s*=\s*regs\.final_head_contract\s*;' -Reason "Top final-head contract ownership anchor missing"
Require-Regex -Text $topText -Pattern '(?ms)FinalHeadCorePassABTopManaged<\s*u32_t\*\s*>\s*\([\s\S]*?contract' -Reason "Top final-head path must dispatch FinalHeadCorePassABTopManaged with Top-owned contract"
Require-Regex -Text $topText -Pattern '(?ms)run_infer_pipeline[\s\S]*?u32_t\s+topfed_final_scalar_words\s*\[\s*N_NODES\s*\][\s\S]*?TOPFED_FINAL_SCALAR_PRELOAD_LOOP[\s\S]*?FinalHeadCorePassABTopManaged<\s*u32_t\*\s*>\s*\([\s\S]*?topfed_final_scalar_words\s*\)' -Reason "run_infer_pipeline must preload and pass top-fed FinalHead scalar payload"
Forbid-Regex -Text $topText -Pattern '(?ms)run_infer_pipeline[\s\S]{0,2200}\bFinalHead\s*\(' -Reason "run_infer_pipeline regressed to wrapper-owned FinalHead path"

# Top preloads sublayer1 norm params before TransformerLayer dispatch.
Require-Regex -Text $topText -Pattern '(?ms)static\s+inline\s+void\s+top_preload_layer_sublayer1_norm_params\s*\(' -Reason "Top preload helper for sublayer1 norm params missing"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop[\s\S]*?top_preload_layer_sublayer1_norm_params\s*\(' -Reason "Top main loop must preload sublayer1 norm params"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?top_preload_layer_sublayer1_norm_params\s*\(' -Reason "Top bridge loop must preload sublayer1 norm params"
Require-Regex -Text $topText -Pattern '(?ms)TransformerLayer\s*\([\s\S]*?out_prebuilt_from_top_managed\s*,\s*true\s*\)' -Reason "Top main loop must signal sublayer1_norm_preloaded_by_top=true"
Require-Regex -Text $topText -Pattern '(?ms)TransformerLayerTopManagedAttnBridge\s*\([\s\S]*?out_prebuilt_from_top_managed\s*,\s*true\s*\)' -Reason "Top bridge loop must signal sublayer1_norm_preloaded_by_top=true"

# TransformerLayer consumes Top preload with guarded fallback.
Require-Regex -Text $layerText -Pattern '(?ms)TransformerLayerTopManagedAttnBridge\s*\([\s\S]*?bool\s+sublayer1_norm_preloaded_by_top\s*=\s*false' -Reason "TransformerLayerTopManagedAttnBridge preload flag missing"
Require-Regex -Text $layerText -Pattern '(?ms)TransformerLayer\s*\([\s\S]*?bool\s+sublayer1_norm_preloaded_by_top\s*=\s*false' -Reason "TransformerLayer preload flag missing"
Require-Regex -Text $layerText -Pattern '(?ms)if\s*\(\s*!sublayer1_norm_preloaded_by_top\s*\)\s*\{[\s\S]*?load_layer_sublayer1_norm_params\s*\(\s*sram_window' -Reason "TransformerLayerTopManagedAttnBridge fallback load guard missing"
Require-Regex -Text $layerText -Pattern '(?ms)if\s*\(\s*!sublayer1_norm_preloaded_by_top\s*\)\s*\{[\s\S]*?load_layer_sublayer1_norm_params\s*\(\s*sram\s*,' -Reason "TransformerLayer fallback load guard missing"
Require-Regex -Text $topText -Pattern '(?ms)run_p11ac_layer0_top_managed_kv\s*\([\s\S]*?phase_entry_probe_x_base_word[\s\S]*?phase_entry_probe_x_words[\s\S]*?phase_entry_probe_x_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Top run_p11ac helper missing W4-M3 KV phase-entry probe arguments"
Require-Regex -Text $topText -Pattern '(?ms)run_p11ac_layer0_top_managed_kv[\s\S]*?attn_phasea_top_managed_kv_mainline\s*\([\s\S]*?phase_entry_probe_x_base_word[\s\S]*?phase_entry_probe_x_words[\s\S]*?phase_entry_probe_x_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Top run_p11ac helper must pass through W4-M3 KV phase-entry probe arguments"
Require-Regex -Text $phaseAKVText -Pattern '(?ms)attn_phasea_top_managed_kv_mainline\s*\([\s\S]*?phase_entry_probe_x_base_word[\s\S]*?phase_entry_probe_x_words[\s\S]*?phase_entry_probe_x_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Phase-A KV mainline missing W4-M3 KV phase-entry probe signature"
Require-Regex -Text $phaseAKVText -Pattern '(?ms)phase_entry_probe_enabled' -Reason "Phase-A KV mainline missing probe-enable gating anchor"
Require-Regex -Text $phaseAKVText -Pattern '(?ms)probe_valid\s*!=\s*d_model' -Reason "Phase-A KV probe must enforce descriptor-ready full-row gating"
Require-Regex -Text $phaseAKVText -Pattern '(?ms)ATTN_P11AC_PHASE_ENTRY_PROBE_COL_LOOP' -Reason "Phase-A KV mainline missing phase-entry probe loop label"
Require-Regex -Text $phaseAKVText -Pattern '(?ms)phase_entry_probe_owner_ok' -Reason "Phase-A KV mainline missing probe ownership observability anchor"
Require-Regex -Text $phaseAKVText -Pattern '(?ms)phase_entry_probe_compare_ok' -Reason "Phase-A KV mainline missing probe compare observability anchor"
Require-Regex -Text $topText -Pattern '(?ms)run_p11ad_layer0_top_managed_q\s*\([\s\S]*?phase_entry_probe_x_base_word[\s\S]*?phase_entry_probe_x_words[\s\S]*?phase_entry_probe_x_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Top run_p11ad helper missing W4-M3 phase-entry probe arguments"
Require-Regex -Text $topText -Pattern '(?ms)run_p11ad_layer0_top_managed_q[\s\S]*?attn_phasea_top_managed_q_mainline\s*\([\s\S]*?phase_entry_probe_x_base_word[\s\S]*?phase_entry_probe_x_words[\s\S]*?phase_entry_probe_x_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Top run_p11ad helper must pass through W4-M3 phase-entry probe arguments"
Require-Regex -Text $phaseAQText -Pattern '(?ms)attn_phasea_top_managed_q_mainline\s*\([\s\S]*?phase_entry_probe_x_base_word[\s\S]*?phase_entry_probe_x_words[\s\S]*?phase_entry_probe_x_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Phase-A Q mainline missing W4-M3 phase-entry probe signature"
Require-Regex -Text $phaseAQText -Pattern '(?ms)phase_entry_probe_enabled' -Reason "Phase-A Q mainline missing probe-enable gating anchor"
Require-Regex -Text $phaseAQText -Pattern '(?ms)ATTN_P11AD_PHASE_ENTRY_PROBE_COL_LOOP' -Reason "Phase-A Q mainline missing phase-entry probe loop label"
Require-Regex -Text $phaseAQText -Pattern '(?ms)phase_entry_probe_owner_ok' -Reason "Phase-A Q mainline missing probe ownership observability anchor"
Require-Regex -Text $phaseAQText -Pattern '(?ms)phase_entry_probe_compare_ok' -Reason "Phase-A Q mainline missing probe compare observability anchor"
Require-Regex -Text $topText -Pattern '(?ms)run_p11ae_layer0_top_managed_qk_score\s*\([\s\S]*?phase_entry_probe_q_base_word[\s\S]*?phase_entry_probe_k_base_word[\s\S]*?phase_entry_probe_q_words[\s\S]*?phase_entry_probe_k_words[\s\S]*?phase_entry_probe_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Top run_p11ae helper missing W4-M1 phase-entry probe arguments"
Require-Regex -Text $topText -Pattern '(?ms)run_p11ae_layer0_top_managed_qk_score[\s\S]*?attn_phaseb_top_managed_qk_score_mainline\s*\([\s\S]*?phase_entry_probe_q_base_word[\s\S]*?phase_entry_probe_k_base_word[\s\S]*?phase_entry_probe_q_words[\s\S]*?phase_entry_probe_k_words[\s\S]*?phase_entry_probe_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Top run_p11ae helper must pass through W4-M1 phase-entry probe arguments"
Require-Regex -Text $topText -Pattern '(?ms)run_p11ae_layer0_top_managed_qk_score\s*\([\s\S]*?score_tile_bridge_base_word[\s\S]*?score_tile_bridge_words[\s\S]*?score_tile_bridge_words_valid[\s\S]*?score_tile_bridge_key_begin[\s\S]*?score_tile_bridge_visible[\s\S]*?score_tile_bridge_owner_ok[\s\S]*?score_tile_bridge_consumed[\s\S]*?score_tile_bridge_compare_ok' -Reason "Top run_p11ae helper missing W4-B2 score-tile bridge arguments"
Require-Regex -Text $topText -Pattern '(?ms)run_p11ae_layer0_top_managed_qk_score[\s\S]*?attn_phaseb_top_managed_qk_score_mainline\s*\([\s\S]*?score_tile_bridge_base_word[\s\S]*?score_tile_bridge_words[\s\S]*?score_tile_bridge_words_valid[\s\S]*?score_tile_bridge_key_begin[\s\S]*?score_tile_bridge_visible[\s\S]*?score_tile_bridge_owner_ok[\s\S]*?score_tile_bridge_consumed[\s\S]*?score_tile_bridge_compare_ok' -Reason "Top run_p11ae helper must pass through W4-B2 score-tile bridge arguments"
Require-Regex -Text $qkScoreText -Pattern '(?ms)attn_phaseb_top_managed_qk_score_mainline\s*\([\s\S]*?phase_entry_probe_q_base_word[\s\S]*?phase_entry_probe_k_base_word[\s\S]*?phase_entry_probe_q_words[\s\S]*?phase_entry_probe_k_words[\s\S]*?phase_entry_probe_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "QK-score mainline missing W4-M1 phase-entry probe signature"
Require-Regex -Text $qkScoreText -Pattern '(?ms)phase_entry_probe_enabled' -Reason "QK-score mainline missing probe-enable gating anchor"
Require-Regex -Text $qkScoreText -Pattern '(?ms)ATTN_P11AE_PHASE_ENTRY_PROBE_COL_LOOP' -Reason "QK-score mainline missing phase-entry probe loop label"
Require-Regex -Text $qkScoreText -Pattern '(?ms)phase_entry_probe_owner_ok' -Reason "QK-score mainline missing probe ownership observability anchor"
Require-Regex -Text $qkScoreText -Pattern '(?ms)phase_entry_probe_compare_ok' -Reason "QK-score mainline missing probe compare observability anchor"
Require-Regex -Text $qkScoreText -Pattern '(?ms)attn_phaseb_top_managed_qk_score_mainline\s*\([\s\S]*?score_tile_bridge_base_word[\s\S]*?score_tile_bridge_words[\s\S]*?score_tile_bridge_words_valid[\s\S]*?score_tile_bridge_key_begin[\s\S]*?score_tile_bridge_visible[\s\S]*?score_tile_bridge_owner_ok[\s\S]*?score_tile_bridge_consumed[\s\S]*?score_tile_bridge_compare_ok' -Reason "QK-score mainline missing W4-B2 score-tile bridge signature"
Require-Regex -Text $qkScoreText -Pattern '(?ms)score_tile_bridge_enabled' -Reason "QK-score mainline missing W4-B2 bridge-enable anchor"
Require-Regex -Text $qkScoreText -Pattern '(?ms)score_tile_bridge_selected' -Reason "QK-score mainline missing W4-B2 bridge-select anchor"
Require-Regex -Text $qkScoreText -Pattern '(?ms)score_tile_bridge_owner_ok' -Reason "QK-score mainline missing W4-B2 bridge ownership observability anchor"
Require-Regex -Text $qkScoreText -Pattern '(?ms)score_tile_bridge_compare_ok' -Reason "QK-score mainline missing W4-B2 bridge compare observability anchor"
Require-Regex -Text $qkScoreText -Pattern '(?ms)score_tile_bridge_consumed' -Reason "QK-score mainline missing W4-B2 bridge consumed observability anchor"
Require-Regex -Text $topText -Pattern '(?ms)run_p11af_layer0_top_managed_softmax_out\s*\([\s\S]*?phase_entry_probe_v_base_word[\s\S]*?phase_entry_probe_v_words[\s\S]*?phase_entry_probe_v_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Top run_p11af helper missing W4-M2 phase-entry probe arguments"
Require-Regex -Text $topText -Pattern '(?ms)run_p11af_layer0_top_managed_softmax_out[\s\S]*?attn_phaseb_top_managed_softmax_out_mainline\s*\([\s\S]*?phase_entry_probe_v_base_word[\s\S]*?phase_entry_probe_v_words[\s\S]*?phase_entry_probe_v_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "Top run_p11af helper must pass through W4-M2 phase-entry probe arguments"
Require-Regex -Text $topText -Pattern '(?ms)run_p11af_layer0_top_managed_softmax_out\s*\([\s\S]*?phase_tile_bridge_v_base_word[\s\S]*?phase_tile_bridge_v_words[\s\S]*?phase_tile_bridge_v_words_valid[\s\S]*?phase_tile_bridge_d_tile_idx[\s\S]*?phase_tile_bridge_visible[\s\S]*?phase_tile_bridge_owner_ok[\s\S]*?phase_tile_bridge_consumed[\s\S]*?phase_tile_bridge_compare_ok' -Reason "Top run_p11af helper missing W4-B1 phase-B tile bridge arguments"
Require-Regex -Text $topText -Pattern '(?ms)run_p11af_layer0_top_managed_softmax_out[\s\S]*?attn_phaseb_top_managed_softmax_out_mainline\s*\([\s\S]*?phase_tile_bridge_v_base_word[\s\S]*?phase_tile_bridge_v_words[\s\S]*?phase_tile_bridge_v_words_valid[\s\S]*?phase_tile_bridge_d_tile_idx[\s\S]*?phase_tile_bridge_visible[\s\S]*?phase_tile_bridge_owner_ok[\s\S]*?phase_tile_bridge_consumed[\s\S]*?phase_tile_bridge_compare_ok' -Reason "Top run_p11af helper must pass through W4-B1 phase-B tile bridge arguments"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)attn_phaseb_top_managed_softmax_out_mainline\s*\([\s\S]*?phase_entry_probe_v_base_word[\s\S]*?phase_entry_probe_v_words[\s\S]*?phase_entry_probe_v_words_valid[\s\S]*?phase_entry_probe_visible[\s\S]*?phase_entry_probe_owner_ok[\s\S]*?phase_entry_probe_compare_ok' -Reason "SoftmaxOut mainline missing W4-M2 phase-entry probe signature"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)phase_entry_probe_enabled' -Reason "SoftmaxOut mainline missing probe-enable gating anchor"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)ATTN_P11AF_PHASE_ENTRY_PROBE_COL_LOOP' -Reason "SoftmaxOut mainline missing phase-entry probe loop label"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)phase_entry_probe_owner_ok' -Reason "SoftmaxOut mainline missing probe ownership observability anchor"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)phase_entry_probe_compare_ok' -Reason "SoftmaxOut mainline missing probe compare observability anchor"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)attn_phaseb_top_managed_softmax_out_mainline\s*\([\s\S]*?phase_tile_bridge_v_base_word[\s\S]*?phase_tile_bridge_v_words[\s\S]*?phase_tile_bridge_v_words_valid[\s\S]*?phase_tile_bridge_d_tile_idx[\s\S]*?phase_tile_bridge_visible[\s\S]*?phase_tile_bridge_owner_ok[\s\S]*?phase_tile_bridge_consumed[\s\S]*?phase_tile_bridge_compare_ok' -Reason "SoftmaxOut mainline missing W4-B1 tile bridge signature"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)phase_tile_bridge_enabled' -Reason "SoftmaxOut mainline missing W4-B1 tile bridge enable anchor"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)phase_tile_bridge_selected' -Reason "SoftmaxOut mainline missing W4-B1 tile bridge select anchor"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)ATTN_P11AF_TILE_BRIDGE_COMPARE_LOOP' -Reason "SoftmaxOut mainline missing W4-B1 tile bridge compare loop label"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)phase_tile_bridge_consumed' -Reason "SoftmaxOut mainline missing W4-B1 tile bridge consumed observability anchor"
Require-Regex -Text $softmaxOutText -Pattern '(?ms)phase_tile_bridge_compare_ok' -Reason "SoftmaxOut mainline missing W4-B1 tile bridge compare observability anchor"

$preprocPath = Join-Path $repo "src/blocks/PreprocEmbedSPE.h"
$layernormPath = Join-Path $repo "src/blocks/LayerNormBlock.h"
$finalheadPath = Join-Path $repo "src/blocks/FinalHead.h"
$ffnPath = Join-Path $repo "src/blocks/FFNLayer0.h"
Require-True -Condition (Test-Path $preprocPath) -Reason "required file missing: src/blocks/PreprocEmbedSPE.h"
Require-True -Condition (Test-Path $layernormPath) -Reason "required file missing: src/blocks/LayerNormBlock.h"
Require-True -Condition (Test-Path $finalheadPath) -Reason "required file missing: src/blocks/FinalHead.h"
Require-True -Condition (Test-Path $ffnPath) -Reason "required file missing: src/blocks/FFNLayer0.h"
$preprocText = Get-Content -Path $preprocPath -Raw
$layernormBlockText = Get-Content -Path $layernormPath -Raw
$finalheadText = Get-Content -Path $finalheadPath -Raw
$ffnText = Get-Content -Path $ffnPath -Raw

Require-Regex -Text $preprocText -Pattern '(?ms)PreprocEmbedSPECoreWindow\s*\([\s\S]*?const\s+PreprocBlockContract&\s+contract\s*,\s*const\s+u32_t\*\s+topfed_in_words\s*=\s*0' -Reason "PreprocEmbedSPECoreWindow top-fed payload argument missing"
Require-Regex -Text $preprocText -Pattern '(?ms)topfed_in_words\s*!=\s*0\)\s*\?\s*topfed_in_words\[linear_idx\]\s*:\s*sram\[in_base\s*\+\s*linear_idx\]' -Reason "PreprocEmbedSPECoreWindow must consume top-fed input payload when provided"
Require-Regex -Text $layernormBlockText -Pattern '(?ms)LayerNormBlockCoreWindow\s*\([\s\S]*?const\s+LayerNormBlockContract&\s+contract\s*,\s*const\s+u32_t\*\s+topfed_gamma_words\s*=\s*0\s*,\s*const\s+u32_t\*\s+topfed_beta_words\s*=\s*0' -Reason "LayerNormBlockCoreWindow top-fed affine payload arguments missing"
Require-Regex -Text $layernormBlockText -Pattern '(?ms)topfed_gamma_words\s*!=\s*0\)\s*\?\s*topfed_gamma_words\[c\]\s*:\s*sram\[gamma_base\s*\+\s*c\]' -Reason "LayerNormBlockCoreWindow must consume top-fed gamma payload when provided"
Require-Regex -Text $layernormBlockText -Pattern '(?ms)topfed_beta_words\s*!=\s*0\)\s*\?\s*topfed_beta_words\[c\]\s*:\s*sram\[beta_base\s*\+\s*c\]' -Reason "LayerNormBlockCoreWindow must consume top-fed beta payload when provided"
Require-Regex -Text $finalheadText -Pattern '(?ms)FinalHeadCorePassABTopManaged\s*\([\s\S]*?u32_t\s+outmode_word\s*,\s*const\s+u32_t\*\s+topfed_final_scalar_words\s*=\s*0' -Reason "FinalHeadCorePassABTopManaged top-fed scalar argument missing"
Require-Regex -Text $finalheadText -Pattern '(?ms)topfed_final_scalar_words\s*!=\s*0\)\s*\?\s*topfed_final_scalar_words\[t\]\s*:\s*sram\[final_scalar_base\s*\+\s*t\]' -Reason "FinalHead pass-B must consume top-fed scalar payload when provided"
Require-Regex -Text $ffnText -Pattern '(?ms)FFNLayer0CoreWindow\s*\([\s\S]*?u32_t\s+layer_id\s*=\s*\(u32_t\)0\s*,\s*const\s+u32_t\*\s+topfed_x_words\s*=\s*0\s*,\s*const\s+u32_t\*\s+topfed_w1_weight_words\s*=\s*0\s*,\s*u32_t\s+topfed_w1_weight_words_valid\s*=\s*0\s*,\s*const\s+u32_t\*\s+topfed_w2_input_words\s*=\s*0\s*,\s*u32_t\s+topfed_w2_input_words_valid\s*=\s*0\s*,\s*const\s+u32_t\*\s+topfed_w2_weight_words\s*=\s*0\s*,\s*u32_t\s+topfed_w2_weight_words_valid\s*=\s*0\s*,\s*const\s+u32_t\*\s+topfed_w2_bias_words\s*=\s*0\s*,\s*u32_t\s+topfed_w2_bias_words_valid\s*=\s*0\s*,\s*u32_t\s+fallback_policy_flags\s*=\s*\(u32_t\)FFN_POLICY_NONE[\s\S]*?u32_t\s+topfed_x_words_valid_override\s*=\s*0[\s\S]*?const\s+u32_t\*\s+topfed_w1_bias_words\s*=\s*0[\s\S]*?u32_t\s+topfed_w1_bias_words_valid\s*=\s*0[\s\S]*?u32_t\*\s+fallback_policy_reject_stage\s*=\s*0' -Reason "FFNLayer0CoreWindow fallback policy arguments missing"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*topfed_x_words\s*!=\s*0\s*&&\s*x_idx\s*<\s*topfed_x_valid\s*\)\s*\{\s*x_tile\[i\]\s*=\s*topfed_x_words\[x_idx\]' -Reason "FFNLayer0CoreWindow must consume top-fed x payload when provided"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*topfed_w1_weight_words\s*!=\s*0\s*&&\s*w1_idx\s*<\s*topfed_w1_valid\s*\)\s*\{\s*w_tile\[i\]\s*=\s*topfed_w1_weight_words\[w1_idx\]' -Reason "FFNLayer0CoreWindow must consume top-fed W1 weight payload when provided"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*topfed_w1_bias_words\s*!=\s*0\s*&&\s*j\s*<\s*topfed_w1_bias_valid\s*\)\s*\{\s*acc\s*=\s*ffn_bias_from_word\s*\(\s*topfed_w1_bias_words\[j\]\s*\)' -Reason "FFNLayer0CoreWindow must consume top-fed W1 bias payload when provided"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*topfed_w2_input_words\s*!=\s*0\s*&&\s*a_idx\s*<\s*topfed_w2_input_valid\s*\)\s*\{\s*a_tile\[k\]\s*=\s*topfed_w2_input_words\[a_idx\]' -Reason "FFNLayer0CoreWindow must consume top-fed W2 input payload when provided"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*topfed_w2_weight_words\s*!=\s*0\s*&&\s*w2_idx\s*<\s*topfed_w2_weight_valid\s*\)\s*\{\s*w_tile\[k\]\s*=\s*topfed_w2_weight_words\[w2_idx\]' -Reason "FFNLayer0CoreWindow must consume top-fed W2 weight payload when provided"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*topfed_w2_bias_words\s*!=\s*0\s*&&\s*i\s*<\s*topfed_w2_bias_valid\s*\)\s*\{\s*acc\s*=\s*ffn_bias_from_word\s*\(\s*topfed_w2_bias_words\[i\]\s*\)' -Reason "FFNLayer0CoreWindow must consume top-fed W2 bias payload when provided"
Require-Regex -Text $ffnText -Pattern '(?ms)const\s+bool\s+require_w1_topfed\s*=\s*\(\(\(uint32_t\)fallback_policy_flags\.to_uint\(\)\s*&\s*\(uint32_t\)FFN_POLICY_REQUIRE_W1_TOPFED\)\s*!=\s*0u\)' -Reason "FFNLayer0CoreWindow must define explicit require_w1_topfed fallback policy gating"
Require-Regex -Text $ffnText -Pattern '(?ms)const\s+bool\s+w1_bias_descriptor_ready\s*=\s*\(\s*topfed_w1_bias_words\s*!=\s*0\s*\)\s*&&\s*\(\s*topfed_w1_bias_raw_valid\s*>=\s*expected_w1_bias_words\s*\)' -Reason "FFNLayer0CoreWindow must define W1 bias descriptor-ready gating"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*require_w1_topfed[\s\S]*?!\(w1_input_descriptor_ready\s*&&\s*w1_weight_descriptor_ready\s*&&\s*w1_bias_descriptor_ready\)\s*\)\s*\{[\s\S]*?fallback_policy_reject_flag[\s\S]*?return\s*;' -Reason "FFNLayer0CoreWindow must reject W1 stage when strict top-fed x/weight/bias descriptors are not ready"
Require-Regex -Text $ffnText -Pattern '(?ms)const\s+bool\s+require_w2_topfed\s*=\s*\(\(\(uint32_t\)fallback_policy_flags\.to_uint\(\)\s*&\s*\(uint32_t\)FFN_POLICY_REQUIRE_W2_TOPFED\)\s*!=\s*0u\)' -Reason "FFNLayer0CoreWindow must define explicit require_w2_topfed fallback policy gating"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*require_w2_topfed\s*&&\s*!\(w2_input_descriptor_ready\s*&&\s*w2_weight_descriptor_ready\s*&&\s*w2_bias_descriptor_ready\)\s*\)\s*\{[\s\S]*?fallback_policy_reject_flag[\s\S]*?return\s*;' -Reason "FFNLayer0CoreWindow must reject W2 stage when strict top-fed descriptors are not ready"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*fallback_policy_reject_stage\s*!=\s*0\s*\)\s*\{\s*\*fallback_policy_reject_stage\s*=\s*\(u32_t\)FFN_REJECT_STAGE_NONE\s*;\s*\}' -Reason "FFNLayer0CoreWindow must reset fallback reject stage observability"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*fallback_policy_reject_stage\s*!=\s*0\s*\)\s*\{\s*\*fallback_policy_reject_stage\s*=\s*\(u32_t\)FFN_REJECT_STAGE_W1\s*;\s*\}' -Reason "FFNLayer0CoreWindow must set W1 reject stage observability"
Require-Regex -Text $ffnText -Pattern '(?ms)if\s*\(\s*fallback_policy_reject_stage\s*!=\s*0\s*\)\s*\{\s*\*fallback_policy_reject_stage\s*=\s*\(u32_t\)FFN_REJECT_STAGE_W2\s*;\s*\}' -Reason "FFNLayer0CoreWindow must set W2 reject stage observability"
Require-Regex -Text $layerText -Pattern '(?ms)u32_t\s+topfed_ffn_w1_words\s*\[\s*FFN_W1_WEIGHT_WORDS\s*\]' -Reason "TransformerLayer top-fed W1 weight buffer missing"
Require-Regex -Text $layerText -Pattern '(?ms)u32_t\s+topfed_ffn_w1_bias_words\s*\[\s*FFN_W1_BIAS_WORDS\s*\]' -Reason "TransformerLayer top-fed W1 bias buffer missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W1_PRELOAD_BRIDGE_LOOP' -Reason "TransformerLayer bridge W1 preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W1_PRELOAD_LOOP' -Reason "TransformerLayer pointer-path W1 preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_PRELOAD_BRIDGE_LOOP' -Reason "TransformerLayer bridge W1 bias preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_PRELOAD_LOOP' -Reason "TransformerLayer pointer-path W1 bias preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)u32_t\s+topfed_ffn_w2_input_words\s*\[\s*FFN_W2_INPUT_WORDS\s*\]' -Reason "TransformerLayer top-fed W2 input buffer missing"
Require-Regex -Text $layerText -Pattern '(?ms)u32_t\s+topfed_ffn_w2_words\s*\[\s*FFN_W2_WEIGHT_WORDS\s*\]' -Reason "TransformerLayer top-fed W2 weight buffer missing"
Require-Regex -Text $layerText -Pattern '(?ms)u32_t\s+topfed_ffn_w2_bias_words\s*\[\s*FFN_W2_BIAS_WORDS\s*\]' -Reason "TransformerLayer top-fed W2 bias buffer missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_PRELOAD_BRIDGE_LOOP' -Reason "TransformerLayer bridge W2 input preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_PRELOAD_BRIDGE_LOOP' -Reason "TransformerLayer bridge W2 weight preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_PRELOAD_BRIDGE_LOOP' -Reason "TransformerLayer bridge W2 bias preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_PRELOAD_LOOP' -Reason "TransformerLayer pointer-path W2 input preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_PRELOAD_LOOP' -Reason "TransformerLayer pointer-path W2 weight preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_PRELOAD_LOOP' -Reason "TransformerLayer pointer-path W2 bias preload loop label missing"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0TopManagedWindowBridge<\s*FFN_STAGE_W1\s*>\s*\([\s\S]*?topfed_ffn_x_words\s*,[\s\S]*?topfed_ffn_w1_words\s*,[\s\S]*?\(u32_t\)w1_weight_words[\s\S]*?\(u32_t\)FFN_POLICY_REQUIRE_W1_TOPFED[\s\S]*?\(u32_t\)ffn_x_words[\s\S]*?topfed_ffn_w1_bias_words\s*,[\s\S]*?\(u32_t\)w1_bias_words' -Reason "TransformerLayerTopManagedAttnBridge must dispatch strict top-fed W1 policy with x and W1-bias descriptors"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0TopManagedWindowBridge<\s*FFN_STAGE_RELU\s*>\s*\(' -Reason "TransformerLayerTopManagedAttnBridge must stage-dispatch FFN ReLU"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0TopManagedWindowBridge<\s*FFN_STAGE_W2\s*>\s*\([\s\S]*?topfed_ffn_w2_input_words\s*,[\s\S]*?\(u32_t\)w2_input_words[\s\S]*?topfed_ffn_w2_words\s*,[\s\S]*?\(u32_t\)w2_weight_words[\s\S]*?topfed_ffn_w2_bias_words\s*,[\s\S]*?\(u32_t\)w2_bias_words\s*,[\s\S]*?\(u32_t\)FFN_POLICY_REQUIRE_W2_TOPFED' -Reason "TransformerLayerTopManagedAttnBridge must dispatch strict top-fed W2 fallback policy flag"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0<\s*FFN_STAGE_W1\s*>\s*\([\s\S]*?topfed_ffn_x_words\s*,[\s\S]*?topfed_ffn_w1_words\s*,[\s\S]*?\(u32_t\)w1_weight_words[\s\S]*?\(u32_t\)FFN_POLICY_REQUIRE_W1_TOPFED[\s\S]*?\(u32_t\)ffn_x_words[\s\S]*?topfed_ffn_w1_bias_words\s*,[\s\S]*?\(u32_t\)w1_bias_words' -Reason "TransformerLayer pointer path must dispatch strict top-fed W1 policy with x and W1-bias descriptors"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0<\s*FFN_STAGE_RELU\s*>\s*\(' -Reason "TransformerLayer pointer path must stage-dispatch FFN ReLU"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0<\s*FFN_STAGE_W2\s*>\s*\([\s\S]*?topfed_ffn_w2_input_words\s*,[\s\S]*?\(u32_t\)w2_input_words[\s\S]*?topfed_ffn_w2_words\s*,[\s\S]*?\(u32_t\)w2_weight_words[\s\S]*?topfed_ffn_w2_bias_words\s*,[\s\S]*?\(u32_t\)w2_bias_words\s*,[\s\S]*?\(u32_t\)FFN_POLICY_REQUIRE_W2_TOPFED' -Reason "TransformerLayer pointer path must dispatch strict top-fed W2 fallback policy flag"

Write-Log "guard: Top-owned preproc/layernorm/final-head contract dispatch anchors OK"
Write-Log "guard: G4 infer ingest contractized base/len dispatch anchors OK"
Write-Log "guard: OP_INFER preflight reject path maps invalid span to ERR_MEM_RANGE"
Write-Log "guard: G4-E cross-command ingest metadata surface helpers anchored"
Write-Log "guard: G4-F commit-time diagnostics helper + error mapping anchors OK"
Write-Log "guard: G4-G accepted-commit metadata record harmonization anchors OK"
Write-Log "guard: G5 wave1/wave2 top-fed payload migration anchors OK"
Write-Log "guard: G5 wave3 FFN top-fed payload migration anchors OK"
Write-Log "guard: G5 wave3.5 FFN W1 top-fed weight payload migration anchors OK"
Write-Log "guard: G5 FFN closure campaign W2 top-fed input/weight/bias anchors OK"
Write-Log "guard: G5 FFN W1 fallback policy strict top-fed gating anchors OK"
Write-Log "guard: G5 FFN fallback policy strict W2 top-fed gating anchors OK"
Write-Log "guard: G6 FFN W1 top-fed bias descriptor + reject-stage observability anchors OK"
Write-Log "guard: G7 FFN W1 strict policy now requires top-fed x/weight/bias descriptor readiness"
Write-Log "guard: W4-M1 QK-score phase-entry caller-fed descriptor probe anchors OK"
Write-Log "guard: W4-B2 QkScore bounded score-tile bridge anchors OK"
Write-Log "guard: W4-M2 SoftmaxOut phase-entry caller-fed V-tile probe anchors OK"
Write-Log "guard: W4-B1 SoftmaxOut bounded tile bridge anchors OK"
Write-Log "guard: W4-M3 Phase-A KV phase-entry caller-fed x-row probe anchors OK"
Write-Log "guard: W4-M3 KV probe descriptor-ready full-row gating anchor OK"
Write-Log "guard: W4-M3 Phase-A Q phase-entry caller-fed x-row probe anchors OK"
Write-Log "guard: Top preloaded sublayer1 norm params before layer dispatch anchors OK"
Write-Log "guard: TransformerLayer guarded preload fallback anchors OK"
Write-Log "PASS: check_top_managed_sram_boundary_regression"
Write-Summary -Status "PASS" -Detail "top-managed SRAM boundary regression anchors passed"
exit 0
