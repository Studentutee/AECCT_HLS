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
Require-True -Condition (Test-Path $topPath) -Reason "required file missing: src/Top.h"
Require-True -Condition (Test-Path $layerPath) -Reason "required file missing: src/blocks/TransformerLayer.h"

$topText = Get-Content -Path $topPath -Raw
$layerText = Get-Content -Path $layerPath -Raw

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

Write-Log "guard: Top-owned preproc/layernorm/final-head contract dispatch anchors OK"
Write-Log "guard: G4 infer ingest contractized base/len dispatch anchors OK"
Write-Log "guard: OP_INFER preflight reject path maps invalid span to ERR_MEM_RANGE"
Write-Log "guard: G4-E cross-command ingest metadata surface helpers anchored"
Write-Log "guard: G4-F commit-time diagnostics helper + error mapping anchors OK"
Write-Log "guard: G4-G accepted-commit metadata record harmonization anchors OK"
Write-Log "guard: Top preloaded sublayer1 norm params before layer dispatch anchors OK"
Write-Log "guard: TransformerLayer guarded preload fallback anchors OK"
Write-Log "PASS: check_top_managed_sram_boundary_regression"
Write-Summary -Status "PASS" -Detail "top-managed SRAM boundary regression anchors passed"
exit 0
