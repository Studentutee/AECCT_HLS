param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11anb_deeper",
    [ValidateSet("pre", "post")]
    [string]$Phase = "pre"
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

$logPath = Join-Path $outDirAbs "check_p11anb_attnlayer0_deeper_boundary.log"
$summaryPath = Join-Path $outDirAbs "check_p11anb_attnlayer0_deeper_boundary_summary.txt"
if (-not (Test-Path $logPath)) {
    New-Item -ItemType File -Path $logPath -Force > $null
}

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
        ("phase: {0}" -f $Phase),
        ("repo_root: {0}" -f $repo),
        ("out_dir: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $outDirAbs)),
        ("log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $logPath)),
        ("detail: {0}" -f $Detail)
    ) | Set-Content -Path $summaryPath -Encoding UTF8
}

function Fail-Check {
    param([string]$Reason)
    Write-Log "FAIL: check_p11anb_attnlayer0_deeper_boundary"
    Write-Log $Reason
    Write-Summary -Status "FAIL" -Detail $Reason
    exit 1
}

function Require-Regex {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Reason
    )
    if (-not ([System.Text.RegularExpressions.Regex]::IsMatch($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline))) {
        Fail-Check $Reason
    }
}

Add-Content -Path $logPath -Value ("===== check_p11anb_attnlayer0_deeper_boundary phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11anb_deeper] phase={0}" -f $Phase)

$attnRel = "src/blocks/AttnLayer0.h"
$transformerRel = "src/blocks/TransformerLayer.h"
$topRel = "src/Top.h"
$tbRel = "tb/tb_p11anb_attnlayer0_boundary_seam_contract.cpp"
$runnerRel = "scripts/local/run_p11anb_attnlayer0_boundary_seam_contract.ps1"

foreach ($rel in @($attnRel, $transformerRel, $topRel, $tbRel, $runnerRel)) {
    if (-not (Test-Path (Join-Path $repo $rel))) {
        Fail-Check ("required file missing: {0}" -f $rel)
    }
}

$attnText = Get-Content -Path (Join-Path $repo $attnRel) -Raw
$transformerText = Get-Content -Path (Join-Path $repo $transformerRel) -Raw
$topText = Get-Content -Path (Join-Path $repo $topRel) -Raw
$tbText = Get-Content -Path (Join-Path $repo $tbRel) -Raw
$runnerText = Get-Content -Path (Join-Path $repo $runnerRel) -Raw

Require-Regex -Text $attnText -Pattern '(?ms)struct\s+AttnLayer0PrebuiltHandoffDesc\s*\{[\s\S]*?out_topfed_payload_enable[\s\S]*?out_topfed_payload_words[\s\S]*?out_topfed_payload_words_valid' -Reason "AttnLayer0 prebuilt descriptor missing topfed OUT payload fields"
Require-Regex -Text $attnText -Pattern '(?ms)make_attn_layer0_prebuilt_handoff_desc\s*\([\s\S]*?bool\s+out_topfed_payload_enable[\s\S]*?const\s+u32_t\*\s+out_topfed_payload_words[\s\S]*?u32_t\s+out_topfed_payload_words_valid' -Reason "AttnLayer0 helper overload missing topfed OUT payload args"
Require-Regex -Text $attnText -Pattern '(?ms)if\s+constexpr\s*\(\s*STAGE_MODE\s*==\s*ATTN_STAGE_OUT[\s\S]*?out_topfed_ready[\s\S]*?ATTN_OUT_TOPFED_PAYLOAD_CONSUME_LOOP[\s\S]*?ATTN_OUT_TOPFED_INVALID_FALLBACK_LOOP' -Reason "AttnLayer0 OUT stage missing deeper consume/fallback loops for topfed payload"
Require-Regex -Text $attnText -Pattern '(?ms)if\s*\(\s*prebuilt_handoff\.out_prebuilt_from_top_managed\s*\)\s*\{[\s\S]*?return;' -Reason "AttnLayer0 legacy out_prebuilt anti-fallback guard missing"

Require-Regex -Text $transformerText -Pattern '(?ms)TransformerLayerTopManagedAttnBridge\s*\([\s\S]*?bool\s+attn_out_topfed_payload_enable\s*=\s*false[\s\S]*?const\s+u32_t\*\s+attn_out_topfed_payload_words\s*=\s*0[\s\S]*?u32_t\s+attn_out_topfed_payload_words_valid' -Reason "TransformerLayer deep bridge missing Attn OUT topfed payload args"
Require-Regex -Text $transformerText -Pattern '(?ms)TransformerLayer\s*\([\s\S]*?bool\s+attn_out_topfed_payload_enable\s*=\s*false[\s\S]*?const\s+u32_t\*\s+attn_out_topfed_payload_words\s*=\s*0[\s\S]*?u32_t\s+attn_out_topfed_payload_words_valid' -Reason "TransformerLayer pointer path missing Attn OUT topfed payload args"
Require-Regex -Text $transformerText -Pattern '(?ms)make_attn_layer0_prebuilt_handoff_desc\s*\([\s\S]*?attn_out_topfed_payload_enable[\s\S]*?attn_out_topfed_payload_words[\s\S]*?attn_out_topfed_payload_words_valid' -Reason "TransformerLayer -> AttnLayer0 mapping missing Attn OUT topfed payload forwarding"
Require-Regex -Text $topText -Pattern '(?ms)top_dispatch_transformer_layer\s*\([\s\S]*?bool\s+attn_out_topfed_payload_enable\s*=\s*false[\s\S]*?const\s+u32_t\*\s+attn_out_topfed_payload_words\s*=\s*0[\s\S]*?u32_t\s+attn_out_topfed_payload_words_valid' -Reason "Top caller pointer dispatch missing Attn OUT topfed payload args"
Require-Regex -Text $topText -Pattern '(?ms)top_dispatch_transformer_layer_top_managed_attn_bridge\s*\([\s\S]*?bool\s+attn_out_topfed_payload_enable\s*=\s*false[\s\S]*?const\s+u32_t\*\s+attn_out_topfed_payload_words\s*=\s*0[\s\S]*?u32_t\s+attn_out_topfed_payload_words_valid' -Reason "Top caller deep bridge dispatch missing Attn OUT topfed payload args"
Require-Regex -Text $topText -Pattern '(?ms)top_dispatch_transformer_layer\s*\([\s\S]*?TransformerLayer\([\s\S]*?attn_out_topfed_payload_enable[\s\S]*?attn_out_topfed_payload_words[\s\S]*?attn_out_topfed_payload_words_valid' -Reason "Top caller pointer dispatch missing forwarding to TransformerLayer"
Require-Regex -Text $topText -Pattern '(?ms)top_dispatch_transformer_layer_top_managed_attn_bridge\s*\([\s\S]*?TransformerLayerTopManagedAttnBridge\([\s\S]*?attn_out_topfed_payload_enable[\s\S]*?attn_out_topfed_payload_words[\s\S]*?attn_out_topfed_payload_words_valid' -Reason "Top caller deep bridge dispatch missing forwarding to TransformerLayer"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop\s*\([\s\S]*?bool\s+lid0_local_only_attn_out_payload_enable\s*=\s*false[\s\S]*?bool\s+lid0_local_only_attn_out_payload_descriptor_valid\s*=\s*true' -Reason "run_transformer_layer_loop missing local-only Attn OUT payload hook args"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge\s*\([\s\S]*?bool\s+lid0_local_only_attn_out_payload_enable\s*=\s*false[\s\S]*?bool\s+lid0_local_only_attn_out_payload_descriptor_valid\s*=\s*true' -Reason "run_transformer_layer_loop_top_managed_attn_bridge missing local-only Attn OUT payload hook args"
Require-Regex -Text $topText -Pattern '(?ms)top_make_runloop_lid0_local_only_attn_out_payload_handoff\s*\([\s\S]*?attn_out_topfed_payload_enable_for_layer[\s\S]*?attn_out_topfed_payload_words_for_layer[\s\S]*?attn_out_topfed_payload_words_valid_for_layer' -Reason "run-loop hook missing Attn OUT payload handoff assembly"
Require-Regex -Text $topText -Pattern '(?ms)p11ax_attn_out_payload_gate_taken_count[\s\S]*?p11ax_attn_out_payload_fallback_seen_count[\s\S]*?p11ax_lid_nonzero_attn_out_payload_fallback_seen_count' -Reason "TopRegs missing loop-level Attn OUT payload observability counters"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop\s*\([\s\S]*?bool\s+lid0_local_only_qkscore_mask_handoff_enable\s*=\s*false[\s\S]*?bool\s+lid0_local_only_qkscore_mask_handoff_descriptor_valid\s*=\s*true' -Reason "run_transformer_layer_loop missing local-only QkScore MASK handoff hook args"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge\s*\([\s\S]*?bool\s+lid0_local_only_qkscore_mask_handoff_enable\s*=\s*false[\s\S]*?bool\s+lid0_local_only_qkscore_mask_handoff_descriptor_valid\s*=\s*true' -Reason "run_transformer_layer_loop_top_managed_attn_bridge missing local-only QkScore MASK handoff hook args"
Require-Regex -Text $topText -Pattern '(?ms)top_make_runloop_lid0_local_only_qkscore_mask_family_handoff\s*\(' -Reason "run-loop hook missing QkScore MASK family handoff assembly helper"
Require-Regex -Text $topText -Pattern '(?ms)p11ay_qkscore_mask_handoff_gate_taken_count[\s\S]*?p11ay_qkscore_mask_handoff_fallback_seen_count[\s\S]*?p11ay_lid_nonzero_qkscore_mask_handoff_fallback_seen_count' -Reason "TopRegs missing loop-level QkScore MASK handoff observability counters"

Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_CONSUME PASS' -Reason "TB missing topfed deeper consume PASS banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_INVALID_FALLBACK PASS' -Reason "TB missing topfed invalid fallback PASS banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_DISABLED_FALLBACK PASS' -Reason "TB missing topfed disabled fallback PASS banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_POINTER_MAPPING_CONSUME' -Reason "TB missing TransformerLayer pointer mapping consume banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_POINTER_INVALID_FALLBACK' -Reason "TB missing TransformerLayer pointer invalid fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_POINTER_DISABLED_FALLBACK' -Reason "TB missing TransformerLayer pointer disabled fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_DEEP_BRIDGE_MAPPING_CONSUME' -Reason "TB missing TransformerLayer deep bridge mapping consume banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_DEEP_BRIDGE_INVALID_FALLBACK' -Reason "TB missing TransformerLayer deep bridge invalid fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_DEEP_BRIDGE_DISABLED_FALLBACK' -Reason "TB missing TransformerLayer deep bridge disabled fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TRANSFORMER_ATTN_OUT_TOPFED_MAPPING_EXPECTED_COMPARE PASS' -Reason "TB missing TransformerLayer expected compare PASS banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_POINTER_CHAIN_CONSUME' -Reason "TB missing Top caller pointer chain consume banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_POINTER_CHAIN_INVALID_FALLBACK' -Reason "TB missing Top caller pointer chain invalid fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_POINTER_CHAIN_DISABLED_FALLBACK' -Reason "TB missing Top caller pointer chain disabled fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_CHAIN_CONSUME' -Reason "TB missing Top caller deep bridge chain consume banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_CHAIN_INVALID_FALLBACK' -Reason "TB missing Top caller deep bridge chain invalid fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_CHAIN_DISABLED_FALLBACK' -Reason "TB missing Top caller deep bridge chain disabled fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_TOP_CALLER_ATTN_OUT_TOPFED_CHAIN_EXPECTED_COMPARE PASS' -Reason "TB missing Top caller chain expected compare PASS banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_CONSUME' -Reason "TB missing loop caller pointer hook consume banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_INVALID_FALLBACK' -Reason "TB missing loop caller pointer hook invalid fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_DISABLED_FALLBACK' -Reason "TB missing loop caller pointer hook disabled fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_LID_NONZERO_FALLBACK' -Reason "TB missing loop caller pointer hook lid!=0 fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_CONSUME' -Reason "TB missing loop caller deep bridge hook consume banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_INVALID_FALLBACK' -Reason "TB missing loop caller deep bridge hook invalid fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_DISABLED_FALLBACK' -Reason "TB missing loop caller deep bridge hook disabled fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_LID_NONZERO_FALLBACK' -Reason "TB missing loop caller deep bridge hook lid!=0 fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_ATTN_OUT_TOPFED_HOOK_EXPECTED_COMPARE PASS' -Reason "TB missing loop caller hook expected compare PASS banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_CONSUME' -Reason "TB missing loop caller QkScore MASK pointer hook consume banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_INVALID_FALLBACK' -Reason "TB missing loop caller QkScore MASK pointer hook invalid fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_DISABLED_FALLBACK' -Reason "TB missing loop caller QkScore MASK pointer hook disabled fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_LID_NONZERO_FALLBACK' -Reason "TB missing loop caller QkScore MASK pointer hook lid!=0 fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_CONSUME' -Reason "TB missing loop caller QkScore MASK deep bridge hook consume banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_INVALID_FALLBACK' -Reason "TB missing loop caller QkScore MASK deep bridge hook invalid fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_DISABLED_FALLBACK' -Reason "TB missing loop caller QkScore MASK deep bridge hook disabled fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_LID_NONZERO_FALLBACK' -Reason "TB missing loop caller QkScore MASK deep bridge hook lid!=0 fallback banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_LOOP_CALLER_QKSCORE_MASK_HOOK_EXPECTED_COMPARE PASS' -Reason "TB missing loop caller QkScore MASK hook expected compare PASS banner"

Require-Regex -Text $runnerText -Pattern '(?ms)OUT_TOPFED_PAYLOAD_CONSUME PASS' -Reason "runner missing topfed consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)OUT_TOPFED_PAYLOAD_INVALID_FALLBACK PASS' -Reason "runner missing topfed invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)OUT_TOPFED_PAYLOAD_DISABLED_FALLBACK PASS' -Reason "runner missing topfed disabled fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TRANSFORMER_ATTN_OUT_TOPFED_POINTER_MAPPING_CONSUME PASS' -Reason "runner missing TransformerLayer pointer mapping consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TRANSFORMER_ATTN_OUT_TOPFED_POINTER_INVALID_FALLBACK PASS' -Reason "runner missing TransformerLayer pointer invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TRANSFORMER_ATTN_OUT_TOPFED_POINTER_DISABLED_FALLBACK PASS' -Reason "runner missing TransformerLayer pointer disabled fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TRANSFORMER_ATTN_OUT_TOPFED_DEEP_BRIDGE_MAPPING_CONSUME PASS' -Reason "runner missing TransformerLayer deep bridge mapping consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TRANSFORMER_ATTN_OUT_TOPFED_DEEP_BRIDGE_INVALID_FALLBACK PASS' -Reason "runner missing TransformerLayer deep bridge invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TRANSFORMER_ATTN_OUT_TOPFED_DEEP_BRIDGE_DISABLED_FALLBACK PASS' -Reason "runner missing TransformerLayer deep bridge disabled fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TRANSFORMER_ATTN_OUT_TOPFED_MAPPING_EXPECTED_COMPARE PASS' -Reason "runner missing TransformerLayer expected compare gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TOP_CALLER_ATTN_OUT_TOPFED_POINTER_CHAIN_CONSUME PASS' -Reason "runner missing Top caller pointer chain consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TOP_CALLER_ATTN_OUT_TOPFED_POINTER_CHAIN_INVALID_FALLBACK PASS' -Reason "runner missing Top caller pointer chain invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TOP_CALLER_ATTN_OUT_TOPFED_POINTER_CHAIN_DISABLED_FALLBACK PASS' -Reason "runner missing Top caller pointer chain disabled fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_CHAIN_CONSUME PASS' -Reason "runner missing Top caller deep bridge chain consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_CHAIN_INVALID_FALLBACK PASS' -Reason "runner missing Top caller deep bridge chain invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_CHAIN_DISABLED_FALLBACK PASS' -Reason "runner missing Top caller deep bridge chain disabled fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)TOP_CALLER_ATTN_OUT_TOPFED_CHAIN_EXPECTED_COMPARE PASS' -Reason "runner missing Top caller chain expected compare gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_CONSUME PASS' -Reason "runner missing loop caller pointer hook consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_INVALID_FALLBACK PASS' -Reason "runner missing loop caller pointer hook invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_DISABLED_FALLBACK PASS' -Reason "runner missing loop caller pointer hook disabled fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_ATTN_OUT_TOPFED_POINTER_HOOK_LID_NONZERO_FALLBACK PASS' -Reason "runner missing loop caller pointer hook lid!=0 fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_CONSUME PASS' -Reason "runner missing loop caller deep bridge hook consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_INVALID_FALLBACK PASS' -Reason "runner missing loop caller deep bridge hook invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_DISABLED_FALLBACK PASS' -Reason "runner missing loop caller deep bridge hook disabled fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_ATTN_OUT_TOPFED_DEEP_BRIDGE_HOOK_LID_NONZERO_FALLBACK PASS' -Reason "runner missing loop caller deep bridge hook lid!=0 fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_ATTN_OUT_TOPFED_HOOK_EXPECTED_COMPARE PASS' -Reason "runner missing loop caller hook expected compare gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_CONSUME PASS' -Reason "runner missing loop caller QkScore MASK pointer hook consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_INVALID_FALLBACK PASS' -Reason "runner missing loop caller QkScore MASK pointer hook invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_DISABLED_FALLBACK PASS' -Reason "runner missing loop caller QkScore MASK pointer hook disabled fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_QKSCORE_MASK_POINTER_HOOK_LID_NONZERO_FALLBACK PASS' -Reason "runner missing loop caller QkScore MASK pointer hook lid!=0 fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_CONSUME PASS' -Reason "runner missing loop caller QkScore MASK deep bridge hook consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_INVALID_FALLBACK PASS' -Reason "runner missing loop caller QkScore MASK deep bridge hook invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_DISABLED_FALLBACK PASS' -Reason "runner missing loop caller QkScore MASK deep bridge hook disabled fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_QKSCORE_MASK_DEEP_BRIDGE_HOOK_LID_NONZERO_FALLBACK PASS' -Reason "runner missing loop caller QkScore MASK deep bridge hook lid!=0 fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)LOOP_CALLER_QKSCORE_MASK_HOOK_EXPECTED_COMPARE PASS' -Reason "runner missing loop caller QkScore MASK hook expected compare gate"

Write-Log "PASS: check_p11anb_attnlayer0_deeper_boundary"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
