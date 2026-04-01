param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11ao",
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

$logPath = Join-Path $outDirAbs "check_p11ao_ffn_deep_boundary.log"
$summaryPath = Join-Path $outDirAbs "check_p11ao_ffn_deep_boundary_summary.txt"
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
    Write-Log "FAIL: check_p11ao_ffn_deep_boundary"
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

Add-Content -Path $logPath -Value ("===== check_p11ao_ffn_deep_boundary phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11ao] phase={0}" -f $Phase)

$topRel = "src/Top.h"
$layerRel = "src/blocks/TransformerLayer.h"
$ffnRel = "src/blocks/FFNLayer0.h"
$wrapperRel = "src/blocks/TopManagedAttentionChainCatapultTop.h"
foreach ($rel in @($topRel, $layerRel, $ffnRel, $wrapperRel)) {
    if (-not (Test-Path (Join-Path $repo $rel))) {
        Fail-Check ("required file missing: {0}" -f $rel)
    }
}

$topText = Get-Content -Path (Join-Path $repo $topRel) -Raw
$layerText = Get-Content -Path (Join-Path $repo $layerRel) -Raw
$ffnText = Get-Content -Path (Join-Path $repo $ffnRel) -Raw
$wrapperText = Get-Content -Path (Join-Path $repo $wrapperRel) -Raw

# Old deep raw-pointer FFN site remains for backward-compatible local path.
Require-Regex -Text $ffnText -Pattern '(?ms)template<\s*unsigned\s+STAGE_MODE\s*>\s*static\s+inline\s+void\s+FFNLayer0\s*\(\s*u32_t\*\s*sram\s*,' -Reason "baseline FFNLayer0(u32_t* sram, ...) signature missing"

# New FFN deep bridge must be array-shaped.
Require-Regex -Text $ffnText -Pattern '(?ms)template<\s*unsigned\s+STAGE_MODE\s*,\s*uint32_t\s+SRAM_WORDS\s*>\s*static\s+inline\s+void\s+FFNLayer0TopManagedWindowBridge\s*\(\s*u32_t\s*\(&\s*sram_window\s*\)\s*\[\s*SRAM_WORDS\s*\]' -Reason "FFNLayer0 deep bridge missing array-shaped signature"

# Transformer deep bridge must use the new FFN array-shaped bridge.
Require-Regex -Text $layerText -Pattern '(?ms)TransformerLayerTopManagedAttnBridge\s*\(\s*u32_t\s*\(&\s*sram_window\s*\)\s*\[\s*SRAM_WORDS\s*\]' -Reason "TransformerLayer deep bridge missing array-shaped signature"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0TopManagedWindowBridge<\s*FFN_STAGE_W1\s*>\s*\(' -Reason "TransformerLayer deep bridge missing FFN W1 stage dispatch"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0TopManagedWindowBridge<\s*FFN_STAGE_RELU\s*>\s*\(' -Reason "TransformerLayer deep bridge missing FFN RELU stage dispatch"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0TopManagedWindowBridge<\s*FFN_STAGE_W2\s*>\s*\(' -Reason "TransformerLayer deep bridge missing FFN W2 stage dispatch"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0<\s*FFN_STAGE_W1\s*>\s*\(' -Reason "TransformerLayer pointer path missing FFN W1 stage dispatch"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0<\s*FFN_STAGE_RELU\s*>\s*\(' -Reason "TransformerLayer pointer path missing FFN RELU stage dispatch"
Require-Regex -Text $layerText -Pattern '(?ms)FFNLayer0<\s*FFN_STAGE_W2\s*>\s*\(' -Reason "TransformerLayer pointer path missing FFN W2 stage dispatch"
Require-Regex -Text $layerText -Pattern '(?ms)struct\s+TransformerLayerFfnTopfedHandoffDesc\s*\{[\s\S]*?topfed_w2_input_words[\s\S]*?topfed_w2_weight_words[\s\S]*?topfed_w2_bias_words' -Reason "TransformerLayer FFN handoff descriptor missing W2 payload fields"
Require-Regex -Text $layerText -Pattern '(?ms)transformer_layer_select_topfed_words\s*\([\s\S]*?ffn_topfed_handoff_desc\.topfed_w2_input_words[\s\S]*?selected_topfed_ffn_w2_input_words' -Reason "TransformerLayer missing W2 input seam selector anchor"
Require-Regex -Text $layerText -Pattern '(?ms)transformer_layer_select_topfed_words\s*\([\s\S]*?ffn_topfed_handoff_desc\.topfed_w2_weight_words[\s\S]*?selected_topfed_ffn_w2_words' -Reason "TransformerLayer missing W2 weight seam selector anchor"
Require-Regex -Text $layerText -Pattern '(?ms)transformer_layer_select_topfed_words\s*\([\s\S]*?ffn_topfed_handoff_desc\.topfed_w2_bias_words[\s\S]*?selected_topfed_ffn_w2_bias_words' -Reason "TransformerLayer missing W2 bias seam selector anchor"

$bridgeBodyMatch = [System.Text.RegularExpressions.Regex]::Match(
    $layerText,
    '(?ms)static\s+inline\s+void\s+TransformerLayerTopManagedAttnBridge\s*\(.*?\)\s*\{(?<body>.*?)\n\}')
if (-not $bridgeBodyMatch.Success) {
    Fail-Check "failed to parse TransformerLayerTopManagedAttnBridge body"
}
$bridgeBody = $bridgeBodyMatch.Groups["body"].Value
if ([System.Text.RegularExpressions.Regex]::IsMatch($bridgeBody, '(?ms)\bFFNLayer0<\s*FFN_STAGE_[A-Z0-9_]+\s*>\s*\(\s*sram_window\s*,')) {
    Fail-Check "deep bridge still calls pointer-path FFNLayer0<...>(sram_window,...) directly"
}

# Wrapper and Top dispatch path must remain through the deep bridge loop.
Require-Regex -Text $wrapperText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge\s*\(\s*regs_\s*,\s*sram_\s*\)' -Reason "TopManagedAttentionChainCatapultTop is not using the deep-bridge loop"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge\s*\(\s*TopRegs&\s+regs\s*,\s*u32_t\s*\(&\s*sram\s*\)\s*\[\s*SRAM_WORDS\s*\](?:\s*,\s*bool\s+lid0_local_only_ffn_handoff_enable\s*=\s*false\s*,\s*bool\s+lid0_local_only_ffn_handoff_descriptor_valid\s*=\s*true)?' -Reason "Top deep-bridge loop function missing"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?(TransformerLayerTopManagedAttnBridge|top_dispatch_transformer_layer_top_managed_attn_bridge)\s*\(' -Reason "Top deep-bridge loop does not dispatch TransformerLayerTopManagedAttnBridge (direct or helper path)"
Require-Regex -Text $topText -Pattern '(?ms)top_make_transformer_layer_ffn_topfed_handoff_desc\s*\(' -Reason "Top-side FFN handoff assembly helper missing"
Require-Regex -Text $topText -Pattern '(?ms)top_make_lid0_local_only_ffn_fixed_handoff_desc\s*\(' -Reason "Top-side lid0 local-only fixed FFN preload/handoff helper missing"
Require-Regex -Text $topText -Pattern '(?ms)top_make_runloop_lid0_local_only_ffn_handoff_desc\s*\(' -Reason "Top run-loop lid0 local-only FFN handoff helper missing"
Require-Regex -Text $topText -Pattern '(?ms)run_pipeline_transformer_layer_loop_with_local_ffn_handoff\s*\(' -Reason "Top pipeline-level lid0 local-only FFN handoff bridge helper missing"
Require-Regex -Text $topText -Pattern '(?ms)run_pipeline_transformer_layer_loop_top_managed_attn_bridge_with_local_ffn_handoff\s*\(' -Reason "Top pipeline-level deep-bridge lid0 local-only FFN handoff helper missing"
Require-Regex -Text $topText -Pattern '(?ms)top_dispatch_transformer_layer\s*\(' -Reason "Top pointer-path transformer dispatch helper missing"
Require-Regex -Text $topText -Pattern '(?ms)top_dispatch_transformer_layer_top_managed_attn_bridge\s*\(' -Reason "Top deep-bridge transformer dispatch helper missing"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop\s*\(\s*TopRegs&\s+regs\s*,\s*u32_t\*\s+sram\s*(?:,\s*bool\s+lid0_local_only_ffn_handoff_enable\s*=\s*false\s*,\s*bool\s+lid0_local_only_ffn_handoff_descriptor_valid\s*=\s*true)?\s*\)' -Reason "Top pointer loop missing lid0 local-only handoff control args"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop[\s\S]*?top_make_runloop_lid0_local_only_ffn_handoff_desc\s*\([\s\S]*?top_dispatch_transformer_layer\s*\(' -Reason "Top run_transformer_layer_loop missing lid0 FFN handoff assembly->dispatch wiring"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?top_make_runloop_lid0_local_only_ffn_handoff_desc\s*\([\s\S]*?top_dispatch_transformer_layer_top_managed_attn_bridge\s*\(' -Reason "Top deep-bridge loop missing lid0 FFN handoff assembly->dispatch wiring"
Require-Regex -Text $topText -Pattern '(?ms)run_infer_pipeline[\s\S]*?run_pipeline_transformer_layer_loop_with_local_ffn_handoff\s*\(' -Reason "run_infer_pipeline missing pipeline-level lid0 FFN handoff bridge dispatch"
Require-Regex -Text $topText -Pattern '(?ms)run_infer_pipeline_top_managed_attn_bridge[\s\S]*?run_pipeline_transformer_layer_loop_top_managed_attn_bridge_with_local_ffn_handoff\s*\(' -Reason "run_infer_pipeline_top_managed_attn_bridge missing pipeline-level deep-bridge lid0 FFN handoff dispatch"

Write-Log "PASS: check_p11ao_ffn_deep_boundary"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
