param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11ap",
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

$logPath = Join-Path $outDirAbs "check_p11ap_active_chain_residual_rawptr.log"
$summaryPath = Join-Path $outDirAbs "check_p11ap_active_chain_residual_rawptr_summary.txt"
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
    Write-Log "FAIL: check_p11ap_active_chain_residual_rawptr"
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

Add-Content -Path $logPath -Value ("===== check_p11ap_active_chain_residual_rawptr phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11ap] phase={0}" -f $Phase)

$wrapperRel = "src/blocks/TopManagedAttentionChainCatapultTop.h"
$topRel = "src/Top.h"
$layerRel = "src/blocks/TransformerLayer.h"
$attnRel = "src/blocks/AttnLayer0.h"
$ffnRel = "src/blocks/FFNLayer0.h"
foreach ($rel in @($wrapperRel, $topRel, $layerRel, $attnRel, $ffnRel)) {
    if (-not (Test-Path (Join-Path $repo $rel))) {
        Fail-Check ("required file missing: {0}" -f $rel)
    }
}

$wrapperText = Get-Content -Path (Join-Path $repo $wrapperRel) -Raw
$topText = Get-Content -Path (Join-Path $repo $topRel) -Raw
$layerText = Get-Content -Path (Join-Path $repo $layerRel) -Raw
$attnText = Get-Content -Path (Join-Path $repo $attnRel) -Raw
$ffnText = Get-Content -Path (Join-Path $repo $ffnRel) -Raw

# Public wrapper boundary must stay raw-pointer free.
$runSig = [System.Text.RegularExpressions.Regex]::Match(
    $wrapperText,
    '(?ms)bool\s+CCS_BLOCK\(run\)\s*\((?<sig>.*?)\)\s*\{')
if (-not $runSig.Success) {
    Fail-Check "failed to parse TopManagedAttentionChainCatapultTop::run signature"
}
if ([System.Text.RegularExpressions.Regex]::IsMatch($runSig.Groups["sig"].Value, '(?ms)\bu32_t\s*\*')) {
    Fail-Check "raw pointer found in TopManagedAttentionChainCatapultTop public run() boundary"
}

# Active chain dispatch path.
Require-Regex -Text $wrapperText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge\s*\(\s*regs_\s*,\s*sram_\s*\)' -Reason "wrapper is not dispatching deep bridge loop"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?TransformerLayerTopManagedAttnBridge\s*\(' -Reason "Top deep bridge loop does not dispatch TransformerLayerTopManagedAttnBridge"
Require-Regex -Text $layerText -Pattern '(?ms)TransformerLayerTopManagedAttnBridge[\s\S]*?AttnLayer0TopManagedWindowBridge<\s*ATTN_STAGE_FULL\s*>\s*\(' -Reason "Transformer deep bridge missing Attn bridge call"
Require-Regex -Text $layerText -Pattern '(?ms)TransformerLayerTopManagedAttnBridge[\s\S]*?FFNLayer0TopManagedWindowBridge<\s*FFN_STAGE_FULL\s*>\s*\(' -Reason "Transformer deep bridge missing FFN bridge call"

# Bridge-level residual raw-pointer elimination on active chain.
Require-Regex -Text $attnText -Pattern '(?ms)AttnLayer0TopManagedWindowBridge[\s\S]*?AttnLayer0CoreWindow<\s*STAGE_MODE\s*,\s*u32_t\s*\(&\)\s*\[\s*SRAM_WORDS\s*\]\s*>\s*\(' -Reason "Attn bridge is not using array-shaped Attn core entry"
Require-Regex -Text $ffnText -Pattern '(?ms)FFNLayer0TopManagedWindowBridge[\s\S]*?FFNLayer0CoreWindow<\s*STAGE_MODE\s*,\s*u32_t\s*\(&\)\s*\[\s*SRAM_WORDS\s*\]\s*>\s*\(' -Reason "FFN bridge is not using array-shaped FFN core entry"

if ([System.Text.RegularExpressions.Regex]::IsMatch($attnText, '(?ms)AttnLayer0TopManagedWindowBridge[\s\S]*?AttnLayer0<\s*STAGE_MODE\s*>\s*\(')) {
    Fail-Check "Attn active bridge still calls pointer-facing AttnLayer0 wrapper"
}
if ([System.Text.RegularExpressions.Regex]::IsMatch($ffnText, '(?ms)FFNLayer0TopManagedWindowBridge[\s\S]*?FFNLayer0<\s*STAGE_MODE\s*>\s*\(')) {
    Fail-Check "FFN active bridge still calls pointer-facing FFNLayer0 wrapper"
}

# Legacy pointer signatures remain for backward compatibility only.
Require-Regex -Text $attnText -Pattern '(?ms)template<\s*unsigned\s+STAGE_MODE\s*>\s*static\s+inline\s+void\s+AttnLayer0\s*\(\s*u32_t\*\s*sram\s*,' -Reason "legacy AttnLayer0(u32_t* sram, ...) wrapper missing"
Require-Regex -Text $ffnText -Pattern '(?ms)template<\s*unsigned\s+STAGE_MODE\s*>\s*static\s+inline\s+void\s+FFNLayer0\s*\(\s*u32_t\*\s*sram\s*,' -Reason "legacy FFNLayer0(u32_t* sram, ...) wrapper missing"

Write-Log "active_chain_remaining_raw_pointer_sites: none detected on targeted Attn+FFN synth-facing chain"
Write-Log "inactive_legacy_pointer_sites: AttnLayer0(u32_t*), FFNLayer0(u32_t*)"
Write-Log "PASS: check_p11ap_active_chain_residual_rawptr"
Write-Summary -Status "PASS" -Detail "active chain Attn+FFN deep bridges no longer call pointer wrappers"
exit 0
