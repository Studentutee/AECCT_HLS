param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11an",
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

$logPath = Join-Path $outDirAbs "check_p11an_attn_deep_boundary.log"
$summaryPath = Join-Path $outDirAbs "check_p11an_attn_deep_boundary_summary.txt"
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
    Write-Log "FAIL: check_p11an_attn_deep_boundary"
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

Add-Content -Path $logPath -Value ("===== check_p11an_attn_deep_boundary phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11an] phase={0}" -f $Phase)

$topRel = "src/Top.h"
$attnRel = "src/blocks/AttnLayer0.h"
$layerRel = "src/blocks/TransformerLayer.h"
$wrapperRel = "src/blocks/TopManagedAttentionChainCatapultTop.h"
foreach ($rel in @($topRel, $attnRel, $layerRel, $wrapperRel)) {
    if (-not (Test-Path (Join-Path $repo $rel))) {
        Fail-Check ("required file missing: {0}" -f $rel)
    }
}

$topText = Get-Content -Path (Join-Path $repo $topRel) -Raw
$attnText = Get-Content -Path (Join-Path $repo $attnRel) -Raw
$layerText = Get-Content -Path (Join-Path $repo $layerRel) -Raw
$wrapperText = Get-Content -Path (Join-Path $repo $wrapperRel) -Raw

# Old deep raw-pointer site remains for backward-compatible path.
Require-Regex -Text $attnText -Pattern '(?ms)static\s+inline\s+void\s+AttnLayer0\s*\(\s*u32_t\*\s*sram\s*,' -Reason "baseline AttnLayer0(u32_t* sram, ...) signature missing"

# New bridge in AttnLayer0 must be array-shaped.
Require-Regex -Text $attnText -Pattern '(?ms)AttnLayer0TopManagedWindowBridge\s*\(\s*u32_t\s*\(&\s*sram_window\s*\)\s*\[\s*SRAM_WORDS\s*\]' -Reason "AttnLayer0 deep bridge missing array-shaped signature"

# TransformerLayer must provide deep Attn bridge entry and use AttnLayer0TopManagedWindowBridge.
Require-Regex -Text $layerText -Pattern '(?ms)TransformerLayerTopManagedAttnBridge\s*\(\s*u32_t\s*\(&\s*sram_window\s*\)\s*\[\s*SRAM_WORDS\s*\]' -Reason "TransformerLayer deep bridge missing array-shaped signature"
Require-Regex -Text $layerText -Pattern '(?ms)AttnLayer0TopManagedWindowBridge<\s*ATTN_STAGE_FULL\s*>\s*\(' -Reason "TransformerLayer deep bridge does not call AttnLayer0TopManagedWindowBridge"

$bridgeBodyMatch = [System.Text.RegularExpressions.Regex]::Match(
    $layerText,
    '(?ms)static\s+inline\s+void\s+TransformerLayerTopManagedAttnBridge\s*\(.*?\)\s*\{(?<body>.*?)\n\}')
if (-not $bridgeBodyMatch.Success) {
    Fail-Check "failed to parse TransformerLayerTopManagedAttnBridge body"
}
$bridgeBody = $bridgeBodyMatch.Groups["body"].Value
if ([System.Text.RegularExpressions.Regex]::IsMatch($bridgeBody, '(?ms)\bAttnLayer0<\s*ATTN_STAGE_FULL\s*>\s*\(\s*sram\s*,')) {
    Fail-Check "deep bridge still calls AttnLayer0<...>(sram,...) directly"
}

# Top-managed wrapper must call new deep-bridge layer loop variant.
Require-Regex -Text $wrapperText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge\s*\(\s*regs_\s*,\s*sram_\s*\)' -Reason "TopManagedAttentionChainCatapultTop is not using the new deep-bridge loop"

# Top must provide the deep-bridge loop and dispatch through TransformerLayerTopManagedAttnBridge.
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge\s*\(\s*TopRegs&\s+regs\s*,\s*u32_t\s*\(&\s*sram\s*\)\s*\[\s*SRAM_WORDS\s*\]' -Reason "Top deep-bridge loop function missing"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?TransformerLayerTopManagedAttnBridge\s*\(' -Reason "Top deep-bridge loop does not dispatch TransformerLayerTopManagedAttnBridge"

Write-Log "PASS: check_p11an_attn_deep_boundary"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
