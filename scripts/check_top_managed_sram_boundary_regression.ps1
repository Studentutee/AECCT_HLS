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
Write-Log "guard: Top preloaded sublayer1 norm params before layer dispatch anchors OK"
Write-Log "guard: TransformerLayer guarded preload fallback anchors OK"
Write-Log "PASS: check_top_managed_sram_boundary_regression"
Write-Summary -Status "PASS" -Detail "top-managed SRAM boundary regression anchors passed"
exit 0
