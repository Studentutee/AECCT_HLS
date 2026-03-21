param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11as\preflight",
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

$logPath = Join-Path $outDirAbs "check_p11as_corrected_chain_launch_pack.log"
$summaryPath = Join-Path $outDirAbs "check_p11as_corrected_chain_launch_pack_summary.txt"
if (-not (Test-Path $logPath)) {
    New-Item -ItemType File -Path $logPath -Force > $null
}
Add-Content -Path $logPath -Value ("===== check_p11as_corrected_chain_launch_pack phase={0} =====" -f $Phase) -Encoding UTF8

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
    Write-Log "FAIL: check_p11as_corrected_chain_launch_pack"
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
    if (-not ([System.Text.RegularExpressions.Regex]::IsMatch(
                $Text,
                $Pattern,
                [System.Text.RegularExpressions.RegexOptions]::Singleline))) {
        Fail-Check $Reason
    }
}

function Parse-TclListTokens {
    param(
        [string]$TclText,
        [string]$VarName
    )
    $m = [System.Text.RegularExpressions.Regex]::Match(
        $TclText,
        ('(?ms)set\s+{0}\s+\[list(?<body>.*?)\]' -f [System.Text.RegularExpressions.Regex]::Escape($VarName)))
    if (-not $m.Success) {
        return ,@()
    }
    $body = $m.Groups["body"].Value
    $tokens = [System.Text.RegularExpressions.Regex]::Matches($body, '"([^"]+)"')
    $out = @()
    foreach ($t in $tokens) {
        $out += $t.Groups[1].Value
    }
    return ,$out
}

Write-Log ("[p11as] phase={0}" -f $Phase)

$required = @(
    "src/blocks/TopManagedAttentionChainCatapultTop.h",
    "src/Top.h",
    "src/blocks/TransformerLayer.h",
    "src/blocks/FFNLayer0.h",
    "src/blocks/AttnLayer0.h",
    "include/AttnTopManagedPackets.h",
    "src/catapult/p11as_top_managed_attention_chain_entry.cpp",
    "scripts/catapult/p11as_corrected_chain_filelist.f",
    "scripts/catapult/p11as_corrected_chain_project.tcl",
    "scripts/check_p11am_catapult_top_surface.ps1",
    "scripts/check_p11an_attn_deep_boundary.ps1",
    "scripts/check_p11ao_ffn_deep_boundary.ps1",
    "scripts/check_p11ap_active_chain_residual_rawptr.ps1"
)
foreach ($rel in $required) {
    Require-True -Condition (Test-Path (Join-Path $repo $rel)) -Reason ("required file missing: {0}" -f $rel)
}

$wrapperText = Get-Content -Path (Join-Path $repo "src/blocks/TopManagedAttentionChainCatapultTop.h") -Raw
$topText = Get-Content -Path (Join-Path $repo "src/Top.h") -Raw
$layerText = Get-Content -Path (Join-Path $repo "src/blocks/TransformerLayer.h") -Raw
$filelistText = Get-Content -Path (Join-Path $repo "scripts/catapult/p11as_corrected_chain_filelist.f") -Raw
$projectTclText = Get-Content -Path (Join-Path $repo "scripts/catapult/p11as_corrected_chain_project.tcl") -Raw
$entryText = Get-Content -Path (Join-Path $repo "src/catapult/p11as_top_managed_attention_chain_entry.cpp") -Raw
$pktText = Get-Content -Path (Join-Path $repo "include/AttnTopManagedPackets.h") -Raw

# Synth entry existence check.
Require-Regex -Text $wrapperText -Pattern '(?m)^\s*#pragma\s+hls_design\s+top\s*$' -Reason "missing '#pragma hls_design top' on canonical corrected entry class"
Require-Regex -Text $wrapperText -Pattern '(?m)^\s*class\s+TopManagedAttentionChainCatapultTop\s*\{' -Reason "missing TopManagedAttentionChainCatapultTop class declaration"
Require-Regex -Text $wrapperText -Pattern '(?m)^\s*bool\s+CCS_BLOCK\(run\)\s*\(' -Reason "missing TopManagedAttentionChainCatapultTop::run boundary"
Require-Regex -Text $entryText -Pattern '(?m)^\s*#include\s+"blocks/TopManagedAttentionChainCatapultTop\.h"\s*$' -Reason "p11as entry TU does not include canonical corrected top wrapper"
Write-Log "P11AS_SYNTH_ENTRY_FOUND PASS"

# Filelist completeness check.
$fileEntries = @()
foreach ($raw in ($filelistText -split "`r?`n")) {
    $line = $raw.Trim()
    if ($line.Length -eq 0) { continue }
    if ($line.StartsWith("#")) { continue }
    $fileEntries += $line
}
Require-True -Condition ($fileEntries.Count -gt 0) -Reason "p11as filelist is empty"
foreach ($entry in $fileEntries) {
    $abs = Join-Path $repo $entry
    Require-True -Condition (Test-Path $abs) -Reason ("filelist entry missing on disk: {0}" -f $entry)
}
Require-True -Condition ($fileEntries.Count -eq 1) -Reason "p11as filelist must stay single-entry and corrected-chain focused"
Require-True -Condition ($fileEntries[0] -eq "src/catapult/p11as_top_managed_attention_chain_entry.cpp") -Reason "p11as filelist canonical source mismatch"
if ($fileEntries -match '(^|[\\/])tb([\\/]|$)') {
    Fail-Check "p11as filelist must not include testbench files"
}
Write-Log "P11AS_FILELIST_COMPLETENESS PASS"

# Include/macro check from project Tcl.
$includeDirs = Parse-TclListTokens -TclText $projectTclText -VarName "p11as_include_dirs"
$defineMacros = Parse-TclListTokens -TclText $projectTclText -VarName "p11as_define_macros"
Require-True -Condition ($includeDirs.Count -gt 0) -Reason "p11as project tcl include dir list missing or empty"
foreach ($inc in $includeDirs) {
    $absInc = Join-RepoPath -RepoRootPath $repo -Path $inc
    Require-True -Condition (Test-Path $absInc) -Reason ("include dir missing: {0}" -f $inc)
}
Require-True -Condition ($defineMacros.Count -gt 0) -Reason "p11as project tcl define macro list missing or empty"
Require-True -Condition ($defineMacros -contains "__SYNTHESIS__") -Reason "p11as project tcl missing __SYNTHESIS__ macro"
Require-Regex -Text $projectTclText -Pattern '(?m)^\s*set\s+p11as_top_entry\s+"TopManagedAttentionChainCatapultTop::run"\s*$' -Reason "project tcl corrected-entry declaration mismatch"
Require-Regex -Text $projectTclText -Pattern 'solution\s+design\s+set\s+\$p11as_top_entry\s+-top' -Reason "project tcl missing set-top command for corrected entry"
Write-Log "P11AS_INCLUDE_MACRO_CHECK PASS"

# Corrected active path reachability check.
Require-Regex -Text $wrapperText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge\s*\(\s*regs_\s*,\s*sram_\s*\)' -Reason "wrapper does not dispatch corrected deep-bridge loop"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge\s*\(\s*TopRegs&\s+regs\s*,\s*u32_t\s*\(&\s*sram\s*\)\s*\[\s*SRAM_WORDS\s*\]' -Reason "Top corrected deep-bridge loop definition missing"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?run_p11ad_layer0_top_managed_q\s*\(' -Reason "Top corrected deep-bridge loop missing run_p11ad call edge"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?run_p11ac_layer0_top_managed_kv\s*\(' -Reason "Top corrected deep-bridge loop missing run_p11ac call edge"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?run_p11ae_layer0_top_managed_qk_score\s*\(' -Reason "Top corrected deep-bridge loop missing run_p11ae call edge"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?run_p11af_layer0_top_managed_softmax_out\s*\(' -Reason "Top corrected deep-bridge loop missing run_p11af call edge"
Require-Regex -Text $topText -Pattern '(?ms)run_transformer_layer_loop_top_managed_attn_bridge[\s\S]*?TransformerLayerTopManagedAttnBridge\s*\(' -Reason "Top corrected deep-bridge loop missing TransformerLayerTopManagedAttnBridge dispatch"
Require-Regex -Text $layerText -Pattern '(?ms)TransformerLayerTopManagedAttnBridge[\s\S]*?FFNLayer0TopManagedWindowBridge<\s*FFN_STAGE_FULL\s*>\s*\(' -Reason "TransformerLayer corrected bridge missing FFNLayer0TopManagedWindowBridge call"
Require-Regex -Text $pktText -Pattern '(?m)^\s*static\s+const\s+unsigned\s+ATTN_TOP_MANAGED_WORK_TILE_WORDS\s*=\s*4u\s*;' -Reason "tile-word contract drift: expected ATTN_TOP_MANAGED_WORK_TILE_WORDS=4"
Write-Log "P11AS_CORRECTED_PATH_REACHABILITY PASS"

# Corrected-chain-only fallback exclusion check.
if ([System.Text.RegularExpressions.Regex]::IsMatch(
        $wrapperText,
        '(?ms)\brun_transformer_layer_loop\s*\(\s*regs_\s*,\s*sram_\s*\)')) {
    Fail-Check "wrapper main entry regressed to legacy run_transformer_layer_loop path"
}
Require-Regex -Text $wrapperText -Pattern '(?ms)return\s*\(\s*mainline_all_taken\s*&&\s*!fallback_taken\s*\)\s*;' -Reason "wrapper return contract no longer enforces mainline && !fallback"
Write-Log "P11AS_FALLBACK_EXCLUSION PASS"

# Reuse existing chain checkers to keep continuity with accepted AN/AO/AP evidence gates.
$amLog = Join-Path $outDirAbs "reuse_check_p11am.log"
$anLog = Join-Path $outDirAbs "reuse_check_p11an.log"
$aoLog = Join-Path $outDirAbs "reuse_check_p11ao.log"
$apLog = Join-Path $outDirAbs "reuse_check_p11ap.log"

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts/check_p11am_catapult_top_surface.ps1") -OutDir $outDirAbs -Phase $Phase *> $amLog
if ($LASTEXITCODE -ne 0) { Fail-Check "reused checker failed: check_p11am_catapult_top_surface" }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts/check_p11an_attn_deep_boundary.ps1") -OutDir $outDirAbs -Phase $Phase *> $anLog
if ($LASTEXITCODE -ne 0) { Fail-Check "reused checker failed: check_p11an_attn_deep_boundary" }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts/check_p11ao_ffn_deep_boundary.ps1") -OutDir $outDirAbs -Phase $Phase *> $aoLog
if ($LASTEXITCODE -ne 0) { Fail-Check "reused checker failed: check_p11ao_ffn_deep_boundary" }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts/check_p11ap_active_chain_residual_rawptr.ps1") -OutDir $outDirAbs -Phase $Phase *> $apLog
if ($LASTEXITCODE -ne 0) { Fail-Check "reused checker failed: check_p11ap_active_chain_residual_rawptr" }

Require-True -Condition (Select-String -Path $amLog -SimpleMatch -Quiet "PASS: check_p11am_catapult_top_surface") -Reason "missing p11am PASS evidence in reused checker log"
Require-True -Condition (Select-String -Path $anLog -SimpleMatch -Quiet "PASS: check_p11an_attn_deep_boundary") -Reason "missing p11an PASS evidence in reused checker log"
Require-True -Condition (Select-String -Path $aoLog -SimpleMatch -Quiet "PASS: check_p11ao_ffn_deep_boundary") -Reason "missing p11ao PASS evidence in reused checker log"
Require-True -Condition (Select-String -Path $apLog -SimpleMatch -Quiet "PASS: check_p11ap_active_chain_residual_rawptr") -Reason "missing p11ap PASS evidence in reused checker log"
Require-True -Condition (Select-String -Path $apLog -SimpleMatch -Quiet "active_chain_remaining_raw_pointer_sites: none detected on targeted Attn+FFN synth-facing chain") -Reason "missing residual raw-pointer none-detected evidence in reused checker log"
Write-Log "P11AS_REUSED_CHECKERS PASS"

Write-Log "PASS: check_p11as_corrected_chain_launch_pack"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
