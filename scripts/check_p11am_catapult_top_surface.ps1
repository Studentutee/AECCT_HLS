param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11am",
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

$logPath = Join-Path $outDirAbs "check_p11am_catapult_top_surface.log"
$summaryPath = Join-Path $outDirAbs "check_p11am_catapult_top_surface_summary.txt"
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

    $lines = @(
        ("status: {0}" -f $Status),
        ("phase: {0}" -f $Phase),
        ("repo_root: {0}" -f $repo),
        ("out_dir: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $outDirAbs)),
        ("log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $logPath)),
        ("detail: {0}" -f $Detail)
    )
    $lines | Set-Content -Path $summaryPath -Encoding UTF8
}

function Fail-Check {
    param([string]$Reason)
    Write-Log "FAIL: check_p11am_catapult_top_surface"
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
    if (-not ([System.Text.RegularExpressions.Regex]::IsMatch($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline))) {
        Fail-Check $Reason
    }
}

Add-Content -Path $logPath -Value ("===== check_p11am_catapult_top_surface phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11am] phase={0}" -f $Phase)

$topRel = "src/blocks/TopManagedAttentionChainCatapultTop.h"
$tbRel = "tb/tb_top_managed_catapult_compile_prep_p11am.cpp"
$runnerRel = "scripts/local/run_p11am_catapult_compile_surface.ps1"

foreach ($rel in @($topRel, $tbRel, $runnerRel)) {
    Require-True -Condition (Test-Path (Join-Path $repo $rel)) -Reason ("required file missing: {0}" -f $rel)
}

$topText = Get-Content -Path (Join-Path $repo $topRel) -Raw
$tbText = Get-Content -Path (Join-Path $repo $tbRel) -Raw
$runnerText = Get-Content -Path (Join-Path $repo $runnerRel) -Raw

Require-Regex -Text $topText -Pattern '(?m)^\s*#pragma\s+hls_design\s+top\s*$' -Reason "missing '#pragma hls_design top' in new wrapper"
Require-Regex -Text $topText -Pattern '(?m)^\s*#pragma\s+hls_design\s+interface\s*$' -Reason "missing '#pragma hls_design interface' in new wrapper"
Require-Regex -Text $topText -Pattern '(?m)^\s*bool\s+CCS_BLOCK\(run\)\s*\(' -Reason "missing same-line CCS_BLOCK(run) method in new wrapper"

$runSig = [System.Text.RegularExpressions.Regex]::Match(
    $topText,
    '(?ms)bool\s+CCS_BLOCK\(run\)\s*\((?<args>.*?)\)\s*\{')
Require-True -Condition $runSig.Success -Reason "failed to parse TopManagedAttentionChainCatapultTop::run signature"
$runArgs = $runSig.Groups["args"].Value
if ([System.Text.RegularExpressions.Regex]::IsMatch($runArgs, '\*')) {
    Fail-Check "raw pointer found in TopManagedAttentionChainCatapultTop public run() boundary"
}
if ([System.Text.RegularExpressions.Regex]::IsMatch($runArgs, '\bu32_t\s*\*\s*sram\b')) {
    Fail-Check "whole-SRAM pointer contract leaked to new public boundary"
}

$publicBlock = [System.Text.RegularExpressions.Regex]::Match(
    $topText,
    '(?ms)public:\s*(?<body>.*?)\s*private:')
Require-True -Condition $publicBlock.Success -Reason "failed to parse public/private section split in wrapper"
if ([System.Text.RegularExpressions.Regex]::IsMatch($publicBlock.Groups["body"].Value, '\bu32_t\s*\*\s*')) {
    Fail-Check "raw pointer detected in wrapper public section"
}

Require-Regex -Text $runnerText -Pattern '(?m)/D__SYNTHESIS__' -Reason "AM runner missing __SYNTHESIS__ compile-surface invocation"
Require-Regex -Text $runnerText -Pattern 'check_p11am_catapult_top_surface\.ps1\s+-OutDir' -Reason "AM runner missing p11am surface checker calls"

Require-Regex -Text $tbText -Pattern '\bPASS:\s*tb_top_managed_catapult_compile_prep_p11am\b' -Reason "AM TB missing fixed PASS banner"
Require-Regex -Text $tbText -Pattern '\bTopManagedAttentionChainCatapultTop\b' -Reason "AM TB is not using the new wrapper entrypoint"

Write-Log "PASS: check_p11am_catapult_top_surface"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
