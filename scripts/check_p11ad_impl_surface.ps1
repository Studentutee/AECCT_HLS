param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11ad_impl",
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

$logPath = Join-Path $outDirAbs "check_p11ad_impl_surface.log"
$summaryPath = Join-Path $outDirAbs "check_p11ad_impl_surface_summary.txt"
if (-not (Test-Path $logPath)) {
    New-Item -ItemType File -Path $logPath -Force > $null
}
Add-Content -Path $logPath -Value ("===== check_p11ad_impl_surface phase={0} =====" -f $Phase) -Encoding UTF8

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
    Write-Log "FAIL: check_p11ad_impl_surface"
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

function Require-TextContains {
    param(
        [string]$Text,
        [string]$Needle,
        [string]$Reason
    )
    if ($Text -notmatch [System.Text.RegularExpressions.Regex]::Escape($Needle)) {
        Fail-Check $Reason
    }
}

$required = @(
    "include/AttnTopManagedPackets.h",
    "src/Top.h",
    "src/blocks/TransformerLayer.h",
    "src/blocks/AttnLayer0.h",
    "src/blocks/AttnPhaseATopManagedQ.h",
    "tb/tb_q_path_impl_p11ad.cpp",
    "scripts/check_p11ad_impl_surface.ps1",
    "scripts/local/run_p11ad_impl_q_path.ps1",
    "docs/milestones/P00-011AD_report.md"
)
foreach ($rel in $required) {
    Require-True -Condition (Test-Path (Join-Path $repo $rel)) -Reason ("required file missing: {0}" -f $rel)
}

$pktText = Get-Content -Path (Join-Path $repo "include/AttnTopManagedPackets.h") -Raw
Require-TextContains -Text $pktText -Needle "ATTN_PKT_WQ" -Reason "packet kind WQ missing"
Require-TextContains -Text $pktText -Needle "ATTN_PKT_Q" -Reason "packet kind Q missing"

$topText = Get-Content -Path (Join-Path $repo "src/Top.h") -Raw
Require-TextContains -Text $topText -Needle "P11AD_MAINLINE_TOP_Q_CALLSITE" -Reason "Top Q mainline callsite marker missing"
Require-TextContains -Text $topText -Needle "run_p11ad_layer0_top_managed_q" -Reason "Top Q wrapper missing"
Require-TextContains -Text $topText -Needle "attn_phasea_top_managed_q_mainline" -Reason "Top-managed Q helper call missing"
Require-TextContains -Text $topText -Needle "p11ad_mainline_q_path_taken" -Reason "TopRegs AD telemetry missing"
Require-TextContains -Text $topText -Needle "p11ad_q_fallback_taken" -Reason "TopRegs AD fallback telemetry missing"

$trText = Get-Content -Path (Join-Path $repo "src/blocks/TransformerLayer.h") -Raw
Require-TextContains -Text $trText -Needle "q_prebuilt_from_top_managed" -Reason "TransformerLayer Q hook propagation missing"

$attnText = Get-Content -Path (Join-Path $repo "src/blocks/AttnLayer0.h") -Raw
Require-TextContains -Text $attnText -Needle "q_prebuilt_from_top_managed" -Reason "AttnLayer0 Q hook missing"
Require-TextContains -Text $attnText -Needle "skip_q_materialization" -Reason "AttnLayer0 skip_q_materialization marker missing"

$runnerText = Get-Content -Path (Join-Path $repo "scripts/local/run_p11ad_impl_q_path.ps1") -Raw
Require-TextContains -Text $runnerText -Needle "PASS: run_p11ad_impl_q_path" -Reason "runner PASS banner missing"
Require-TextContains -Text $runnerText -Needle "MAINLINE_Q_PATH_TAKEN PASS" -Reason "runner must gate on mainline Q path banner"
Require-TextContains -Text $runnerText -Needle "FALLBACK_NOT_TAKEN PASS" -Reason "runner must gate on fallback-not-taken banner"
Require-TextContains -Text $runnerText -Needle "fallback_taken = false" -Reason "runner must gate on fallback false evidence"

if ($Phase -eq "post") {
    $reportText = Get-Content -Path (Join-Path $repo "docs/milestones/P00-011AD_report.md") -Raw
    Require-TextContains -Text $reportText -Needle "local-only" -Reason "report missing local-only wording"
    Require-TextContains -Text $reportText -Needle "not Catapult closure" -Reason "report missing not Catapult closure wording"
    Require-TextContains -Text $reportText -Needle "not SCVerify closure" -Reason "report missing not SCVerify closure wording"
}

Write-Log "PASS: check_p11ad_impl_surface"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
