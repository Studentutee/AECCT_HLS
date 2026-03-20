param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11ae_impl",
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

$logPath = Join-Path $outDirAbs "check_p11ae_impl_surface.log"
$summaryPath = Join-Path $outDirAbs "check_p11ae_impl_surface_summary.txt"
if (-not (Test-Path $logPath)) {
    New-Item -ItemType File -Path $logPath -Force > $null
}
Add-Content -Path $logPath -Value ("===== check_p11ae_impl_surface phase={0} =====" -f $Phase) -Encoding UTF8

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
    Write-Log "FAIL: check_p11ae_impl_surface"
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
    "src/blocks/AttnPhaseBTopManagedQkScore.h",
    "src/Top.h",
    "src/blocks/TransformerLayer.h",
    "src/blocks/AttnLayer0.h",
    "tb/tb_qk_score_impl_p11ae.cpp",
    "scripts/check_p11ae_impl_surface.ps1",
    "scripts/local/run_p11ae_impl_qk_score.ps1",
    "docs/milestones/P00-011AE_report.md"
)
foreach ($rel in $required) {
    Require-True -Condition (Test-Path (Join-Path $repo $rel)) -Reason ("required file missing: {0}" -f $rel)
}

$topText = Get-Content -Path (Join-Path $repo "src/Top.h") -Raw
Require-TextContains -Text $topText -Needle "P11AE_MAINLINE_TOP_SCORE_CALLSITE" -Reason "Top AE score callsite marker missing"
Require-TextContains -Text $topText -Needle "run_p11ae_layer0_top_managed_qk_score" -Reason "Top AE wrapper missing"
Require-TextContains -Text $topText -Needle "p11ae_mainline_score_path_taken" -Reason "TopRegs AE telemetry missing"
Require-TextContains -Text $topText -Needle "p11ae_score_fallback_taken" -Reason "TopRegs AE fallback telemetry missing"

$loopStart = $topText.IndexOf("TOP_LAYER_ORCHESTRATION_LOOP:")
$loopEnd = $topText.IndexOf("TransformerLayer(", $loopStart)
Require-True -Condition ($loopStart -ge 0 -and $loopEnd -gt $loopStart) -Reason "unable to locate transformer layer orchestration call window"
$loopWindow = $topText.Substring($loopStart, $loopEnd - $loopStart)
$idxAd = $loopWindow.IndexOf("run_p11ad_layer0_top_managed_q")
$idxAc = $loopWindow.IndexOf("run_p11ac_layer0_top_managed_kv")
$idxAe = $loopWindow.IndexOf("run_p11ae_layer0_top_managed_qk_score")
Require-True -Condition ($idxAd -ge 0 -and $idxAc -ge 0 -and $idxAe -ge 0) -Reason "AD/AC/AE runtime callsites must exist in layer loop"
Require-True -Condition ($idxAd -lt $idxAc -and $idxAc -lt $idxAe) -Reason "AE insertion must keep AD->AC ordering and remain additive in layer loop"

$trText = Get-Content -Path (Join-Path $repo "src/blocks/TransformerLayer.h") -Raw
Require-TextContains -Text $trText -Needle "score_prebuilt_from_top_managed" -Reason "TransformerLayer AE hook missing"

$attnText = Get-Content -Path (Join-Path $repo "src/blocks/AttnLayer0.h") -Raw
Require-TextContains -Text $attnText -Needle "score_prebuilt_from_top_managed" -Reason "AttnLayer0 AE hook missing"

$runnerText = Get-Content -Path (Join-Path $repo "scripts/local/run_p11ae_impl_qk_score.ps1") -Raw
Require-TextContains -Text $runnerText -Needle "PASS: run_p11ae_impl_qk_score" -Reason "runner PASS banner missing"
Require-TextContains -Text $runnerText -Needle "QK_SCORE_MAINLINE PASS" -Reason "runner must gate on score mainline banner"
Require-TextContains -Text $runnerText -Needle "MAINLINE_SCORE_PATH_TAKEN PASS" -Reason "runner must gate on AE mainline path banner"
Require-TextContains -Text $runnerText -Needle "FALLBACK_NOT_TAKEN PASS" -Reason "runner must gate on fallback-not-taken banner"
Require-TextContains -Text $runnerText -Needle "fallback_taken = false" -Reason "runner must gate on fallback false evidence"

if ($Phase -eq "post") {
    $reportText = Get-Content -Path (Join-Path $repo "docs/milestones/P00-011AE_report.md") -Raw
    Require-TextContains -Text $reportText -Needle "local-only" -Reason "report missing local-only wording"
    Require-TextContains -Text $reportText -Needle "not Catapult closure" -Reason "report missing not Catapult closure wording"
    Require-TextContains -Text $reportText -Needle "not SCVerify closure" -Reason "report missing not SCVerify closure wording"
}

Write-Log "PASS: check_p11ae_impl_surface"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
