param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11af_impl",
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

$logPath = Join-Path $outDirAbs "check_p11af_impl_surface.log"
$summaryPath = Join-Path $outDirAbs "check_p11af_impl_surface_summary.txt"
if (-not (Test-Path $logPath)) {
    New-Item -ItemType File -Path $logPath -Force > $null
}
Add-Content -Path $logPath -Value ("===== check_p11af_impl_surface phase={0} =====" -f $Phase) -Encoding UTF8

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
    Write-Log "FAIL: check_p11af_impl_surface"
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
    "src/blocks/AttnPhaseBTopManagedSoftmaxOut.h",
    "src/Top.h",
    "src/blocks/TransformerLayer.h",
    "src/blocks/AttnLayer0.h",
    "tb/tb_softmax_out_impl_p11af.cpp",
    "scripts/check_p11af_impl_surface.ps1",
    "scripts/local/run_p11af_impl_softmax_out.ps1",
    "docs/milestones/P00-011AF_report.md"
)
foreach ($rel in $required) {
    Require-True -Condition (Test-Path (Join-Path $repo $rel)) -Reason ("required file missing: {0}" -f $rel)
}

$topText = Get-Content -Path (Join-Path $repo "src/Top.h") -Raw
Require-TextContains -Text $topText -Needle "P11AF_MAINLINE_TOP_SOFTMAX_OUT_CALLSITE" -Reason "Top AF softmax/out callsite marker missing"
Require-TextContains -Text $topText -Needle "run_p11af_layer0_top_managed_softmax_out" -Reason "Top AF wrapper missing"
Require-TextContains -Text $topText -Needle "p11af_mainline_softmax_output_path_taken" -Reason "TopRegs AF telemetry missing"
Require-TextContains -Text $topText -Needle "p11af_softmax_output_fallback_taken" -Reason "TopRegs AF fallback telemetry missing"

$idxAe = $topText.IndexOf("run_p11ae_layer0_top_managed_qk_score")
$idxAf = $topText.IndexOf("run_p11af_layer0_top_managed_softmax_out")
Require-True -Condition ($idxAe -ge 0 -and $idxAf -ge 0) -Reason "AE/AF callsites must exist"
Require-True -Condition ($idxAe -lt $idxAf) -Reason "AF insertion must remain after AE"

$trText = Get-Content -Path (Join-Path $repo "src/blocks/TransformerLayer.h") -Raw
Require-TextContains -Text $trText -Needle "out_prebuilt_from_top_managed" -Reason "TransformerLayer AF hook missing"

$attnText = Get-Content -Path (Join-Path $repo "src/blocks/AttnLayer0.h") -Raw
Require-TextContains -Text $attnText -Needle "out_prebuilt_from_top_managed" -Reason "AttnLayer0 AF hook missing"
Require-TextContains -Text $attnText -Needle "score_prebuilt_from_top_managed" -Reason "AttnLayer0 AE/AF gating contract missing"

$afText = Get-Content -Path (Join-Path $repo "src/blocks/AttnPhaseBTopManagedSoftmaxOut.h") -Raw
Require-TextContains -Text $afText -Needle "ATTN_P11AF_KEY_TOKEN_LOOP" -Reason "AF helper must keep single-pass key-token loop marker"
Require-TextContains -Text $afText -Needle "attn_phaseb_softmax_score_ch_t" -Reason "AF score channel split typedef missing"
Require-TextContains -Text $afText -Needle "attn_phaseb_softmax_v_ch_t" -Reason "AF V channel split typedef missing"
Require-TextContains -Text $afText -Needle "score_ch.nb_read(score_pkt)" -Reason "AF score consume must read from score_ch"
Require-TextContains -Text $afText -Needle "v_ch.nb_read(v_pkt)" -Reason "AF V consume must read from v_ch"

$runnerText = Get-Content -Path (Join-Path $repo "scripts/local/run_p11af_impl_softmax_out.ps1") -Raw
Require-TextContains -Text $runnerText -Needle "PASS: run_p11af_impl_softmax_out" -Reason "runner PASS banner missing"
Require-TextContains -Text $runnerText -Needle "SOFTMAX_MAINLINE PASS" -Reason "runner must gate on softmax mainline banner"
Require-TextContains -Text $runnerText -Needle "MAINLINE_SOFTMAX_OUTPUT_PATH_TAKEN PASS" -Reason "runner must gate on AF mainline path banner"
Require-TextContains -Text $runnerText -Needle "FALLBACK_NOT_TAKEN PASS" -Reason "runner must gate on fallback-not-taken banner"
Require-TextContains -Text $runnerText -Needle "fallback_taken = false" -Reason "runner must gate on fallback false evidence"

if ($Phase -eq "post") {
    $reportText = Get-Content -Path (Join-Path $repo "docs/milestones/P00-011AF_report.md") -Raw
    Require-TextContains -Text $reportText -Needle "local-only" -Reason "report missing local-only wording"
    Require-TextContains -Text $reportText -Needle "not Catapult closure" -Reason "report missing not Catapult closure wording"
    Require-TextContains -Text $reportText -Needle "not SCVerify closure" -Reason "report missing not SCVerify closure wording"
}

Write-Log "PASS: check_p11af_impl_surface"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0

