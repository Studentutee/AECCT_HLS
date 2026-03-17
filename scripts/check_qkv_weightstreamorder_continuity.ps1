param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11v",
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

$logPath = Join-Path $outDirAbs "check_qkv_weightstreamorder_continuity.log"
$summaryPath = Join-Path $outDirAbs "check_qkv_weightstreamorder_continuity_summary.txt"
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
    Write-Log "FAIL: check_qkv_weightstreamorder_continuity"
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

function Assert-NoOverclaim {
    param(
        [string]$Text,
        [string]$DocLabel
    )

    $forbiddenPatterns = @(
        '(?i)\bfull Catapult closure achieved\b',
        '(?i)\bfull SCVerify closure achieved\b',
        '(?i)\bCatapult compile success\b',
        '(?i)\bSCVerify success\b',
        '(?i)\bCatapult closure achieved\b',
        '(?i)\bSCVerify closure achieved\b'
    )
    foreach ($pattern in $forbiddenPatterns) {
        if ([System.Text.RegularExpressions.Regex]::IsMatch($Text, $pattern)) {
            Fail-Check ("overclaim wording detected in {0}: pattern {1}" -f $DocLabel, $pattern)
        }
    }
}

function Require-BaselineContinuity {
    param(
        [string]$Text,
        [string]$Milestone
    )

    $m = [System.Text.RegularExpressions.Regex]::Escape($Milestone)
    $semantic = 'retain|retained|remain|remains|valid|baseline|authoritative'
    $pattern = ("(?is){0}[^\r\n]{{0,140}}({1})|({1})[^\r\n]{{0,140}}{0}" -f $m, $semantic)
    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($Text, $pattern)) {
        Fail-Check ("continuity wording missing for {0}" -f $Milestone)
    }
}

Add-Content -Path $logPath -Value ("===== check_qkv_weightstreamorder_continuity phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11v] phase={0}" -f $Phase)

$shapeRel = "src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h"
$fenceRel = "src/blocks/TernaryLiveQkvWeightStreamOrderContinuityFence.h"
$leafRel = "src/blocks/TernaryLiveQkvLeafKernel.h"
$wsoRel = "gen/WeightStreamOrder.h"
$wsoGenRel = "gen/include/WeightStreamOrder.h"
$tbP11rRel = "tb/tb_ternary_live_leaf_top_compile_prep_p11r.cpp"
$tbP11sRel = "tb/tb_ternary_live_leaf_top_compile_prep_family_p11s.cpp"
$runnerRel = "scripts/local/run_p11l_local_regression.ps1"
$reportRel = "docs/milestones/P00-011V_report.md"
$handoffRulesRel = "docs/process/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md"
$statusRel = "docs/process/PROJECT_STATUS_zhTW.txt"
$traceRel = "docs/milestones/TRACEABILITY_MAP_v12.1.md"
$closureRel = "docs/milestones/CLOSURE_MATRIX_v12.1.md"

$mustExistPre = @(
    $shapeRel,
    $fenceRel,
    $leafRel,
    $wsoRel,
    $wsoGenRel,
    $tbP11rRel,
    $tbP11sRel,
    $runnerRel,
    $handoffRulesRel,
    $statusRel,
    $traceRel,
    $closureRel
)
foreach ($rel in $mustExistPre) {
    $abs = Join-Path $repo $rel
    Require-True -Condition (Test-Path $abs) -Reason ("required file missing: {0}" -f $rel)
}

$shapeText = Get-Content -Path (Join-Path $repo $shapeRel) -Raw
$fenceText = Get-Content -Path (Join-Path $repo $fenceRel) -Raw
$leafText = Get-Content -Path (Join-Path $repo $leafRel) -Raw
$runnerText = Get-Content -Path (Join-Path $repo $runnerRel) -Raw

Require-Regex -Text $fenceText -Pattern '(?m)^\s*#include\s+"gen/WeightStreamOrder.h"\s*$' -Reason "continuity fence must include gen/WeightStreamOrder.h"
Require-Regex -Text $fenceText -Pattern '(?m)^\s*#include\s+"TernaryLiveQkvLeafKernelShapeConfig.h"\s*$' -Reason "continuity fence must include shape SSOT header"
Require-Regex -Text $leafText -Pattern '(?m)^\s*#include\s+"TernaryLiveQkvWeightStreamOrderContinuityFence.h"\s*$' -Reason "leaf kernel must include continuity fence header"

$matrixTargets = @(
    @{ Id = "QLM_L0_WQ"; Alias = "kQkvWoMetaL0Wq"; Label = "L0_WQ" },
    @{ Id = "QLM_L0_WK"; Alias = "kQkvWoMetaL0Wk"; Label = "L0_WK" },
    @{ Id = "QLM_L0_WV"; Alias = "kQkvWoMetaL0Wv"; Label = "L0_WV" }
)

foreach ($m in $matrixTargets) {
    $idEsc = [System.Text.RegularExpressions.Regex]::Escape($m.Id)
    Require-Regex -Text $fenceText -Pattern ("kQuantLinearMeta\s*\[\s*\(uint32_t\)\s*{0}\s*\]" -f $idEsc) -Reason ("continuity fence must directly reference kQuantLinearMeta entry for {0}" -f $m.Label)
}

$requiredFields = @(
    "matrix_id",
    "rows",
    "cols",
    "num_weights",
    "payload_words_2b",
    "last_word_valid_count"
)

foreach ($m in $matrixTargets) {
    $idEsc = [System.Text.RegularExpressions.Regex]::Escape($m.Id)
    $aliasEsc = [System.Text.RegularExpressions.Regex]::Escape($m.Alias)
    $matrixExpr = ("(?:{0}|kQuantLinearMeta\s*\[\s*\(uint32_t\)\s*{1}\s*\])" -f $aliasEsc, $idEsc)
    foreach ($field in $requiredFields) {
        $fieldEsc = [System.Text.RegularExpressions.Regex]::Escape($field)
        $pattern = ("(?s)static_assert\s*\([^;]*{0}[^;]*\.{1}[^;]*\)" -f $matrixExpr, $fieldEsc)
        Require-Regex -Text $fenceText -Pattern $pattern -Reason ("missing continuity static_assert semantics for {0}.{1}" -f $m.Label, $field)
    }
}

$forbiddenDefinitionPatterns = @(
    '(?m)^\s*static\s+constexpr\s+QuantLinearMeta\s+\w+\s*\[',
    '(?m)^\s*QuantLinearMeta\s+\w+\s*\[[^\]]+\]\s*=\s*\{'
)
foreach ($pattern in $forbiddenDefinitionPatterns) {
    if ([System.Text.RegularExpressions.Regex]::IsMatch($fenceText, $pattern)) {
        Fail-Check ("second metadata definition point detected in continuity fence: pattern {0}" -f $pattern)
    }
}

if (-not [System.Text.RegularExpressions.Regex]::IsMatch($runnerText, "check_qkv_weightstreamorder_continuity\.ps1[^\r\n]*'pre'")) {
    Fail-Check "run_p11l_local_regression must invoke check_qkv_weightstreamorder_continuity pre phase"
}
if (-not [System.Text.RegularExpressions.Regex]::IsMatch($runnerText, "check_qkv_weightstreamorder_continuity\.ps1[^\r\n]*'post'")) {
    Fail-Check "run_p11l_local_regression must invoke check_qkv_weightstreamorder_continuity post phase"
}

Assert-NoOverclaim -Text $shapeText -DocLabel $shapeRel
Assert-NoOverclaim -Text $fenceText -DocLabel $fenceRel
Assert-NoOverclaim -Text $leafText -DocLabel $leafRel

if ($Phase -eq "post") {
    $reportAbs = Join-Path $repo $reportRel
    Require-True -Condition (Test-Path $reportAbs) -Reason "P00-011V report missing in post phase"
    $reportText = Get-Content -Path $reportAbs -Raw

    $requiredSections = @(
        "Summary",
        "Scope",
        "Files changed",
        "Exact commands executed",
        "Actual execution evidence excerpt",
        "Result / verdict wording",
        "Limitations",
        "Why useful for later WeightStreamOrder continuity fence but not closure"
    )
    foreach ($section in $requiredSections) {
        Require-Regex -Text $reportText -Pattern ("(?m)^##\s+{0}\s*$" -f [System.Text.RegularExpressions.Regex]::Escape($section)) -Reason ("P00-011V report missing section: {0}" -f $section)
    }

    $requiredCommandMarkers = @(
        "scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11v -Phase pre",
        "scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11v -Phase post",
        "scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11v",
        "scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11v",
        "scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11v"
    )
    foreach ($marker in $requiredCommandMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011V report missing command marker: {0}" -f $marker)
    }

    $requiredEvidenceMarkers = @(
        "PASS: check_qkv_weightstreamorder_continuity",
        "PASS: run_p11r_compile_prep",
        "PASS: run_p11s_compile_prep_family"
    )
    foreach ($marker in $requiredEvidenceMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011V report missing evidence marker: {0}" -f $marker)
    }

    $statusText = Get-Content -Path (Join-Path $repo $statusRel) -Raw
    $traceText = Get-Content -Path (Join-Path $repo $traceRel) -Raw
    $closureText = Get-Content -Path (Join-Path $repo $closureRel) -Raw
    $handoffText = Get-Content -Path (Join-Path $repo $handoffRulesRel) -Raw

    foreach ($pair in @(
            @{ Label = $statusRel; Text = $statusText },
            @{ Label = $traceRel; Text = $traceText },
            @{ Label = $closureRel; Text = $closureText },
            @{ Label = $handoffRulesRel; Text = $handoffText })) {
        Require-TextContains -Text $pair.Text -Needle "P00-011V" -Reason ("governance doc missing P00-011V: {0}" -f $pair.Label)
        Assert-NoOverclaim -Text $pair.Text -DocLabel $pair.Label
    }

    $combinedText = $statusText + "`n" + $traceText + "`n" + $closureText + "`n" + $handoffText + "`n" + $reportText
    Require-Regex -Text $combinedText -Pattern '(?i)\blocal-only\b' -Reason "required local-only wording missing for P00-011V"
    Require-Regex -Text $combinedText -Pattern '(?i)\bnot Catapult closure\b' -Reason "required not Catapult closure wording missing for P00-011V"
    Require-Regex -Text $combinedText -Pattern '(?i)\bnot SCVerify closure\b' -Reason "required not SCVerify closure wording missing for P00-011V"

    foreach ($id in @("P00-011Q", "P00-011R", "P00-011S", "P00-011T", "P00-011U")) {
        Require-BaselineContinuity -Text $combinedText -Milestone $id
    }

    Assert-NoOverclaim -Text $reportText -DocLabel $reportRel
}

Write-Log "PASS: check_qkv_weightstreamorder_continuity"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
