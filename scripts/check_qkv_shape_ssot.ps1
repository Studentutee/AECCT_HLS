param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11t",
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

$logPath = Join-Path $outDirAbs "check_qkv_shape_ssot.log"
$summaryPath = Join-Path $outDirAbs "check_qkv_shape_ssot_summary.txt"
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
    Write-Log "FAIL: check_qkv_shape_ssot"
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

function Assert-NoMojibakeMarkers {
    param(
        [string]$Text,
        [string]$DocLabel
    )

    $badPatterns = @(
        ([string][char]0xFFFD),
        ([string][char]0x00C3),
        ([string][char]0x00C2),
        (([string][char]0x00EF) + ([string][char]0x00BC)),
        ([string][char]0x00E5),
        ([string][char]0x00E7),
        ([string][char]0x00E6),
        ([string][char]0x00E9)
    )
    foreach ($p in $badPatterns) {
        if ($Text -match [System.Text.RegularExpressions.Regex]::Escape($p)) {
            Fail-Check ("mojibake marker detected in {0}: {1}" -f $DocLabel, $p)
        }
    }
}

Add-Content -Path $logPath -Value ("===== check_qkv_shape_ssot phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11t] phase={0}" -f $Phase)

$shapeRel = "src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h"
$kernelRel = "src/blocks/TernaryLiveQkvLeafKernel.h"
$prepTopRel = "src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h"
$tbP11rRel = "tb/tb_ternary_live_leaf_top_compile_prep_p11r.cpp"
$tbP11sRel = "tb/tb_ternary_live_leaf_top_compile_prep_family_p11s.cpp"
$reportRel = "docs/milestones/P00-011T_report.md"
$handoffRulesRel = "docs/process/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md"
$statusRel = "docs/process/PROJECT_STATUS_zhTW.txt"
$traceRel = "docs/milestones/TRACEABILITY_MAP_v12.1.md"
$closureRel = "docs/milestones/CLOSURE_MATRIX_v12.1.md"

$mustExistPre = @(
    $shapeRel,
    $kernelRel,
    $prepTopRel,
    $tbP11rRel,
    $tbP11sRel,
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
$kernelText = Get-Content -Path (Join-Path $repo $kernelRel) -Raw
$prepTopText = Get-Content -Path (Join-Path $repo $prepTopRel) -Raw

$shapeConstantsNumeric = @(
    "kQkvCtSupportedL0WqRows",
    "kQkvCtSupportedL0WqCols",
    "kQkvCtSupportedL0WkRows",
    "kQkvCtSupportedL0WkCols",
    "kQkvCtSupportedL0WvRows",
    "kQkvCtSupportedL0WvCols"
)
foreach ($name in $shapeConstantsNumeric) {
    Require-Regex -Text $shapeText -Pattern ("(?m)^\s*static\s+constexpr\s+uint32_t\s+{0}\s*=\s*\d+u\s*;" -f [System.Text.RegularExpressions.Regex]::Escape($name)) -Reason ("missing compile-time SSOT constant: {0}" -f $name)
}

$shapeConstantsExpr = @(
    "kQkvCtSupportedL0WqPayloadWords",
    "kQkvCtSupportedL0WkPayloadWords",
    "kQkvCtSupportedL0WvPayloadWords"
)
foreach ($name in $shapeConstantsExpr) {
    Require-Regex -Text $shapeText -Pattern ("(?m)^\s*static\s+constexpr\s+uint32_t\s+{0}\s*=\s*[^;]+\s*;" -f [System.Text.RegularExpressions.Regex]::Escape($name)) -Reason ("missing compile-time SSOT constant: {0}" -f $name)
}

$aliasPairs = @(
    @{ Legacy = "kTernaryLiveL0WqRows"; Source = "kQkvCtSupportedL0WqRows" },
    @{ Legacy = "kTernaryLiveL0WqCols"; Source = "kQkvCtSupportedL0WqCols" },
    @{ Legacy = "kTernaryLiveL0WqPayloadWords"; Source = "kQkvCtSupportedL0WqPayloadWords" },
    @{ Legacy = "kTernaryLiveL0WkRows"; Source = "kQkvCtSupportedL0WkRows" },
    @{ Legacy = "kTernaryLiveL0WkCols"; Source = "kQkvCtSupportedL0WkCols" },
    @{ Legacy = "kTernaryLiveL0WkPayloadWords"; Source = "kQkvCtSupportedL0WkPayloadWords" },
    @{ Legacy = "kTernaryLiveL0WvRows"; Source = "kQkvCtSupportedL0WvRows" },
    @{ Legacy = "kTernaryLiveL0WvCols"; Source = "kQkvCtSupportedL0WvCols" },
    @{ Legacy = "kTernaryLiveL0WvPayloadWords"; Source = "kQkvCtSupportedL0WvPayloadWords" }
)
foreach ($p in $aliasPairs) {
    Require-Regex -Text $shapeText -Pattern ("(?m)^\s*static\s+constexpr\s+uint32_t\s+{0}\s*=\s*{1}\s*;" -f [System.Text.RegularExpressions.Regex]::Escape($p.Legacy), [System.Text.RegularExpressions.Regex]::Escape($p.Source)) -Reason ("legacy alias must re-export from SSOT: {0} -> {1}" -f $p.Legacy, $p.Source)
}

Require-TextContains -Text $shapeText -Needle "compile-time shape SSOT" -Reason "shape config must state compile-time shape SSOT role"
Require-TextContains -Text $shapeText -Needle "Runtime metadata/config performs validation against these compile-time supported shapes only." -Reason "shape config must state runtime validation boundary"
Require-TextContains -Text $shapeText -Needle "does not imply runtime-variable top interfaces" -Reason "shape config must state no runtime-variable top interface meaning"

Require-Regex -Text $kernelText -Pattern '(?m)^\s*#include\s+"TernaryLiveQkvLeafKernelShapeConfig.h"\s*$' -Reason "leaf kernel must include shape SSOT header"
Require-Regex -Text $prepTopText -Pattern '(?m)^\s*#include\s+"TernaryLiveQkvLeafKernelShapeConfig.h"\s*$' -Reason "catapult-prep top must include shape SSOT header"

$legacyDefinePattern = '(?m)^\s*static\s+constexpr\s+uint32_t\s+kTernaryLiveL0W[qkv](Rows|Cols|PayloadWords)\s*='
if ([System.Text.RegularExpressions.Regex]::IsMatch($kernelText, $legacyDefinePattern)) {
    Fail-Check "leaf kernel must not remain active definition point for duplicated local QKV shape constants"
}

Require-TextContains -Text $kernelText -Needle "Runtime metadata validates against compile-time SSOT for this supported build." -Reason "leaf kernel must clarify runtime-vs-compile-time validation boundary"
if ([System.Text.RegularExpressions.Regex]::IsMatch($kernelText, '(?i)runtime-variable top array sizing')) {
    Fail-Check "leaf kernel must not claim runtime-variable top array sizing"
}
if ([System.Text.RegularExpressions.Regex]::IsMatch($prepTopText, '(?i)runtime-variable top array sizing')) {
    Fail-Check "catapult-prep top must not claim runtime-variable top array sizing"
}

Assert-NoMojibakeMarkers -Text $shapeText -DocLabel $shapeRel
Assert-NoMojibakeMarkers -Text $kernelText -DocLabel $kernelRel
Assert-NoMojibakeMarkers -Text $prepTopText -DocLabel $prepTopRel

Assert-NoOverclaim -Text $shapeText -DocLabel $shapeRel
Assert-NoOverclaim -Text $kernelText -DocLabel $kernelRel
Assert-NoOverclaim -Text $prepTopText -DocLabel $prepTopRel

if ($Phase -eq "post") {
    $reportAbs = Join-Path $repo $reportRel
    Require-True -Condition (Test-Path $reportAbs) -Reason "P00-011T report missing in post phase"
    $reportText = Get-Content -Path $reportAbs -Raw

    $requiredSections = @(
        "Summary",
        "Scope",
        "Files changed",
        "Exact commands executed",
        "Actual execution evidence excerpt",
        "Result / verdict wording",
        "Limitations",
        "Why useful for later tile/generalization work but not closure"
    )
    foreach ($section in $requiredSections) {
        Require-Regex -Text $reportText -Pattern ("(?m)^##\s+{0}\s*$" -f [System.Text.RegularExpressions.Regex]::Escape($section)) -Reason ("P00-011T report missing section: {0}" -f $section)
    }

    $requiredCommandMarkers = @(
        "scripts/check_handoff_surface.ps1 -OutDir build\p11t -Phase pre",
        "scripts/check_compile_prep_surface.ps1 -OutDir build\p11t -Phase pre",
        "scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11t -Phase pre",
        "scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11t -Phase pre",
        "scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11t",
        "scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11t",
        "scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11t",
        "scripts/check_handoff_surface.ps1 -OutDir build\p11t -Phase post",
        "scripts/check_compile_prep_surface.ps1 -OutDir build\p11t -Phase post",
        "scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11t -Phase post",
        "scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11t -Phase post"
    )
    foreach ($marker in $requiredCommandMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011T report missing command marker: {0}" -f $marker)
    }

    $requiredEvidenceMarkers = @(
        "PASS: check_qkv_shape_ssot",
        "PASS: run_p11r_compile_prep",
        "PASS: run_p11s_compile_prep_family",
        "PASS: run_p11l_local_regression"
    )
    foreach ($marker in $requiredEvidenceMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011T report missing evidence marker: {0}" -f $marker)
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
        Require-TextContains -Text $pair.Text -Needle "P00-011T" -Reason ("governance doc missing P00-011T: {0}" -f $pair.Label)
        Assert-NoOverclaim -Text $pair.Text -DocLabel $pair.Label
    }

    $requiredWording = @(
        "QKV shape SSOT consolidation",
        "compile-time shape SSOT",
        "runtime validation only",
        "not Catapult closure",
        "not SCVerify closure",
        "P00-011Q handoff freeze remains authoritative",
        "P00-011R WQ compile-prep probe remains valid baseline",
        "P00-011S WK/WV family compile-prep expansion remains valid baseline"
    )
    $combinedGovernanceText = $statusText + "`n" + $traceText + "`n" + $closureText + "`n" + $handoffText + "`n" + $reportText
    foreach ($phrase in $requiredWording) {
        Require-TextContains -Text $combinedGovernanceText -Needle $phrase -Reason ("required wording missing for P00-011T: {0}" -f $phrase)
    }

    Assert-NoOverclaim -Text $reportText -DocLabel $reportRel
}

Write-Log "PASS: check_qkv_shape_ssot"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
