param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11u",
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

$logPath = Join-Path $outDirAbs "check_qkv_payload_metadata_ssot.log"
$summaryPath = Join-Path $outDirAbs "check_qkv_payload_metadata_ssot_summary.txt"
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
    Write-Log "FAIL: check_qkv_payload_metadata_ssot"
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

function Get-ConstExpr {
    param(
        [string]$Text,
        [string]$ConstName,
        [string]$Label
    )

    $name = [System.Text.RegularExpressions.Regex]::Escape($ConstName)
    $m = [System.Text.RegularExpressions.Regex]::Match(
        $Text,
        ("(?m)^\s*static\s+constexpr\s+uint32_t\s+{0}\s*=\s*(?<expr>[^;]+)\s*;" -f $name))
    if (-not $m.Success) {
        Fail-Check ("missing constant definition in {0}: {1}" -f $Label, $ConstName)
    }
    return $m.Groups["expr"].Value.Trim()
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

Add-Content -Path $logPath -Value ("===== check_qkv_payload_metadata_ssot phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11u] phase={0}" -f $Phase)

$shapeRel = "src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h"
$leafRel = "src/blocks/TernaryLiveQkvLeafKernel.h"
$attnRel = "src/blocks/AttnLayer0.h"
$tbP11rRel = "tb/tb_ternary_live_leaf_top_compile_prep_p11r.cpp"
$tbP11sRel = "tb/tb_ternary_live_leaf_top_compile_prep_family_p11s.cpp"
$runnerRel = "scripts/local/run_p11l_local_regression.ps1"
$reportRel = "docs/milestones/P00-011U_report.md"
$handoffRulesRel = "docs/process/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md"
$statusRel = "docs/process/PROJECT_STATUS_zhTW.txt"
$traceRel = "docs/milestones/TRACEABILITY_MAP_v12.1.md"
$closureRel = "docs/milestones/CLOSURE_MATRIX_v12.1.md"

$mustExistPre = @(
    $shapeRel,
    $leafRel,
    $attnRel,
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
$leafText = Get-Content -Path (Join-Path $repo $leafRel) -Raw
$attnText = Get-Content -Path (Join-Path $repo $attnRel) -Raw
$runnerText = Get-Content -Path (Join-Path $repo $runnerRel) -Raw

$expectedConstNames = @(
    "kQkvCtExpectedL0WqNumWeights",
    "kQkvCtExpectedL0WqPayloadWords",
    "kQkvCtExpectedL0WqLastWordValidCount",
    "kQkvCtExpectedL0WkNumWeights",
    "kQkvCtExpectedL0WkPayloadWords",
    "kQkvCtExpectedL0WkLastWordValidCount",
    "kQkvCtExpectedL0WvNumWeights",
    "kQkvCtExpectedL0WvPayloadWords",
    "kQkvCtExpectedL0WvLastWordValidCount"
)
foreach ($name in $expectedConstNames) {
    [void](Get-ConstExpr -Text $shapeText -ConstName $name -Label $shapeRel)
}

$pairs = @(
    @{ Supported = "kQkvCtSupportedL0WqPayloadWords"; Expected = "kQkvCtExpectedL0WqPayloadWords" },
    @{ Supported = "kQkvCtSupportedL0WkPayloadWords"; Expected = "kQkvCtExpectedL0WkPayloadWords" },
    @{ Supported = "kQkvCtSupportedL0WvPayloadWords"; Expected = "kQkvCtExpectedL0WvPayloadWords" }
)
foreach ($pair in $pairs) {
    $supportedExpr = Get-ConstExpr -Text $shapeText -ConstName $pair.Supported -Label $shapeRel
    $expectedExpr = Get-ConstExpr -Text $shapeText -ConstName $pair.Expected -Label $shapeRel
    $supportedUsesExpected = $supportedExpr -match [System.Text.RegularExpressions.Regex]::Escape($pair.Expected)
    $sharedExpr = ($supportedExpr -eq $expectedExpr)
    if (-not ($supportedUsesExpected -or $sharedExpr)) {
        Fail-Check ("{0} must derive from {1} or share the same source expression chain" -f $pair.Supported, $pair.Expected)
    }
}

$leafExpectedNeedles = @(
    "kQkvCtExpectedL0WqNumWeights",
    "kQkvCtExpectedL0WqPayloadWords",
    "kQkvCtExpectedL0WqLastWordValidCount",
    "kQkvCtExpectedL0WkNumWeights",
    "kQkvCtExpectedL0WkPayloadWords",
    "kQkvCtExpectedL0WkLastWordValidCount",
    "kQkvCtExpectedL0WvNumWeights",
    "kQkvCtExpectedL0WvPayloadWords",
    "kQkvCtExpectedL0WvLastWordValidCount"
)
foreach ($needle in $leafExpectedNeedles) {
    Require-TextContains -Text $leafText -Needle $needle -Reason ("leaf kernel must consume payload-metadata expectations: {0}" -f $needle)
}

$attnExpectedNeedles = @(
    "kQkvCtExpectedL0WqPayloadWords",
    "kQkvCtExpectedL0WkPayloadWords",
    "kQkvCtExpectedL0WvPayloadWords",
    "kQkvCtPackedWordElems"
)
foreach ($needle in $attnExpectedNeedles) {
    Require-TextContains -Text $attnText -Needle $needle -Reason ("AttnLayer0 metadata guard must consume payload-metadata expectations: {0}" -f $needle)
}

$payloadGuardFiles = @(
    @{ Label = $leafRel; Text = $leafText },
    @{ Label = $attnRel; Text = $attnText }
)
$forbiddenGuardHardcodePatterns = @(
    '(?m)\bmeta\.num_weights\s*(==|!=|<=|>=|<|>)\s*1024u\b',
    '(?m)\bmeta\.payload_words_2b\s*(==|!=|<=|>=|<|>)\s*64u\b',
    '(?m)\bmeta\.last_word_valid_count\s*(==|!=|<=|>=|<|>)\s*16u\b'
)
foreach ($f in $payloadGuardFiles) {
    foreach ($pattern in $forbiddenGuardHardcodePatterns) {
        if ([System.Text.RegularExpressions.Regex]::IsMatch($f.Text, $pattern)) {
            Fail-Check ("payload metadata guard hardcode detected in {0}: pattern {1}" -f $f.Label, $pattern)
        }
    }
}

if (-not [System.Text.RegularExpressions.Regex]::IsMatch($runnerText, "check_qkv_payload_metadata_ssot\.ps1[^\r\n]*'pre'")) {
    Fail-Check "run_p11l_local_regression must invoke check_qkv_payload_metadata_ssot pre phase"
}
if (-not [System.Text.RegularExpressions.Regex]::IsMatch($runnerText, "check_qkv_payload_metadata_ssot\.ps1[^\r\n]*'post'")) {
    Fail-Check "run_p11l_local_regression must invoke check_qkv_payload_metadata_ssot post phase"
}

Assert-NoOverclaim -Text $shapeText -DocLabel $shapeRel
Assert-NoOverclaim -Text $leafText -DocLabel $leafRel
Assert-NoOverclaim -Text $attnText -DocLabel $attnRel

if ($Phase -eq "post") {
    $reportAbs = Join-Path $repo $reportRel
    Require-True -Condition (Test-Path $reportAbs) -Reason "P00-011U report missing in post phase"
    $reportText = Get-Content -Path $reportAbs -Raw

    $requiredSections = @(
        "Summary",
        "Scope",
        "Files changed",
        "Exact commands executed",
        "Actual execution evidence excerpt",
        "Result / verdict wording",
        "Limitations",
        "Why useful for later payload-metadata SSOT bridge but not closure"
    )
    foreach ($section in $requiredSections) {
        Require-Regex -Text $reportText -Pattern ("(?m)^##\s+{0}\s*$" -f [System.Text.RegularExpressions.Regex]::Escape($section)) -Reason ("P00-011U report missing section: {0}" -f $section)
    }

    $requiredCommandMarkers = @(
        "scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11u -Phase pre",
        "scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11u -Phase post",
        "scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11u",
        "scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11u",
        "scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11u"
    )
    foreach ($marker in $requiredCommandMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011U report missing command marker: {0}" -f $marker)
    }

    $requiredEvidenceMarkers = @(
        "PASS: check_qkv_payload_metadata_ssot",
        "PASS: run_p11r_compile_prep",
        "PASS: run_p11s_compile_prep_family",
        "PASS: run_p11l_local_regression"
    )
    foreach ($marker in $requiredEvidenceMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011U report missing evidence marker: {0}" -f $marker)
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
        Require-TextContains -Text $pair.Text -Needle "P00-011U" -Reason ("governance doc missing P00-011U: {0}" -f $pair.Label)
        Assert-NoOverclaim -Text $pair.Text -DocLabel $pair.Label
    }

    $combinedText = $statusText + "`n" + $traceText + "`n" + $closureText + "`n" + $handoffText + "`n" + $reportText
    Require-Regex -Text $combinedText -Pattern '(?i)\blocal-only\b' -Reason "required local-only wording missing for P00-011U"
    Require-Regex -Text $combinedText -Pattern '(?i)\bnot Catapult closure\b' -Reason "required not Catapult closure wording missing for P00-011U"
    Require-Regex -Text $combinedText -Pattern '(?i)\bnot SCVerify closure\b' -Reason "required not SCVerify closure wording missing for P00-011U"

    foreach ($id in @("P00-011Q", "P00-011R", "P00-011S", "P00-011T")) {
        Require-BaselineContinuity -Text $combinedText -Milestone $id
    }

    Assert-NoOverclaim -Text $reportText -DocLabel $reportRel
}

Write-Log "PASS: check_qkv_payload_metadata_ssot"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
