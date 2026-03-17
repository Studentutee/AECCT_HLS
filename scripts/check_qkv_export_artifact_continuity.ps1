param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11w",
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

$logPath = Join-Path $outDirAbs "check_qkv_export_artifact_continuity.log"
$summaryPath = Join-Path $outDirAbs "check_qkv_export_artifact_continuity_summary.txt"
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
    Write-Log "FAIL: check_qkv_export_artifact_continuity"
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

function Parse-U32Token {
    param(
        [string]$Token,
        [string]$Label
    )

    $t = $Token.Trim()
    if ($t -match '^(?<n>\d+)(?:u)?$') {
        return [uint32]$Matches['n']
    }
    Fail-Check ("cannot parse uint32 token for {0}: {1}" -f $Label, $Token)
}

function Evaluate-PayloadWordsExpr {
    param(
        [string]$Expr,
        [string]$Label
    )

    $e = $Expr.Trim()
    if ($e -match '^(?<n>\d+)(?:u)?$') {
        return [uint32]$Matches['n']
    }
    if ($e -match '^ternary_payload_words_2b\s*\(\s*(?<n>\d+)(?:u)?\s*\)$') {
        $n = [uint32]$Matches['n']
        return [uint32][Math]::Floor((([double]$n) + 15.0) / 16.0)
    }
    Fail-Check ("unsupported payload_words_2b expression for {0}: {1}" -f $Label, $Expr)
}

function Evaluate-LastWordValidCountExpr {
    param(
        [string]$Expr,
        [string]$Label
    )

    $e = $Expr.Trim()
    if ($e -match '^(?<n>\d+)(?:u)?$') {
        return [uint32]$Matches['n']
    }
    if ($e -match '^ternary_last_word_valid_count\s*\(\s*(?<n>\d+)(?:u)?\s*\)$') {
        $n = [uint32]$Matches['n']
        $rem = [uint32]($n % 16)
        if ($rem -eq 0) {
            return [uint32]16
        }
        return $rem
    }
    Fail-Check ("unsupported last_word_valid_count expression for {0}: {1}" -f $Label, $Expr)
}

function Get-WsoQkvExpectedMeta {
    param(
        [string]$WsoText,
        [string]$MatrixEnum
    )

    $enumEsc = [System.Text.RegularExpressions.Regex]::Escape($MatrixEnum)
    $pattern = ("(?m)\{{\s*{0}\s*,\s*\d+u?\s*,\s*\d+u?\s*,\s*(?<rows>\d+u?)\s*,\s*(?<cols>\d+u?)\s*,\s*(?<num>\d+u?)\s*,\s*(?<payload>[^,]+?)\s*,\s*(?<last>[^,]+?)\s*,\s*QLAYOUT_TERNARY_W_OUT_IN\s*\}}" -f $enumEsc)
    $m = [System.Text.RegularExpressions.Regex]::Match($WsoText, $pattern)
    if (-not $m.Success) {
        Fail-Check ("cannot locate QuantLinearMeta row for {0} in gen/include/WeightStreamOrder.h" -f $MatrixEnum)
    }

    $rows = Parse-U32Token -Token $m.Groups['rows'].Value -Label ("{0}.rows" -f $MatrixEnum)
    $cols = Parse-U32Token -Token $m.Groups['cols'].Value -Label ("{0}.cols" -f $MatrixEnum)
    $num = Parse-U32Token -Token $m.Groups['num'].Value -Label ("{0}.num_weights" -f $MatrixEnum)
    $payload = Evaluate-PayloadWordsExpr -Expr $m.Groups['payload'].Value -Label ("{0}.payload_words_2b" -f $MatrixEnum)
    $last = Evaluate-LastWordValidCountExpr -Expr $m.Groups['last'].Value -Label ("{0}.last_word_valid_count" -f $MatrixEnum)

    return [ordered]@{
        rows = $rows
        cols = $cols
        num_weights = $num
        payload_words_2b = $payload
        last_word_valid_count = $last
    }
}

function Get-JsonUInt32Field {
    param(
        [object]$JsonObj,
        [string]$Field,
        [string]$Label
    )

    $has = $null -ne $JsonObj.PSObject.Properties[$Field]
    if (-not $has) {
        Fail-Check ("missing JSON field {0} in {1}" -f $Field, $Label)
    }
    $valueObj = $JsonObj.$Field
    try {
        return [uint32]$valueObj
    }
    catch {
        Fail-Check ("JSON field {0} is not uint32-compatible in {1}: {2}" -f $Field, $Label, $valueObj)
    }
}

Add-Content -Path $logPath -Value ("===== check_qkv_export_artifact_continuity phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11w] phase={0}" -f $Phase)

$jsonRel = "gen/ternary_p11c_export.json"
$wsoRel = "gen/WeightStreamOrder.h"
$wsoGenRel = "gen/include/WeightStreamOrder.h"
$shapeRel = "src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h"
$fenceRel = "src/blocks/TernaryLiveQkvWeightStreamOrderContinuityFence.h"
$runnerRel = "scripts/local/run_p11l_local_regression.ps1"
$payloadCheckerRel = "scripts/check_qkv_payload_metadata_ssot.ps1"
$wsoCheckerRel = "scripts/check_qkv_weightstreamorder_continuity.ps1"
$tbExportRel = "tb/tb_ternary_export_p11c.cpp"
$reportRel = "docs/milestones/P00-011W_report.md"
$handoffRulesRel = "docs/process/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md"
$statusRel = "docs/process/PROJECT_STATUS_zhTW.txt"
$traceRel = "docs/milestones/TRACEABILITY_MAP_v12.1.md"
$closureRel = "docs/milestones/CLOSURE_MATRIX_v12.1.md"

$mustExistPre = @(
    $jsonRel,
    $wsoRel,
    $wsoGenRel,
    $shapeRel,
    $fenceRel,
    $runnerRel,
    $payloadCheckerRel,
    $wsoCheckerRel,
    $tbExportRel,
    $handoffRulesRel,
    $statusRel,
    $traceRel,
    $closureRel
)
foreach ($rel in $mustExistPre) {
    $abs = Join-Path $repo $rel
    Require-True -Condition (Test-Path $abs) -Reason ("required file missing: {0}" -f $rel)
}

$jsonText = Get-Content -Path (Join-Path $repo $jsonRel) -Raw
$wsoText = Get-Content -Path (Join-Path $repo $wsoGenRel) -Raw
$shapeText = Get-Content -Path (Join-Path $repo $shapeRel) -Raw
$fenceText = Get-Content -Path (Join-Path $repo $fenceRel) -Raw
$runnerText = Get-Content -Path (Join-Path $repo $runnerRel) -Raw
$tbExportText = Get-Content -Path (Join-Path $repo $tbExportRel) -Raw

$jsonObj = $null
try {
    $jsonObj = $jsonText | ConvertFrom-Json
}
catch {
    Fail-Check ("invalid JSON parse for {0}: {1}" -f $jsonRel, $_.Exception.Message)
}

Require-True -Condition ($null -ne $jsonObj) -Reason ("failed to parse {0}" -f $jsonRel)
Require-True -Condition ($null -ne $jsonObj.matrices) -Reason ("missing matrices array in {0}" -f $jsonRel)
$allMatrices = @($jsonObj.matrices)
Require-True -Condition ($allMatrices.Count -gt 0) -Reason ("empty matrices array in {0}" -f $jsonRel)

$qkvMap = [ordered]@{
    L0_WQ = "QLM_L0_WQ"
    L0_WK = "QLM_L0_WK"
    L0_WV = "QLM_L0_WV"
}

foreach ($pair in $qkvMap.GetEnumerator()) {
    $matrixName = $pair.Key
    $matrixEnum = $pair.Value
    $records = @($allMatrices | Where-Object { $_.matrix_id -eq $matrixName })
    if ($records.Count -ne 1) {
        Fail-Check ("JSON must contain exactly one record for matrix_id={0}; found {1}" -f $matrixName, $records.Count)
    }

    $expected = Get-WsoQkvExpectedMeta -WsoText $wsoText -MatrixEnum $matrixEnum
    $record = $records[0]
    $recordLabel = ("{0} record in {1}" -f $matrixName, $jsonRel)

    $rows = Get-JsonUInt32Field -JsonObj $record -Field "rows" -Label $recordLabel
    $cols = Get-JsonUInt32Field -JsonObj $record -Field "cols" -Label $recordLabel
    $num = Get-JsonUInt32Field -JsonObj $record -Field "num_weights" -Label $recordLabel
    $payload = Get-JsonUInt32Field -JsonObj $record -Field "payload_words_2b" -Label $recordLabel
    $last = Get-JsonUInt32Field -JsonObj $record -Field "last_word_valid_count" -Label $recordLabel

    if ($rows -ne $expected.rows) {
        Fail-Check ("rows mismatch for {0}: json={1}, expected={2}" -f $matrixName, $rows, $expected.rows)
    }
    if ($cols -ne $expected.cols) {
        Fail-Check ("cols mismatch for {0}: json={1}, expected={2}" -f $matrixName, $cols, $expected.cols)
    }
    if ($num -ne $expected.num_weights) {
        Fail-Check ("num_weights mismatch for {0}: json={1}, expected={2}" -f $matrixName, $num, $expected.num_weights)
    }
    if ($payload -ne $expected.payload_words_2b) {
        Fail-Check ("payload_words_2b mismatch for {0}: json={1}, expected={2}" -f $matrixName, $payload, $expected.payload_words_2b)
    }
    if ($last -ne $expected.last_word_valid_count) {
        Fail-Check ("last_word_valid_count mismatch for {0}: json={1}, expected={2}" -f $matrixName, $last, $expected.last_word_valid_count)
    }
}

Require-Regex -Text $fenceText -Pattern 'kQuantLinearMeta\s*\[\s*\(uint32_t\)\s*QLM_L0_WQ\s*\]' -Reason "continuity fence must reference QLM_L0_WQ metadata"
Require-Regex -Text $fenceText -Pattern 'kQuantLinearMeta\s*\[\s*\(uint32_t\)\s*QLM_L0_WK\s*\]' -Reason "continuity fence must reference QLM_L0_WK metadata"
Require-Regex -Text $fenceText -Pattern 'kQuantLinearMeta\s*\[\s*\(uint32_t\)\s*QLM_L0_WV\s*\]' -Reason "continuity fence must reference QLM_L0_WV metadata"
foreach ($field in @("matrix_id", "rows", "cols", "num_weights", "payload_words_2b", "last_word_valid_count")) {
    Require-Regex -Text $fenceText -Pattern ("(?s)static_assert\s*\([^;]*{0}" -f [System.Text.RegularExpressions.Regex]::Escape($field)) -Reason ("continuity fence missing static_assert for field: {0}" -f $field)
}

$forbiddenSecondSourcePatterns = @(
    '(?m)^\s*static\s+constexpr\s+QuantLinearMeta\s+\w+\s*\[',
    '(?m)^\s*QuantLinearMeta\s+\w+\s*\[[^\]]+\]\s*=\s*\{'
)
foreach ($pattern in $forbiddenSecondSourcePatterns) {
    if ([System.Text.RegularExpressions.Regex]::IsMatch($fenceText, $pattern)) {
        Fail-Check ("second metadata definition point detected in continuity fence: pattern {0}" -f $pattern)
    }
}

Require-Regex -Text $tbExportText -Pattern 'kQuantLinearMeta\s*\[\s*i\s*\]' -Reason "tb_ternary_export_p11c must consume authoritative kQuantLinearMeta table"

if (-not [System.Text.RegularExpressions.Regex]::IsMatch($runnerText, "check_qkv_export_artifact_continuity\.ps1[^\r\n]*'pre'")) {
    Fail-Check "run_p11l_local_regression must invoke check_qkv_export_artifact_continuity pre phase"
}
if (-not [System.Text.RegularExpressions.Regex]::IsMatch($runnerText, "check_qkv_export_artifact_continuity\.ps1[^\r\n]*'post'")) {
    Fail-Check "run_p11l_local_regression must invoke check_qkv_export_artifact_continuity post phase"
}

Assert-NoOverclaim -Text $shapeText -DocLabel $shapeRel
Assert-NoOverclaim -Text $fenceText -DocLabel $fenceRel
Assert-NoOverclaim -Text $tbExportText -DocLabel $tbExportRel

if ($Phase -eq "post") {
    $reportAbs = Join-Path $repo $reportRel
    Require-True -Condition (Test-Path $reportAbs) -Reason "P00-011W report missing in post phase"
    $reportText = Get-Content -Path $reportAbs -Raw

    $requiredSections = @(
        "Summary",
        "Scope",
        "Files changed",
        "Exact commands executed",
        "Actual execution evidence excerpt",
        "Result / verdict wording",
        "Limitations",
        "Why useful for later exported-artifact/loader-facing continuity fence but not closure"
    )
    foreach ($section in $requiredSections) {
        Require-Regex -Text $reportText -Pattern ("(?m)^##\s+{0}\s*$" -f [System.Text.RegularExpressions.Regex]::Escape($section)) -Reason ("P00-011W report missing section: {0}" -f $section)
    }

    $requiredCommandMarkers = @(
        "scripts/check_qkv_export_artifact_continuity.ps1 -OutDir build\p11w -Phase pre",
        "scripts/check_qkv_export_artifact_continuity.ps1 -OutDir build\p11w -Phase post",
        "scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11w",
        "scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11w",
        "scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11w"
    )
    foreach ($marker in $requiredCommandMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011W report missing command marker: {0}" -f $marker)
    }

    $requiredEvidenceMarkers = @(
        "PASS: check_qkv_export_artifact_continuity",
        "PASS: run_p11r_compile_prep",
        "PASS: run_p11s_compile_prep_family"
    )
    foreach ($marker in $requiredEvidenceMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011W report missing evidence marker: {0}" -f $marker)
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
        Require-TextContains -Text $pair.Text -Needle "P00-011W" -Reason ("governance doc missing P00-011W: {0}" -f $pair.Label)
        Assert-NoOverclaim -Text $pair.Text -DocLabel $pair.Label
    }

    $combinedText = $statusText + "`n" + $traceText + "`n" + $closureText + "`n" + $handoffText + "`n" + $reportText
    Require-Regex -Text $combinedText -Pattern '(?i)\blocal-only\b' -Reason "required local-only wording missing for P00-011W"
    Require-Regex -Text $combinedText -Pattern '(?i)\bnot Catapult closure\b' -Reason "required not Catapult closure wording missing for P00-011W"
    Require-Regex -Text $combinedText -Pattern '(?i)\bnot SCVerify closure\b' -Reason "required not SCVerify closure wording missing for P00-011W"

    foreach ($id in @("P00-011Q", "P00-011R", "P00-011S", "P00-011T", "P00-011U", "P00-011V")) {
        Require-BaselineContinuity -Text $combinedText -Milestone $id
    }

    Assert-NoOverclaim -Text $reportText -DocLabel $reportRel
}

Write-Log "PASS: check_qkv_export_artifact_continuity"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
