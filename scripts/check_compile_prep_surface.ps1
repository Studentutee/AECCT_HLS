param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11r",
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

$logPath = Join-Path $outDirAbs "check_compile_prep_surface.log"
$summaryPath = Join-Path $outDirAbs "check_compile_prep_surface_summary.txt"
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
    Write-Log "FAIL: check_compile_prep_surface"
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

function Assert-McScverifyIncludeOrder {
    param(
        [string]$Text,
        [string]$RelPath
    )

    if ($Text -notmatch 'mc_scverify\.h') {
        return
    }

    Require-Regex -Text $Text -Pattern '#if\s+__has_include\s*\(\s*<mc_scverify\.h>\s*\)[\s\S]*?#include\s*<mc_scverify\.h>[\s\S]*?#endif' -Reason ("mc_scverify include must be guarded by __has_include in {0}" -f $RelPath)

    $lines = $Text -split "`r?`n"
    $mcLine = -1
    $lastIncludeLine = -1
    $firstTypeOrFuncLine = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        if ($line -match '^\s*#\s*include\b') {
            $lastIncludeLine = $i
        }
        if ($line -match 'mc_scverify\.h') {
            $mcLine = $i
        }
        if ($firstTypeOrFuncLine -lt 0 -and ($line -match '^\s*(class|struct)\s+[A-Za-z_][A-Za-z0-9_]*\b' -or $line -match '^\s*CCS_MAIN\s*\(')) {
            $firstTypeOrFuncLine = $i
        }
    }

    Require-True -Condition ($mcLine -ge 0) -Reason ("failed to locate mc_scverify include line in {0}" -f $RelPath)
    Require-True -Condition ($mcLine -eq $lastIncludeLine) -Reason ("mc_scverify include must be after last include in {0}" -f $RelPath)
    if ($firstTypeOrFuncLine -ge 0) {
        Require-True -Condition ($mcLine -lt $firstTypeOrFuncLine) -Reason ("mc_scverify include must appear before class/function definition in {0}" -f $RelPath)
    }
}

Add-Content -Path $logPath -Value ("===== check_compile_prep_surface phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11r] phase={0}" -f $Phase)

$topRel = "src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h"
$tbRel = "tb/tb_ternary_live_leaf_top_compile_prep_p11r.cpp"
$runnerRel = "scripts/local/run_p11r_compile_prep.ps1"
$reportRel = "docs/milestones/P00-011R_report.md"
$handoffRulesRel = "docs/process/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md"
$statusRel = "docs/process/PROJECT_STATUS_zhTW.txt"
$traceRel = "docs/milestones/TRACEABILITY_MAP_v12.1.md"
$closureRel = "docs/milestones/CLOSURE_MATRIX_v12.1.md"

$mustExistPre = @(
    $topRel,
    $tbRel,
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

$topText = Get-Content -Path (Join-Path $repo $topRel) -Raw
$tbText = Get-Content -Path (Join-Path $repo $tbRel) -Raw
$runnerText = Get-Content -Path (Join-Path $repo $runnerRel) -Raw

Require-Regex -Text $topText -Pattern '(?m)^\s*#pragma\s+hls_design\s+top\s*$' -Reason "compile-prep top missing '#pragma hls_design top'"
Require-Regex -Text $topText -Pattern '(?m)^\s*#pragma\s+hls_design\s+interface\s*$' -Reason "compile-prep top missing '#pragma hls_design interface'"
Require-Regex -Text $topText -Pattern '(?m)^\s*bool\s+CCS_BLOCK\(run\)\s*\(' -Reason "compile-prep top missing same-line 'CCS_BLOCK(run)' run method"
Require-Regex -Text $topText -Pattern '(?ms)#ifndef\s+CCS_BLOCK\s*#define\s+CCS_BLOCK\(name\)\s+name\s*#endif' -Reason "compile-prep top missing CCS_BLOCK fallback macro"
Assert-McScverifyIncludeOrder -Text $topText -RelPath $topRel

$runMatch = [System.Text.RegularExpressions.Regex]::Match(
    $topText,
    'bool\s+CCS_BLOCK\(run\)\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\s*\}',
    [System.Text.RegularExpressions.RegexOptions]::Singleline)
Require-True -Condition $runMatch.Success -Reason "failed to parse compile-prep top run() body"
$runBody = $runMatch.Groups['body'].Value
Require-Regex -Text $runBody -Pattern 'return\s+ternary_live_l0_wq_materialize_row_kernel_split\s*\(' -Reason "compile-prep top run() must forward to accepted split materialize function"
if ([System.Text.RegularExpressions.Regex]::IsMatch($runBody, '\b(for|while|if|switch)\s*\(')) {
    Fail-Check "compile-prep top run() must remain thin-wrapper only (control logic detected)"
}

$macroP11m = ('AECCT_LOCAL_' + 'P11M_WQ_SPLIT_TOP_ENABLE')
$macroP11n = ('AECCT_LOCAL_' + 'P11N_WK_WV_SPLIT_TOP_ENABLE')
foreach ($macroName in @($macroP11m, $macroP11n)) {
    if ($topText -match [System.Text.RegularExpressions.Regex]::Escape($macroName)) {
        Fail-Check ("macro leakage detected in compile-prep top: {0}" -f $macroName)
    }
    if ($tbText -match [System.Text.RegularExpressions.Regex]::Escape($macroName)) {
        Fail-Check ("macro leakage detected in compile-prep TB: {0}" -f $macroName)
    }
}

Assert-McScverifyIncludeOrder -Text $tbText -RelPath $tbRel
Require-Regex -Text $tbText -Pattern '(?m)^\s*class\s+[A-Za-z_][A-Za-z0-9_]*\s*\{' -Reason "compile-prep TB must be class-based"
Require-Regex -Text $tbText -Pattern '\brun_all\s*\(' -Reason "compile-prep TB missing run_all()"
Require-Regex -Text $tbText -Pattern '\bCCS_MAIN\s*\(' -Reason "compile-prep TB missing CCS_MAIN"
Require-Regex -Text $tbText -Pattern '\bCCS_RETURN\s*\(' -Reason "compile-prep TB missing CCS_RETURN"
Require-TextContains -Text $tbText -Needle "PASS: tb_ternary_live_leaf_top_compile_prep_p11r" -Reason "compile-prep TB missing fixed PASS banner"
if ([System.Text.RegularExpressions.Regex]::IsMatch($tbText, '\bstd::exit\s*\(')) {
    Fail-Check "compile-prep TB must not use std::exit"
}
if ([System.Text.RegularExpressions.Regex]::IsMatch($tbText, '(?i)local smoke|handoff freeze')) {
    Fail-Check "compile-prep TB must not self-label as local smoke or handoff freeze"
}
if ([System.Text.RegularExpressions.Regex]::IsMatch($tbText, 'PASS:\s*tb_ternary_live_leaf_top_smoke_')) {
    Fail-Check "compile-prep TB contains unrelated PASS banner wording"
}

Require-TextContains -Text $runnerText -Needle "build_p11r_compile_prep.log" -Reason "local runner missing build log contract"
Require-TextContains -Text $runnerText -Needle "run_p11r_compile_prep.log" -Reason "local runner missing run log contract"
Require-TextContains -Text $runnerText -Needle "PASS: tb_ternary_live_leaf_top_compile_prep_p11r" -Reason "local runner missing TB PASS gate"
Require-TextContains -Text $runnerText -Needle "PASS: run_p11r_compile_prep" -Reason "local runner missing final PASS banner"

Assert-NoOverclaim -Text $topText -DocLabel $topRel
Assert-NoOverclaim -Text $tbText -DocLabel $tbRel

if ($Phase -eq "post") {
    $reportAbs = Join-Path $repo $reportRel
    Require-True -Condition (Test-Path $reportAbs) -Reason "P00-011R report missing in post phase"
    $reportText = Get-Content -Path $reportAbs -Raw

    $requiredSections = @(
        "Summary",
        "Scope",
        "Files changed",
        "Exact commands executed",
        "Actual execution evidence excerpt",
        "Result / verdict wording",
        "Limitations",
        "Why useful for later Catapult run but not closure"
    )
    foreach ($section in $requiredSections) {
        Require-Regex -Text $reportText -Pattern ("(?m)^##\s+{0}\s*$" -f [System.Text.RegularExpressions.Regex]::Escape($section)) -Reason ("P00-011R report missing section: {0}" -f $section)
    }

    $requiredCommandMarkers = @(
        "scripts/check_handoff_surface.ps1 -OutDir build\p11r -Phase pre",
        "scripts/check_compile_prep_surface.ps1 -OutDir build\p11r -Phase pre",
        "scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11r",
        "scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11r",
        "scripts/check_handoff_surface.ps1 -OutDir build\p11r -Phase post",
        "scripts/check_compile_prep_surface.ps1 -OutDir build\p11r -Phase post"
    )
    foreach ($marker in $requiredCommandMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011R report missing command marker: {0}" -f $marker)
    }

    $statusText = Get-Content -Path (Join-Path $repo $statusRel) -Raw
    $traceText = Get-Content -Path (Join-Path $repo $traceRel) -Raw
    $closureText = Get-Content -Path (Join-Path $repo $closureRel) -Raw
    $handoffText = Get-Content -Path (Join-Path $repo $handoffRulesRel) -Raw

    foreach ($pair in @(
            @{ Label = $statusRel; Text = $statusText },
            @{ Label = $traceRel; Text = $traceText },
            @{ Label = $closureRel; Text = $closureText })) {
        Require-TextContains -Text $pair.Text -Needle "P00-011R" -Reason ("governance doc missing P00-011R: {0}" -f $pair.Label)
        Assert-NoOverclaim -Text $pair.Text -DocLabel $pair.Label
    }
    Require-TextContains -Text $handoffText -Needle "P00-011R" -Reason "handoff rules missing P00-011R entry"

    $requiredWording = @(
        "first Catapult-facing compile-prep probe",
        "single-slice representative",
        "local compiler evidence only",
        "not Catapult closure",
        "not SCVerify closure",
        "accepted local-only progress remains valid",
        "P00-011Q freeze boundary remains authoritative"
    )
    $combinedGovernanceText = $statusText + "`n" + $traceText + "`n" + $closureText + "`n" + $handoffText + "`n" + $reportText
    foreach ($phrase in $requiredWording) {
        Require-TextContains -Text $combinedGovernanceText -Needle $phrase -Reason ("required wording missing for P00-011R: {0}" -f $phrase)
    }

    Assert-NoOverclaim -Text $reportText -DocLabel $reportRel
}

Write-Log "PASS: check_compile_prep_surface"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
