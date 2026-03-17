param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11s",
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

$logPath = Join-Path $outDirAbs "check_compile_prep_family_surface.log"
$summaryPath = Join-Path $outDirAbs "check_compile_prep_family_surface_summary.txt"
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
    Write-Log "FAIL: check_compile_prep_family_surface"
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

function Get-FunctionReturnType {
    param(
        [string]$Text,
        [string]$FunctionName,
        [string]$DocLabel
    )

    $fn = [System.Text.RegularExpressions.Regex]::Escape($FunctionName)
    $m = [System.Text.RegularExpressions.Regex]::Match(
        $Text,
        ("(?ms)\bstatic\s+inline\s+(?<rt>[A-Za-z_][A-Za-z0-9_:<>]*)\s+{0}\s*\(" -f $fn))
    if (-not $m.Success) {
        Fail-Check ("failed to parse return type for {0} in {1}" -f $FunctionName, $DocLabel)
    }
    return $m.Groups["rt"].Value
}

function Assert-TopClassForwarding {
    param(
        [string]$TopText,
        [string]$ClassName,
        [string]$SplitFuncName,
        [string]$ExpectedReturnType
    )

    $classEsc = [System.Text.RegularExpressions.Regex]::Escape($ClassName)
    $classMatch = [System.Text.RegularExpressions.Regex]::Match(
        $TopText,
        ("(?ms)#pragma\s+hls_design\s+top\s*class\s+{0}\s*\{{(?<body>.*?)\n\}};" -f $classEsc))
    if (-not $classMatch.Success) {
        Fail-Check ("missing compile-prep class with top pragma: {0}" -f $ClassName)
    }
    $body = $classMatch.Groups["body"].Value

    Require-Regex -Text $body -Pattern '(?m)^\s*#pragma\s+hls_design\s+interface\s*$' -Reason ("missing '#pragma hls_design interface' in {0}" -f $ClassName)
    $runDeclMatch = [System.Text.RegularExpressions.Regex]::Match(
        $body,
        '(?m)^\s*(?<rt>[A-Za-z_][A-Za-z0-9_:<>]*)\s+CCS_BLOCK\(run\)\s*\(')
    if (-not $runDeclMatch.Success) {
        Fail-Check ("missing same-line CCS_BLOCK(run) method declaration in {0}" -f $ClassName)
    }

    $runReturnType = $runDeclMatch.Groups["rt"].Value
    if ($runReturnType -ne $ExpectedReturnType) {
        Fail-Check ("{0} run return type mismatch: got '{1}', expect '{2}' from accepted split/materialize contract" -f $ClassName, $runReturnType, $ExpectedReturnType)
    }

    $runMatch = [System.Text.RegularExpressions.Regex]::Match(
        $body,
        '(?ms)(?<rt>[A-Za-z_][A-Za-z0-9_:<>]*)\s+CCS_BLOCK\(run\)\s*\((?<args>[\s\S]*?)\)\s*\{(?<runBody>[\s\S]*?)\n\s*\}')
    if (-not $runMatch.Success) {
        Fail-Check ("failed to parse run() body in {0}" -f $ClassName)
    }
    $runBody = $runMatch.Groups["runBody"].Value
    $fnEsc = [System.Text.RegularExpressions.Regex]::Escape($SplitFuncName)
    Require-Regex -Text $runBody -Pattern ("(?ms)^\s*return\s+{0}\s*\(" -f $fnEsc) -Reason ("{0} run() must forward to {1}" -f $ClassName, $SplitFuncName)
    if ([System.Text.RegularExpressions.Regex]::IsMatch($runBody, '\b(for|while|if|switch)\s*\(')) {
        Fail-Check ("{0} run() must remain thin-wrapper only (control logic detected)" -f $ClassName)
    }
    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($runBody, ("(?ms)^\s*return\s+{0}\s*\([\s\S]*\)\s*;\s*$" -f $fnEsc))) {
        Fail-Check ("{0} run() must only contain a direct return-forwarding statement" -f $ClassName)
    }
}

Add-Content -Path $logPath -Value ("===== check_compile_prep_family_surface phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11s] phase={0}" -f $Phase)

$topRel = "src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h"
$tbRel = "tb/tb_ternary_live_leaf_top_compile_prep_family_p11s.cpp"
$runnerRel = "scripts/local/run_p11s_compile_prep_family.ps1"
$reportRel = "docs/milestones/P00-011S_report.md"
$handoffRulesRel = "docs/process/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md"
$statusRel = "docs/process/PROJECT_STATUS_zhTW.txt"
$traceRel = "docs/milestones/TRACEABILITY_MAP_v12.1.md"
$closureRel = "docs/milestones/CLOSURE_MATRIX_v12.1.md"
$kernelRel = "src/blocks/TernaryLiveQkvLeafKernel.h"

$mustExistPre = @(
    $topRel,
    $tbRel,
    $runnerRel,
    $handoffRulesRel,
    $statusRel,
    $traceRel,
    $closureRel,
    $kernelRel
)
foreach ($rel in $mustExistPre) {
    $abs = Join-Path $repo $rel
    Require-True -Condition (Test-Path $abs) -Reason ("required file missing: {0}" -f $rel)
}

$topText = Get-Content -Path (Join-Path $repo $topRel) -Raw
$tbText = Get-Content -Path (Join-Path $repo $tbRel) -Raw
$runnerText = Get-Content -Path (Join-Path $repo $runnerRel) -Raw
$kernelText = Get-Content -Path (Join-Path $repo $kernelRel) -Raw

Require-Regex -Text $topText -Pattern '(?ms)#ifndef\s+CCS_BLOCK\s*#define\s+CCS_BLOCK\(name\)\s+name\s*#endif' -Reason "compile-prep top missing CCS_BLOCK fallback macro"
Assert-McScverifyIncludeOrder -Text $topText -RelPath $topRel

$wqRt = Get-FunctionReturnType -Text $kernelText -FunctionName 'ternary_live_l0_wq_materialize_row_kernel_split' -DocLabel $kernelRel
$wkRt = Get-FunctionReturnType -Text $kernelText -FunctionName 'ternary_live_l0_wk_materialize_row_kernel_split' -DocLabel $kernelRel
$wvRt = Get-FunctionReturnType -Text $kernelText -FunctionName 'ternary_live_l0_wv_materialize_row_kernel_split' -DocLabel $kernelRel

Assert-TopClassForwarding -TopText $topText -ClassName 'TernaryLiveL0WqRowTopCatapultPrep' -SplitFuncName 'ternary_live_l0_wq_materialize_row_kernel_split' -ExpectedReturnType $wqRt
Assert-TopClassForwarding -TopText $topText -ClassName 'TernaryLiveL0WkRowTopCatapultPrep' -SplitFuncName 'ternary_live_l0_wk_materialize_row_kernel_split' -ExpectedReturnType $wkRt
Assert-TopClassForwarding -TopText $topText -ClassName 'TernaryLiveL0WvRowTopCatapultPrep' -SplitFuncName 'ternary_live_l0_wv_materialize_row_kernel_split' -ExpectedReturnType $wvRt

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
Require-Regex -Text $tbText -Pattern '(?m)^\s*class\s+[A-Za-z_][A-Za-z0-9_]*\s*\{' -Reason "compile-prep family TB must be class-based"
Require-Regex -Text $tbText -Pattern '\brun_all\s*\(' -Reason "compile-prep family TB missing run_all()"
Require-Regex -Text $tbText -Pattern '\brun_wk_subtest\s*\(' -Reason "compile-prep family TB missing run_wk_subtest()"
Require-Regex -Text $tbText -Pattern '\brun_wv_subtest\s*\(' -Reason "compile-prep family TB missing run_wv_subtest()"
Require-Regex -Text $tbText -Pattern '\bCCS_MAIN\s*\(' -Reason "compile-prep family TB missing CCS_MAIN"
Require-Regex -Text $tbText -Pattern '\bCCS_RETURN\s*\(' -Reason "compile-prep family TB missing CCS_RETURN"
Require-TextContains -Text $tbText -Needle "PASS: tb_ternary_live_leaf_top_compile_prep_family_p11s" -Reason "compile-prep family TB missing fixed PASS banner"
if ([System.Text.RegularExpressions.Regex]::IsMatch($tbText, '\bstd::exit\s*\(')) {
    Fail-Check "compile-prep family TB must not use std::exit"
}
if ([System.Text.RegularExpressions.Regex]::IsMatch($tbText, '(?i)local smoke|handoff freeze')) {
    Fail-Check "compile-prep family TB must not self-label as local smoke or handoff freeze"
}

Require-TextContains -Text $runnerText -Needle "build_p11s_compile_prep_family.log" -Reason "local runner missing build log contract"
Require-TextContains -Text $runnerText -Needle "run_p11s_compile_prep_family.log" -Reason "local runner missing run log contract"
Require-TextContains -Text $runnerText -Needle "PASS: tb_ternary_live_leaf_top_compile_prep_family_p11s" -Reason "local runner missing TB PASS gate"
Require-TextContains -Text $runnerText -Needle "PASS: run_p11s_compile_prep_family" -Reason "local runner missing final PASS banner"

Assert-NoMojibakeMarkers -Text $tbText -DocLabel $tbRel
Assert-NoMojibakeMarkers -Text $runnerText -DocLabel $runnerRel
Assert-NoMojibakeMarkers -Text $topText -DocLabel $topRel

Assert-NoOverclaim -Text $topText -DocLabel $topRel
Assert-NoOverclaim -Text $tbText -DocLabel $tbRel

if ($Phase -eq "post") {
    $reportAbs = Join-Path $repo $reportRel
    Require-True -Condition (Test-Path $reportAbs) -Reason "P00-011S report missing in post phase"
    $reportText = Get-Content -Path $reportAbs -Raw

    $requiredSections = @(
        "Summary",
        "Scope",
        "Files changed",
        "Exact commands executed",
        "Actual execution evidence excerpt",
        "Result / verdict wording",
        "Limitations",
        "Why useful for later Catapult family prep but not closure"
    )
    foreach ($section in $requiredSections) {
        Require-Regex -Text $reportText -Pattern ("(?m)^##\s+{0}\s*$" -f [System.Text.RegularExpressions.Regex]::Escape($section)) -Reason ("P00-011S report missing section: {0}" -f $section)
    }

    $requiredCommandMarkers = @(
        "scripts/check_handoff_surface.ps1 -OutDir build\p11s -Phase pre",
        "scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11s -Phase pre",
        "scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11s",
        "scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11s",
        "scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11s",
        "scripts/check_handoff_surface.ps1 -OutDir build\p11s -Phase post",
        "scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11s -Phase post"
    )
    foreach ($marker in $requiredCommandMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011S report missing command marker: {0}" -f $marker)
    }

    $requiredEvidenceMarkers = @(
        "PASS: check_compile_prep_family_surface",
        "PASS: tb_ternary_live_leaf_top_compile_prep_family_p11s",
        "PASS: run_p11s_compile_prep_family",
        "PASS: run_p11r_compile_prep",
        "PASS: run_p11l_local_regression"
    )
    foreach ($marker in $requiredEvidenceMarkers) {
        Require-TextContains -Text $reportText -Needle $marker -Reason ("P00-011S report missing evidence marker: {0}" -f $marker)
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
        Require-TextContains -Text $pair.Text -Needle "P00-011S" -Reason ("governance doc missing P00-011S: {0}" -f $pair.Label)
        Assert-NoOverclaim -Text $pair.Text -DocLabel $pair.Label
    }

    $requiredWording = @(
        "WK/WV family compile-prep expansion",
        "local compiler evidence only",
        "not Catapult closure",
        "not SCVerify closure",
        "P00-011Q handoff freeze remains authoritative",
        "P00-011R WQ compile-prep probe remains valid baseline"
    )
    $combinedGovernanceText = $statusText + "`n" + $traceText + "`n" + $closureText + "`n" + $handoffText + "`n" + $reportText
    foreach ($phrase in $requiredWording) {
        Require-TextContains -Text $combinedGovernanceText -Needle $phrase -Reason ("required wording missing for P00-011S: {0}" -f $phrase)
    }

    Assert-NoOverclaim -Text $reportText -DocLabel $reportRel
}

Write-Log "PASS: check_compile_prep_family_surface"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
