param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11q",
    [ValidateSet("pre", "post")]
    [string]$Phase = "pre"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRelativePath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )

    $baseUri = New-Object System.Uri(([System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'))
    $targetUri = New-Object System.Uri([System.IO.Path]::GetFullPath($TargetPath))
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('\', '/')
}

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

$repo = [System.IO.Path]::GetFullPath((Resolve-Path $RepoRoot).Path)
$outDirAbs = Join-RepoPath -RepoRootPath $repo -Path $OutDir
New-Item -ItemType Directory -Force -Path $outDirAbs > $null

$logPath = Join-Path $outDirAbs "check_handoff_surface.log"
$summaryPath = Join-Path $outDirAbs "check_handoff_surface_summary.txt"
if (-not (Test-Path $logPath)) {
    New-Item -ItemType File -Path $logPath -Force > $null
}
Add-Content -Path $logPath -Value ("===== check_handoff_surface phase={0} =====" -f $Phase) -Encoding UTF8

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
    Write-Log "FAIL: check_handoff_surface"
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
        '(?i)\bSCVerify success\b'
    )
    foreach ($pattern in $forbiddenPatterns) {
        if ([System.Text.RegularExpressions.Regex]::IsMatch($Text, $pattern)) {
            Fail-Check ("overclaim wording detected in {0}: pattern {1}" -f $DocLabel, $pattern)
        }
    }
}

Write-Log ("[p11q] phase={0}" -f $Phase)

$trackedRaw = & git -C $repo ls-files
if ($LASTEXITCODE -ne 0) {
    Fail-Check "git ls-files failed"
}
$trackedSet = New-Object System.Collections.Generic.HashSet[string] ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($item in $trackedRaw) {
    [void]$trackedSet.Add(($item -replace '\\', '/'))
}

$allowPendingTrack = New-Object System.Collections.Generic.HashSet[string] ([System.StringComparer]::OrdinalIgnoreCase)
[void]$allowPendingTrack.Add("docs/process/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md")
[void]$allowPendingTrack.Add("scripts/check_handoff_surface.ps1")
[void]$allowPendingTrack.Add("docs/milestones/P00-011Q_report.md")

function Require-TrackedFile {
    param([string]$RelPath)
    $normalized = $RelPath -replace '\\', '/'
    $abs = Join-Path $repo $normalized
    if ($trackedSet.Contains($normalized)) {
        Require-True -Condition (Test-Path $abs) -Reason ("required file path not found on disk: {0}" -f $normalized)
        return
    }

    if ($allowPendingTrack.Contains($normalized) -and (Test-Path $abs)) {
        Write-Log ("[p11q][WARN] pending-track file accepted in working tree: {0}" -f $normalized)
        return
    }

    Fail-Check ("required tracked file missing: {0}" -f $normalized)
}

$handoffDocRel = "docs/process/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md"
$statusDocRel = "docs/process/PROJECT_STATUS_zhTW.txt"
$traceDocRel = "docs/milestones/TRACEABILITY_MAP_v12.1.md"
$closureDocRel = "docs/milestones/CLOSURE_MATRIX_v12.1.md"
$runScriptRel = "scripts/local/run_p11l_local_regression.ps1"

if ($Phase -eq "pre") {
    $requiredPreTracked = @(
        $handoffDocRel,
        $statusDocRel,
        $traceDocRel,
        $closureDocRel,
        "docs/process/SYNTHESIS_RULES.md",
        "docs/process/EVIDENCE_BUNDLE_RULES.md",
        "docs/milestones/P00-011O_report.md",
        "docs/milestones/P00-011P_report.md",
        "tb/tb_ternary_live_leaf_smoke_p11j.cpp",
        "tb/tb_ternary_live_leaf_top_smoke_p11k.cpp",
        "tb/tb_ternary_live_leaf_top_smoke_p11l_b.cpp",
        "tb/tb_ternary_live_leaf_top_smoke_p11l_c.cpp",
        "tb/tb_ternary_live_source_integration_smoke_p11m.cpp",
        "tb/tb_ternary_live_family_source_integration_smoke_p11n.cpp",
        $runScriptRel,
        "scripts/check_design_purity.ps1",
        "scripts/check_interface_lock.ps1",
        "scripts/check_macro_hygiene.ps1",
        "scripts/check_repo_hygiene.ps1",
        "scripts/check_handoff_surface.ps1"
    )
    foreach ($path in $requiredPreTracked) {
        Require-TrackedFile -RelPath $path
    }

    $handoffDocAbs = Join-Path $repo $handoffDocRel
    $handoffText = Get-Content -Path $handoffDocAbs -Raw

    $requiredSections = @(
        "Purpose",
        "What This Document Is / Is Not",
        "Current Accepted Local-Only Family Scope",
        "Accepted Handoff Surface",
        "Role Classification",
        "Allowed Local-Only Macros / Knobs",
        "Explicit Deferred Items",
        "Non-Goals",
        "How Later Catapult-Prep Work Should Interpret This Boundary"
    )
    foreach ($section in $requiredSections) {
        Require-Regex -Text $handoffText -Pattern ("(?m)^##\s+{0}\s*$" -f [System.Text.RegularExpressions.Regex]::Escape($section)) -Reason ("handoff doc missing section: {0}" -f $section)
    }

    $requiredRoles = @(
        "design-side helper",
        "local top wrapper",
        "split-interface local top",
        "TB-only smoke drivers",
        "one-shot regression scripts",
        "governance / evidence documents"
    )
    foreach ($role in $requiredRoles) {
        Require-TextContains -Text $handoffText -Needle $role -Reason ("handoff doc missing role keyword: {0}" -f $role)
    }

    $requiredPhrases = @(
        "local-only progress is valid",
        "local smoke / local static checks != full Catapult closure",
        "Catapult / SCVerify deferred by design",
        "deferred items are intentional"
    )
    foreach ($phrase in $requiredPhrases) {
        Require-TextContains -Text $handoffText -Needle $phrase -Reason ("handoff doc missing fixed phrase: {0}" -f $phrase)
    }

    $macroP11m = ("AECCT_LOCAL_{0}" -f "P11M_WQ_SPLIT_TOP_ENABLE")
    $macroP11n = ("AECCT_LOCAL_{0}" -f "P11N_WK_WV_SPLIT_TOP_ENABLE")
    $allowedMacros = @(
        $macroP11m,
        $macroP11n
    )
    $macroWhitelist = @{
        $macroP11m = @(
            "src/blocks/AttnLayer0.h",
            "tb/tb_ternary_live_source_integration_smoke_p11m.cpp",
            "scripts/local/run_p11l_local_regression.ps1",
            "scripts/check_design_purity.ps1",
            "scripts/check_macro_hygiene.ps1"
        )
        $macroP11n = @(
            "src/blocks/AttnLayer0.h",
            "tb/tb_ternary_live_family_source_integration_smoke_p11n.cpp",
            "scripts/local/run_p11l_local_regression.ps1",
            "scripts/check_design_purity.ps1",
            "scripts/check_macro_hygiene.ps1"
        )
    }

    $scanRoots = @("src", "tb", "scripts")
    $scanExtensions = @(".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx", ".ps1", ".py", ".sh", ".cmd", ".bat")
    foreach ($root in $scanRoots) {
        $absRoot = Join-Path $repo $root
        if (-not (Test-Path $absRoot)) {
            continue
        }
        $files = Get-ChildItem -Path $absRoot -Recurse -File
        foreach ($file in $files) {
            if ($scanExtensions -notcontains $file.Extension.ToLowerInvariant()) {
                continue
            }
            $rel = Get-RepoRelativePath -BasePath $repo -TargetPath $file.FullName
            $lines = Get-Content -Path $file.FullName
            for ($i = 0; $i -lt $lines.Count; $i++) {
                $line = $lines[$i]
                $macroMatches = [System.Text.RegularExpressions.Regex]::Matches($line, '\bAECCT_LOCAL_[A-Z0-9_]+\b')
                foreach ($match in $macroMatches) {
                    $macro = $match.Value
                    if ($allowedMacros -notcontains $macro) {
                        Fail-Check ("unapproved local macro token: {0} at {1}:{2}" -f $macro, $rel, ($i + 1))
                    }
                    if ($macroWhitelist[$macro] -notcontains $rel) {
                        Fail-Check ("approved local macro out of allowed context: {0} at {1}:{2}" -f $macro, $rel, ($i + 1))
                    }
                }
            }
        }
    }

    $localTbFiles = @(
        "tb/tb_ternary_live_leaf_smoke_p11j.cpp",
        "tb/tb_ternary_live_leaf_top_smoke_p11k.cpp",
        "tb/tb_ternary_live_leaf_top_smoke_p11l_b.cpp",
        "tb/tb_ternary_live_leaf_top_smoke_p11l_c.cpp",
        "tb/tb_ternary_live_source_integration_smoke_p11m.cpp",
        "tb/tb_ternary_live_family_source_integration_smoke_p11n.cpp"
    )
    foreach ($relPath in $localTbFiles) {
        $absPath = Join-Path $repo $relPath
        $tbText = Get-Content -Path $absPath -Raw
        if ($tbText -match 'mc_scverify\.h') {
            Require-Regex -Text $tbText -Pattern '__has_include\s*\(\s*[<"]mc_scverify\.h[>"]\s*\)' -Reason ("mc_scverify include must be guarded by __has_include in {0}" -f $relPath)
            Require-Regex -Text $tbText -Pattern '(?m)^\s*#\s*else\b' -Reason ("mc_scverify guarded include requires fallback #else path in {0}" -f $relPath)
        }
    }

    $runScriptAbs = Join-Path $repo $runScriptRel
    $runScriptText = Get-Content -Path $runScriptAbs -Raw
    $coverageMarkers = @(
        "tb_ternary_live_leaf_smoke_p11j.cpp",
        "tb_ternary_live_leaf_top_smoke_p11k.cpp",
        "tb_ternary_live_leaf_top_smoke_p11l_b.cpp",
        "tb_ternary_live_leaf_top_smoke_p11l_c.cpp",
        "tb_ternary_live_source_integration_smoke_p11m.cpp",
        "tb_ternary_live_family_source_integration_smoke_p11n.cpp",
        "PASS: run_p11l_local_regression"
    )
    foreach ($marker in $coverageMarkers) {
        Require-TextContains -Text $runScriptText -Needle $marker -Reason ("one-shot script missing coverage/wording marker: {0}" -f $marker)
    }

    $governanceDocs = @($statusDocRel, $traceDocRel, $closureDocRel)
    foreach ($docRel in $governanceDocs) {
        $docAbs = Join-Path $repo $docRel
        $docText = Get-Content -Path $docAbs -Raw
        Require-TextContains -Text $docText -Needle "P00-011Q" -Reason ("governance doc missing P00-011Q: {0}" -f $docRel)
        Require-Regex -Text $docText -Pattern '(?i)local-only|local smoke' -Reason ("governance doc missing local-only semantics: {0}" -f $docRel)
        Require-TextContains -Text $docText -Needle "Catapult / SCVerify deferred" -Reason ("governance doc missing deferred semantics: {0}" -f $docRel)
        Assert-NoOverclaim -Text $docText -DocLabel $docRel
    }

    Assert-NoOverclaim -Text $handoffText -DocLabel $handoffDocRel
}
elseif ($Phase -eq "post") {
    $reportRel = "docs/milestones/P00-011Q_report.md"
    Require-TrackedFile -RelPath $reportRel
    $reportAbs = Join-Path $repo $reportRel
    $reportText = Get-Content -Path $reportAbs -Raw

    $reportSections = @(
        "Summary",
        "Scope",
        "Files changed",
        "Exact commands executed",
        "Actual execution evidence excerpt",
        "Result / verdict wording",
        "Limitations",
        "Why useful for later Catapult-prep but not closure"
    )
    foreach ($section in $reportSections) {
        Require-Regex -Text $reportText -Pattern ("(?m)^##\s+{0}\s*$" -f [System.Text.RegularExpressions.Regex]::Escape($section)) -Reason ("P00-011Q report missing section: {0}" -f $section)
    }

    $reportMustContain = @(
        "scripts/check_handoff_surface.ps1 -OutDir build\p11q -Phase pre",
        "scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11q",
        "scripts/check_handoff_surface.ps1 -OutDir build\p11q -Phase post",
        "PASS: check_handoff_surface",
        "PASS: run_p11l_local_regression",
        "Catapult / SCVerify deferred"
    )
    foreach ($needle in $reportMustContain) {
        Require-TextContains -Text $reportText -Needle $needle -Reason ("P00-011Q report missing required content: {0}" -f $needle)
    }

    $postRequiredLogs = @(
        "build/p11q/check_handoff_surface.log",
        "build/p11q/check_handoff_surface_summary.txt",
        "build/p11q/run_p11p_regression.log",
        "build/p11q/run_p11j.log",
        "build/p11q/run_p11k.log",
        "build/p11q/run_p11l_b.log",
        "build/p11q/run_p11l_c.log",
        "build/p11q/run_p11m_baseline.log",
        "build/p11q/run_p11m_macro.log",
        "build/p11q/run_p11n_baseline.log",
        "build/p11q/run_p11n_macro.log"
    )
    foreach ($relLog in $postRequiredLogs) {
        $absLog = Join-Path $repo ($relLog -replace '/', '\')
        Require-True -Condition (Test-Path $absLog) -Reason ("required post artifact missing: {0}" -f $relLog)
    }

    $governanceDocs = @($statusDocRel, $traceDocRel, $closureDocRel)
    foreach ($docRel in $governanceDocs) {
        $docAbs = Join-Path $repo $docRel
        $docText = Get-Content -Path $docAbs -Raw
        Require-TextContains -Text $docText -Needle "P00-011Q" -Reason ("governance doc not synced to P00-011Q: {0}" -f $docRel)
        Require-TextContains -Text $docText -Needle "Catapult / SCVerify deferred" -Reason ("governance doc missing deferred semantics in post phase: {0}" -f $docRel)
        Assert-NoOverclaim -Text $docText -DocLabel $docRel
    }

    Assert-NoOverclaim -Text $reportText -DocLabel $reportRel
}

Write-Log "PASS: check_handoff_surface"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
