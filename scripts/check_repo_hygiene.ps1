param(
    [string]$RepoRoot = ".",
    [ValidateSet("pre", "post")]
    [string]$Phase = "pre",
    [string]$BuildDir = "build\p11n"
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

function Add-Finding {
    param(
        [System.Collections.Generic.List[string]]$Findings,
        [string]$Message
    )

    $Findings.Add($Message)
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
$buildDirAbs = Join-RepoPath -RepoRootPath $repo -Path $BuildDir
$buildDirRel = Get-RepoRelativePath -BasePath $repo -TargetPath $buildDirAbs

$findings = New-Object System.Collections.Generic.List[string]

if ($Phase -eq "pre") {
    $trackedPaths = & git -C $repo ls-files
    if ($LASTEXITCODE -ne 0) {
        throw "git ls-files failed under $repo"
    }

    $denyPatterns = @(
        '(^|/)build/',
        '(^|/)(tmp|temp)/',
        '(^|/)(manual|sandbox)/',
        '\.sandbox',
        '_manual\.(log|txt|md|json)$',
        '(^|/).*\.(exe|obj|pdb|ilk|iobj|ipdb|tmp|cache)$'
    )

    foreach ($tracked in $trackedPaths) {
        $rel = ($tracked -replace '\\', '/')
        foreach ($pattern in $denyPatterns) {
            if ($rel -match $pattern) {
                Add-Finding -Findings $findings -Message ("tracked_path_violation: {0}" -f $rel)
                break
            }
        }
    }

    $requiredDocs = @(
        @{ Path = "docs/process/SYNTHESIS_RULES.md"; Marker = "SYNTHESIS_RULES" },
        @{ Path = "docs/process/EVIDENCE_BUNDLE_RULES.md"; Marker = "EVIDENCE_BUNDLE_RULES" },
        @{ Path = "docs/milestones/P00-011M_report.md"; Marker = "P00-011M" },
        @{ Path = "docs/milestones/P00-011N_report.md"; Marker = "P00-011N" },
        @{ Path = "docs/milestones/P00-011O_report.md"; Marker = "P00-011O" }
    )

    foreach ($doc in $requiredDocs) {
        $abs = Join-Path $repo $doc.Path
        if (-not (Test-Path $abs)) {
            Add-Finding -Findings $findings -Message ("doc_presence: missing {0}" -f $doc.Path)
            continue
        }
        $text = Get-Content -Path $abs -Raw
        if ($text -notmatch [System.Text.RegularExpressions.Regex]::Escape($doc.Marker)) {
            Add-Finding -Findings $findings -Message ("doc_marker: missing marker '{0}' in {1}" -f $doc.Marker, $doc.Path)
        }
    }
}
elseif ($Phase -eq "post") {
    $manifestPath = Join-Path $buildDirAbs "EVIDENCE_MANIFEST_p11p.txt"
    $summaryPath = Join-Path $buildDirAbs "EVIDENCE_SUMMARY_p11p.md"
    $warningPath = Join-Path $buildDirAbs "warning_summary_p11p.txt"
    $verdictPath = Join-Path $buildDirAbs "verdict_p11p.json"

    $requiredBundle = @($manifestPath, $summaryPath, $warningPath, $verdictPath)
    foreach ($bundle in $requiredBundle) {
        if (-not (Test-Path $bundle)) {
            Add-Finding -Findings $findings -Message ("bundle_missing: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $bundle))
            continue
        }
        if ((Get-Item $bundle).Length -le 0) {
            Add-Finding -Findings $findings -Message ("bundle_empty: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $bundle))
        }
    }

    if (Test-Path $verdictPath) {
        try {
            $verdictRaw = Get-Content -Path $verdictPath -Raw
            $verdict = $verdictRaw | ConvertFrom-Json
            $requiredKeys = @("task_id", "overall", "prechecks", "regression", "compares", "artifacts")
            $keySet = @($verdict.PSObject.Properties.Name)
            foreach ($k in $requiredKeys) {
                if ($keySet -notcontains $k) {
                    Add-Finding -Findings $findings -Message ("verdict_key_missing: {0}" -f $k)
                }
            }
        }
        catch {
            Add-Finding -Findings $findings -Message ("verdict_parse_failure: {0}" -f $_.Exception.Message)
        }
    }

    if (Test-Path $manifestPath) {
        $manifest = Get-Content -Path $manifestPath -Raw
        $requiredEntries = @(
            ("{0}/run_p11p_regression.log" -f $buildDirRel),
            ("{0}/warning_summary_p11p.txt" -f $buildDirRel),
            ("{0}/EVIDENCE_SUMMARY_p11p.md" -f $buildDirRel),
            ("{0}/verdict_p11p.json" -f $buildDirRel),
            ("{0}/run_p11j.log" -f $buildDirRel),
            ("{0}/run_p11k.log" -f $buildDirRel),
            ("{0}/run_p11l_b.log" -f $buildDirRel),
            ("{0}/run_p11l_c.log" -f $buildDirRel),
            ("{0}/run_p11m_baseline.log" -f $buildDirRel),
            ("{0}/run_p11m_macro.log" -f $buildDirRel),
            ("{0}/run_p11n_baseline.log" -f $buildDirRel),
            ("{0}/run_p11n_macro.log" -f $buildDirRel)
        )

        foreach ($entry in $requiredEntries) {
            if ($manifest -notmatch [System.Text.RegularExpressions.Regex]::Escape($entry)) {
                Add-Finding -Findings $findings -Message ("manifest_entry_missing: {0}" -f $entry)
            }
        }
    }
}

if ($findings.Count -gt 0) {
    Write-Host "FAIL: check_repo_hygiene"
    foreach ($f in $findings) {
        Write-Host $f
    }
    exit 1
}

Write-Host "PASS: check_repo_hygiene"
exit 0
