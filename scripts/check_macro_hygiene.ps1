param(
    [string]$RepoRoot = "."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRelativePath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )

    $baseUri = New-Object System.Uri(($BasePath.TrimEnd('\') + '\'))
    $targetUri = New-Object System.Uri($TargetPath)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('\', '/')
}

function Add-Finding {
    param(
        [System.Collections.Generic.List[string]]$Findings,
        [string]$Message
    )

    $Findings.Add($Message)
}

function Require-Line {
    param(
        [string[]]$Lines,
        [string]$Pattern,
        [string]$Label,
        [System.Collections.Generic.List[string]]$Findings
    )

    $line = $Lines | Where-Object { $_ -match $Pattern } | Select-Object -First 1
    if (-not $line) {
        Add-Finding -Findings $Findings -Message "build_command: missing $Label"
        return $null
    }
    return $line
}

$repo = (Resolve-Path $RepoRoot).Path
$findings = New-Object System.Collections.Generic.List[string]

$allowedMacros = @(
    "AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE",
    "AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE"
)

$macroWhitelist = @{
    "AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE" = @(
        "src/blocks/AttnLayer0.h",
        "tb/tb_ternary_live_source_integration_smoke_p11m.cpp",
        "scripts/local/run_p11l_local_regression.ps1",
        "scripts/check_design_purity.ps1",
        "scripts/check_macro_hygiene.ps1"
    )
    "AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE" = @(
        "src/blocks/AttnLayer0.h",
        "tb/tb_ternary_live_family_source_integration_smoke_p11n.cpp",
        "scripts/local/run_p11l_local_regression.ps1",
        "scripts/check_design_purity.ps1",
        "scripts/check_macro_hygiene.ps1"
    )
}

$scanRoots = @("src", "tb", "scripts")
$scanExtensions = @(".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx", ".ps1", ".py", ".sh", ".cmd", ".bat")

$scanFiles = New-Object System.Collections.Generic.List[string]
foreach ($root in $scanRoots) {
    $absRoot = Join-Path $repo $root
    if (-not (Test-Path $absRoot)) {
        continue
    }
    Get-ChildItem -Path $absRoot -Recurse -File | ForEach-Object {
        if ($scanExtensions -contains $_.Extension.ToLowerInvariant()) {
            $scanFiles.Add($_.FullName)
        }
    }
}

$uniqueFiles = $scanFiles | Sort-Object -Unique
foreach ($file in $uniqueFiles) {
    $rel = Get-RepoRelativePath -BasePath $repo -TargetPath $file
    $lines = Get-Content -Path $file
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        $matches = [System.Text.RegularExpressions.Regex]::Matches($line, '\bAECCT_LOCAL_[A-Z0-9_]+\b')
        foreach ($match in $matches) {
            $macro = $match.Value
            if ($allowedMacros -notcontains $macro) {
                Add-Finding -Findings $findings -Message ("macro_token: unapproved macro {0} at {1}:{2}" -f $macro, $rel, ($i + 1))
                continue
            }
            if ($macroWhitelist[$macro] -notcontains $rel) {
                Add-Finding -Findings $findings -Message ("macro_scope: {0} is outside strict whitelist at {1}:{2}" -f $macro, $rel, ($i + 1))
            }
        }
    }
}

$runScriptRel = "scripts/local/run_p11l_local_regression.ps1"
$runScript = Join-Path $repo $runScriptRel
if (-not (Test-Path $runScript)) {
    Add-Finding -Findings $findings -Message "build_command: missing $runScriptRel"
}
else {
    $runLines = Get-Content -Path $runScript
    $macroCommandLines = $runLines | Where-Object { $_ -match '/DAECCT_LOCAL_' }
    if (($macroCommandLines | Measure-Object).Count -ne 2) {
        Add-Finding -Findings $findings -Message "build_command: expected exactly 2 local-macro build command lines in $runScriptRel"
    }

    foreach ($cmdLine in $macroCommandLines) {
        if ($cmdLine -notmatch '/DAECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE=1' -and
            $cmdLine -notmatch '/DAECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE=1') {
            Add-Finding -Findings $findings -Message ("build_command: unapproved local macro command in ${runScriptRel}: {0}" -f $cmdLine.Trim())
        }
    }

    $p11mBaselineLine = Require-Line -Lines $runLines -Pattern 'Invoke-ClBuild\s+''tb\\tb_ternary_live_source_integration_smoke_p11m\.cpp''\s+\$exeP11mBaseline\s+\$logBuildP11mBaseline' -Label "p11m baseline build command" -Findings $findings
    if ($p11mBaselineLine -and $p11mBaselineLine -match '/DAECCT_LOCAL_') {
        Add-Finding -Findings $findings -Message "build_command: p11m baseline build must not carry local macro"
    }

    $p11nBaselineLine = Require-Line -Lines $runLines -Pattern 'Invoke-ClBuild\s+''tb\\tb_ternary_live_family_source_integration_smoke_p11n\.cpp''\s+\$exeP11nBaseline\s+\$logBuildP11nBaseline' -Label "p11n baseline build command" -Findings $findings
    if ($p11nBaselineLine -and $p11nBaselineLine -match '/DAECCT_LOCAL_') {
        Add-Finding -Findings $findings -Message "build_command: p11n baseline build must not carry local macro"
    }

    if (-not (Require-Line -Lines $runLines -Pattern 'Invoke-ClBuild\s+''tb\\tb_ternary_live_source_integration_smoke_p11m\.cpp''\s+\$exeP11mMacro\s+\$logBuildP11mMacro\s+@\(''/DAECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE=1''\)' -Label "p11m macro build command with approved macro" -Findings $findings)) {
        # finding is added by Require-Line
    }

    if (-not (Require-Line -Lines $runLines -Pattern 'Invoke-ClBuild\s+''tb\\tb_ternary_live_family_source_integration_smoke_p11n\.cpp''\s+\$exeP11nMacro\s+\$logBuildP11nMacro\s+@\(''/DAECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE=1''\)' -Label "p11n macro build command with approved macro" -Findings $findings)) {
        # finding is added by Require-Line
    }
}

$docChecks = @(
    @{ Path = "docs/milestones/P00-011M_report.md"; Marker = "P00-011M"; Required = $true },
    @{ Path = "docs/milestones/P00-011N_report.md"; Marker = "P00-011N"; Required = $true },
    @{ Path = "docs/process/SYNTHESIS_RULES.md"; Marker = "synthesis"; Required = $true },
    @{ Path = "docs/milestones/P00-011O_report.md"; Marker = "P00-011O"; Required = $false }
)

foreach ($doc in $docChecks) {
    $abs = Join-Path $repo $doc.Path
    if (-not (Test-Path $abs)) {
        if ($doc.Required) {
            Add-Finding -Findings $findings -Message ("doc_presence: missing {0}" -f $doc.Path)
        }
        continue
    }
    $text = Get-Content -Path $abs -Raw
    if ($text -notmatch [System.Text.RegularExpressions.Regex]::Escape($doc.Marker)) {
        Add-Finding -Findings $findings -Message ("doc_marker: missing recognizable marker '{0}' in {1}" -f $doc.Marker, $doc.Path)
    }
}

if ($findings.Count -gt 0) {
    Write-Host "FAIL: check_macro_hygiene"
    foreach ($item in $findings) {
        Write-Host $item
    }
    exit 1
}

Write-Host "PASS: check_macro_hygiene"
exit 0
