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
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString())
}

function Add-Finding {
    param(
        [System.Collections.Generic.List[string]]$Findings,
        [string]$RelPath,
        [int]$LineNo,
        [string]$RuleName,
        [string]$LineText
    )

    $Findings.Add(("{0}:{1}: {2}: {3}" -f $RelPath, $LineNo, $RuleName, $LineText.Trim()))
}

$repo = (Resolve-Path $RepoRoot).Path
$textExt = @(".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx")
$scanFiles = New-Object System.Collections.Generic.List[string]

$roots = @("src", "include", "gen\include")
foreach ($root in $roots) {
    $absRoot = Join-Path $repo $root
    if (-not (Test-Path $absRoot)) {
        continue
    }
    Get-ChildItem -Path $absRoot -Recurse -File | ForEach-Object {
        if ($textExt -contains $_.Extension.ToLowerInvariant()) {
            if ($_.FullName -notmatch '[\\/]third_party[\\/]') {
                $scanFiles.Add($_.FullName)
            }
        }
    }
}

$genRoot = Join-Path $repo "gen"
if (Test-Path $genRoot) {
    Get-ChildItem -Path $genRoot -File | ForEach-Object {
        if ($textExt -contains $_.Extension.ToLowerInvariant()) {
            if ($_.FullName -notmatch '[\\/]third_party[\\/]') {
                $scanFiles.Add($_.FullName)
            }
        }
    }
}

$rules = @(
    @{ Name = "include_step0"; Pattern = '#include\s*["<][^">]*step0\.h[">]' },
    @{ Name = "include_weights_h"; Pattern = '#include\s*["<]weights\.h[">]' },
    @{ Name = "include_data_weights_path"; Pattern = '#include\s*["<][^">]*data[\\/]+weights[\\/][^">]*[">]' },
    @{ Name = "trace_macro"; Pattern = '\bAECCT_.*TRACE_MODE\b' },
    @{ Name = "host_cmath"; Pattern = '#include\s*<cmath>' },
    @{ Name = "host_std_math"; Pattern = '\bstd::(?:sqrt|exp|log|pow)\b' },
    @{ Name = "union_keyword"; Pattern = '\bunion\b' },
    @{ Name = "float_keyword"; Pattern = '\bfloat\b' },
    @{ Name = "host_or_bringup_marker"; Pattern = '\b(?:HOST_ONLY|BRINGUP_ONLY|BRING_UP_ONLY)\b' }
)

$allowedLocalMacros = @(
    "AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE",
    "AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE"
)
$allowedLocalMacroRelPath = "src/blocks/AttnLayer0.h"

$findings = New-Object System.Collections.Generic.List[string]
$uniqueFiles = $scanFiles | Sort-Object -Unique
foreach ($file in $uniqueFiles) {
    $rel = (Get-RepoRelativePath -BasePath $repo -TargetPath $file).Replace('\', '/')
    $lines = Get-Content -Path $file
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        foreach ($rule in $rules) {
            if ($line -notmatch $rule.Pattern) {
                continue
            }
            if ($rule.Name -eq "float_keyword" -and $line -match "ac_ieee_float") {
                continue
            }
            Add-Finding -Findings $findings -RelPath $rel -LineNo ($i + 1) -RuleName $rule.Name -LineText $line
        }

        $macroMatches = [System.Text.RegularExpressions.Regex]::Matches($line, '\bAECCT_LOCAL_[A-Z0-9_]+\b')
        foreach ($match in $macroMatches) {
            $macroToken = $match.Value
            if ($allowedLocalMacros -notcontains $macroToken) {
                Add-Finding -Findings $findings -RelPath $rel -LineNo ($i + 1) -RuleName "unapproved_local_macro" -LineText $line
                continue
            }
            if ($rel -ne $allowedLocalMacroRelPath) {
                Add-Finding -Findings $findings -RelPath $rel -LineNo ($i + 1) -RuleName "local_macro_outside_approved_design_path" -LineText $line
            }
        }
    }
}

if ($findings.Count -gt 0) {
    Write-Host "FAIL: check_design_purity"
    foreach ($item in $findings) {
        Write-Host $item
    }
    exit 1
}

Write-Host "PASS: check_design_purity"
exit 0
