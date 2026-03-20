param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11aeaf_catapult_progress",
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

$logPath = Join-Path $outDirAbs "check_p11aeaf_catapult_progress_surface.log"
$summaryPath = Join-Path $outDirAbs "check_p11aeaf_catapult_progress_surface_summary.txt"
if (-not (Test-Path $logPath)) {
    New-Item -ItemType File -Path $logPath -Force > $null
}
Add-Content -Path $logPath -Value ("===== check_p11aeaf_catapult_progress_surface phase={0} =====" -f $Phase) -Encoding UTF8

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
    @(
        ("status: {0}" -f $Status),
        ("phase: {0}" -f $Phase),
        ("repo_root: {0}" -f $repo),
        ("out_dir: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $outDirAbs)),
        ("log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $logPath)),
        ("detail: {0}" -f $Detail)
    ) | Set-Content -Path $summaryPath -Encoding UTF8
}

function Fail-Check {
    param([string]$Reason)
    Write-Log "FAIL: check_p11aeaf_catapult_progress_surface"
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

$required = @(
    "scripts/check_compile_prep_surface.ps1",
    "scripts/local/run_p11r_compile_prep.ps1",
    "scripts/check_p11aeaf_catapult_progress_surface.ps1",
    "scripts/local/run_p11aeaf_catapult_progress.ps1",
    "docs/milestones/P00-011AEAF_catapult_progress_report.md"
)
foreach ($rel in $required) {
    Require-True -Condition (Test-Path (Join-Path $repo $rel)) -Reason ("required file missing: {0}" -f $rel)
}

$runnerText = Get-Content -Path (Join-Path $repo "scripts/local/run_p11aeaf_catapult_progress.ps1") -Raw
Require-TextContains -Text $runnerText -Needle "run_p11r_compile_prep.ps1" -Reason "runner must call compile-prep probe runner"
Require-TextContains -Text $runnerText -Needle "PASS: run_p11aeaf_catapult_progress" -Reason "runner PASS banner missing"

if ($Phase -eq "post") {
    $reportText = Get-Content -Path (Join-Path $repo "docs/milestones/P00-011AEAF_catapult_progress_report.md") -Raw
    Require-TextContains -Text $reportText -Needle "local-only" -Reason "report missing local-only wording"
    Require-TextContains -Text $reportText -Needle "Catapult-facing progress" -Reason "report missing Catapult-facing progress wording"
    Require-TextContains -Text $reportText -Needle "not Catapult closure" -Reason "report missing not Catapult closure wording"
    Require-TextContains -Text $reportText -Needle "not SCVerify closure" -Reason "report missing not SCVerify closure wording"
}

Write-Log "PASS: check_p11aeaf_catapult_progress_surface"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
