param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11ae_prep",
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
    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRootPath $Path))
}

$repo = [System.IO.Path]::GetFullPath((Resolve-Path $RepoRoot).Path)
$outDirAbs = Join-RepoPath -RepoRootPath $repo -Path $OutDir
New-Item -ItemType Directory -Force -Path $outDirAbs > $null
$logPath = Join-Path $outDirAbs "check_p11ae_prep_surface.log"
$summaryPath = Join-Path $outDirAbs "check_p11ae_prep_surface_summary.txt"

function Fail-Check([string]$Reason) {
    "FAIL: check_p11ae_prep_surface" | Tee-Object -FilePath $logPath -Append
    $Reason | Tee-Object -FilePath $logPath -Append
    @("status: FAIL", "phase: $Phase", "detail: $Reason") | Set-Content -Path $summaryPath -Encoding UTF8
    exit 1
}

$required = @(
    "tb/tb_qk_score_scaffold_p11ae_prep.cpp",
    "scripts/check_p11ae_prep_surface.ps1",
    "scripts/local/run_p11ae_prep_qk_score.ps1",
    "docs/milestones/P00-011AE-prep_report.md"
)
foreach ($rel in $required) {
    if (-not (Test-Path (Join-Path $repo $rel))) {
        Fail-Check "required file missing: $rel"
    }
}

$tbText = Get-Content -Path (Join-Path $repo "tb/tb_qk_score_scaffold_p11ae_prep.cpp") -Raw
if ($tbText -match 'AttnTopManagedPackets\.h' -or $tbText -match 'AttnPhaseATopManagedKv\.h') {
    Fail-Check "prep TB must not depend on AC-only design headers"
}
if ($tbText -match '#include\s+"src/') {
    Fail-Check "prep TB must not include src/ design headers"
}

if ($Phase -eq "post") {
    $reportText = Get-Content -Path (Join-Path $repo "docs/milestones/P00-011AE-prep_report.md") -Raw
    if ($reportText -notmatch 'local-only') { Fail-Check "report missing local-only wording" }
}

"PASS: check_p11ae_prep_surface" | Tee-Object -FilePath $logPath -Append
@("status: PASS", "phase: $Phase", "detail: all checks passed") | Set-Content -Path $summaryPath -Encoding UTF8
exit 0
