param(
    [string]$RepoRoot = ".",
    [string]$StateRoot = "build/agent_state"
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

function Add-Finding {
    param(
        [System.Collections.Generic.List[string]]$Findings,
        [string]$Message
    )

    $Findings.Add($Message)
}

$repo = [System.IO.Path]::GetFullPath((Resolve-Path $RepoRoot).Path)
$findings = New-Object System.Collections.Generic.List[string]

$agentsPath = Join-Path $repo "AGENTS.md"
if (-not (Test-Path $agentsPath)) {
    Add-Finding -Findings $findings -Message "missing_required_file: AGENTS.md"
}
else {
    $agentsRaw = Get-Content -Path $agentsPath -Raw
    if ([string]::IsNullOrWhiteSpace($agentsRaw)) {
        Add-Finding -Findings $findings -Message "empty_required_file: AGENTS.md"
    }
}

$handoffTemplatePath = Join-Path $repo "docs/handoff/AGENT_COMPLETION_TEMPLATE.md"
if (-not (Test-Path $handoffTemplatePath)) {
    Add-Finding -Findings $findings -Message "missing_required_file: docs/handoff/AGENT_COMPLETION_TEMPLATE.md"
}
else {
    $handoffTemplateRaw = Get-Content -Path $handoffTemplatePath -Raw
    if ([string]::IsNullOrWhiteSpace($handoffTemplateRaw)) {
        Add-Finding -Findings $findings -Message "empty_required_file: docs/handoff/AGENT_COMPLETION_TEMPLATE.md"
    }
}

$bootstrapScriptPath = Join-Path $repo "scripts/init_agent_state.ps1"
if (-not (Test-Path $bootstrapScriptPath)) {
    Add-Finding -Findings $findings -Message "missing_required_script: scripts/init_agent_state.ps1"
}

$stateRootAbs = Join-RepoPath -RepoRootPath $repo -Path $StateRoot
$stateRootRel = Get-RepoRelativePath -BasePath $repo -TargetPath $stateRootAbs
$sessionTag = "selfcheck_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")

if ($findings.Count -eq 0) {
    & $bootstrapScriptPath -RepoRoot $repo -StateRoot $StateRoot -SessionTag $sessionTag
    if ($LASTEXITCODE -ne 0) {
        Add-Finding -Findings $findings -Message ("bootstrap_failed: exit_code={0}" -f $LASTEXITCODE)
    }
}

$sessionDir = Join-Path $stateRootAbs $sessionTag
$expectedFiles = @(
    "WORKING_MEMORY.md",
    "TASK_QUEUE.md",
    "BLOCKERS.md",
    "SESSION_SUMMARY.md"
)
foreach ($name in $expectedFiles) {
    $path = Join-Path $sessionDir $name
    if (-not (Test-Path $path)) {
        Add-Finding -Findings $findings -Message ("missing_initialized_template: {0}/{1}" -f ($stateRootRel -replace '\\', '/'), $sessionTag + "/" + $name)
        continue
    }
    if ((Get-Item $path).Length -le 0) {
        Add-Finding -Findings $findings -Message ("empty_initialized_template: {0}/{1}" -f ($stateRootRel -replace '\\', '/'), $sessionTag + "/" + $name)
    }
}

$latestSessionPath = Join-Path $stateRootAbs "LATEST_SESSION.txt"
if (-not (Test-Path $latestSessionPath)) {
    Add-Finding -Findings $findings -Message ("missing_initialized_file: {0}/LATEST_SESSION.txt" -f ($stateRootRel -replace '\\', '/'))
}

$probePath = Join-Path $sessionDir ".ignore_probe"
"probe" | Set-Content -Path $probePath -Encoding ASCII
$probeRel = Get-RepoRelativePath -BasePath $repo -TargetPath $probePath
& git -C $repo check-ignore -q -- $probeRel
if ($LASTEXITCODE -ne 0) {
    Add-Finding -Findings $findings -Message ("local_only_path_not_ignored: {0}" -f $probeRel)
}
Remove-Item -LiteralPath $probePath -ErrorAction SilentlyContinue

if ($findings.Count -gt 0) {
    Write-Host "FAIL: check_agent_tooling"
    foreach ($finding in $findings) {
        Write-Host $finding
    }
    exit 1
}

Write-Host "PASS: check_agent_tooling"
Write-Host ("validated_state_root: {0}" -f ($stateRootRel -replace '\\', '/'))
Write-Host ("validated_session: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $sessionDir))
exit 0
