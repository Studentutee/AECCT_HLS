param(
    [string]$RepoRoot = ".",
    [string]$StateRoot = "build/agent_state",
    [string]$SessionTag = ""
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
$stateRootAbs = Join-RepoPath -RepoRootPath $repo -Path $StateRoot
$createdAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if ([string]::IsNullOrWhiteSpace($SessionTag)) {
    $SessionTag = "session_$timestamp"
}

$sessionDir = Join-Path $stateRootAbs $SessionTag
New-Item -ItemType Directory -Force -Path $sessionDir > $null

$workingMemoryPath = Join-Path $sessionDir "WORKING_MEMORY.md"
$taskQueuePath = Join-Path $sessionDir "TASK_QUEUE.md"
$blockersPath = Join-Path $sessionDir "BLOCKERS.md"
$sessionSummaryPath = Join-Path $sessionDir "SESSION_SUMMARY.md"
$latestSessionPath = Join-Path $stateRootAbs "LATEST_SESSION.txt"

$workingMemoryContent = @"
# WORKING_MEMORY

- session_tag: $SessionTag
- created_at: $createdAt
- owner_agent: <fill_me>
- objective: <fill_me>

## Current Context
- upstream_handoff:
- accepted_constraints:
- assumptions:

## Active Notes
- note_1:
- note_2:

## Decision Log
- [time] decision:

## Evidence Pointers
- command_log:
- local_log_paths:
- repo_artifact_paths:

## Next Resume Point
- next_step_1:
- next_step_2:
"@

$taskQueueContent = @"
# TASK_QUEUE

- session_tag: $SessionTag
- updated_at: $createdAt

| task_id | status | priority | owner | description | done_when | evidence_target |
| --- | --- | --- | --- | --- | --- | --- |
| T001 | todo | high | <agent> | <task> | <acceptance> | <log/report path> |
| T002 | todo | medium | <agent> | <task> | <acceptance> | <log/report path> |

## Queue Notes
- Keep task states explicit: todo, in_progress, blocked, done.
- Move finished items to session summary with evidence links.
"@

$blockersContent = @"
# BLOCKERS

- session_tag: $SessionTag
- updated_at: $createdAt

| blocker_id | status | discovered_at | impact | owner | unblock_condition | next_check |
| --- | --- | --- | --- | --- | --- | --- |
| B001 | open | <time> | <impact> | <owner> | <condition> | <time> |

## Blocker Notes
- Record exact failing command and shortest log excerpt.
- Mark blocker as resolved only after rerun evidence is captured.
"@

$sessionSummaryContent = @"
# SESSION_SUMMARY

- session_tag: $SessionTag
- started_at: $createdAt
- ended_at:

## Summary
- what_changed:
- why:

## Exact Files Changed
- repo_tracked:
- local_only:

## Exact Commands Run
- command_1:
- command_2:

## Actual Execution Evidence
- log_excerpt_1:
- log_excerpt_2:

## Governance Posture
- scope:
- closure_posture: not Catapult closure; not SCVerify closure
- ownership_boundary_check: Top is sole production shared-SRAM owner

## Residual Risks
- risk_1:

## Recommended Next Step
- next_step_1:
"@

$workingMemoryContent | Set-Content -Path $workingMemoryPath -Encoding ASCII
$taskQueueContent | Set-Content -Path $taskQueuePath -Encoding ASCII
$blockersContent | Set-Content -Path $blockersPath -Encoding ASCII
$sessionSummaryContent | Set-Content -Path $sessionSummaryPath -Encoding ASCII
$SessionTag | Set-Content -Path $latestSessionPath -Encoding ASCII

$stateRootRel = Get-RepoRelativePath -BasePath $repo -TargetPath $stateRootAbs
$sessionDirRel = Get-RepoRelativePath -BasePath $repo -TargetPath $sessionDir
$workingMemoryRel = Get-RepoRelativePath -BasePath $repo -TargetPath $workingMemoryPath
$taskQueueRel = Get-RepoRelativePath -BasePath $repo -TargetPath $taskQueuePath
$blockersRel = Get-RepoRelativePath -BasePath $repo -TargetPath $blockersPath
$sessionSummaryRel = Get-RepoRelativePath -BasePath $repo -TargetPath $sessionSummaryPath
$latestSessionRel = Get-RepoRelativePath -BasePath $repo -TargetPath $latestSessionPath

Write-Host "PASS: init_agent_state"
Write-Host ("session_tag: {0}" -f $SessionTag)
Write-Host ("state_root: {0}" -f $stateRootRel)
Write-Host ("session_dir: {0}" -f $sessionDirRel)
Write-Host "local_only_files_do_not_git_add:"
Write-Host ("- {0}" -f $workingMemoryRel)
Write-Host ("- {0}" -f $taskQueueRel)
Write-Host ("- {0}" -f $blockersRel)
Write-Host ("- {0}" -f $sessionSummaryRel)
Write-Host ("- {0}" -f $latestSessionRel)
Write-Host "quick_start:"
Write-Host ("1) update {0} for current context and assumptions." -f $workingMemoryRel)
Write-Host ("2) manage actionable items in {0} and blockers in {1}." -f $taskQueueRel, $blockersRel)
Write-Host ("3) prepare final handoff with docs/handoff/AGENT_COMPLETION_TEMPLATE.md and mirror key facts in {0}." -f $sessionSummaryRel)
exit 0
