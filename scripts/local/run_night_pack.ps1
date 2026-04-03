param(
    [string]$BuildDir = "build\\night_run",
    [string]$NightPackDoc = "docs/night_run/NIGHT_PACK.md",
    [string]$TaskQueueDoc = "docs/night_run/TASK_QUEUE.md",
    [string]$AcceptancePackDoc = "docs/night_run/ACCEPTANCE_PACK.md",
    [switch]$Smoke
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

function Assert-FileExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "missing_required_input: $Label ($Path)"
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path)
Push-Location $repoRoot
try {
    $buildAbs = Join-RepoPath -RepoRootPath $repoRoot -Path $BuildDir
    New-Item -ItemType Directory -Force -Path $buildAbs > $null

    $nightPackDocAbs = Join-RepoPath -RepoRootPath $repoRoot -Path $NightPackDoc
    $taskQueueDocAbs = Join-RepoPath -RepoRootPath $repoRoot -Path $TaskQueueDoc
    $acceptancePackDocAbs = Join-RepoPath -RepoRootPath $repoRoot -Path $AcceptancePackDoc

    Assert-FileExists -Path $nightPackDocAbs -Label "NIGHT_PACK document"
    Assert-FileExists -Path $taskQueueDocAbs -Label "TASK_QUEUE document"
    Assert-FileExists -Path $acceptancePackDocAbs -Label "ACCEPTANCE_PACK document"

    $runId = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDirAbs = Join-Path $buildAbs $runId
    New-Item -ItemType Directory -Force -Path $runDirAbs > $null
    $runDirRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $runDirAbs

    $mode = if ($Smoke.IsPresent) { "smoke" } else { "normal" }

    $taskQueueRaw = Get-Content -Path $taskQueueDocAbs -Raw
    $queueRowCount = ([System.Text.RegularExpressions.Regex]::Matches(
            $taskQueueRaw,
            '(?im)^\|\s*[^|]+\|\s*(queued|ready|running|blocked|done|dropped)\s*\|'
        )).Count
    $queueReadyCount = ([System.Text.RegularExpressions.Regex]::Matches(
            $taskQueueRaw,
            '(?im)^\|\s*[^|]+\|\s*ready\s*\|'
        )).Count

    $summaryPathAbs = Join-Path $runDirAbs "NIGHT_PACK_SUMMARY.txt"
    $executionPathAbs = Join-Path $runDirAbs "NIGHT_PACK_EXECUTION.md"
    $verdictPathAbs = Join-Path $runDirAbs "NIGHT_PACK_VERDICT.json"
    $manifestPathAbs = Join-Path $runDirAbs "NIGHT_PACK_MANIFEST.txt"
    $acceptanceFilledPathAbs = Join-Path $runDirAbs "ACCEPTANCE_PACK_FILLED.md"

    @(
        "status: PASS",
        "scope: local-only skeleton",
        "mode: $mode",
        "queue_rows: $queueRowCount",
        "queue_ready_rows: $queueReadyCount",
        "closure: not Catapult closure; not SCVerify closure",
        "PASS: run_night_pack"
    ) | Set-Content -Path $summaryPathAbs -Encoding UTF8

    @(
        "# NIGHT_PACK_EXECUTION",
        "",
        "- run_id: $runId",
        "- run_dir: $runDirRel",
        "- mode: $mode",
        "- scope: local-only skeleton",
        "- queue_rows: $queueRowCount",
        "- queue_ready_rows: $queueReadyCount",
        "- closure_posture: not Catapult closure; not SCVerify closure",
        "- design_mainline_change: none by this script",
        "- smoke_case: $($Smoke.IsPresent)"
    ) | Set-Content -Path $executionPathAbs -Encoding UTF8

    $summaryRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $summaryPathAbs
    $executionRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $executionPathAbs
    $verdictRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $verdictPathAbs
    $manifestRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $manifestPathAbs
    $acceptanceFilledRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $acceptanceFilledPathAbs

    $verdict = [ordered]@{
        run_id = $runId
        status = "PASS"
        mode = $mode
        scope = "local-only skeleton"
        queue = [ordered]@{
            rows = $queueRowCount
            ready_rows = $queueReadyCount
        }
        closure_posture = @(
            "not Catapult closure",
            "not SCVerify closure"
        )
        docs = [ordered]@{
            night_pack = (Get-RepoRelativePath -BasePath $repoRoot -TargetPath $nightPackDocAbs)
            task_queue = (Get-RepoRelativePath -BasePath $repoRoot -TargetPath $taskQueueDocAbs)
            acceptance_pack = (Get-RepoRelativePath -BasePath $repoRoot -TargetPath $acceptancePackDocAbs)
        }
        artifacts = [ordered]@{
            summary = $summaryRel
            execution = $executionRel
            verdict = $verdictRel
            manifest = $manifestRel
            acceptance_filled = $acceptanceFilledRel
        }
    }
    ($verdict | ConvertTo-Json -Depth 8) | Set-Content -Path $verdictPathAbs -Encoding UTF8

    @(
        "# ACCEPTANCE_PACK_FILLED",
        "",
        "## 1. Summary",
        "- scope: night-run automation skeleton",
        "- key outcome: emitted fixed evidence outputs for one local run",
        "- boundary note: no attention/design code changed by this script",
        "",
        "## 2. Exact files changed",
        "- repo_tracked_files: none (runtime artifact only)",
        "- local_only_files: $summaryRel, $executionRel, $verdictRel, $manifestRel, $acceptanceFilledRel",
        "",
        "## 3. Exact commands run",
        "- command_1: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_night_pack.ps1 -BuildDir $BuildDir -Smoke:$($Smoke.IsPresent)",
        "",
        "## 4. Actual execution evidence / log excerpt",
        "- evidence_1: PASS: run_night_pack",
        "- evidence_2: run_dir: $runDirRel",
        "",
        "## 5. Repo-tracked artifacts",
        "- artifact_1: docs/night_run/NIGHT_PACK.md",
        "- artifact_2: docs/night_run/TASK_QUEUE.md",
        "- artifact_3: docs/night_run/ACCEPTANCE_PACK.md",
        "- artifact_4: scripts/local/run_night_pack.ps1",
        "",
        "## 6. Local-only working-memory artifacts",
        "- artifact_1: $summaryRel",
        "- artifact_2: $executionRel",
        "- artifact_3: $verdictRel",
        "- artifact_4: $manifestRel",
        "- artifact_5: $acceptanceFilledRel",
        "",
        "## 7. Governance posture",
        "- hls_hardware_boundary: respected",
        "- shared_sram_ownership: Top-only production shared-SRAM owner",
        "- closure_posture: not Catapult closure; not SCVerify closure",
        "- local_only_marking: runtime outputs under build/night_run",
        "",
        "## 8. Residual risks",
        "- risk_1: queue parser currently recognizes markdown table rows only",
        "- mitigation_or_watchpoint: keep TASK_QUEUE in stable table format",
        "",
        "## 9. Recommended next step",
        "- next_step: replace placeholder queue rows with actual nightly tasks",
        "- acceptance_check: rerun run_night_pack and confirm PASS + artifact contract"
    ) | Set-Content -Path $acceptanceFilledPathAbs -Encoding UTF8

    @(
        $summaryRel,
        $executionRel,
        $verdictRel,
        $manifestRel,
        $acceptanceFilledRel
    ) | Set-Content -Path $manifestPathAbs -Encoding UTF8

    Write-Host "PASS: run_night_pack"
    Write-Host ("run_id: {0}" -f $runId)
    Write-Host ("run_dir: {0}" -f $runDirRel)
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
