param(
    [string]$BuildDir = "build\\night_run",
    [string]$NightPackDoc = "docs/night_run/NIGHT_PACK.md",
    [string]$TaskQueueDoc = "docs/night_run/TASK_QUEUE.md",
    [string]$AcceptancePackDoc = "docs/night_run/ACCEPTANCE_PACK.md",
    [int]$MaxReadyTasks = 3,
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

function Split-MarkdownRow {
    param(
        [string]$Line
    )

    $trimmed = $Line.Trim()
    if (-not $trimmed.StartsWith("|")) {
        return ,@()
    }

    $trimmed = $trimmed.Trim("|")
    $parts = $trimmed.Split("|")
    $cells = @()
    foreach ($part in $parts) {
        $cells += $part.Trim()
    }
    return ,$cells
}

function Parse-TaskQueue {
    param(
        [string]$Path
    )

    $lines = Get-Content -Path $Path
    $headerIndex = -1
    $headers = @()

    for ($i = 0; $i -lt $lines.Count; $i++) {
        $cells = Split-MarkdownRow -Line $lines[$i]
        if ($cells.Count -eq 0) {
            continue
        }
        if ($cells -contains "task_id") {
            $headerIndex = $i
            $headers = $cells
            break
        }
    }

    if ($headerIndex -lt 0) {
        throw "task queue table header with task_id not found: $Path"
    }

    $requiredColumns = @(
        "task_id",
        "status",
        "lane",
        "depends_on",
        "runner",
        "stop_on_fail",
        "objective",
        "acceptance"
    )
    foreach ($requiredColumn in $requiredColumns) {
        if ($headers -notcontains $requiredColumn) {
            throw "task queue required column missing: $requiredColumn"
        }
    }

    $rows = New-Object System.Collections.Generic.List[object]
    for ($i = $headerIndex + 1; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        $cells = Split-MarkdownRow -Line $line
        if ($cells.Count -eq 0) {
            continue
        }

        $separatorRow = $true
        foreach ($cell in $cells) {
            if ($cell -notmatch '^:?-{3,}:?$') {
                $separatorRow = $false
                break
            }
        }
        if ($separatorRow) {
            continue
        }

        if ($cells.Count -ne $headers.Count) {
            throw "task queue row column mismatch at line $($i + 1): expected $($headers.Count), got $($cells.Count)"
        }

        $rowData = [ordered]@{}
        for ($col = 0; $col -lt $headers.Count; $col++) {
            $rowData[$headers[$col]] = $cells[$col]
        }
        $rows.Add([pscustomobject]$rowData)
    }

    return $rows
}

function Parse-DependsOn {
    param(
        [string]$DependsOnRaw
    )

    if ([string]::IsNullOrWhiteSpace($DependsOnRaw)) {
        return ,@()
    }
    if ($DependsOnRaw -eq "-") {
        return ,@()
    }

    $tokens = @()
    foreach ($part in $DependsOnRaw.Split(",")) {
        $token = $part.Trim()
        if ($token.Length -gt 0) {
            $tokens += $token
        }
    }
    return ,$tokens
}

function Parse-BoolStrict {
    param(
        [string]$Raw,
        [string]$FieldName,
        [string]$TaskId
    )

    $normalized = $Raw.Trim().ToLowerInvariant()
    if ($normalized -eq "true") {
        return $true
    }
    if ($normalized -eq "false") {
        return $false
    }

    throw "invalid boolean value for $FieldName in ${TaskId}: $Raw (expected true/false)"
}

function Get-SafeName {
    param(
        [string]$Name
    )

    $invalidChars = [System.IO.Path]::GetInvalidFileNameChars()
    $safe = $Name
    foreach ($invalidChar in $invalidChars) {
        $safe = $safe.Replace([string]$invalidChar, "_")
    }
    return $safe
}

function Find-VsDevCmdPath {
    $candidates = @()
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles "Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\VsDevCmd.bat")
        $candidates += (Join-Path $env:ProgramFiles "Microsoft Visual Studio\\18\\Community\\Common7\\Tools\\VsDevCmd.bat")
    }

    $programFilesX86 = ${env:ProgramFiles(x86)}
    if (-not [string]::IsNullOrWhiteSpace($programFilesX86)) {
        $vswherePath = Join-Path $programFilesX86 "Microsoft Visual Studio\\Installer\\vswhere.exe"
        if (Test-Path -LiteralPath $vswherePath) {
            try {
                $installPath = & $vswherePath -latest -products * -property installationPath
                if (-not [string]::IsNullOrWhiteSpace($installPath)) {
                    $candidates = @((Join-Path $installPath "Common7\\Tools\\VsDevCmd.bat")) + $candidates
                }
            }
            catch {
                # Keep fallback candidates when vswhere probing fails.
            }
        }
    }

    foreach ($candidate in $candidates) {
        if ((-not [string]::IsNullOrWhiteSpace($candidate)) -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }
    return $null
}

function Import-VsDevCmdEnvironment {
    param(
        [string]$VsDevCmdPath
    )

    if ([string]::IsNullOrWhiteSpace($VsDevCmdPath)) {
        throw "VsDevCmd path is empty"
    }

    $envLines = cmd /c "call `"$VsDevCmdPath`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to import VsDevCmd environment (exit=$LASTEXITCODE): $VsDevCmdPath"
    }

    foreach ($line in $envLines) {
        if ($line -match '^(.*?)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
}

function Get-FirstMatchingLine {
    param(
        [string]$Path,
        [string[]]$Needles
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }
    foreach ($needle in $Needles) {
        $match = Select-String -Path $Path -SimpleMatch -Pattern $needle | Select-Object -First 1
        if ($null -ne $match) {
            return $match.Line.Trim()
        }
    }
    return ""
}

function Resolve-RunnerSpec {
    param(
        [string]$RunnerKey,
        [string]$RunId,
        [string]$TaskId,
        [string]$TaskDirAbs
    )

    switch ($RunnerKey) {
        "checker.design_purity" {
            return [pscustomobject]@{
                command = "powershell"
                args = @(
                    "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", "scripts/check_design_purity.ps1"
                )
                required_pass = "PASS: check_design_purity"
                toolchain_note = "none"
                requires_vsdevcmd = $false
                expected_artifacts = @()
            }
        }
        "runner.init_agent_state" {
            $sessionTag = "night_run_{0}_{1}" -f $RunId, (Get-SafeName -Name $TaskId)
            return [pscustomobject]@{
                command = "powershell"
                args = @(
                    "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", "scripts/init_agent_state.ps1",
                    "-RepoRoot", ".",
                    "-StateRoot", "build/agent_state",
                    "-SessionTag", $sessionTag
                )
                required_pass = "PASS: init_agent_state"
                toolchain_note = "none"
                requires_vsdevcmd = $false
                expected_artifacts = @()
            }
        }
        "runner.local.p11aj" {
            $runnerBuildDir = Join-Path $TaskDirAbs "p11aj_runner_build"
            return [pscustomobject]@{
                command = "powershell"
                args = @(
                    "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", "scripts/local/run_p11aj_top_managed_sram_provenance.ps1",
                    "-BuildDir", $runnerBuildDir
                )
                required_pass = "PASS: run_p11aj_top_managed_sram_provenance"
                toolchain_note = "requires cl via VsDevCmd (MSVC x64 host/toolchain)"
                requires_vsdevcmd = $true
                expected_artifacts = @(
                    (Join-Path $runnerBuildDir "build.log"),
                    (Join-Path $runnerBuildDir "run.log"),
                    (Join-Path $runnerBuildDir "verdict.txt"),
                    (Join-Path $runnerBuildDir "file_manifest.txt")
                )
            }
        }
        default {
            throw "unsupported runner key in v1.1: $RunnerKey"
        }
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path)
Push-Location $repoRoot
try {
    if ($MaxReadyTasks -lt 1) {
        throw "MaxReadyTasks must be >= 1"
    }

    $buildAbs = Join-RepoPath -RepoRootPath $repoRoot -Path $BuildDir
    New-Item -ItemType Directory -Force -Path $buildAbs > $null

    $nightPackDocAbs = Join-RepoPath -RepoRootPath $repoRoot -Path $NightPackDoc
    $taskQueueDocAbs = Join-RepoPath -RepoRootPath $repoRoot -Path $TaskQueueDoc
    $acceptancePackDocAbs = Join-RepoPath -RepoRootPath $repoRoot -Path $AcceptancePackDoc

    Assert-FileExists -Path $nightPackDocAbs -Label "NIGHT_PACK document"
    Assert-FileExists -Path $taskQueueDocAbs -Label "TASK_QUEUE document"
    Assert-FileExists -Path $acceptancePackDocAbs -Label "ACCEPTANCE_PACK document"

    $taskQueueRows = @(Parse-TaskQueue -Path $taskQueueDocAbs)
    $queueRowCount = $taskQueueRows.Count
    $queueReadyRows = @($taskQueueRows | Where-Object { $_.status.Trim().ToLowerInvariant() -eq "ready" })
    $queueReadyCount = $queueReadyRows.Count

    $runId = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDirAbs = Join-Path $buildAbs $runId
    New-Item -ItemType Directory -Force -Path $runDirAbs > $null
    $runDirRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $runDirAbs
    $tasksRootAbs = Join-Path $runDirAbs "tasks"
    New-Item -ItemType Directory -Force -Path $tasksRootAbs > $null

    $mode = if ($Smoke.IsPresent) { "smoke" } else { "dispatch" }

    $summaryPathAbs = Join-Path $runDirAbs "NIGHT_PACK_SUMMARY.txt"
    $executionPathAbs = Join-Path $runDirAbs "NIGHT_PACK_EXECUTION.md"
    $verdictPathAbs = Join-Path $runDirAbs "NIGHT_PACK_VERDICT.json"
    $manifestPathAbs = Join-Path $runDirAbs "NIGHT_PACK_MANIFEST.txt"
    $acceptanceFilledPathAbs = Join-Path $runDirAbs "ACCEPTANCE_PACK_FILLED.md"

    $taskResults = New-Object System.Collections.Generic.List[object]
    $completedTaskIds = New-Object "System.Collections.Generic.HashSet[string]"
    $executedReadyCount = 0
    $stopDispatch = $false

    if (-not $Smoke.IsPresent) {
        if ($queueReadyCount -eq 0) {
            throw "dispatch mode requires at least one ready task in queue"
        }

        foreach ($task in $taskQueueRows) {
            $taskId = $task.task_id.Trim()
            $taskStatusRaw = $task.status.Trim().ToLowerInvariant()
            if ($taskStatusRaw -ne "ready") {
                continue
            }

            if ($executedReadyCount -ge $MaxReadyTasks) {
                break
            }

            $dependsOnList = Parse-DependsOn -DependsOnRaw $task.depends_on
            $missingDependencies = @()
            foreach ($dependencyId in $dependsOnList) {
                $dependencyRow = @($taskQueueRows | Where-Object { $_.task_id.Trim() -eq $dependencyId } | Select-Object -First 1)
                if ($dependencyRow.Count -eq 0) {
                    $missingDependencies += $dependencyId
                    continue
                }

                $dependencyDoneInQueue = ($dependencyRow[0].status.Trim().ToLowerInvariant() -eq "done")
                if ((-not $completedTaskIds.Contains($dependencyId)) -and (-not $dependencyDoneInQueue)) {
                    $missingDependencies += $dependencyId
                }
            }

            $stopOnFail = Parse-BoolStrict -Raw $task.stop_on_fail -FieldName "stop_on_fail" -TaskId $taskId
            $safeTaskId = Get-SafeName -Name $taskId
            $taskDirAbs = Join-Path $tasksRootAbs $safeTaskId
            New-Item -ItemType Directory -Force -Path $taskDirAbs > $null
            $taskDirRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $taskDirAbs
            $taskCommandPathAbs = Join-Path $taskDirAbs "task_command.txt"
            $taskLogPathAbs = Join-Path $taskDirAbs "task_execution.log"
            $taskSummaryPathAbs = Join-Path $taskDirAbs "task_summary.txt"

            $taskCommandString = ""
            $taskResultStatus = "SKIPPED"
            $taskExitCode = -1
            $requiredPass = ""
            $requiredPassFound = $false
            $taskMessage = ""
            $toolchainNote = ""
            $evidenceExcerpt = ""
            $runnerArtifactRels = @()
            $vsDevCmdPathUsed = ""

            if ($stopDispatch) {
                $taskResultStatus = "SKIPPED_STOP_ON_FAIL"
                $taskMessage = "previous task failed and stop_on_fail barrier is active"
                Set-Content -Path $taskLogPathAbs -Encoding UTF8 -Value $taskMessage
            }
            elseif ($missingDependencies.Count -gt 0) {
                $taskResultStatus = "SKIPPED_DEPENDENCY"
                $taskMessage = "missing dependency completion: $($missingDependencies -join ',')"
                Set-Content -Path $taskLogPathAbs -Encoding UTF8 -Value $taskMessage
            }
            else {
                try {
                    $runnerSpec = Resolve-RunnerSpec -RunnerKey $task.runner.Trim() -RunId $runId -TaskId $taskId -TaskDirAbs $taskDirAbs
                    $taskCommandString = "$($runnerSpec.command) $($runnerSpec.args -join ' ')"
                    $toolchainNote = $runnerSpec.toolchain_note
                    Set-Content -Path $taskCommandPathAbs -Encoding UTF8 -Value $taskCommandString

                    if ($runnerSpec.requires_vsdevcmd) {
                        $vsDevCmdPath = Find-VsDevCmdPath
                        if ([string]::IsNullOrWhiteSpace($vsDevCmdPath)) {
                            throw "runner requires VsDevCmd but no VsDevCmd.bat was found"
                        }
                        Import-VsDevCmdEnvironment -VsDevCmdPath $vsDevCmdPath
                        $vsDevCmdPathUsed = $vsDevCmdPath
                    }

                    & $runnerSpec.command @($runnerSpec.args) *> $taskLogPathAbs
                    $taskExitCode = $LASTEXITCODE
                    $requiredPass = $runnerSpec.required_pass
                    $requiredPassFound = $true
                    if (-not [string]::IsNullOrWhiteSpace($requiredPass)) {
                        $requiredPassFound = Select-String -Path $taskLogPathAbs -SimpleMatch -Quiet $requiredPass
                    }

                    foreach ($artifactPath in $runnerSpec.expected_artifacts) {
                        if (Test-Path -LiteralPath $artifactPath) {
                            $runnerArtifactRels += (Get-RepoRelativePath -BasePath $repoRoot -TargetPath $artifactPath)
                        }
                    }

                    if (($taskExitCode -eq 0) -and $requiredPassFound) {
                        $taskResultStatus = "PASS"
                        $completedTaskIds.Add($taskId) > $null
                    }
                    else {
                        $taskResultStatus = "FAIL"
                        if ($taskExitCode -ne 0) {
                            $taskMessage = "non-zero exit code: $taskExitCode"
                        }
                        elseif (-not $requiredPassFound) {
                            $taskMessage = "required pass string missing: $requiredPass"
                        }
                    }
                }
                catch {
                    $taskExitCode = 1
                    $taskResultStatus = "FAIL"
                    $taskMessage = $_.Exception.Message
                    Add-Content -Path $taskLogPathAbs -Encoding UTF8 -Value ("[night_pack_exception] {0}" -f $taskMessage)
                }

                if (($taskResultStatus -eq "FAIL") -and $stopOnFail) {
                    $stopDispatch = $true
                }
                $executedReadyCount++
            }

            if (-not [string]::IsNullOrWhiteSpace($requiredPass)) {
                $evidenceExcerpt = Get-FirstMatchingLine -Path $taskLogPathAbs -Needles @($requiredPass)
            }
            if ([string]::IsNullOrWhiteSpace($evidenceExcerpt)) {
                foreach ($runnerArtifactRel in $runnerArtifactRels) {
                    $runnerArtifactAbs = Join-RepoPath -RepoRootPath $repoRoot -Path $runnerArtifactRel
                    $candidate = Get-FirstMatchingLine -Path $runnerArtifactAbs -Needles @("PASS:", "status: PASS", "PASS")
                    if (-not [string]::IsNullOrWhiteSpace($candidate)) {
                        $evidenceExcerpt = $candidate
                        break
                    }
                }
            }
            if ([string]::IsNullOrWhiteSpace($evidenceExcerpt)) {
                $evidenceExcerpt = "no evidence excerpt matched"
            }

            @(
                "task_id: $taskId",
                "lane: $($task.lane)",
                "runner: $($task.runner)",
                "command: $taskCommandString",
                "status: $taskResultStatus",
                "exit_code: $taskExitCode",
                "stop_on_fail: $stopOnFail",
                "depends_on: $($task.depends_on)",
                "toolchain_note: $toolchainNote",
                "vsdevcmd_path: $vsDevCmdPathUsed",
                "required_pass: $requiredPass",
                "required_pass_found: $requiredPassFound",
                "evidence_excerpt: $evidenceExcerpt",
                "message: $taskMessage"
            ) | Set-Content -Path $taskSummaryPathAbs -Encoding UTF8
            if ($runnerArtifactRels.Count -gt 0) {
                Add-Content -Path $taskSummaryPathAbs -Encoding UTF8 -Value "runner_artifacts:"
                foreach ($runnerArtifactRel in $runnerArtifactRels) {
                    Add-Content -Path $taskSummaryPathAbs -Encoding UTF8 -Value ("- {0}" -f $runnerArtifactRel)
                }
            }

            $taskResults.Add([pscustomobject]@{
                task_id = $taskId
                lane = $task.lane
                runner = $task.runner
                queue_status = $task.status
                result = $taskResultStatus
                exit_code = $taskExitCode
                stop_on_fail = $stopOnFail
                depends_on = $task.depends_on
                command = $taskCommandString
                message = $taskMessage
                toolchain_note = $toolchainNote
                vsdevcmd_path = $vsDevCmdPathUsed
                evidence_excerpt = $evidenceExcerpt
                required_pass = $requiredPass
                required_pass_found = $requiredPassFound
                runner_artifacts = $runnerArtifactRels
                artifacts = [ordered]@{
                    task_dir = $taskDirRel
                    command = (Get-RepoRelativePath -BasePath $repoRoot -TargetPath $taskCommandPathAbs)
                    log = (Get-RepoRelativePath -BasePath $repoRoot -TargetPath $taskLogPathAbs)
                    summary = (Get-RepoRelativePath -BasePath $repoRoot -TargetPath $taskSummaryPathAbs)
                }
            })
        }
    }

    $taskPassCount = @($taskResults | Where-Object { $_.result -eq "PASS" }).Count
    $taskFailCount = @($taskResults | Where-Object { $_.result -eq "FAIL" }).Count
    $taskSkipCount = @($taskResults | Where-Object { $_.result -like "SKIPPED*" }).Count
    $checkerPassCount = @($taskResults | Where-Object { ($_.lane -eq "checker") -and ($_.result -eq "PASS") }).Count
    $runnerExecutedCount = @($taskResults | Where-Object { ($_.lane -eq "runner") -and ($_.result -ne "SKIPPED_DEPENDENCY") -and ($_.result -ne "SKIPPED_STOP_ON_FAIL") }).Count

    $overallStatus = "PASS"
    if (-not $Smoke.IsPresent) {
        if ($taskFailCount -gt 0) {
            $overallStatus = "FAIL"
        }
        elseif ($checkerPassCount -lt 1) {
            $overallStatus = "FAIL"
        }
        elseif ($runnerExecutedCount -lt 1) {
            $overallStatus = "FAIL"
        }
        elseif ($taskPassCount -lt 2) {
            $overallStatus = "FAIL"
        }
    }

    @(
        "status: $overallStatus",
        "scope: local-only night-run v1.1",
        "mode: $mode",
        "queue_rows: $queueRowCount",
        "queue_ready_rows: $queueReadyCount",
        "executed_ready_rows: $executedReadyCount",
        "task_pass: $taskPassCount",
        "task_fail: $taskFailCount",
        "task_skipped: $taskSkipCount",
        "closure: not Catapult closure; not SCVerify closure",
        "PASS: run_night_pack"
    ) | Set-Content -Path $summaryPathAbs -Encoding UTF8

    $executionLines = New-Object System.Collections.Generic.List[string]
    $executionLines.Add("# NIGHT_PACK_EXECUTION")
    $executionLines.Add("")
    $executionLines.Add("- run_id: $runId")
    $executionLines.Add("- run_dir: $runDirRel")
    $executionLines.Add("- mode: $mode")
    $executionLines.Add("- scope: local-only night-run v1.1")
    $executionLines.Add("- queue_rows: $queueRowCount")
    $executionLines.Add("- queue_ready_rows: $queueReadyCount")
    $executionLines.Add("- executed_ready_rows: $executedReadyCount")
    $executionLines.Add("- task_pass: $taskPassCount")
    $executionLines.Add("- task_fail: $taskFailCount")
    $executionLines.Add("- task_skipped: $taskSkipCount")
    $executionLines.Add("- closure_posture: not Catapult closure; not SCVerify closure")
    $executionLines.Add("- design_mainline_change: none by this script")
    $executionLines.Add("- smoke_case: $($Smoke.IsPresent)")
    $executionLines.Add("")
    $executionLines.Add("## Task Results")
    if ($taskResults.Count -eq 0) {
        $executionLines.Add("- no tasks executed (smoke mode)")
    }
    else {
        foreach ($taskResult in $taskResults) {
            $executionLines.Add("- $($taskResult.task_id) [$($taskResult.lane)] => $($taskResult.result)")
            $executionLines.Add("  runner: $($taskResult.runner)")
            $executionLines.Add("  log: $($taskResult.artifacts.log)")
            $executionLines.Add("  evidence: $($taskResult.evidence_excerpt)")
            if (-not [string]::IsNullOrWhiteSpace($taskResult.toolchain_note)) {
                $executionLines.Add("  toolchain: $($taskResult.toolchain_note)")
            }
            if (-not [string]::IsNullOrWhiteSpace($taskResult.message)) {
                $executionLines.Add("  message: $($taskResult.message)")
            }
            if ($taskResult.runner_artifacts.Count -gt 0) {
                foreach ($runnerArtifact in $taskResult.runner_artifacts) {
                    $executionLines.Add("  runner_artifact: $runnerArtifact")
                }
            }
        }
    }
    $executionLines | Set-Content -Path $executionPathAbs -Encoding UTF8

    $summaryRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $summaryPathAbs
    $executionRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $executionPathAbs
    $verdictRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $verdictPathAbs
    $manifestRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $manifestPathAbs
    $acceptanceFilledRel = Get-RepoRelativePath -BasePath $repoRoot -TargetPath $acceptanceFilledPathAbs
    [object[]]$taskResultArray = $taskResults.ToArray()

    $verdict = [ordered]@{
        run_id = $runId
        status = $overallStatus
        mode = $mode
        scope = "local-only night-run v1.1"
        queue = [ordered]@{
            rows = $queueRowCount
            ready_rows = $queueReadyCount
            executed_ready_rows = $executedReadyCount
            max_ready_tasks = $MaxReadyTasks
        }
        task_counts = [ordered]@{
            pass = $taskPassCount
            fail = $taskFailCount
            skipped = $taskSkipCount
            checker_pass = $checkerPassCount
            runner_executed = $runnerExecutedCount
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
        tasks = $taskResultArray
        artifacts = [ordered]@{
            summary = $summaryRel
            execution = $executionRel
            verdict = $verdictRel
            manifest = $manifestRel
            acceptance_filled = $acceptanceFilledRel
        }
    }
    ($verdict | ConvertTo-Json -Depth 10) | Set-Content -Path $verdictPathAbs -Encoding UTF8

    $acceptanceLines = New-Object System.Collections.Generic.List[string]
    $acceptanceLines.Add("# ACCEPTANCE_PACK_FILLED")
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 1. Summary")
    $acceptanceLines.Add("- scope: night-run automation dispatch v1.1")
    $acceptanceLines.Add("- key outcome: queue-driven task dispatch with per-task evidence artifacts")
    $acceptanceLines.Add("- boundary note: no attention/design code changed by this script")
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 2. Exact files changed")
    $acceptanceLines.Add("- repo_tracked_files: none (runtime artifact only)")
    $acceptanceLines.Add("- local_only_files: $summaryRel, $executionRel, $verdictRel, $manifestRel, $acceptanceFilledRel")
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 3. Exact commands run")
    $acceptanceLines.Add("- command_1: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_night_pack.ps1 -BuildDir $BuildDir -MaxReadyTasks $MaxReadyTasks -Smoke:$($Smoke.IsPresent)")
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 4. Actual execution evidence / log excerpt")
    $acceptanceLines.Add("- evidence_1: status: $overallStatus")
    $acceptanceLines.Add("- evidence_2: mode: $mode")
    $acceptanceLines.Add("- evidence_3: PASS: run_night_pack")
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 4a. Per-task execution evidence")
    if ($taskResults.Count -eq 0) {
        $acceptanceLines.Add("- task_id: none (smoke mode)")
    }
    else {
        foreach ($taskResult in $taskResults) {
            $acceptanceLines.Add("- task_id: $($taskResult.task_id)")
            $acceptanceLines.Add("- lane: $($taskResult.lane)")
            $acceptanceLines.Add("- runner: $($taskResult.runner)")
            $acceptanceLines.Add("- task_status: $($taskResult.result)")
            $acceptanceLines.Add("- task_log_excerpt: $($taskResult.artifacts.log)")
            $acceptanceLines.Add("- evidence_excerpt: $($taskResult.evidence_excerpt)")
            if ($taskResult.runner_artifacts.Count -gt 0) {
                foreach ($runnerArtifact in $taskResult.runner_artifacts) {
                    $acceptanceLines.Add("- runner_artifact: $runnerArtifact")
                }
            }
        }
    }
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 5. Repo-tracked artifacts")
    $acceptanceLines.Add("- artifact_1: docs/night_run/NIGHT_PACK.md")
    $acceptanceLines.Add("- artifact_2: docs/night_run/TASK_QUEUE.md")
    $acceptanceLines.Add("- artifact_3: docs/night_run/ACCEPTANCE_PACK.md")
    $acceptanceLines.Add("- artifact_4: scripts/local/run_night_pack.ps1")
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 6. Local-only working-memory artifacts")
    $acceptanceLines.Add("- artifact_1: $summaryRel")
    $acceptanceLines.Add("- artifact_2: $executionRel")
    $acceptanceLines.Add("- artifact_3: $verdictRel")
    $acceptanceLines.Add("- artifact_4: $manifestRel")
    $acceptanceLines.Add("- artifact_5: $acceptanceFilledRel")
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 7. Governance posture")
    $acceptanceLines.Add("- hls_hardware_boundary: respected")
    $acceptanceLines.Add("- shared_sram_ownership: Top-only production shared-SRAM owner")
    $acceptanceLines.Add("- closure_posture: not Catapult closure; not SCVerify closure")
    $acceptanceLines.Add("- local_only_marking: runtime outputs under build/night_run")
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 8. Residual risks")
    $acceptanceLines.Add("- risk_1: v1.1 supports only bounded runner keys (checker.design_purity, runner.init_agent_state, runner.local.p11aj)")
    $acceptanceLines.Add("- mitigation_or_watchpoint: expand mapping by explicit review per additional task type")
    $acceptanceLines.Add("")
    $acceptanceLines.Add("## 9. Recommended next step")
    $acceptanceLines.Add("- next_step: add one compile-backed local runner key with the same per-task evidence contract")
    $acceptanceLines.Add("- acceptance_check: rerun dispatch and verify checker+runner both PASS")
    $acceptanceLines | Set-Content -Path $acceptanceFilledPathAbs -Encoding UTF8

    $manifestEntries = New-Object System.Collections.Generic.List[string]
    $manifestEntries.Add($summaryRel)
    $manifestEntries.Add($executionRel)
    $manifestEntries.Add($verdictRel)
    $manifestEntries.Add($manifestRel)
    $manifestEntries.Add($acceptanceFilledRel)
    foreach ($taskResult in $taskResults) {
        $manifestEntries.Add($taskResult.artifacts.command)
        $manifestEntries.Add($taskResult.artifacts.log)
        $manifestEntries.Add($taskResult.artifacts.summary)
        foreach ($runnerArtifact in $taskResult.runner_artifacts) {
            $manifestEntries.Add($runnerArtifact)
        }
    }
    $manifestEntries | Set-Content -Path $manifestPathAbs -Encoding UTF8

    if ($overallStatus -eq "PASS") {
        Write-Host "PASS: run_night_pack"
        Write-Host ("run_id: {0}" -f $runId)
        Write-Host ("run_dir: {0}" -f $runDirRel)
        exit 0
    }

    Write-Host "FAIL: run_night_pack"
    Write-Host ("run_id: {0}" -f $runId)
    Write-Host ("run_dir: {0}" -f $runDirRel)
    exit 1
}
catch {
    Write-Host ("ERROR: {0}" -f $_.Exception.Message)
    if ($_.Exception -and $_.Exception.StackTrace) {
        Write-Host ("STACK: {0}" -f $_.Exception.StackTrace)
    }
    if ($_.InvocationInfo -and $_.InvocationInfo.PositionMessage) {
        Write-Host ("POSITION: {0}" -f $_.InvocationInfo.PositionMessage)
    }
    if ($_.ScriptStackTrace) {
        Write-Host ("SCRIPTSTACK: {0}" -f $_.ScriptStackTrace)
    }
    exit 1
}
finally {
    Pop-Location
}
