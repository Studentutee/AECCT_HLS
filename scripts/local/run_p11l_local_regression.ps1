param(
    [string]$BuildDir = "build\p11n"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-ClBuild {
    param(
        [string]$Source,
        [string]$ExeOut,
        [string]$LogOut,
        [string[]]$ExtraArgs = @()
    )

    $args = @(
        '/nologo',
        '/std:c++14',
        '/EHsc',
        '/utf-8',
        '/I.',
        '/Iinclude',
        '/Isrc',
        '/Igen\include',
        '/Ithird_party\ac_types',
        '/Idata\weights'
    ) + $ExtraArgs + @(
        $Source,
        "/Fe:$ExeOut"
    )
    & cl @args *> $LogOut
    if ($LASTEXITCODE -ne 0) {
        throw "build failed ($Source), exit=$LASTEXITCODE"
    }
}

function Invoke-ExeRun {
    param(
        [string]$ExePath,
        [string]$LogOut
    )

    cmd /c $ExePath > $LogOut 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "run failed ($ExePath), exit=$LASTEXITCODE"
    }
}

function Require-PassString {
    param(
        [string]$LogPath,
        [string]$Needle
    )

    if (-not (Select-String -Path $LogPath -SimpleMatch -Quiet $Needle)) {
        throw "required PASS string missing in $LogPath : $Needle"
    }
}

function Read-KvSig {
    param(
        [string]$LogPath
    )

    $line = Select-String -Path $LogPath -Pattern '^\[p11m\]\[KV_SIG\] K=0x[0-9A-Fa-f]{16} V=0x[0-9A-Fa-f]{16}$' | Select-Object -First 1
    if (-not $line) {
        throw "KV_SIG line missing in $LogPath"
    }
    if ($line.Line -notmatch '^\[p11m\]\[KV_SIG\] K=(0x[0-9A-Fa-f]{16}) V=(0x[0-9A-Fa-f]{16})$') {
        throw "KV_SIG parse failure in $LogPath"
    }
    return @{
        K = $Matches[1].ToUpperInvariant()
        V = $Matches[2].ToUpperInvariant()
    }
}

function Read-P11nSig {
    param(
        [string]$LogPath
    )

    $wkLine = Select-String -Path $LogPath -Pattern '^\[p11n\]\[WK_SIG\] K=0x[0-9A-Fa-f]{16}$' | Select-Object -First 1
    if (-not $wkLine) {
        throw "WK_SIG line missing in $LogPath"
    }
    $wvLine = Select-String -Path $LogPath -Pattern '^\[p11n\]\[WV_SIG\] V=0x[0-9A-Fa-f]{16}$' | Select-Object -First 1
    if (-not $wvLine) {
        throw "WV_SIG line missing in $LogPath"
    }

    $wkMatch = [regex]::Match($wkLine.Line, '^\[p11n\]\[WK_SIG\] K=(0x[0-9A-Fa-f]{16})$')
    if (-not $wkMatch.Success) {
        throw "WK_SIG parse failure in $LogPath"
    }
    $wvMatch = [regex]::Match($wvLine.Line, '^\[p11n\]\[WV_SIG\] V=(0x[0-9A-Fa-f]{16})$')
    if (-not $wvMatch.Success) {
        throw "WV_SIG parse failure in $LogPath"
    }

    return @{
        WK = $wkMatch.Groups[1].Value.ToUpperInvariant()
        WV = $wvMatch.Groups[1].Value.ToUpperInvariant()
    }
}

function To-RepoRelativePath {
    param(
        [string]$RepoRoot,
        [string]$Path
    )

    $root = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\')
    $full = [System.IO.Path]::GetFullPath($Path)
    if ($full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        $rel = $full.Substring($root.Length).TrimStart('\', '/')
        return ($rel -replace '\\', '/')
    }
    return ($Path -replace '\\', '/')
}

function Invoke-CheckScript {
    param(
        [string]$RepoRoot,
        [string]$ScriptRelPath,
        [string]$CheckKey,
        [System.Collections.IDictionary]$StatusTable,
        [string[]]$ExtraArgs = @()
    )

    $scriptPath = Join-Path $RepoRoot $ScriptRelPath
    if (-not (Test-Path $scriptPath)) {
        $StatusTable[$CheckKey] = 'FAIL'
        throw "check script missing: $ScriptRelPath"
    }

    Write-Host ("[p11p][PRECHECK] start {0}" -f $CheckKey)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $scriptPath -RepoRoot $RepoRoot @ExtraArgs
    if ($LASTEXITCODE -ne 0) {
        $StatusTable[$CheckKey] = 'FAIL'
        throw "check script failed: $ScriptRelPath (exit=$LASTEXITCODE)"
    }
    $StatusTable[$CheckKey] = 'PASS'
    Write-Host ("[p11p][PRECHECK] pass {0}" -f $CheckKey)
}

function Write-WarningSummaryP11P {
    param(
        [string]$RepoRoot,
        [string[]]$BuildLogs,
        [string]$OutPath
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $perLog = [ordered]@{}
    $totalWarnings = 0

    $lines.Add('[p11p][WARN_SUMMARY] begin')
    foreach ($logPath in $BuildLogs) {
        if (-not (Test-Path $logPath)) {
            throw "required build log missing for warning summary: $logPath"
        }
        $warnHits = Select-String -Path $logPath -Pattern '\bwarning\b' -AllMatches -CaseSensitive:$false
        $warnCount = ($warnHits | Measure-Object).Count
        $totalWarnings += $warnCount

        $rel = To-RepoRelativePath -RepoRoot $RepoRoot -Path $logPath
        $perLog[$rel] = $warnCount
        $lines.Add(("[p11p][WARN_SUMMARY] {0}: warnings={1}" -f $rel, $warnCount))

        if ($warnCount -gt 0) {
            $samples = $warnHits | Select-Object -First 2
            foreach ($sample in $samples) {
                $lines.Add(("[p11p][WARN_SAMPLE] {0}: {1}" -f $rel, $sample.Line.Trim()))
            }
        }
    }

    $lines.Add(("[p11p][WARN_SUMMARY] total_warnings={0}" -f $totalWarnings))
    $lines.Add('[p11p][WARN_SUMMARY] policy=allowlist-only-nonblocking')
    $lines.Add('[p11p][WARN_SUMMARY] end')

    $lines | Set-Content -Path $OutPath -Encoding UTF8
    foreach ($line in $lines) {
        Write-Host $line
    }

    return [ordered]@{
        total_warnings = $totalWarnings
        per_log = $perLog
        policy = 'allowlist-only-nonblocking'
    }
}

function Write-EvidenceManifest {
    param(
        [string]$OutPath,
        [string]$TaskId,
        [string]$Overall,
        [hashtable]$Artifacts
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('# EVIDENCE_MANIFEST_p11p')
    $lines.Add(("task_id: {0}" -f $TaskId))
    $lines.Add(("overall: {0}" -f $Overall))
    $lines.Add(("build_dir: {0}" -f $Artifacts['build_dir']))
    $lines.Add(("one_shot_run_log: {0}" -f $Artifacts['one_shot_run_log']))
    $lines.Add(("warning_summary: {0}" -f $Artifacts['warning_summary']))
    $lines.Add(("summary_markdown: {0}" -f $Artifacts['summary_markdown']))
    $lines.Add(("verdict_json: {0}" -f $Artifacts['verdict_json']))
    $lines.Add('core_raw_run_logs:')
    foreach ($item in $Artifacts['core_raw_run_logs']) {
        $lines.Add(("- {0}" -f $item))
    }

    $lines | Set-Content -Path $OutPath -Encoding UTF8
}

function Write-EvidenceSummary {
    param(
        [string]$OutPath,
        [string]$TaskId,
        [string]$Overall,
        [hashtable]$Prechecks,
        [hashtable]$Regression,
        [hashtable]$Compares,
        [hashtable]$Artifacts,
        [hashtable]$WarningSummary
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add(("# {0} Evidence Summary" -f $TaskId))
    $lines.Add('')
    $lines.Add(('overall: `{0}`' -f $Overall))
    $lines.Add('')

    $lines.Add('## Prechecks')
    foreach ($k in $Prechecks.Keys) {
        $lines.Add(('- `{0}`: `{1}`' -f $k, $Prechecks[$k]))
    }
    $lines.Add('')

    $lines.Add('## Regression')
    foreach ($k in $Regression.Keys) {
        $lines.Add(('- `{0}`: `{1}`' -f $k, $Regression[$k]))
    }
    $lines.Add('')

    $lines.Add('## Compares')
    foreach ($k in $Compares.Keys) {
        $cmpJson = $Compares[$k] | ConvertTo-Json -Compress -Depth 8
        $lines.Add(('- `{0}`: `{1}`' -f $k, $cmpJson))
    }
    $lines.Add('')

    $lines.Add('## Warning Summary')
    $lines.Add(('- `total_warnings`: `{0}`' -f $WarningSummary['total_warnings']))
    $lines.Add(('- `policy`: `{0}`' -f $WarningSummary['policy']))
    $lines.Add('')

    $lines.Add('## Artifacts')
    foreach ($k in $Artifacts.Keys) {
        $v = $Artifacts[$k]
        if ($v -is [System.Array]) {
            $lines.Add(('- `{0}`:' -f $k))
            foreach ($entry in $v) {
                $lines.Add(('  - `{0}`' -f $entry))
            }
        }
        else {
            $lines.Add(('- `{0}`: `{1}`' -f $k, $v))
        }
    }

    $lines | Set-Content -Path $OutPath -Encoding UTF8
}

function Write-VerdictJson {
    param(
        [string]$OutPath,
        [string]$TaskId,
        [string]$Overall,
        [hashtable]$Prechecks,
        [hashtable]$Regression,
        [hashtable]$Compares,
        [hashtable]$Artifacts
    )

    $payload = [ordered]@{
        task_id = $TaskId
        overall = $Overall
        prechecks = $Prechecks
        regression = $Regression
        compares = $Compares
        artifacts = $Artifacts
    }
    $payload | ConvertTo-Json -Depth 12 | Set-Content -Path $OutPath -Encoding UTF8
}

$taskId = 'P00-011P'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $prechecks = [ordered]@{}
    $regression = [ordered]@{}
    $compares = [ordered]@{}

    $exeP11j = Join-Path $BuildDir 'tb_ternary_live_leaf_smoke_p11j.exe'
    $exeP11k = Join-Path $BuildDir 'tb_ternary_live_leaf_top_smoke_p11k.exe'
    $exeP11lb = Join-Path $BuildDir 'tb_ternary_live_leaf_top_smoke_p11l_b.exe'
    $exeP11lc = Join-Path $BuildDir 'tb_ternary_live_leaf_top_smoke_p11l_c.exe'
    $exeP11mBaseline = Join-Path $BuildDir 'tb_ternary_live_source_integration_smoke_p11m_baseline.exe'
    $exeP11mMacro = Join-Path $BuildDir 'tb_ternary_live_source_integration_smoke_p11m_macro.exe'
    $exeP11nBaseline = Join-Path $BuildDir 'tb_ternary_live_family_source_integration_smoke_p11n_baseline.exe'
    $exeP11nMacro = Join-Path $BuildDir 'tb_ternary_live_family_source_integration_smoke_p11n_macro.exe'

    $logBuildP11j = Join-Path $BuildDir 'build_p11j.log'
    $logBuildP11k = Join-Path $BuildDir 'build_p11k.log'
    $logBuildP11lb = Join-Path $BuildDir 'build_p11l_b.log'
    $logBuildP11lc = Join-Path $BuildDir 'build_p11l_c.log'
    $logBuildP11mBaseline = Join-Path $BuildDir 'build_p11m_baseline.log'
    $logBuildP11mMacro = Join-Path $BuildDir 'build_p11m_macro.log'
    $logBuildP11nBaseline = Join-Path $BuildDir 'build_p11n_baseline.log'
    $logBuildP11nMacro = Join-Path $BuildDir 'build_p11n_macro.log'

    $logRunP11j = Join-Path $BuildDir 'run_p11j.log'
    $logRunP11k = Join-Path $BuildDir 'run_p11k.log'
    $logRunP11lb = Join-Path $BuildDir 'run_p11l_b.log'
    $logRunP11lc = Join-Path $BuildDir 'run_p11l_c.log'
    $logRunP11mBaseline = Join-Path $BuildDir 'run_p11m_baseline.log'
    $logRunP11mMacro = Join-Path $BuildDir 'run_p11m_macro.log'
    $logRunP11nBaseline = Join-Path $BuildDir 'run_p11n_baseline.log'
    $logRunP11nMacro = Join-Path $BuildDir 'run_p11n_macro.log'

    $warningSummaryPath = Join-Path $BuildDir 'warning_summary_p11p.txt'
    $evidenceSummaryPath = Join-Path $BuildDir 'EVIDENCE_SUMMARY_p11p.md'
    $evidenceManifestPath = Join-Path $BuildDir 'EVIDENCE_MANIFEST_p11p.txt'
    $verdictJsonPath = Join-Path $BuildDir 'verdict_p11p.json'
    $expectedOneShotRunLog = Join-Path $BuildDir 'run_p11p_regression.log'

    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_repo_hygiene.ps1' -CheckKey 'check_repo_hygiene_pre' -StatusTable $prechecks -ExtraArgs @('-Phase', 'pre', '-BuildDir', $BuildDir)
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_design_purity.ps1' -CheckKey 'check_design_purity' -StatusTable $prechecks
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_interface_lock.ps1' -CheckKey 'check_interface_lock' -StatusTable $prechecks
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_macro_hygiene.ps1' -CheckKey 'check_macro_hygiene' -StatusTable $prechecks
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_payload_metadata_ssot.ps1' -CheckKey 'check_qkv_payload_metadata_ssot_pre' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'pre')
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_weightstreamorder_continuity.ps1' -CheckKey 'check_qkv_weightstreamorder_continuity_pre' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'pre')
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_export_artifact_continuity.ps1' -CheckKey 'check_qkv_export_artifact_continuity_pre' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'pre')
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_export_consumer_semantics.ps1' -CheckKey 'check_qkv_export_consumer_semantics_pre' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'pre')
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_runtime_handoff_continuity.ps1' -CheckKey 'check_qkv_runtime_handoff_continuity_pre' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'pre')

    Invoke-ClBuild 'tb\tb_ternary_live_leaf_smoke_p11j.cpp' $exeP11j $logBuildP11j
    Invoke-ClBuild 'tb\tb_ternary_live_leaf_top_smoke_p11k.cpp' $exeP11k $logBuildP11k
    Invoke-ClBuild 'tb\tb_ternary_live_leaf_top_smoke_p11l_b.cpp' $exeP11lb $logBuildP11lb
    Invoke-ClBuild 'tb\tb_ternary_live_leaf_top_smoke_p11l_c.cpp' $exeP11lc $logBuildP11lc
    Invoke-ClBuild 'tb\tb_ternary_live_source_integration_smoke_p11m.cpp' $exeP11mBaseline $logBuildP11mBaseline
    Invoke-ClBuild 'tb\tb_ternary_live_source_integration_smoke_p11m.cpp' $exeP11mMacro $logBuildP11mMacro @('/DAECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE=1')
    Invoke-ClBuild 'tb\tb_ternary_live_family_source_integration_smoke_p11n.cpp' $exeP11nBaseline $logBuildP11nBaseline
    Invoke-ClBuild 'tb\tb_ternary_live_family_source_integration_smoke_p11n.cpp' $exeP11nMacro $logBuildP11nMacro @('/DAECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE=1')

    Invoke-ExeRun $exeP11j $logRunP11j
    Invoke-ExeRun $exeP11k $logRunP11k
    Invoke-ExeRun $exeP11lb $logRunP11lb
    Invoke-ExeRun $exeP11lc $logRunP11lc
    Invoke-ExeRun $exeP11mBaseline $logRunP11mBaseline
    Invoke-ExeRun $exeP11mMacro $logRunP11mMacro
    Invoke-ExeRun $exeP11nBaseline $logRunP11nBaseline
    Invoke-ExeRun $exeP11nMacro $logRunP11nMacro

    Require-PassString $logRunP11j 'PASS: tb_ternary_live_leaf_smoke_p11j'
    Require-PassString $logRunP11k 'PASS: tb_ternary_live_leaf_top_smoke_p11k'
    Require-PassString $logRunP11lb 'PASS: tb_ternary_live_leaf_top_smoke_p11l_b'
    Require-PassString $logRunP11lc 'PASS: tb_ternary_live_leaf_top_smoke_p11l_c'
    Require-PassString $logRunP11mBaseline 'PASS: tb_ternary_live_source_integration_smoke_p11m'
    Require-PassString $logRunP11mMacro 'PASS: tb_ternary_live_source_integration_smoke_p11m'
    Require-PassString $logRunP11mMacro '[p11m][PASS] source-side WQ integration path exact-match equivalent to split-interface local top'
    Require-PassString $logRunP11mMacro '[p11m][PASS] K/V fallback retained under WQ-only integration slice'
    Require-PassString $logRunP11nBaseline 'PASS: tb_ternary_live_family_source_integration_smoke_p11n'
    Require-PassString $logRunP11nMacro 'PASS: tb_ternary_live_family_source_integration_smoke_p11n'
    Require-PassString $logRunP11nBaseline '[p11n][PASS] WK/WV fallback retained under WK/WV-only integration slice'
    Require-PassString $logRunP11nMacro '[p11n][PASS] WK/WV fallback retained under WK/WV-only integration slice'
    Require-PassString $logRunP11nMacro '[p11n][PASS] source-side WK integration path exact-match equivalent to split-interface local top'
    Require-PassString $logRunP11nMacro '[p11n][PASS] source-side WV integration path exact-match equivalent to split-interface local top'

    $regression['p11j'] = 'PASS'
    $regression['p11k'] = 'PASS'
    $regression['p11l_b'] = 'PASS'
    $regression['p11l_c'] = 'PASS'
    $regression['p11m_baseline'] = 'PASS'
    $regression['p11m_macro'] = 'PASS'
    $regression['p11n_baseline'] = 'PASS'
    $regression['p11n_macro'] = 'PASS'
    $regression['final_pass_string'] = 'PASS'

    $kvBaseline = Read-KvSig $logRunP11mBaseline
    $kvMacro = Read-KvSig $logRunP11mMacro
    if ($kvBaseline.K -ne $kvMacro.K -or $kvBaseline.V -ne $kvMacro.V) {
        throw "KV signature mismatch baseline vs macro: baseline(K=$($kvBaseline.K),V=$($kvBaseline.V)) macro(K=$($kvMacro.K),V=$($kvMacro.V))"
    }
    $compares['p11m_kv_signature_equal'] = [ordered]@{
        status = 'PASS'
        baseline = [ordered]@{ K = $kvBaseline.K; V = $kvBaseline.V }
        macro = [ordered]@{ K = $kvMacro.K; V = $kvMacro.V }
    }

    $p11nBaseline = Read-P11nSig $logRunP11nBaseline
    $p11nMacro = Read-P11nSig $logRunP11nMacro
    if ($p11nBaseline.WK -ne $p11nMacro.WK -or $p11nBaseline.WV -ne $p11nMacro.WV) {
        throw "P11N signature mismatch baseline vs macro: baseline(WK=$($p11nBaseline.WK),WV=$($p11nBaseline.WV)) macro(WK=$($p11nMacro.WK),WV=$($p11nMacro.WV))"
    }
    $compares['p11n_wk_wv_signature_equal'] = [ordered]@{
        status = 'PASS'
        baseline = [ordered]@{ WK = $p11nBaseline.WK; WV = $p11nBaseline.WV }
        macro = [ordered]@{ WK = $p11nMacro.WK; WV = $p11nMacro.WV }
    }

    $allowedBuildLogs = @(
        $logBuildP11j,
        $logBuildP11k,
        $logBuildP11lb,
        $logBuildP11lc,
        $logBuildP11mBaseline,
        $logBuildP11mMacro,
        $logBuildP11nBaseline,
        $logBuildP11nMacro
    )
    $warningSummary = Write-WarningSummaryP11P -RepoRoot $repoRoot -BuildLogs $allowedBuildLogs -OutPath $warningSummaryPath

    $artifacts = [ordered]@{
        build_dir = To-RepoRelativePath -RepoRoot $repoRoot -Path $BuildDir
        one_shot_run_log = To-RepoRelativePath -RepoRoot $repoRoot -Path $expectedOneShotRunLog
        warning_summary = To-RepoRelativePath -RepoRoot $repoRoot -Path $warningSummaryPath
        summary_markdown = To-RepoRelativePath -RepoRoot $repoRoot -Path $evidenceSummaryPath
        manifest = To-RepoRelativePath -RepoRoot $repoRoot -Path $evidenceManifestPath
        verdict_json = To-RepoRelativePath -RepoRoot $repoRoot -Path $verdictJsonPath
        core_raw_run_logs = @(
            (To-RepoRelativePath -RepoRoot $repoRoot -Path $logRunP11j),
            (To-RepoRelativePath -RepoRoot $repoRoot -Path $logRunP11k),
            (To-RepoRelativePath -RepoRoot $repoRoot -Path $logRunP11lb),
            (To-RepoRelativePath -RepoRoot $repoRoot -Path $logRunP11lc),
            (To-RepoRelativePath -RepoRoot $repoRoot -Path $logRunP11mBaseline),
            (To-RepoRelativePath -RepoRoot $repoRoot -Path $logRunP11mMacro),
            (To-RepoRelativePath -RepoRoot $repoRoot -Path $logRunP11nBaseline),
            (To-RepoRelativePath -RepoRoot $repoRoot -Path $logRunP11nMacro)
        )
    }

    $prechecks['check_repo_hygiene_post'] = 'PENDING'
    $prechecks['check_qkv_payload_metadata_ssot_post'] = 'PENDING'
    $prechecks['check_qkv_weightstreamorder_continuity_post'] = 'PENDING'
    $prechecks['check_qkv_export_artifact_continuity_post'] = 'PENDING'
    $prechecks['check_qkv_export_consumer_semantics_post'] = 'PENDING'
    $prechecks['check_qkv_runtime_handoff_continuity_post'] = 'PENDING'
    $overall = 'PENDING_POSTCHECK'

    Write-EvidenceManifest -OutPath $evidenceManifestPath -TaskId $taskId -Overall $overall -Artifacts $artifacts
    Write-EvidenceSummary -OutPath $evidenceSummaryPath -TaskId $taskId -Overall $overall -Prechecks $prechecks -Regression $regression -Compares $compares -Artifacts $artifacts -WarningSummary $warningSummary
    Write-VerdictJson -OutPath $verdictJsonPath -TaskId $taskId -Overall $overall -Prechecks $prechecks -Regression $regression -Compares $compares -Artifacts $artifacts

    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_payload_metadata_ssot.ps1' -CheckKey 'check_qkv_payload_metadata_ssot_post' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'post')
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_weightstreamorder_continuity.ps1' -CheckKey 'check_qkv_weightstreamorder_continuity_post' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'post')
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_export_artifact_continuity.ps1' -CheckKey 'check_qkv_export_artifact_continuity_post' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'post')
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_export_consumer_semantics.ps1' -CheckKey 'check_qkv_export_consumer_semantics_post' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'post')
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_qkv_runtime_handoff_continuity.ps1' -CheckKey 'check_qkv_runtime_handoff_continuity_post' -StatusTable $prechecks -ExtraArgs @('-OutDir', $BuildDir, '-Phase', 'post')
    Invoke-CheckScript -RepoRoot $repoRoot -ScriptRelPath 'scripts/check_repo_hygiene.ps1' -CheckKey 'check_repo_hygiene_post' -StatusTable $prechecks -ExtraArgs @('-Phase', 'post', '-BuildDir', $BuildDir)

    $overall = 'PASS'
    Write-EvidenceManifest -OutPath $evidenceManifestPath -TaskId $taskId -Overall $overall -Artifacts $artifacts
    Write-EvidenceSummary -OutPath $evidenceSummaryPath -TaskId $taskId -Overall $overall -Prechecks $prechecks -Regression $regression -Compares $compares -Artifacts $artifacts -WarningSummary $warningSummary
    Write-VerdictJson -OutPath $verdictJsonPath -TaskId $taskId -Overall $overall -Prechecks $prechecks -Regression $regression -Compares $compares -Artifacts $artifacts

    Write-Host "PASS: run_p11l_local_regression"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
