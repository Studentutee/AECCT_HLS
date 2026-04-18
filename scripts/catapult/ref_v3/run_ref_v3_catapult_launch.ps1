param(
    [string]$BuildDir = "",
    [string]$RemoteHost = "140.124.41.193",
    [int]$RemotePort = 2190,
    [string]$RemoteUser = "peter",
    [string]$IdentityFile = ".codex_ssh/id_codex",
    [string]$KnownHostsFile = ".codex_ssh/known_hosts",
    [string]$RemoteRepoRoot = "/home/peter/AECCT/AECCT_HLS-master",
    [string]$CatapultCmd = "/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult",
    [switch]$PrepareOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Resolve-PathFromRepo {
    param(
        [string]$RepoRoot,
        [string]$InputPath
    )

    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        return [System.IO.Path]::GetFullPath($InputPath)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $InputPath))
}

function Copy-RemoteArtifact {
    param(
        [string[]]$ScpArgsCommon,
        [string]$RemoteUser,
        [string]$RemoteHost,
        [string]$RemoteOutDir,
        [string]$ArtifactName,
        [string]$LocalDir
    )

    $remoteSpec = "{0}@{1}:{2}/{3}" -f $RemoteUser, $RemoteHost, $RemoteOutDir, $ArtifactName
    $localPath = Join-Path $LocalDir $ArtifactName
    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & scp @ScpArgsCommon $remoteSpec $localPath *> $null
    $scpExit = $LASTEXITCODE
    $ErrorActionPreference = $prevErrorAction
    return ($scpExit -eq 0)
}

$repoRoot = (Resolve-Path (Join-Path (Join-Path $PSScriptRoot "..\..") "..")).Path
if ($BuildDir -eq "") {
    $BuildDir = "build\ref_v3\catapult_compile_first_remote_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$buildAbs = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
$identityAbs = Resolve-PathFromRepo -RepoRoot $repoRoot -InputPath $IdentityFile
$knownHostsAbs = Resolve-PathFromRepo -RepoRoot $repoRoot -InputPath $KnownHostsFile
$remoteRunLeaf = Split-Path $BuildDir -Leaf
$remoteProjectTcl = "$RemoteRepoRoot/scripts/catapult/ref_v3/project.tcl"
$remoteOutDir = "$RemoteRepoRoot/build/ref_v3/$remoteRunLeaf"

$runLog = Join-Path $buildAbs "run.log"
$verdictPath = Join-Path $buildAbs "verdict.txt"
$envSnapshotPath = Join-Path $buildAbs "env_snapshot.txt"
$exactCmdPath = Join-Path $buildAbs "exact_commands.sh"
$sshCmdPath = Join-Path $buildAbs "exact_commands_ssh.ps1"
$remoteScriptPath = Join-Path $buildAbs "remote_script.tcsh"
$sshConsoleLog = Join-Path $buildAbs "ssh_console.log"
$artifactManifestPath = Join-Path $buildAbs "artifact_manifest.txt"

New-Item -ItemType Directory -Force -Path $buildAbs > $null

if (-not (Test-Path -LiteralPath $identityAbs)) {
    throw "identity file not found: $identityAbs"
}
if (-not (Test-Path -LiteralPath $knownHostsAbs)) {
    throw "known_hosts file not found: $knownHostsAbs"
}

$sshPrefix = "ssh -tt -i $identityAbs -o UserKnownHostsFile=$knownHostsAbs -o StrictHostKeyChecking=yes -p $RemotePort $RemoteUser@$RemoteHost"

$remoteScriptTemplate = @'
set REPO_ROOT=__REMOTE_REPO_ROOT__
set OUTDIR=__REMOTE_OUTDIR__
set CATAPULT_BIN=__CATAPULT_BIN__
set PROJECT_TCL=__PROJECT_TCL__

mkdir -p $OUTDIR/project
cd $REPO_ROOT

setenv AECCT_REFV3_REPO_ROOT $REPO_ROOT
setenv AECCT_REFV3_CATAPULT_OUTDIR $OUTDIR/project

echo MODE=interactive_tty_login >! $OUTDIR/remote_env_snapshot.txt
echo WHOAMI=`whoami` >> $OUTDIR/remote_env_snapshot.txt
echo HOSTNAME=`hostname` >> $OUTDIR/remote_env_snapshot.txt
echo SHELL=$SHELL >> $OUTDIR/remote_env_snapshot.txt
if ($?TERM) then
  echo TERM=$TERM >> $OUTDIR/remote_env_snapshot.txt
else
  echo TERM= >> $OUTDIR/remote_env_snapshot.txt
endif
if ($?LM_LICENSE_FILE) then
  echo LM_LICENSE_FILE=$LM_LICENSE_FILE >> $OUTDIR/remote_env_snapshot.txt
else
  echo LM_LICENSE_FILE= >> $OUTDIR/remote_env_snapshot.txt
endif
if ($?MGLS_LICENSE_FILE) then
  echo MGLS_LICENSE_FILE=$MGLS_LICENSE_FILE >> $OUTDIR/remote_env_snapshot.txt
else
  echo MGLS_LICENSE_FILE= >> $OUTDIR/remote_env_snapshot.txt
endif
if ($?CDS_LIC_FILE) then
  echo CDS_LIC_FILE=$CDS_LIC_FILE >> $OUTDIR/remote_env_snapshot.txt
else
  echo CDS_LIC_FILE= >> $OUTDIR/remote_env_snapshot.txt
endif

echo "echo ""exit"" | $CATAPULT_BIN -shell |& tee $OUTDIR/license_probe.log" >! $OUTDIR/exact_command_remote.txt
echo "$CATAPULT_BIN -shell -file $PROJECT_TCL -logfile $OUTDIR/catapult_internal.log |& tee $OUTDIR/catapult_console.log" >> $OUTDIR/exact_command_remote.txt

if (! -f $PROJECT_TCL) then
  echo "PROJECT_TCL_MISSING $PROJECT_TCL" |& tee $OUTDIR/catapult_console.log
  set CATAPULT_RC=99
  echo $CATAPULT_RC >! $OUTDIR/catapult_exit_code.txt
  echo 99 >! $OUTDIR/license_probe_exit_code.txt
  echo NOT_FOUND >! $OUTDIR/messages_path.txt
  echo "# Error=UNKNOWN" >! $OUTDIR/grep_summary.txt
  echo "Compilation aborted=UNKNOWN" >> $OUTDIR/grep_summary.txt
  echo "Completed transformation=UNKNOWN" >> $OUTDIR/grep_summary.txt
  echo "project.tcl missing before catapult invocation" >! $OUTDIR/grep_messages_tail.txt
  exit $CATAPULT_RC
endif

echo "exit" | $CATAPULT_BIN -shell |& tee $OUTDIR/license_probe.log
set LICENSE_PROBE_RC=$status
$CATAPULT_BIN -shell -file $PROJECT_TCL -logfile $OUTDIR/catapult_internal.log |& tee $OUTDIR/catapult_console.log
set CATAPULT_RC=$status
echo $CATAPULT_RC >! $OUTDIR/catapult_exit_code.txt
echo $LICENSE_PROBE_RC >! $OUTDIR/license_probe_exit_code.txt

set MSG=""
if (-f $REPO_ROOT/Catapult_15/SIF/messages.txt) then
  set MSG=$REPO_ROOT/Catapult_15/SIF/messages.txt
else if (-f $OUTDIR/project/messages.txt) then
  set MSG=$OUTDIR/project/messages.txt
endif
if ("$MSG" != "") then
  echo $MSG >! $OUTDIR/messages_path.txt
  cp $MSG $OUTDIR/messages_latest.txt
  set ERROR_COUNT=`grep -c "# Error" $MSG`
  set ABORT_COUNT=`grep -c "Compilation aborted" $MSG`
  set COMPILE_DONE_COUNT=`grep -c "Completed transformation" $MSG`
  echo "# Error=$ERROR_COUNT" >! $OUTDIR/grep_summary.txt
  echo "Compilation aborted=$ABORT_COUNT" >> $OUTDIR/grep_summary.txt
  echo "Completed transformation=$COMPILE_DONE_COUNT" >> $OUTDIR/grep_summary.txt
  grep -n "# Error\|Compilation aborted\|Completed transformation\|Warning" $MSG | tail -n 200 >! $OUTDIR/grep_messages_tail.txt
else
  echo NOT_FOUND >! $OUTDIR/messages_path.txt
  echo "# Error=UNKNOWN" >! $OUTDIR/grep_summary.txt
  echo "Compilation aborted=UNKNOWN" >> $OUTDIR/grep_summary.txt
  echo "Completed transformation=UNKNOWN" >> $OUTDIR/grep_summary.txt
  echo "messages.txt not found" >! $OUTDIR/grep_messages_tail.txt
endif

echo REMOTE_OUTDIR=$OUTDIR >! $OUTDIR/remote_manifest.txt
echo PROJECT_TCL=$PROJECT_TCL >> $OUTDIR/remote_manifest.txt
echo TOP_TARGET=aecct_ref::ref_v3::RefV3CatapultTop >> $OUTDIR/remote_manifest.txt
echo ENTRY_TU=AECCT_ac_ref/catapult/ref_v3/ref_v3_catapult_top_entry.cpp >> $OUTDIR/remote_manifest.txt
echo LICENSE_PROBE_RC=$LICENSE_PROBE_RC >> $OUTDIR/remote_manifest.txt
echo CATAPULT_RC=$CATAPULT_RC >> $OUTDIR/remote_manifest.txt
if ("$MSG" != "") then
  echo MESSAGES_PATH=$MSG >> $OUTDIR/remote_manifest.txt
else
  echo MESSAGES_PATH=NOT_FOUND >> $OUTDIR/remote_manifest.txt
endif

exit $CATAPULT_RC
'@
$remoteScript = $remoteScriptTemplate.Replace("__REMOTE_REPO_ROOT__", $RemoteRepoRoot)
$remoteScript = $remoteScript.Replace("__REMOTE_OUTDIR__", $remoteOutDir)
$remoteScript = $remoteScript.Replace("__CATAPULT_BIN__", $CatapultCmd)
$remoteScript = $remoteScript.Replace("__PROJECT_TCL__", $remoteProjectTcl)
$remoteScript = $remoteScript.TrimStart()
$remoteScript | Set-Content -Path $remoteScriptPath -Encoding UTF8

@(
    "local_repo_root=$repoRoot",
    "local_build_dir=$buildAbs",
    "remote_host=$RemoteHost",
    "remote_port=$RemotePort",
    "remote_user=$RemoteUser",
    "identity_file=$identityAbs",
    "known_hosts_file=$knownHostsAbs",
    "remote_repo_root=$RemoteRepoRoot",
    "remote_project_tcl=$remoteProjectTcl",
    "remote_outdir=$remoteOutDir",
    "canonical_catapult_bin=$CatapultCmd",
    "canonical_top_target=aecct_ref::ref_v3::RefV3CatapultTop",
    "entry_tu=AECCT_ac_ref/catapult/ref_v3/ref_v3_catapult_top_entry.cpp"
) | Set-Content -Path $envSnapshotPath -Encoding UTF8

@(
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "cat <<'TCSSCRIPT' | $sshPrefix",
    $remoteScript.TrimEnd(),
    "TCSSCRIPT"
) | Set-Content -Path $exactCmdPath -Encoding UTF8

@(
    "@'",
    $remoteScript.TrimEnd(),
    "'@ | $sshPrefix"
) | Set-Content -Path $sshCmdPath -Encoding UTF8

if ($PrepareOnly) {
    @(
        "REFV3_STATUS PREPARED_ONLY",
        "REFV3_REASON prepare-only requested; remote command not executed",
        "REFV3_REMOTE_TARGET=$RemoteUser@$RemoteHost`:$RemotePort",
        "REFV3_REMOTE_OUTDIR=$remoteOutDir"
    ) | Set-Content -Path $runLog -Encoding UTF8

    @(
        "task: REF_V3_CATAPULT_COMPILE_FIRST_REMOTE",
        "status: PREPARED_ONLY",
        "scope: compile-first remote launch pack",
        "closure: not Catapult closure; not SCVerify closure"
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        "run_log=$runLog",
        "verdict=$verdictPath",
        "env_snapshot=$envSnapshotPath",
        "remote_script=$remoteScriptPath",
        "exact_commands_sh=$exactCmdPath",
        "exact_commands_ps1=$sshCmdPath",
        "remote_outdir=$remoteOutDir"
    ) | Set-Content -Path $artifactManifestPath -Encoding UTF8

    Write-Host "Prepared ref_v3 remote compile-first launch pack: $buildAbs" -ForegroundColor Green
    exit 0
}

$sshArgs = @(
    "-tt",
    "-i", $identityAbs,
    "-o", "UserKnownHostsFile=$knownHostsAbs",
    "-o", "StrictHostKeyChecking=yes",
    "-p", $RemotePort.ToString(),
    "$RemoteUser@$RemoteHost"
)

$prevErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$remoteScript | & ssh @sshArgs 2>&1 | Tee-Object -FilePath $sshConsoleLog
$sshExit = $LASTEXITCODE
$ErrorActionPreference = $prevErrorAction

$scpArgsCommon = @(
    "-i", $identityAbs,
    "-o", "UserKnownHostsFile=$knownHostsAbs",
    "-o", "StrictHostKeyChecking=yes",
    "-P", $RemotePort.ToString()
)

$remoteArtifacts = @(
    "remote_env_snapshot.txt",
    "license_probe.log",
    "license_probe_exit_code.txt",
    "catapult_console.log",
    "catapult_internal.log",
    "catapult_exit_code.txt",
    "messages_path.txt",
    "messages_latest.txt",
    "grep_summary.txt",
    "grep_messages_tail.txt",
    "remote_manifest.txt",
    "exact_command_remote.txt"
)

$copiedArtifacts = New-Object System.Collections.Generic.List[string]
$missingArtifacts = New-Object System.Collections.Generic.List[string]
foreach ($artifact in $remoteArtifacts) {
    if (Copy-RemoteArtifact -ScpArgsCommon $scpArgsCommon `
            -RemoteUser $RemoteUser `
            -RemoteHost $RemoteHost `
            -RemoteOutDir $remoteOutDir `
            -ArtifactName $artifact `
            -LocalDir $buildAbs) {
        $copiedArtifacts.Add($artifact)
    } else {
        $missingArtifacts.Add($artifact)
    }
}

$messagesPath = "NOT_FOUND"
$messagesPathFile = Join-Path $buildAbs "messages_path.txt"
if (Test-Path -LiteralPath $messagesPathFile) {
    $messagesPath = (Get-Content -LiteralPath $messagesPathFile -TotalCount 1).Trim()
    if ($messagesPath -eq "") {
        $messagesPath = "NOT_FOUND"
    }
}

$grepSummary = "NOT_FOUND"
$grepSummaryPath = Join-Path $buildAbs "grep_summary.txt"
if (Test-Path -LiteralPath $grepSummaryPath) {
    $grepSummary = ((Get-Content -LiteralPath $grepSummaryPath) -join "; ")
}

$firstBlocker = "not-detected"
$catapultConsolePath = Join-Path $buildAbs "catapult_console.log"
if (Test-Path -LiteralPath $catapultConsolePath) {
    $catapultHit = Select-String -Path $catapultConsolePath `
        -Pattern "PROJECT_TCL_MISSING|Compilation aborted|cannot open source file|# Error" `
        | Select-Object -First 1
    if ($null -ne $catapultHit) {
        $firstBlocker = $catapultHit.Line.Trim()
    }
}
if ($firstBlocker -eq "not-detected" -and (Test-Path -LiteralPath $sshConsoleLog)) {
    $blockerPatterns = @(
        "PROJECT_TCL_MISSING",
        "Compilation aborted",
        "cannot open source file",
        "unrecognized token",
        "invalid command name",
        "can't open",
        "can't read"
    )
    foreach ($pattern in $blockerPatterns) {
        $hit = Select-String -Path $sshConsoleLog -Pattern $pattern | Select-Object -First 1
        if ($null -ne $hit) {
            $firstBlocker = $hit.Line.Trim()
            break
        }
    }
}

$statusToken = if ($sshExit -eq 0) { "EXECUTED" } else { "FAILED" }
@(
    "REFV3_STATUS $statusToken",
    "REFV3_REMOTE_TARGET=$RemoteUser@$RemoteHost`:$RemotePort",
    "REFV3_REMOTE_OUTDIR=$remoteOutDir",
    "REFV3_SSH_EXIT=$sshExit",
    "REFV3_MESSAGES_PATH=$messagesPath",
    "REFV3_GREP_SUMMARY=$grepSummary",
    "REFV3_FIRST_BLOCKER=$firstBlocker",
    ("REFV3_COPIED_ARTIFACTS={0}" -f ($copiedArtifacts -join ",")),
    ("REFV3_MISSING_ARTIFACTS={0}" -f ($missingArtifacts -join ","))
) | Set-Content -Path $runLog -Encoding UTF8

$verdictStatus = if ($sshExit -eq 0) { "EXECUTED" } else { "FAIL" }
@(
    "task: REF_V3_CATAPULT_COMPILE_FIRST_REMOTE",
    "status: $verdictStatus",
    "ssh_exit: $sshExit",
    "messages_path: $messagesPath",
    "first_blocker: $firstBlocker",
    "scope: compile-first remote launch execution",
    "posture: not Catapult closure; not SCVerify closure"
) | Set-Content -Path $verdictPath -Encoding UTF8

@(
    "run_log=$runLog",
    "verdict=$verdictPath",
    "env_snapshot=$envSnapshotPath",
    "remote_script=$remoteScriptPath",
    "ssh_console_log=$sshConsoleLog",
    "exact_commands_sh=$exactCmdPath",
    "exact_commands_ps1=$sshCmdPath",
    "messages_path_local=$messagesPathFile",
    "grep_summary_local=$grepSummaryPath",
    "remote_outdir=$remoteOutDir",
    ("copied_artifacts={0}" -f ($copiedArtifacts -join ",")),
    ("missing_artifacts={0}" -f ($missingArtifacts -join ","))
) | Set-Content -Path $artifactManifestPath -Encoding UTF8

if ($sshExit -eq 0) {
    Write-Host "EXECUTED: ref_v3 remote compile-first launch" -ForegroundColor Green
    Write-Host "Artifacts: $buildAbs" -ForegroundColor Green
    exit 0
}

Write-Error "ref_v3 remote compile-first launch failed (ssh_exit=$sshExit)"
exit $sshExit
