param(
    [string]$BuildDir = "",
    [string]$RemoteHost = "140.124.41.193",
    [int]$RemotePort = 2190,
    [string]$RemoteUser = "peter",
    [string]$IdentityFile = ".codex_ssh/id_codex",
    [string]$KnownHostsFile = ".codex_ssh/known_hosts",
    [string]$RemoteRepoRoot = "/home/peter/AECCT/AECCT_HLS-master",
    [string]$CatapultCmd = "/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path (Join-Path $PSScriptRoot "..\..") "..")).Path
if ($BuildDir -eq "") {
    $BuildDir = "build\ref_v3\catapult_compile_check_tty_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$buildAbs = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
$remoteRunLeaf = Split-Path $BuildDir -Leaf
$remoteProjectTcl = "$RemoteRepoRoot/scripts/catapult/ref_v3/project.tcl"
$remoteOutDir = "$RemoteRepoRoot/build/ref_v3/$remoteRunLeaf"
$runLog = Join-Path $buildAbs "run.log"
$envSnapshotPath = Join-Path $buildAbs "env_snapshot.txt"
$exactCmdPath = Join-Path $buildAbs "exact_commands.sh"
$sshCmdPath = Join-Path $buildAbs "exact_commands_ssh.ps1"
$verdictPath = Join-Path $buildAbs "verdict.txt"

New-Item -ItemType Directory -Force -Path $buildAbs > $null

@(
    "local_repo_root=$repoRoot",
    "remote_host=$RemoteHost",
    "remote_port=$RemotePort",
    "remote_user=$RemoteUser",
    "identity_file=$IdentityFile",
    "known_hosts_file=$KnownHostsFile",
    "remote_repo_root=$RemoteRepoRoot",
    "canonical_catapult_bin=$CatapultCmd",
    "validated_mode=ssh -tt interactive login shell",
    "remote_shell_hint=/bin/tcsh",
    "validated_term=xterm-256color",
    "validated_license_env=LM_LICENSE_FILE present",
    "expected_messages_path=$RemoteRepoRoot/Catapult_15/SIF/messages.txt"
) | Set-Content -Path $envSnapshotPath -Encoding UTF8

$remoteScriptLines = @(
    "set REPO_ROOT=$RemoteRepoRoot",
    'set TS=`date +%Y%m%d_%H%M%S`',
    "set OUTDIR=$remoteOutDir",
    "set CATAPULT_BIN=$CatapultCmd",
    "set PROJECT_TCL=$remoteProjectTcl",
    '',
    'mkdir -p $OUTDIR/project',
    'cd $REPO_ROOT',
    '',
    'echo MODE=interactive_tty_login',
    'whoami',
    'hostname',
    'echo SHELL=$SHELL',
    'printenv TERM || true',
    'printenv LM_LICENSE_FILE || true',
    'printenv MGLS_LICENSE_FILE || true',
    'printenv CDS_LIC_FILE || true',
    'command -v catapult || true',
    "$CatapultCmd -version | head -n 3",
    '',
    'setenv AECCT_REFV3_REPO_ROOT $REPO_ROOT',
    'setenv AECCT_REFV3_CATAPULT_OUTDIR $OUTDIR/project',
    '',
    'echo "exit" | $CATAPULT_BIN -shell |& tee $OUTDIR/license_probe.log',
    '$CATAPULT_BIN -shell -file $PROJECT_TCL -logfile $OUTDIR/catapult_internal.log |& tee $OUTDIR/catapult_console.log',
    'echo $status >! $OUTDIR/catapult_exit_code.txt',
    'echo MESSAGES_PATH=/home/peter/AECCT/AECCT_HLS-master/Catapult_15/SIF/messages.txt',
    'exit'
)
$remoteScript = [string]::Join("`n", $remoteScriptLines)
$sshPrefix = "ssh -tt -i $IdentityFile -o UserKnownHostsFile=$KnownHostsFile -o StrictHostKeyChecking=yes -p $RemotePort $RemoteUser@$RemoteHost"

@(
    '#!/usr/bin/env bash',
    'set -euo pipefail',
    "cat <<'TCSSCRIPT' | $sshPrefix",
    $remoteScript,
    'TCSSCRIPT'
) | Set-Content -Path $exactCmdPath -Encoding UTF8

@(
    "@'",
    $remoteScript,
    "'@ | $sshPrefix"
) | Set-Content -Path $sshCmdPath -Encoding UTF8

@(
    'REFV3_STATUS READY_FOR_INTERACTIVE_LOGIN_RETRY',
    'REFV3_REASON validated ssh -tt + tcsh login-shell recipe is now the canonical launch path',
    "REFV3_REMOTE_TARGET=$RemoteUser@$RemoteHost:$RemotePort",
    'REFV3_EXPECTED_NEXT_BLOCKER_AFTER_LICENSE=source compile if go compile is reached'
) | Set-Content -Path $runLog -Encoding UTF8

@(
    'task: REF_V3_CATAPULT_INTERACTIVE_COMPILE_CHECK',
    'status: PREPARED_ONLY',
    'LICENSE_ENV_MODE_MISMATCH=defer_to_true_remote_shell',
    'EXECUTION_DEFERRED_TTY_REQUIRED=false',
    'LICENSE_SHELL_OK=unknown_until_remote_run',
    'GO_COMPILE_REACHED=unknown_until_remote_run',
    'FIRST_TRUE_BLOCKER=unknown_until_remote_run',
    'posture: not Catapult closure; not SCVerify closure; launch-pack refresh only'
) | Set-Content -Path $verdictPath -Encoding UTF8

Write-Host 'Prepared ref_v3 interactive-login launch pack:' -ForegroundColor Green
Write-Host "  $buildAbs" -ForegroundColor Green
Write-Host 'Use exact_commands_ssh.ps1 or exact_commands.sh on a machine that can reach the remote Catapult host.' -ForegroundColor Yellow
