param(
    [string]$BuildDir = "",
    [string]$CatapultCmd = "/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult",
    [switch]$AllowPathCatapult
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-CatapultCommand {
    param(
        [string]$PreferredCmd,
        [bool]$AllowPath
    )

    if ($PreferredCmd -ne "") {
        if (Test-Path $PreferredCmd) {
            return (Resolve-Path $PreferredCmd).Path
        }
        $explicit = Get-Command $PreferredCmd -ErrorAction SilentlyContinue
        if ($explicit) {
            return $explicit.Source
        }
    }

    if ($AllowPath) {
        $fromPath = Get-Command "catapult" -ErrorAction SilentlyContinue
        if ($fromPath) {
            return $fromPath.Source
        }
    }

    return $null
}

function Try-Run {
    param([scriptblock]$Script)
    try {
        return (& $Script)
    }
    catch {
        return ""
    }
}

function Get-LatestMessagesFile {
    param([string[]]$SearchRoots)

    $candidates = @()
    foreach ($root in $SearchRoots) {
        if (-not $root) { continue }
        if (-not (Test-Path $root)) { continue }
        $hit = Get-ChildItem -Path $root -Filter "messages.txt" -File -Recurse -ErrorAction SilentlyContinue
        if ($hit) {
            $candidates += $hit
        }
    }
    if (-not $candidates) { return $null }
    return ($candidates | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).FullName
}

$repoRoot = (Resolve-Path (Join-Path (Join-Path $PSScriptRoot "..\..") "..")).Path
if ($BuildDir -eq "") {
    $BuildDir = "build\ref_v3\catapult_interactive_compile_check_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}
$buildAbs = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))

$projectTcl = Join-Path $repoRoot "scripts/catapult/ref_v3/project.tcl"
$filelistPath = Join-Path $repoRoot "scripts/catapult/ref_v3/filelist.f"
$projectOutdir = Join-Path $buildAbs "project"
$runLog = Join-Path $buildAbs "run.log"
$envSnapshotPath = Join-Path $buildAbs "env_snapshot.txt"
$exactCmdPath = Join-Path $buildAbs "exact_commands.sh"
$licenseProbeConsole = Join-Path $buildAbs "license_probe_console.log"
$licenseProbeInternal = Join-Path $buildAbs "license_probe_internal.log"
$catapultConsole = Join-Path $buildAbs "catapult_console.log"
$catapultInternal = Join-Path $buildAbs "catapult_internal.log"
$messagesPathFile = Join-Path $buildAbs "messages_path.txt"
$messagesCopy = Join-Path $buildAbs "messages_latest.txt"
$grepSummary = Join-Path $buildAbs "grep_summary.txt"
$verdictPath = Join-Path $buildAbs "verdict.txt"
$probeTcl = Join-Path $buildAbs "license_probe.tcl"

Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $buildAbs > $null
    if (-not (Test-Path $projectTcl)) {
        throw "project Tcl missing: $projectTcl"
    }
    if (-not (Test-Path $filelistPath)) {
        throw "filelist missing: $filelistPath"
    }

    $shellValue = if ($env:SHELL) { $env:SHELL } else { "" }
    $termValue = if ($env:TERM) { $env:TERM } else { "" }
    $lmValue = if ($env:LM_LICENSE_FILE) { $env:LM_LICENSE_FILE } else { "" }
    $mglsValue = if ($env:MGLS_LICENSE_FILE) { $env:MGLS_LICENSE_FILE } else { "" }
    $cdsValue = if ($env:CDS_LIC_FILE) { $env:CDS_LIC_FILE } else { "" }
    $pathCatapult = Try-Run { (Get-Command "catapult" -ErrorAction Stop).Source }
    $unameValue = Try-Run { (& uname -a 2>&1 | Out-String).Trim() }
    if (-not $unameValue) { $unameValue = "<not-available>" }

    $resolvedCatapult = Resolve-CatapultCommand -PreferredCmd $CatapultCmd -AllowPath:$AllowPathCatapult
    $versionText = "<not-executed>"
    if ($resolvedCatapult) {
        $versionText = Try-Run { (& $resolvedCatapult -version 2>&1 | Out-String).Trim() }
        if (-not $versionText) {
            $versionText = "<version-command-failed>"
        }
    }

    $lsCanonical = if ($resolvedCatapult -and (Test-Path $resolvedCatapult)) {
        Try-Run { (Get-Item $resolvedCatapult | Format-List FullName,Length,LastWriteTime | Out-String).Trim() }
    } else {
        "<missing>"
    }

    @(
        ("hostname={0}" -f (Try-Run { hostname })),
        ("whoami={0}" -f (Try-Run { whoami })),
        ("pwd={0}" -f (Get-Location).Path),
        ("date={0}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")),
        ("uname -a={0}" -f $unameValue),
        ("SHELL={0}" -f $shellValue),
        ("TERM={0}" -f $termValue),
        ("LM_LICENSE_FILE={0}" -f $lmValue),
        ("MGLS_LICENSE_FILE={0}" -f $mglsValue),
        ("CDS_LIC_FILE={0}" -f $cdsValue),
        ("command -v catapult={0}" -f $(if ($pathCatapult) { $pathCatapult } else { "" })),
        ("canonical_bin={0}" -f $CatapultCmd),
        ("ls -l canonical={0}" -f $lsCanonical),
        ("catapult_version={0}" -f $versionText)
    ) | Set-Content -Path $envSnapshotPath -Encoding UTF8

    @(
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export AECCT_REFV3_REPO_ROOT=<repo_root>",
        "export AECCT_REFV3_CATAPULT_OUTDIR=<outdir>/project",
        "",
        "# License probe in interactive login shell",
        "/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult \",
        "  -shell \",
        "  -logfile <outdir>/license_probe_internal.log <<'EOF' 2>&1 | tee <outdir>/license_probe_console.log",
        "application report -license true",
        "quit",
        "EOF",
        "",
        "# Compile-entry check",
        "/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult \",
        "  -shell \",
        "  -file <repo_root>/scripts/catapult/ref_v3/project.tcl \",
        "  -logfile <outdir>/catapult_internal.log 2>&1 | tee <outdir>/catapult_console.log"
    ) | Set-Content -Path $exactCmdPath -Encoding UTF8

    $isLinuxShell = $false
    if ($PSVersionTable.ContainsKey("Platform")) {
        $isLinuxShell = ($PSVersionTable.Platform -eq "Unix")
    }
    elseif ($env:OS) {
        $isLinuxShell = ($env:OS -notmatch "Windows")
    }
    $looksInteractive = ($termValue -ne "" -and $termValue -ne "dumb")
    $hasLicenseEnv = ($lmValue -ne "" -or $mglsValue -ne "" -or $cdsValue -ne "")
    $interactiveLoginReady = $isLinuxShell -and $looksInteractive -and $hasLicenseEnv
    $licenseEnvModeMismatch = -not $interactiveLoginReady

    $licenseShellOk = $false
    $goCompileReached = $false
    $firstTrueBlocker = "TTY_REQUIRED_INTERACTIVE_LOGIN_UNAVAILABLE"

    if (-not $interactiveLoginReady) {
        @(
            "DEFERRED: interactive login shell not available in current execution environment.",
            "No license probe executed."
        ) | Set-Content -Path $licenseProbeConsole -Encoding UTF8
        @(
            "DEFERRED: interactive login shell not available in current execution environment.",
            "No internal probe log generated."
        ) | Set-Content -Path $licenseProbeInternal -Encoding UTF8
        @(
            "DEFERRED: interactive login shell not available in current execution environment.",
            "No Catapult project run executed."
        ) | Set-Content -Path $catapultConsole -Encoding UTF8
        @(
            "DEFERRED: interactive login shell not available in current execution environment.",
            "No Catapult internal log generated."
        ) | Set-Content -Path $catapultInternal -Encoding UTF8
        "NOT_FOUND_DEFERRED" | Set-Content -Path $messagesPathFile -Encoding UTF8
        @(
            "go compile=0",
            "Error=0",
            "Compilation aborted=0",
            "HIER-=0",
            "CIN-=0",
            "LIB-=0",
            "SCVerify=0",
            "mc_scverify=0",
            "license=0",
            "LIC-=0",
            "note=deferred_before_interactive_login_shell"
        ) | Set-Content -Path $grepSummary -Encoding UTF8

        @(
            "REFV3_STATUS EXECUTION_DEFERRED",
            "REFV3_REASON interactive login shell evidence not satisfied",
            ("REFV3_IS_LINUX_SHELL {0}" -f $isLinuxShell),
            ("REFV3_TERM {0}" -f $termValue),
            ("REFV3_LICENSE_ENV_PRESENT {0}" -f $hasLicenseEnv)
        ) | Set-Content -Path $runLog -Encoding UTF8
    }
    else {
        if (-not $resolvedCatapult) {
            $firstTrueBlocker = "CANONICAL_CATAPULT_BINARY_NOT_FOUND"
            @(
                "REFV3_STATUS BLOCKED",
                "REFV3_REASON canonical catapult binary not found",
                ("REFV3_CANONICAL_BIN {0}" -f $CatapultCmd)
            ) | Set-Content -Path $runLog -Encoding UTF8
        }
        else {
            New-Item -ItemType Directory -Force -Path $projectOutdir > $null
            $env:AECCT_REFV3_REPO_ROOT = $repoRoot
            $env:AECCT_REFV3_CATAPULT_OUTDIR = $projectOutdir

            @(
                "application report -license true",
                "quit"
            ) | Set-Content -Path $probeTcl -Encoding UTF8

            & $resolvedCatapult -shell -file $probeTcl -logfile $licenseProbeInternal 2>&1 | Tee-Object -FilePath $licenseProbeConsole
            $probeExit = $LASTEXITCODE

            $probeText = @(
                Try-Run { Get-Content -Raw $licenseProbeConsole },
                Try-Run { Get-Content -Raw $licenseProbeInternal }
            ) -join "`n"
            $licenseShellOk = ($probeExit -eq 0) -and (
                $probeText -match "Connected to license server" -or
                $probeText -match "LIC-13" -or
                $probeText -match "LIC-14"
            )

            if (-not $licenseShellOk) {
                $firstTrueBlocker = "LICENSE_CHECKOUT_FAIL_OR_NOT_VISIBLE"
            }
            else {
                & $resolvedCatapult -shell -file $projectTcl -logfile $catapultInternal 2>&1 | Tee-Object -FilePath $catapultConsole
                $compileExit = $LASTEXITCODE
                $compileText = @(
                    Try-Run { Get-Content -Raw $catapultConsole },
                    Try-Run { Get-Content -Raw $catapultInternal }
                ) -join "`n"
                $goCompileReached = $compileText -match "REFV3_STAGE compile START" -or
                                    $compileText -match "REFV3_STAGE compile DONE" -or
                                    $compileText -match "go compile"
                if ($compileExit -ne 0 -and -not $goCompileReached) {
                    $firstTrueBlocker = "PROJECT_TCL_OR_COMPILE_SETUP_BLOCKER"
                }
                elseif ($goCompileReached) {
                    $firstTrueBlocker = "NONE_GO_COMPILE_REACHED"
                }
                else {
                    $firstTrueBlocker = "GO_COMPILE_NOT_REACHED"
                }
            }

            $latestMessages = Get-LatestMessagesFile -SearchRoots @($projectOutdir, (Join-Path $repoRoot "build"), $repoRoot)
            if ($latestMessages) {
                $latestMessages | Set-Content -Path $messagesPathFile -Encoding UTF8
                Copy-Item -Path $latestMessages -Destination $messagesCopy -Force
            }
            else {
                "NOT_FOUND" | Set-Content -Path $messagesPathFile -Encoding UTF8
            }

            $patterns = @("go compile", "Error", "Compilation aborted", "HIER-", "CIN-", "LIB-", "SCVerify", "mc_scverify", "license", "LIC-")
            $summary = @()
            foreach ($pattern in $patterns) {
                $count = 0
                if (Test-Path $catapultConsole) {
                    $count += (Select-String -Path $catapultConsole -SimpleMatch -Pattern $pattern -ErrorAction SilentlyContinue | Measure-Object).Count
                }
                if (Test-Path $catapultInternal) {
                    $count += (Select-String -Path $catapultInternal -SimpleMatch -Pattern $pattern -ErrorAction SilentlyContinue | Measure-Object).Count
                }
                if (Test-Path $messagesCopy) {
                    $count += (Select-String -Path $messagesCopy -SimpleMatch -Pattern $pattern -ErrorAction SilentlyContinue | Measure-Object).Count
                }
                $summary += ("{0}={1}" -f $pattern, $count)
            }
            $summary | Set-Content -Path $grepSummary -Encoding UTF8

            @(
                "REFV3_STATUS EXECUTED",
                ("REFV3_CATAPULT_CMD {0}" -f $resolvedCatapult),
                ("REFV3_LICENSE_SHELL_OK {0}" -f $licenseShellOk),
                ("REFV3_GO_COMPILE_REACHED {0}" -f $goCompileReached),
                ("REFV3_FIRST_TRUE_BLOCKER {0}" -f $firstTrueBlocker)
            ) | Set-Content -Path $runLog -Encoding UTF8
        }
    }

    @(
        "task: REF_V3_CATAPULT_INTERACTIVE_COMPILE_CHECK",
        ("LICENSE_ENV_MODE_MISMATCH={0}" -f $licenseEnvModeMismatch),
        ("EXECUTION_DEFERRED_TTY_REQUIRED={0}" -f (-not $interactiveLoginReady)),
        ("LICENSE_SHELL_OK={0}" -f $licenseShellOk),
        ("GO_COMPILE_REACHED={0}" -f $goCompileReached),
        ("FIRST_TRUE_BLOCKER={0}" -f $firstTrueBlocker),
        "posture: not Catapult closure; not SCVerify closure; compile-entry check only"
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    if (-not $interactiveLoginReady) {
        exit 2
    }
    exit 0
}
catch {
    Write-Error $_
    @(
        "task: REF_V3_CATAPULT_INTERACTIVE_COMPILE_CHECK",
        "status: FAIL",
        ("message: {0}" -f $_.Exception.Message)
    ) | Set-Content -Path $verdictPath -Encoding UTF8
    exit 1
}
finally {
    Pop-Location
}
