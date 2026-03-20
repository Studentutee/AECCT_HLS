param(
    [string]$BuildDir = "build\p11aeaf_catapult_progress"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$RunLog,
        [string]$Command,
        [string[]]$StepArgs
    )

    Add-Content -Path $RunLog -Value ("CMD: {0} {1}" -f $Command, ($StepArgs -join " ")) -Encoding UTF8
    $output = & $Command @StepArgs 2>&1
    $exitCode = $LASTEXITCODE
    foreach ($line in @($output)) {
        $text = [string]$line
        Write-Host $text
        Add-Content -Path $RunLog -Value $text -Encoding UTF8
    }
    if ($exitCode -ne 0) {
        throw "step failed ($Command), exit=$exitCode"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $runLog = Join-Path $BuildDir "run.log"
    if (Test-Path $runLog) {
        Remove-Item $runLog -Force
    }
    New-Item -ItemType File -Path $runLog -Force > $null

    $p11rDir = Join-Path $BuildDir "p11r"

    Invoke-Step -RunLog $runLog -Command "powershell" -StepArgs @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts/check_p11aeaf_catapult_progress_surface.ps1",
        "-OutDir", $BuildDir,
        "-Phase", "pre"
    )
    Invoke-Step -RunLog $runLog -Command "powershell" -StepArgs @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11r_compile_prep.ps1",
        "-BuildDir", $p11rDir
    )
    Invoke-Step -RunLog $runLog -Command "powershell" -StepArgs @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts/check_p11aeaf_catapult_progress_surface.ps1",
        "-OutDir", $BuildDir,
        "-Phase", "post"
    )

    Add-Content -Path $runLog -Value "PASS: run_p11aeaf_catapult_progress" -Encoding UTF8
    Write-Host "PASS: run_p11aeaf_catapult_progress"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
