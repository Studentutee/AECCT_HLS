param(
    [string]$BuildDir = "build\p11aeaf_catapult_progress"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Command,
        [string[]]$StepArgs
    )
    & $Command @StepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "step failed ($Command), exit=$LASTEXITCODE"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $runLog = Join-Path $BuildDir "run.log"
    $p11rDir = Join-Path $BuildDir "p11r"

    Invoke-Step -Command "powershell" -StepArgs @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts/check_p11aeaf_catapult_progress_surface.ps1",
        "-OutDir", $BuildDir,
        "-Phase", "pre"
    )
    Invoke-Step -Command "powershell" -StepArgs @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts/local/run_p11r_compile_prep.ps1",
        "-BuildDir", $p11rDir
    )
    Invoke-Step -Command "powershell" -StepArgs @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts/check_p11aeaf_catapult_progress_surface.ps1",
        "-OutDir", $BuildDir,
        "-Phase", "post"
    )

    @(
        "CMD: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11aeaf_catapult_progress_surface.ps1 -OutDir $BuildDir -Phase pre",
        "PASS: check_p11aeaf_catapult_progress_surface",
        "CMD: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11r_compile_prep.ps1 -BuildDir $p11rDir",
        "PASS: run_p11r_compile_prep",
        "CMD: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11aeaf_catapult_progress_surface.ps1 -OutDir $BuildDir -Phase post",
        "PASS: check_p11aeaf_catapult_progress_surface",
        "PASS: run_p11aeaf_catapult_progress"
    ) | Set-Content -Path $runLog -Encoding UTF8

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
