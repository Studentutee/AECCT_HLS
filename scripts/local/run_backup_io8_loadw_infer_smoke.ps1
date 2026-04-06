param(
    [string]$BuildDir = "build\backup_io8_loadw_infer"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-ClBuild {
    param(
        [string]$Source,
        [string]$ExeOut,
        [string]$LogOut
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
        '/Idata\weights',
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $exePath = Join-Path $BuildDir 'tb_backup_io8_loadw_infer_smoke.exe'
    $buildLog = Join-Path $BuildDir 'build_backup_io8_loadw_infer.log'
    $runLog = Join-Path $BuildDir 'run_backup_io8_loadw_infer.log'

    Invoke-ClBuild -Source 'tb\tb_backup_io8_loadw_infer_smoke.cpp' -ExeOut $exePath -LogOut $buildLog
    Invoke-ExeRun -ExePath $exePath -LogOut $runLog

    Require-PassString -LogPath $runLog -Needle 'PASS: tb_backup_io8_loadw_infer_smoke'

    Write-Host 'PASS: run_backup_io8_loadw_infer_smoke'
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
