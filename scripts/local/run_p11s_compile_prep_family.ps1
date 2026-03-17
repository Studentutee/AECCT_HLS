param(
    [string]$BuildDir = "build\p11s"
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

    $exePath = Join-Path $BuildDir 'tb_ternary_live_leaf_top_compile_prep_family_p11s.exe'
    $buildLog = Join-Path $BuildDir 'build_p11s_compile_prep_family.log'
    $runLog = Join-Path $BuildDir 'run_p11s_compile_prep_family.log'

    Invoke-ClBuild -Source 'tb\tb_ternary_live_leaf_top_compile_prep_family_p11s.cpp' -ExeOut $exePath -LogOut $buildLog
    Invoke-ExeRun -ExePath $exePath -LogOut $runLog

    Require-PassString -LogPath $runLog -Needle 'PASS: tb_ternary_live_leaf_top_compile_prep_family_p11s'

    Add-Content -Path $runLog -Value 'PASS: run_p11s_compile_prep_family' -Encoding UTF8
    Write-Host 'PASS: run_p11s_compile_prep_family'
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
