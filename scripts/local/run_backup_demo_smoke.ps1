param(
    [string]$BuildDir = "build\backup_demo"
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
        '/std:c++20',
        '/Zc:gotoScope-',
        '/EHsc',
        '/utf-8',
        '/I.',
        '/Iinclude',
        '/Isrc',
        '/Igen\include',
        '/Ithird_party\ac_types',
        '/Idata\weights',
        '/Idata\trace',
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

    $cases = @(
        @{
            source = 'tb\tb_backup_wave1_memory_packing_smoke.cpp'
            exe = 'tb_backup_wave1_memory_packing_smoke.exe'
            pass = 'PASS: tb_backup_wave1_memory_packing_smoke'
            key = 'wave1'
        },
        @{
            source = 'tb\tb_backup_wave2_quant_linear_smoke.cpp'
            exe = 'tb_backup_wave2_quant_linear_smoke.exe'
            pass = 'PASS: tb_backup_wave2_quant_linear_smoke'
            key = 'wave2'
        },
        @{
            source = 'tb\tb_backup_wave3_io8_boundary_smoke.cpp'
            exe = 'tb_backup_wave3_io8_boundary_smoke.exe'
            pass = 'PASS: tb_backup_wave3_io8_boundary_smoke'
            key = 'wave3'
        },
        @{
            source = 'tb\tb_backup_io8_loadw_infer_smoke.cpp'
            exe = 'tb_backup_io8_loadw_infer_smoke.exe'
            pass = 'PASS: tb_backup_io8_loadw_infer_smoke'
            key = 'io8_loadw_infer'
        }
    )

    foreach ($case in $cases) {
        $exePath = Join-Path $BuildDir $case.exe
        $buildLog = Join-Path $BuildDir ("build_{0}.log" -f $case.key)
        $runLog = Join-Path $BuildDir ("run_{0}.log" -f $case.key)

        Invoke-ClBuild -Source $case.source -ExeOut $exePath -LogOut $buildLog
        Invoke-ExeRun -ExePath $exePath -LogOut $runLog
        Require-PassString -LogPath $runLog -Needle $case.pass

        if ($case.key -eq 'io8_loadw_infer') {
            Require-PassString -LogPath $runLog -Needle 'PASS: tb_backup_io8_loadw_infer_trace_aligned_xpred_compare'
        }
    }

    $summaryLog = Join-Path $BuildDir 'run_backup_demo_summary.log'
    @(
        'PASS: backup_demo_wave1_packing',
        'PASS: backup_demo_wave2_quant_linear',
        'PASS: backup_demo_wave3_io8_boundary',
        'PASS: backup_demo_io8_loadw_infer',
        'PASS: backup_demo_trace_aligned_xpred_compare',
        'PASS: run_backup_demo_smoke'
    ) | Set-Content -Path $summaryLog -Encoding UTF8

    Write-Host 'PASS: run_backup_demo_smoke'
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
