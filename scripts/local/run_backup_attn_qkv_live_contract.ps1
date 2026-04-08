param(
    [string]$BuildDir = "build\backup_attn_qkv_live_contract"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-ClBuild {
    param(
        [string[]]$Sources,
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
        '/IAECCT_ac_ref\include',
        '/IAECCT_ac_ref\src'
    )
    $args += $Sources
    $args += @("/Fe:$ExeOut")

    & cl @args *> $LogOut
    if ($LASTEXITCODE -ne 0) {
        throw "build failed ($($Sources -join ', ')), exit=$LASTEXITCODE"
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

function Require-String {
    param(
        [string]$LogPath,
        [string]$Needle
    )

    if (-not (Select-String -Path $LogPath -SimpleMatch -Quiet $Needle)) {
        throw "required string missing in $LogPath : $Needle"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $exePath = Join-Path $BuildDir 'tb_backup_attn_qkv_live_contract.exe'
    $buildLog = Join-Path $BuildDir 'build_backup_attn_qkv_live_contract.log'
    $runLog = Join-Path $BuildDir 'run_backup_attn_qkv_live_contract.log'

    Invoke-ClBuild -Sources @(
        'tb\tb_backup_attn_qkv_live_contract.cpp',
        'AECCT_ac_ref\src\RefModel.cpp'
    ) -ExeOut $exePath -LogOut $buildLog

    Invoke-ExeRun -ExePath $exePath -LogOut $runLog

    Require-String -LogPath $runLog -Needle 'Q_exact=1'
    Require-String -LogPath $runLog -Needle 'K_exact=1'
    Require-String -LogPath $runLog -Needle 'V_exact=1'
    Require-String -LogPath $runLog -Needle 'Q_equals_input_exact=0'
    Require-String -LogPath $runLog -Needle 'K_equals_input_exact=0'
    Require-String -LogPath $runLog -Needle 'V_equals_input_exact=0'
    Require-String -LogPath $runLog -Needle 'MASKED_SCORE_SEMANTICS_OK=1'
    Require-String -LogPath $runLog -Needle 'MASKED_PROB_NONZERO=0'
    Require-String -LogPath $runLog -Needle 'PASS: tb_backup_attn_qkv_live_contract'

    Write-Host 'PASS: run_backup_attn_qkv_live_contract'
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
