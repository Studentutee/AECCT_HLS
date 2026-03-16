param(
    [string]$BuildDir = "build\p11m"
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

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $exeP11j = Join-Path $BuildDir 'tb_ternary_live_leaf_smoke_p11j.exe'
    $exeP11k = Join-Path $BuildDir 'tb_ternary_live_leaf_top_smoke_p11k.exe'
    $exeP11lb = Join-Path $BuildDir 'tb_ternary_live_leaf_top_smoke_p11l_b.exe'
    $exeP11lc = Join-Path $BuildDir 'tb_ternary_live_leaf_top_smoke_p11l_c.exe'
    $exeP11mBaseline = Join-Path $BuildDir 'tb_ternary_live_source_integration_smoke_p11m_baseline.exe'
    $exeP11mMacro = Join-Path $BuildDir 'tb_ternary_live_source_integration_smoke_p11m_macro.exe'

    $logBuildP11j = Join-Path $BuildDir 'build_p11j.log'
    $logBuildP11k = Join-Path $BuildDir 'build_p11k.log'
    $logBuildP11lb = Join-Path $BuildDir 'build_p11l_b.log'
    $logBuildP11lc = Join-Path $BuildDir 'build_p11l_c.log'
    $logBuildP11mBaseline = Join-Path $BuildDir 'build_p11m_baseline.log'
    $logBuildP11mMacro = Join-Path $BuildDir 'build_p11m_macro.log'

    $logRunP11j = Join-Path $BuildDir 'run_p11j.log'
    $logRunP11k = Join-Path $BuildDir 'run_p11k.log'
    $logRunP11lb = Join-Path $BuildDir 'run_p11l_b.log'
    $logRunP11lc = Join-Path $BuildDir 'run_p11l_c.log'
    $logRunP11mBaseline = Join-Path $BuildDir 'run_p11m_baseline.log'
    $logRunP11mMacro = Join-Path $BuildDir 'run_p11m_macro.log'

    Invoke-ClBuild 'tb\tb_ternary_live_leaf_smoke_p11j.cpp' $exeP11j $logBuildP11j
    Invoke-ClBuild 'tb\tb_ternary_live_leaf_top_smoke_p11k.cpp' $exeP11k $logBuildP11k
    Invoke-ClBuild 'tb\tb_ternary_live_leaf_top_smoke_p11l_b.cpp' $exeP11lb $logBuildP11lb
    Invoke-ClBuild 'tb\tb_ternary_live_leaf_top_smoke_p11l_c.cpp' $exeP11lc $logBuildP11lc
    Invoke-ClBuild 'tb\tb_ternary_live_source_integration_smoke_p11m.cpp' $exeP11mBaseline $logBuildP11mBaseline
    Invoke-ClBuild 'tb\tb_ternary_live_source_integration_smoke_p11m.cpp' $exeP11mMacro $logBuildP11mMacro @('/DAECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE=1')

    Invoke-ExeRun $exeP11j $logRunP11j
    Invoke-ExeRun $exeP11k $logRunP11k
    Invoke-ExeRun $exeP11lb $logRunP11lb
    Invoke-ExeRun $exeP11lc $logRunP11lc
    Invoke-ExeRun $exeP11mBaseline $logRunP11mBaseline
    Invoke-ExeRun $exeP11mMacro $logRunP11mMacro

    Require-PassString $logRunP11j 'PASS: tb_ternary_live_leaf_smoke_p11j'
    Require-PassString $logRunP11k 'PASS: tb_ternary_live_leaf_top_smoke_p11k'
    Require-PassString $logRunP11lb 'PASS: tb_ternary_live_leaf_top_smoke_p11l_b'
    Require-PassString $logRunP11lc 'PASS: tb_ternary_live_leaf_top_smoke_p11l_c'
    Require-PassString $logRunP11mBaseline 'PASS: tb_ternary_live_source_integration_smoke_p11m'
    Require-PassString $logRunP11mMacro 'PASS: tb_ternary_live_source_integration_smoke_p11m'
    Require-PassString $logRunP11mMacro '[p11m][PASS] source-side WQ integration path exact-match equivalent to split-interface local top'
    Require-PassString $logRunP11mMacro '[p11m][PASS] K/V fallback retained under WQ-only integration slice'

    $kvBaseline = Read-KvSig $logRunP11mBaseline
    $kvMacro = Read-KvSig $logRunP11mMacro
    if ($kvBaseline.K -ne $kvMacro.K -or $kvBaseline.V -ne $kvMacro.V) {
        throw "KV signature mismatch baseline vs macro: baseline(K=$($kvBaseline.K),V=$($kvBaseline.V)) macro(K=$($kvMacro.K),V=$($kvMacro.V))"
    }

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
