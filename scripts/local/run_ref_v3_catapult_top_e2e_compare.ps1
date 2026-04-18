param(
    [string]$BuildDir = "build\ref_v3_catapult_top_e2e_compare"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-ClBuild {
    param(
        [string[]]$Sources,
        [string]$ExeOut,
        [string]$LogOut
    )

    $clArgs = @(
        '/nologo',
        '/std:c++17',
        '/EHsc',
        '/utf-8',
        '/I.',
        '/IAECCT_ac_ref\include',
        '/IAECCT_ac_ref\src',
        '/Iinclude',
        '/Isrc',
        '/Igen',
        '/Igen\include',
        '/Ithird_party\ac_types',
        '/Idata\weights',
        '/Idata\trace'
    ) + $Sources + @("/Fe:$ExeOut")

    & cl @clArgs *> $LogOut
    if ($LASTEXITCODE -ne 0) {
        throw "build failed, exit=$LASTEXITCODE"
    }
}

function Invoke-ExeRun {
    param(
        [string]$ExePath,
        [string]$RunLog
    )

    & $ExePath *> $RunLog
    return $LASTEXITCODE
}

function Get-MatchedLines {
    param(
        [string]$Path,
        [string]$Pattern
    )

    $matches = Select-String -Path $Path -Pattern $Pattern
    if ($null -eq $matches) {
        return @()
    }
    return @($matches | ForEach-Object { $_.Line })
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $exePath = Join-Path $BuildDir "tb_ref_v3_catapult_top_e2e_compare.exe"
    $buildLog = Join-Path $BuildDir "build.log"
    $runLog = Join-Path $BuildDir "run.log"
    $summaryLog = Join-Path $BuildDir "summary.log"

    if (Test-Path $buildLog) { Remove-Item -LiteralPath $buildLog -Force }
    if (Test-Path $runLog) { Remove-Item -LiteralPath $runLog -Force }
    if (Test-Path $summaryLog) { Remove-Item -LiteralPath $summaryLog -Force }

    $sources = @(
        "AECCT_ac_ref\tb\ref_v3\tb_ref_v3_catapult_top_e2e_compare.cpp",
        "AECCT_ac_ref\src\RefModel.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3WeightsFp16LocalOnly.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3PreprocBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3AttenKvBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3AttenQSoftResBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3LayerNormBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3FfnLinear0ReluBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3FfnLinear1ResidualBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3FinalPassABlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3FinalPassBBlock.cpp"
    )

    Invoke-ClBuild -Sources $sources -ExeOut $exePath -LogOut $buildLog
    Write-Host "BUILD_OK: tb_ref_v3_catapult_top_e2e_compare"
    $runExit = Invoke-ExeRun -ExePath (Resolve-Path $exePath).Path -RunLog $runLog

    $summaryLines = @()
    $summaryLines += Get-MatchedLines -Path $runLog -Pattern '^\[ref_v3_top_tail_xpred_onecount_compare\]'
    $summaryLines += Get-MatchedLines -Path $runLog -Pattern '^\[ref_v3_top_tail_xpred_onecount_compare_summary\]'
    $summaryLines += Get-MatchedLines -Path $runLog -Pattern '^(PASS|FAIL): tb_ref_v3_catapult_top_e2e_compare'
    if ($runExit -eq 0) {
        $summaryLines += "RUN_OK: tb_ref_v3_catapult_top_e2e_compare"
    } else {
        $summaryLines += "RUN_FAIL: tb_ref_v3_catapult_top_e2e_compare (run_exit=$runExit)"
    }
    $summaryLines | Set-Content -Path $summaryLog -Encoding UTF8

    if ($runExit -eq 0) {
        Write-Host "RUN_OK: tb_ref_v3_catapult_top_e2e_compare"
        Write-Host "PASS: run_ref_v3_catapult_top_e2e_compare"
        exit 0
    }

    Write-Host "RUN_FAIL: tb_ref_v3_catapult_top_e2e_compare (run_exit=$runExit)"
    Write-Host "FAIL: run_ref_v3_catapult_top_e2e_compare (run_exit=$runExit)"
    exit $runExit
}
catch {
    Write-Host ("FAIL: run_ref_v3_catapult_top_e2e_compare ({0})" -f $_.Exception.Message)
    exit 1
}
finally {
    Pop-Location
}
