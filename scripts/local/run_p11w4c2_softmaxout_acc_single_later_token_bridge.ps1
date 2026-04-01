param(
    [string]$BuildDir = "build\\p11w4c2\\softmaxout_acc_single_later_token_bridge"
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$verdictPath = $null
$manifestPath = $null
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $exePath = Join-Path $BuildDir 'tb_w4c2_softmaxout_acc_single_later_token_bridge.exe'
    $buildLog = Join-Path $BuildDir 'build.log'
    $runLog = Join-Path $BuildDir 'run.log'
    $verdictPath = Join-Path $BuildDir 'verdict.txt'
    $manifestPath = Join-Path $BuildDir 'file_manifest.txt'

    Invoke-ClBuild -Source 'tb\tb_w4c2_softmaxout_acc_single_later_token_bridge.cpp' -ExeOut $exePath -LogOut $buildLog
    Invoke-ExeRun -ExePath $exePath -LogOut $runLog

    $requiredPassLines = @(
        'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_BRIDGE_VISIBLE PASS',
        'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_OWNERSHIP_CHECK PASS',
        'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_EXPECTED_COMPARE PASS',
        'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_LEGACY_COMPARE PASS',
        'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_NO_SPURIOUS_TOUCH PASS',
        'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_MISMATCH_REJECT PASS',
        'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_ANTI_FALLBACK PASS',
        'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_LATER_TOKEN_CONSUME_COUNT_EXACT PASS',
        'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_TOKEN_SELECTOR_VISIBLE PASS',
        'PASS: tb_w4c2_softmaxout_acc_single_later_token_bridge'
    )
    foreach ($line in $requiredPassLines) {
        Require-PassString -LogPath $runLog -Needle $line
    }

    Add-Content -Path $runLog -Value 'PASS: run_p11w4c2_softmaxout_acc_single_later_token_bridge' -Encoding UTF8

    @(
        'task: P00-W4-C2-SOFTMAXOUT-ACC-SINGLE-LATER-TOKEN-BRIDGE',
        'status: PASS',
        'banner: PASS: run_p11w4c2_softmaxout_acc_single_later_token_bridge',
        'scope: local-only',
        'closure: not Catapult closure; not SCVerify closure'
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("build_log={0}" -f $buildLog),
        ("run_log={0}" -f $runLog),
        ("verdict={0}" -f $verdictPath),
        ("tb_exe={0}" -f $exePath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host 'PASS: run_p11w4c2_softmaxout_acc_single_later_token_bridge'
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            'task: P00-W4-C2-SOFTMAXOUT-ACC-SINGLE-LATER-TOKEN-BRIDGE',
            'status: FAIL',
            ("message: {0}" -f $_.Exception.Message)
        ) | Set-Content -Path $verdictPath -Encoding UTF8
    }
    if ($manifestPath -and (Test-Path $manifestPath -PathType Leaf) -eq $false) {
        @(
            ("build_dir={0}" -f $BuildDir),
            'status=FAIL'
        ) | Set-Content -Path $manifestPath -Encoding UTF8
    }
    exit 1
}
finally {
    Pop-Location
}
