param(
    [string]$BuildDir = "build\\p11w4b8\\qkscore_family_fullhead_bridge"
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

    $exePath = Join-Path $BuildDir 'tb_w4b8_qkscore_family_fullhead_bridge.exe'
    $buildLog = Join-Path $BuildDir 'build.log'
    $runLog = Join-Path $BuildDir 'run.log'
    $verdictPath = Join-Path $BuildDir 'verdict.txt'
    $manifestPath = Join-Path $BuildDir 'file_manifest.txt'

    Invoke-ClBuild -Source 'tb\tb_w4b8_qkscore_family_fullhead_bridge.cpp' -ExeOut $exePath -LogOut $buildLog
    Invoke-ExeRun -ExePath $exePath -LogOut $runLog

    $requiredPassLines = @(
        'W4B8_QKSCORE_FAMILY_FULLHEAD_BRIDGE_VISIBLE PASS',
        'W4B8_QKSCORE_FAMILY_FULLHEAD_OWNERSHIP_CHECK PASS',
        'W4B8_QKSCORE_FAMILY_FULLHEAD_EXPECTED_COMPARE PASS',
        'W4B8_QKSCORE_FAMILY_FULLHEAD_LEGACY_COMPARE PASS',
        'W4B8_QKSCORE_FAMILY_FULLHEAD_NO_SPURIOUS_TOUCH PASS',
        'W4B8_QKSCORE_FAMILY_FULLHEAD_MULTI_PATH_ANTI_FALLBACK PASS',
        'W4B8_QKSCORE_FAMILY_FULLHEAD_MISMATCH_REJECT PASS',
        'PASS: tb_w4b8_qkscore_family_fullhead_bridge'
    )
    foreach ($line in $requiredPassLines) {
        Require-PassString -LogPath $runLog -Needle $line
    }

    Add-Content -Path $runLog -Value 'PASS: run_p11w4b8_qkscore_family_fullhead_bridge' -Encoding UTF8

    @(
        'task: P00-W4-B8-QKSCORE-FAMILY-FULLHEAD-BRIDGE',
        'status: PASS',
        'banner: PASS: run_p11w4b8_qkscore_family_fullhead_bridge',
        'scope: local-only',
        'closure: not Catapult closure; not SCVerify closure'
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("build_log={0}" -f $buildLog),
        ("run_log={0}" -f $runLog),
        ("verdict={0}" -f $verdictPath),
        ("tb_exe={0}" -f $exePath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host 'PASS: run_p11w4b8_qkscore_family_fullhead_bridge'
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            'task: P00-W4-B8-QKSCORE-FAMILY-FULLHEAD-BRIDGE',
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



