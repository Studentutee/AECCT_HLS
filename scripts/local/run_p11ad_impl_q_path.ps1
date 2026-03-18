param(
    [string]$BuildDir = "build\p11ad_impl\p11ad"
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
$verdictPath = $null
$manifestPath = $null
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $exePath = Join-Path $BuildDir 'tb_q_path_impl_p11ad.exe'
    $buildLog = Join-Path $BuildDir 'build.log'
    $runLog = Join-Path $BuildDir 'run.log'
    $verdictPath = Join-Path $BuildDir 'verdict.txt'
    $manifestPath = Join-Path $BuildDir 'file_manifest.txt'

    Invoke-ClBuild -Source 'tb\tb_q_path_impl_p11ad.cpp' -ExeOut $exePath -LogOut $buildLog
    Invoke-ExeRun -ExePath $exePath -LogOut $runLog

    $requiredPassLines = @(
        'PASS: tb_q_path_impl_p11ad',
        'Q_PATH_MAINLINE PASS',
        'Q_EXPECTED_COMPARE PASS',
        'Q_TARGET_SPAN_WRITE PASS',
        'NO_SPURIOUS_WRITE PASS',
        'SOURCE_PRESERVATION PASS',
        'MAINLINE_Q_PATH_TAKEN PASS',
        'FALLBACK_NOT_TAKEN PASS',
        'fallback_taken = false'
    )
    foreach ($line in $requiredPassLines) {
        Require-PassString -LogPath $runLog -Needle $line
    }

    Add-Content -Path $runLog -Value 'PASS: run_p11ad_impl_q_path' -Encoding UTF8

    @(
        'task: P00-011AD',
        'status: PASS',
        'banner: PASS: run_p11ad_impl_q_path',
        'scope: local-only',
        'closure: not Catapult closure; not SCVerify closure'
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("build_log={0}" -f $buildLog),
        ("run_log={0}" -f $runLog),
        ("verdict={0}" -f $verdictPath),
        ("tb_exe={0}" -f $exePath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host 'PASS: run_p11ad_impl_q_path'
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            'task: P00-011AD',
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
