param(
    [string]$BuildDir = "build\\preproc_channel_trace_compare_pilot"
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
        '/Igen',
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$verdictPath = $null
$manifestPath = $null
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null

    $exePath = Join-Path $BuildDir 'tb_preproc_channel_trace_compare_pilot.exe'
    $buildLog = Join-Path $BuildDir 'build.log'
    $runLog = Join-Path $BuildDir 'run.log'
    $summaryPath = Join-Path $BuildDir 'trace_compare_summary.txt'
    $reportPath = Join-Path $BuildDir 'trace_compare_report.md'
    $verdictPath = Join-Path $BuildDir 'verdict.txt'
    $manifestPath = Join-Path $BuildDir 'file_manifest.txt'

    Invoke-ClBuild -Source 'tb\tb_preproc_channel_trace_compare_pilot.cpp' -ExeOut $exePath -LogOut $buildLog
    Invoke-ExeRun -ExePath $exePath -LogOut $runLog

    $requiredPassLines = @(
        'PREPROC_CHANNEL_TRANSPORT_SKELETON PASS',
        'PREPROC_TRACE_COMPARE PASS',
        'PASS: tb_preproc_channel_trace_compare_pilot'
    )
    foreach ($line in $requiredPassLines) {
        Require-PassString -LogPath $runLog -Needle $line
    }

    $compareLines = Select-String -Path $runLog -Pattern '^TRACE_COMPARE_RESULT '
    if (-not $compareLines -or $compareLines.Count -eq 0) {
        throw "TRACE_COMPARE_RESULT lines missing in $runLog"
    }

    $aggregateLine = Select-String -Path $runLog -Pattern '^TRACE_COMPARE_AGGREGATE '
    if (-not $aggregateLine) {
        throw "TRACE_COMPARE_AGGREGATE line missing in $runLog"
    }

    $summaryOut = @()
    $summaryOut += (Select-String -Path $runLog -Pattern '^TRACE_COMPARE_TARGET ').Line
    $summaryOut += (Select-String -Path $runLog -Pattern '^TRACE_COMPARE_RULE ').Line
    $summaryOut += $compareLines.Line
    $summaryOut += $aggregateLine.Line
    $summaryOut | Set-Content -Path $summaryPath -Encoding UTF8

    $reportOut = @(
        '# Preproc Channel Trace Compare Pilot',
        '',
        '- scope: local-only',
        '- closure: not Catapult closure; not SCVerify closure',
        '',
        '## Compare Summary',
        ''
    )
    $reportOut += $summaryOut
    $reportOut | Set-Content -Path $reportPath -Encoding UTF8

    Add-Content -Path $runLog -Value 'PASS: run_preproc_channel_trace_compare_pilot' -Encoding UTF8

    @(
        'task: PREPROC-CHANNEL-TRACE-COMPARE-PILOT',
        'status: PASS',
        'banner: PASS: run_preproc_channel_trace_compare_pilot',
        'scope: local-only',
        'closure: not Catapult closure; not SCVerify closure'
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("build_log={0}" -f $buildLog),
        ("run_log={0}" -f $runLog),
        ("trace_summary={0}" -f $summaryPath),
        ("trace_report={0}" -f $reportPath),
        ("verdict={0}" -f $verdictPath),
        ("tb_exe={0}" -f $exePath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host 'PASS: run_preproc_channel_trace_compare_pilot'
    exit 0
}
catch {
    if (Test-Path $runLog) {
        $compareLines = Select-String -Path $runLog -Pattern '^TRACE_COMPARE_RESULT '
        $aggregateLine = Select-String -Path $runLog -Pattern '^TRACE_COMPARE_AGGREGATE '
        if ($compareLines -and $aggregateLine) {
            $summaryOut = @()
            $summaryOut += (Select-String -Path $runLog -Pattern '^TRACE_COMPARE_TARGET ').Line
            $summaryOut += (Select-String -Path $runLog -Pattern '^TRACE_COMPARE_RULE ').Line
            $summaryOut += $compareLines.Line
            $summaryOut += $aggregateLine.Line
            $summaryOut | Set-Content -Path $summaryPath -Encoding UTF8

            $reportOut = @(
                '# Preproc Channel Trace Compare Pilot',
                '',
                '- scope: local-only',
                '- closure: not Catapult closure; not SCVerify closure',
                '',
                '## Compare Summary',
                ''
            )
            $reportOut += $summaryOut
            $reportOut | Set-Content -Path $reportPath -Encoding UTF8
        }
    }

    if ($verdictPath) {
        @(
            'task: PREPROC-CHANNEL-TRACE-COMPARE-PILOT',
            'status: FAIL',
            ("message: {0}" -f $_.Exception.Message)
        ) | Set-Content -Path $verdictPath -Encoding UTF8
    }
    if ($manifestPath) {
        @(
            ("build_log={0}" -f $buildLog),
            ("run_log={0}" -f $runLog),
            ("trace_summary={0}" -f $summaryPath),
            ("trace_report={0}" -f $reportPath),
            ("verdict={0}" -f $verdictPath),
            ("tb_exe={0}" -f $exePath),
            'status=FAIL'
        ) | Set-Content -Path $manifestPath -Encoding UTF8
    }
    Write-Host ("FAIL: run_preproc_channel_trace_compare_pilot ({0})" -f $_.Exception.Message)
    exit 1
}
finally {
    Pop-Location
}
