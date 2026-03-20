param(
    [string]$BuildDir = "build\p11am\p11am"
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

function Invoke-ClCompileOnlySynthesis {
    param(
        [string]$Source,
        [string]$ObjOut,
        [string]$LogOut
    )

    $args = @(
        '/nologo',
        '/std:c++14',
        '/EHsc',
        '/utf-8',
        '/D__SYNTHESIS__',
        '/I.',
        '/Iinclude',
        '/Isrc',
        '/Igen\include',
        '/Ithird_party\ac_types',
        '/Idata\weights',
        '/c',
        $Source,
        "/Fo:$ObjOut"
    )
    & cl @args *> $LogOut
    if ($LASTEXITCODE -ne 0) {
        throw "synthesis-surface compile failed ($Source), exit=$LASTEXITCODE"
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

    $tbSource = 'tb\tb_top_managed_catapult_compile_prep_p11am.cpp'
    $exePath = Join-Path $BuildDir 'tb_top_managed_catapult_compile_prep_p11am.exe'
    $objPath = Join-Path $BuildDir 'tb_top_managed_catapult_compile_prep_p11am.synth.obj'
    $buildLog = Join-Path $BuildDir 'build.log'
    $runLog = Join-Path $BuildDir 'run.log'
    $synthBuildLog = Join-Path $BuildDir 'build_synth_surface.log'
    $surfacePreLog = Join-Path $BuildDir 'surface_pre.log'
    $surfacePostLog = Join-Path $BuildDir 'surface_post.log'
    $verdictPath = Join-Path $BuildDir 'verdict.txt'
    $manifestPath = Join-Path $BuildDir 'file_manifest.txt'

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11am_catapult_top_surface.ps1 -OutDir $BuildDir -Phase pre *> $surfacePreLog
    if ($LASTEXITCODE -ne 0) {
        throw "pre surface checker failed, exit=$LASTEXITCODE"
    }

    Invoke-ClBuild -Source $tbSource -ExeOut $exePath -LogOut $buildLog
    Invoke-ExeRun -ExePath $exePath -LogOut $runLog

    $requiredPassLines = @(
        'P11AM_WRAPPER_RUNTIME_SMOKE PASS',
        'P11AM_MAINLINE_PATH_TAKEN PASS',
        'P11AM_FALLBACK_NOT_TAKEN PASS',
        'fallback_taken = false',
        'PASS: tb_top_managed_catapult_compile_prep_p11am'
    )
    foreach ($line in $requiredPassLines) {
        Require-PassString -LogPath $runLog -Needle $line
    }

    Invoke-ClCompileOnlySynthesis -Source $tbSource -ObjOut $objPath -LogOut $synthBuildLog
    if (-not (Test-Path $objPath)) {
        throw "synthesis-surface object missing: $objPath"
    }
    Add-Content -Path $synthBuildLog -Value 'PASS: p11am_synth_surface_compile' -Encoding UTF8

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11am_catapult_top_surface.ps1 -OutDir $BuildDir -Phase post *> $surfacePostLog
    if ($LASTEXITCODE -ne 0) {
        throw "post surface checker failed, exit=$LASTEXITCODE"
    }

    @(
        'task: P00-011AM',
        'status: PASS',
        'banner: PASS: run_p11am_catapult_compile_surface',
        'scope: local-only',
        'posture: Catapult-facing progress only',
        'closure: not Catapult closure; not SCVerify closure'
    ) | Set-Content -Path $verdictPath -Encoding UTF8

    @(
        ("build_log={0}" -f $buildLog),
        ("run_log={0}" -f $runLog),
        ("synth_compile_log={0}" -f $synthBuildLog),
        ("surface_pre_log={0}" -f $surfacePreLog),
        ("surface_post_log={0}" -f $surfacePostLog),
        ("tb_exe={0}" -f $exePath),
        ("synth_obj={0}" -f $objPath),
        ("verdict={0}" -f $verdictPath)
    ) | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host 'PASS: run_p11am_catapult_compile_surface'
    exit 0
}
catch {
    Write-Error $_
    if ($verdictPath) {
        @(
            'task: P00-011AM',
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
