param(
    [string]$BuildDir = "build\ref_v3_binary16_compile_first"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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
        '/DREFV3_SYNTH_ONLY',
        '/I.',
        '/IAECCT_ac_ref\include',
        '/IAECCT_ac_ref\src',
        '/Iinclude',
        '/Isrc',
        '/Igen\include',
        '/Ithird_party\ac_types',
        '/Idata\weights',
        '/c',
        $Source,
        "/Fo:$ObjOut"
    )
    & cl @args *>> $LogOut
    if ($LASTEXITCODE -ne 0) {
        throw "synthesis-surface compile failed ($Source), exit=$LASTEXITCODE"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDir > $null
    $buildLog = Join-Path $BuildDir "build_ref_v3_binary16_compile_first.log"
    if (Test-Path $buildLog) {
        Remove-Item -LiteralPath $buildLog -Force
    }

    $sources = @(
        "AECCT_ac_ref\src\ref_v3\RefV3PreprocBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3AttenKvBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3AttenQSoftResBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3LayerNormBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3FfnBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3FfnLinear0ReluBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3FfnLinear1ResidualBlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3FinalPassABlock.cpp",
        "AECCT_ac_ref\src\ref_v3\RefV3FinalPassBBlock.cpp"
    )

    foreach ($source in $sources) {
        $objName = [System.IO.Path]::GetFileNameWithoutExtension($source) + ".synth.obj"
        $objPath = Join-Path $BuildDir $objName
        $srcLog = Join-Path $BuildDir ([System.IO.Path]::GetFileNameWithoutExtension($source) + ".build.log")
        Invoke-ClCompileOnlySynthesis -Source $source -ObjOut $objPath -LogOut $srcLog
    }

    $summaryLines = New-Object System.Collections.Generic.List[string]
    foreach ($source in $sources) {
        $srcStem = [System.IO.Path]::GetFileNameWithoutExtension($source)
        $srcLog = Join-Path $BuildDir ($srcStem + ".build.log")
        $summaryLines.Add(("[compile] {0}" -f $source))
        $summaryLines.AddRange([string[]](Get-Content -LiteralPath $srcLog))
    }
    $summaryLines.Add("BUILD_OK: ref_v3 binary16 compile-first synth surface")
    Set-Content -LiteralPath $buildLog -Value $summaryLines -Encoding UTF8
    Write-Host "PASS: run_ref_v3_binary16_compile_first"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
