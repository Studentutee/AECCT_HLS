Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path build\p11l_d > $null

    cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_smoke_p11j.cpp /Fe:build\p11l_d\tb_ternary_live_leaf_smoke_p11j.exe > build\p11l_d\build_p11j.log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "build_p11j failed with exit code $LASTEXITCODE" }

    cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11k.cpp /Fe:build\p11l_d\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11l_d\build_p11k.log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "build_p11k failed with exit code $LASTEXITCODE" }

    cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_b.cpp /Fe:build\p11l_d\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11l_d\build_p11l_b.log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "build_p11l_b failed with exit code $LASTEXITCODE" }

    cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_c.cpp /Fe:build\p11l_d\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11l_d\build_p11l_c.log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "build_p11l_c failed with exit code $LASTEXITCODE" }

    cmd /c build\p11l_d\tb_ternary_live_leaf_smoke_p11j.exe > build\p11l_d\run_p11j.log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "run_p11j failed with exit code $LASTEXITCODE" }

    cmd /c build\p11l_d\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11l_d\run_p11k.log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "run_p11k failed with exit code $LASTEXITCODE" }

    cmd /c build\p11l_d\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11l_d\run_p11l_b.log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "run_p11l_b failed with exit code $LASTEXITCODE" }

    cmd /c build\p11l_d\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11l_d\run_p11l_c.log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "run_p11l_c failed with exit code $LASTEXITCODE" }

    if (-not (Select-String -Path build\p11l_d\run_p11j.log -SimpleMatch -Quiet "PASS: tb_ternary_live_leaf_smoke_p11j")) {
        throw "required PASS string missing: PASS: tb_ternary_live_leaf_smoke_p11j"
    }
    if (-not (Select-String -Path build\p11l_d\run_p11k.log -SimpleMatch -Quiet "PASS: tb_ternary_live_leaf_top_smoke_p11k")) {
        throw "required PASS string missing: PASS: tb_ternary_live_leaf_top_smoke_p11k"
    }
    if (-not (Select-String -Path build\p11l_d\run_p11l_b.log -SimpleMatch -Quiet "PASS: tb_ternary_live_leaf_top_smoke_p11l_b")) {
        throw "required PASS string missing: PASS: tb_ternary_live_leaf_top_smoke_p11l_b"
    }
    if (-not (Select-String -Path build\p11l_d\run_p11l_c.log -SimpleMatch -Quiet "PASS: tb_ternary_live_leaf_top_smoke_p11l_c")) {
        throw "required PASS string missing: PASS: tb_ternary_live_leaf_top_smoke_p11l_c"
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
