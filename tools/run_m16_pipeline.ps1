param(
    [string]$Configuration = "Debug",
    [string]$Platform = "x64"
)

$ErrorActionPreference = "Stop"

Write-Host "[M16] step1: generate headers (run #1)"
python tools/gen_headers.py --repo-root . --src-include include --out-root gen --version v11.5
if ($LASTEXITCODE -ne 0) { throw "gen_headers run #1 failed" }
$manifest1 = Get-FileHash -Algorithm SHA256 gen/manifest.json

Write-Host "[M16] step2: generate headers (run #2) reproducibility check"
python tools/gen_headers.py --repo-root . --src-include include --out-root gen --version v11.5
if ($LASTEXITCODE -ne 0) { throw "gen_headers run #2 failed" }
$manifest2 = Get-FileHash -Algorithm SHA256 gen/manifest.json
if ($manifest1.Hash -ne $manifest2.Hash) {
    throw "Reproducibility check failed: manifest hash mismatch"
}

Write-Host "[M16] step3: build"
msbuild AECCT_HLS.vcxproj /p:Configuration=$Configuration /p:Platform=$Platform /m
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "[M16] step4: run regression"
cmd /c ".\build\bin\$Platform\$Configuration\AECCT_HLS.exe"
if ($LASTEXITCODE -ne 0) { throw "regression failed" }

Write-Host "[M16] done"
