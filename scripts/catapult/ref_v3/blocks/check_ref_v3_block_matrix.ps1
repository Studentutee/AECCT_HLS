Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path (Join-Path $PSScriptRoot "..\..\..") "..")).Path
$base = Join-Path $repoRoot "scripts\\catapult\\ref_v3\\blocks"

$expected = @(
    @{ key = "preproc"; project = "Catapult_refv3_preproc"; tb = "AECCT_ac_ref/tb_catapult/ref_v3/blocks/tb_ref_v3_preproc_block_smoke.cpp" },
    @{ key = "attn_kv"; project = "Catapult_refv3_attn_kv"; tb = "AECCT_ac_ref/tb_catapult/ref_v3/blocks/tb_ref_v3_atten_kv_block_smoke.cpp" },
    @{ key = "attn_qsoftres"; project = "Catapult_refv3_attn_qsoftres"; tb = "AECCT_ac_ref/tb_catapult/ref_v3/blocks/tb_ref_v3_atten_qsoftres_block_smoke.cpp" },
    @{ key = "layernorm"; project = "Catapult_refv3_layernorm"; tb = "AECCT_ac_ref/tb_catapult/ref_v3/blocks/tb_ref_v3_layernorm_block_smoke.cpp" },
    @{ key = "ffn_linear0"; project = "Catapult_refv3_ffn_linear0"; tb = "AECCT_ac_ref/tb_catapult/ref_v3/blocks/tb_ref_v3_ffn_linear0_relu_block_smoke.cpp" },
    @{ key = "ffn_linear1"; project = "Catapult_refv3_ffn_linear1"; tb = "AECCT_ac_ref/tb_catapult/ref_v3/blocks/tb_ref_v3_ffn_linear1_residual_block_smoke.cpp" },
    @{ key = "finalA"; project = "Catapult_refv3_finalA"; tb = "AECCT_ac_ref/tb_catapult/ref_v3/blocks/tb_ref_v3_final_pass_a_block_smoke.cpp" },
    @{ key = "finalB"; project = "Catapult_refv3_finalB"; tb = "AECCT_ac_ref/tb_catapult/ref_v3/blocks/tb_ref_v3_final_pass_b_block_smoke.cpp" }
)

$errors = New-Object System.Collections.Generic.List[string]
$checks = New-Object System.Collections.Generic.List[string]

$commonTcl = Join-Path $base "common_ref_v3_block_project.tcl"
if (-not (Test-Path -LiteralPath $commonTcl)) {
    $errors.Add("missing common tcl: $commonTcl")
} else {
    $checks.Add("found common tcl: $commonTcl")
}

$scanFiles = Get-ChildItem -LiteralPath $base -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in @("project.tcl", "filelist.f") }
$forbiddenHits = Select-String -Path ($scanFiles | ForEach-Object { $_.FullName }) `
    -Pattern "PIPELINE_INIT_INTERVAL\\s+1" `
    -AllMatches `
    -SimpleMatch `
    -ErrorAction SilentlyContinue
if ($null -ne $forbiddenHits) {
    foreach ($hit in $forbiddenHits) {
        $errors.Add("forbidden directive found: $($hit.Path):$($hit.LineNumber)")
    }
} else {
    $checks.Add("no PIPELINE_INIT_INTERVAL 1 in block scripts")
}

foreach ($blk in $expected) {
    $dir = Join-Path $base $blk.key
    $projectTcl = Join-Path $dir "project.tcl"
    $filelist = Join-Path $dir "filelist.f"

    if (-not (Test-Path -LiteralPath $dir)) {
        $errors.Add("missing block dir: $dir")
        continue
    }
    if (-not (Test-Path -LiteralPath $projectTcl)) {
        $errors.Add("missing project.tcl: $projectTcl")
    }
    if (-not (Test-Path -LiteralPath $filelist)) {
        $errors.Add("missing filelist.f: $filelist")
    }

    if (Test-Path -LiteralPath $projectTcl) {
        $content = Get-Content -LiteralPath $projectTcl -Raw
        if ($content -notmatch [regex]::Escape($blk.project)) {
            $errors.Add("project name not found in ${projectTcl}: $($blk.project)")
        }
        if ($content -notmatch [regex]::Escape($blk.tb)) {
            $errors.Add("smoke tb path not found in ${projectTcl}: $($blk.tb)")
        }
        if ($content -notlike "*common_ref_v3_block_project.tcl*") {
            $errors.Add("common tcl not sourced in $projectTcl")
        }
    }

    if (Test-Path -LiteralPath $filelist) {
        foreach ($line in Get-Content -LiteralPath $filelist) {
            $t = $line.Trim()
            if ($t -eq "" -or $t.StartsWith("#")) { continue }
            $abs = Join-Path $repoRoot $t
            if (-not (Test-Path -LiteralPath $abs)) {
                $errors.Add("filelist missing file: $abs")
            }
        }
    }
}

$logPath = Join-Path $repoRoot "build\\ref_v3\\blocks\\matrix_check.log"
New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($logPath)) | Out-Null

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("REFV3 block matrix check")
$lines.Add(("time={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")))
$lines.Add("posture=not Catapult closure")
$lines.Add("posture_scverify=not SCVerify closure")
$lines.Add("")
$lines.Add("[checks]")
foreach ($c in $checks) { $lines.Add($c) }
$lines.Add("")
$lines.Add("[errors]")
if ($errors.Count -eq 0) {
    $lines.Add("none")
} else {
    foreach ($e in $errors) { $lines.Add($e) }
}
$lines | Set-Content -Path $logPath -Encoding Ascii

if ($errors.Count -gt 0) {
    Write-Host "REFV3 block matrix check: FAIL" -ForegroundColor Red
    foreach ($e in $errors) {
        Write-Host " - $e" -ForegroundColor Red
    }
    Write-Host "log: $logPath" -ForegroundColor Yellow
    exit 1
}

Write-Host "REFV3 block matrix check: PASS" -ForegroundColor Green
Write-Host "log: $logPath" -ForegroundColor Green
exit 0
