param(
    [string[]]$Blocks = @("all"),
    [string]$RunRoot = "",
    [string]$CatapultCmd = "catapult",
    [switch]$PrepareOnly,
    [switch]$SkipMetrics
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = (Resolve-Path (Join-Path (Join-Path $PSScriptRoot "..\..\..") "..")).Path
$blockRoot = Join-Path $PSScriptRoot ""

$blockMap = [ordered]@{
    preproc = @{
        block_name = "RefV3PreprocBlock"
        project_name = "Catapult_refv3_preproc"
        project_tcl = "scripts/catapult/ref_v3/blocks/preproc/project.tcl"
    }
    attn_kv = @{
        block_name = "RefV3AttenKvBlock"
        project_name = "Catapult_refv3_attn_kv"
        project_tcl = "scripts/catapult/ref_v3/blocks/attn_kv/project.tcl"
    }
    attn_qsoftres = @{
        block_name = "RefV3AttenQSoftResBlock"
        project_name = "Catapult_refv3_attn_qsoftres"
        project_tcl = "scripts/catapult/ref_v3/blocks/attn_qsoftres/project.tcl"
    }
    layernorm = @{
        block_name = "RefV3LayerNormBlock"
        project_name = "Catapult_refv3_layernorm"
        project_tcl = "scripts/catapult/ref_v3/blocks/layernorm/project.tcl"
    }
    ffn_linear0 = @{
        block_name = "RefV3FfnLinear0ReluBlock"
        project_name = "Catapult_refv3_ffn_linear0"
        project_tcl = "scripts/catapult/ref_v3/blocks/ffn_linear0/project.tcl"
    }
    ffn_linear1 = @{
        block_name = "RefV3FfnLinear1ResidualBlock"
        project_name = "Catapult_refv3_ffn_linear1"
        project_tcl = "scripts/catapult/ref_v3/blocks/ffn_linear1/project.tcl"
    }
    finalA = @{
        block_name = "RefV3FinalPassABlock"
        project_name = "Catapult_refv3_finalA"
        project_tcl = "scripts/catapult/ref_v3/blocks/finalA/project.tcl"
    }
    finalB = @{
        block_name = "RefV3FinalPassBBlock"
        project_name = "Catapult_refv3_finalB"
        project_tcl = "scripts/catapult/ref_v3/blocks/finalB/project.tcl"
    }
}

if ($RunRoot -eq "") {
    $RunRoot = Join-Path $repoRoot ("build\\ref_v3\\blocks\\run_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
}
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$selectedBlockKeys = New-Object System.Collections.Generic.List[string]
if ($Blocks.Count -eq 1 -and $Blocks[0] -eq "all") {
    foreach ($k in $blockMap.Keys) {
        $selectedBlockKeys.Add($k)
    }
} else {
    foreach ($raw in $Blocks) {
        $k = $raw.Trim()
        if (-not $blockMap.Contains($k)) {
            throw "Unknown block key: $k"
        }
        $selectedBlockKeys.Add($k)
    }
}

$catapultAvailable = $false
if (-not $PrepareOnly) {
    $cmdProbe = Get-Command $CatapultCmd -ErrorAction SilentlyContinue
    $catapultAvailable = ($null -ne $cmdProbe)
    if (-not $catapultAvailable) {
        Write-Warning "Catapult command not found in PATH: $CatapultCmd. Switching to PrepareOnly mode."
        $PrepareOnly = $true
    }
}

$summaryPath = Join-Path $RunRoot "run_summary.tsv"
$summaryRows = New-Object System.Collections.Generic.List[string]
$summaryRows.Add("block_key`tblock_name`tproject_name`tstatus`tcatapult_exit`tproject_dir`tmanifest_path`tconsole_log")

foreach ($k in $selectedBlockKeys) {
    $cfg = $blockMap[$k]
    $projectTclAbs = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $cfg.project_tcl))
    if (-not (Test-Path -LiteralPath $projectTclAbs)) {
        throw "project.tcl missing: $projectTclAbs"
    }

    $blockOutDir = Join-Path $RunRoot $cfg.project_name
    $projectOutDir = Join-Path $blockOutDir "project"
    New-Item -ItemType Directory -Force -Path $blockOutDir | Out-Null

    $consoleLog = Join-Path $blockOutDir "catapult_console.log"
    $internalLog = Join-Path $blockOutDir "catapult_internal.log"
    $exactCmd = Join-Path $blockOutDir "exact_command.txt"
    $manifestPath = Join-Path $projectOutDir "refv3_block_manifest.txt"

    $cmdText = "{0} -shell -file {1} -logfile {2}" -f $CatapultCmd, $projectTclAbs, $internalLog
    $cmdText | Set-Content -Path $exactCmd -Encoding Ascii

    $status = "PREPARED_ONLY"
    $catapultExit = -1

    if (-not $PrepareOnly) {
        $env:AECCT_REFV3_REPO_ROOT = $repoRoot
        $env:AECCT_REFV3_BLOCK_CATAPULT_OUTDIR = $projectOutDir

        Write-Host ("[{0}] Launching {1}" -f $k, $cfg.block_name) -ForegroundColor Cyan
        $prevError = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $CatapultCmd -shell -file $projectTclAbs -logfile $internalLog 2>&1 | Tee-Object -FilePath $consoleLog
        $catapultExit = $LASTEXITCODE
        $ErrorActionPreference = $prevError
        Remove-Item Env:AECCT_REFV3_REPO_ROOT -ErrorAction SilentlyContinue
        Remove-Item Env:AECCT_REFV3_BLOCK_CATAPULT_OUTDIR -ErrorAction SilentlyContinue

        $status = if ($catapultExit -eq 0) { "EXECUTED_OK" } else { "EXECUTED_FAIL" }
    }

    $summaryRows.Add(("{0}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}`t{7}" -f `
            $k, `
            $cfg.block_name, `
            $cfg.project_name, `
            $status, `
            $catapultExit, `
            $projectOutDir, `
            $manifestPath, `
            $consoleLog))
}

$summaryRows | Set-Content -Path $summaryPath -Encoding Ascii

if (-not $SkipMetrics) {
    $collector = Join-Path $PSScriptRoot "collect_ref_v3_block_metrics.ps1"
    if (Test-Path -LiteralPath $collector) {
        & powershell -ExecutionPolicy Bypass -File $collector -RunRoot $RunRoot
    }
}

$posturePath = Join-Path $RunRoot "governance_posture.txt"
@(
    "scope: ref_v3 block-level Catapult matrix run",
    "status: run complete",
    "posture: not Catapult closure",
    "posture_scverify: not SCVerify closure"
) | Set-Content -Path $posturePath -Encoding Ascii

Write-Host "Run root: $RunRoot" -ForegroundColor Green
Write-Host "Summary:  $summaryPath" -ForegroundColor Green
