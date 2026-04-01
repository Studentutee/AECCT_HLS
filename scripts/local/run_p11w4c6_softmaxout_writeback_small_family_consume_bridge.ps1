param(
    [string]$BuildDir = "build\\p11w4c6\\softmaxout_writeback_small_family_consume_bridge"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonRunner = Join-Path $PSScriptRoot 'run_p11w4cfamily_softmaxout_common.ps1'
$requiredPassLines = @(
    'W4C6_SOFTMAXOUT_WRITEBACK_SMALL_FAMILY_SELECTED_CONSUME_EXACT PASS',
    'W4C6_SOFTMAXOUT_WRITEBACK_SMALL_FAMILY_OWNER_CHECK PASS',
    'W4C6_SOFTMAXOUT_WRITEBACK_SMALL_FAMILY_COMPARE_GATE PASS',
    'W4C6_SOFTMAXOUT_WRITEBACK_SMALL_FAMILY_SELECTOR_CASE_MASK PASS',
    'W4C6_SOFTMAXOUT_WRITEBACK_EXPECTED_LEGACY_COMPARE PASS',
    'W4C6_SOFTMAXOUT_WRITEBACK_MISMATCH_REJECT PASS',
    'W4C6_SOFTMAXOUT_WRITEBACK_NO_SPURIOUS_TOUCH PASS',
    'W4C6_SOFTMAXOUT_WRITEBACK_ANTI_FALLBACK PASS',
    'PASS: tb_w4c6_softmaxout_writeback_small_family_consume_bridge'
)

& $commonRunner `
    -BuildDir $BuildDir `
    -Source 'tb\tb_w4c6_softmaxout_writeback_small_family_consume_bridge.cpp' `
    -ExeName 'tb_w4c6_softmaxout_writeback_small_family_consume_bridge.exe' `
    -TaskId 'P00-W4-C6-SOFTMAXOUT-WRITEBACK-SMALL-FAMILY-SELECTED-CONSUME-BRIDGE' `
    -Banner 'PASS: run_p11w4c6_softmaxout_writeback_small_family_consume_bridge' `
    -RequiredPassLines $requiredPassLines

exit $LASTEXITCODE
