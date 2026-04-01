param(
    [string]$BuildDir = "build\\p11w4c5\\softmaxout_writeback_single_selected_consume_bridge"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonRunner = Join-Path $PSScriptRoot 'run_p11w4cfamily_softmaxout_common.ps1'
$requiredPassLines = @(
    'W4C5_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_VISIBLE PASS',
    'W4C5_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_OWNERSHIP_CHECK PASS',
    'W4C5_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_SELECTOR_CASE_MASK PASS',
    'W4C5_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_WRITEBACK_PATH_VISIBILITY PASS',
    'W4C5_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_WRITEBACK_TOUCH_COUNT_EXACT PASS',
    'W4C5_SOFTMAXOUT_WRITEBACK_SELECTED_CONSUME_EXACT PASS',
    'W4C5_SOFTMAXOUT_WRITEBACK_EXPECTED_LEGACY_COMPARE PASS',
    'W4C5_SOFTMAXOUT_WRITEBACK_MISMATCH_REJECT PASS',
    'W4C5_SOFTMAXOUT_WRITEBACK_NO_SPURIOUS_TOUCH PASS',
    'W4C5_SOFTMAXOUT_WRITEBACK_ANTI_FALLBACK PASS',
    'PASS: tb_w4c5_softmaxout_writeback_single_selected_consume_bridge'
)

& $commonRunner `
    -BuildDir $BuildDir `
    -Source 'tb\tb_w4c5_softmaxout_writeback_single_selected_consume_bridge.cpp' `
    -ExeName 'tb_w4c5_softmaxout_writeback_single_selected_consume_bridge.exe' `
    -TaskId 'P00-W4-C5-SOFTMAXOUT-WRITEBACK-SINGLE-SELECTED-CONSUME-BRIDGE' `
    -Banner 'PASS: run_p11w4c5_softmaxout_writeback_single_selected_consume_bridge' `
    -RequiredPassLines $requiredPassLines

exit $LASTEXITCODE
