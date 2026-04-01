param(
    [string]$BuildDir = "build\\p11w4c4\\softmaxout_writeback_single_selected_probe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonRunner = Join-Path $PSScriptRoot 'run_p11w4cfamily_softmaxout_common.ps1'
$requiredPassLines = @(
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_VISIBLE PASS',
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_OWNERSHIP_CHECK PASS',
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_EXPECTED_COMPARE PASS',
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_LEGACY_COMPARE PASS',
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_NO_SPURIOUS_TOUCH PASS',
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_MISMATCH_REJECT PASS',
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_ANTI_FALLBACK PASS',
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_SELECTOR_CASE_MASK PASS',
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_WRITEBACK_PATH_VISIBILITY PASS',
    'W4C4_SOFTMAXOUT_WRITEBACK_SINGLE_SELECTED_WRITEBACK_TOUCH_COUNT_EXACT PASS',
    'PASS: tb_w4c4_softmaxout_writeback_single_selected_probe'
)

& $commonRunner `
    -BuildDir $BuildDir `
    -Source 'tb\tb_w4c4_softmaxout_writeback_single_selected_probe.cpp' `
    -ExeName 'tb_w4c4_softmaxout_writeback_single_selected_probe.exe' `
    -TaskId 'P00-W4-C4-SOFTMAXOUT-WRITEBACK-SINGLE-SELECTED-PROBE' `
    -Banner 'PASS: run_p11w4c4_softmaxout_writeback_single_selected_probe' `
    -RequiredPassLines $requiredPassLines

exit $LASTEXITCODE
