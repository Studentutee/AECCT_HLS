param(
    [string]$BuildDir = "build\\p11w4c7\\softmaxout_writeback_family_bound_hardening"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonRunner = Join-Path $PSScriptRoot 'run_p11w4cfamily_softmaxout_common.ps1'
$requiredPassLines = @(
    'W4C7_SOFTMAXOUT_WRITEBACK_FAMILY_BOUND_SELECTED_CONSUME_EXACT PASS',
    'W4C7_SOFTMAXOUT_WRITEBACK_FAMILY_BOUND_OWNER_CHECK PASS',
    'W4C7_SOFTMAXOUT_WRITEBACK_FAMILY_BOUND_COMPARE_GATE PASS',
    'W4C7_SOFTMAXOUT_WRITEBACK_FAMILY_BOUND_SELECTOR_CASE_MASK PASS',
    'W4C7_SOFTMAXOUT_WRITEBACK_EXPECTED_LEGACY_COMPARE PASS',
    'W4C7_SOFTMAXOUT_WRITEBACK_MISMATCH_REJECT PASS',
    'W4C7_SOFTMAXOUT_WRITEBACK_OWNER_REJECT PASS',
    'W4C7_SOFTMAXOUT_WRITEBACK_NO_SPURIOUS_TOUCH PASS',
    'W4C7_SOFTMAXOUT_WRITEBACK_ANTI_FALLBACK PASS',
    'PASS: tb_w4c7_softmaxout_writeback_family_bound_hardening'
)

& $commonRunner `
    -BuildDir $BuildDir `
    -Source 'tb\tb_w4c7_softmaxout_writeback_family_bound_hardening.cpp' `
    -ExeName 'tb_w4c7_softmaxout_writeback_family_bound_hardening.exe' `
    -TaskId 'P00-W4-C7-SOFTMAXOUT-WRITEBACK-FAMILY-BOUND-HARDENING' `
    -Banner 'PASS: run_p11w4c7_softmaxout_writeback_family_bound_hardening' `
    -RequiredPassLines $requiredPassLines

exit $LASTEXITCODE
