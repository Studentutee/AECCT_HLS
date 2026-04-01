param(
    [string]$BuildDir = "build\\p11w4c3\\softmaxout_renorm_single_selected_probe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonRunner = Join-Path $PSScriptRoot 'run_p11w4cfamily_softmaxout_common.ps1'
$requiredPassLines = @(
    'W4C3_SOFTMAXOUT_RENORM_SINGLE_SELECTED_VISIBLE PASS',
    'W4C3_SOFTMAXOUT_RENORM_SINGLE_SELECTED_OWNERSHIP_CHECK PASS',
    'W4C3_SOFTMAXOUT_RENORM_SINGLE_SELECTED_EXPECTED_COMPARE PASS',
    'W4C3_SOFTMAXOUT_RENORM_SINGLE_SELECTED_LEGACY_COMPARE PASS',
    'W4C3_SOFTMAXOUT_RENORM_SINGLE_SELECTED_NO_SPURIOUS_TOUCH PASS',
    'W4C3_SOFTMAXOUT_RENORM_SINGLE_SELECTED_MISMATCH_REJECT PASS',
    'W4C3_SOFTMAXOUT_RENORM_SINGLE_SELECTED_ANTI_FALLBACK PASS',
    'W4C3_SOFTMAXOUT_RENORM_SINGLE_SELECTED_SELECTOR_CASE_MASK PASS',
    'W4C3_SOFTMAXOUT_RENORM_SINGLE_SELECTED_RENORM_PATH_VISIBILITY PASS',
    'PASS: tb_w4c3_softmaxout_renorm_single_selected_probe'
)

& $commonRunner `
    -BuildDir $BuildDir `
    -Source 'tb\tb_w4c3_softmaxout_renorm_single_selected_probe.cpp' `
    -ExeName 'tb_w4c3_softmaxout_renorm_single_selected_probe.exe' `
    -TaskId 'P00-W4-C3-SOFTMAXOUT-RENORM-SINGLE-SELECTED-PROBE' `
    -Banner 'PASS: run_p11w4c3_softmaxout_renorm_single_selected_probe' `
    -RequiredPassLines $requiredPassLines

exit $LASTEXITCODE
