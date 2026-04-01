param(
    [string]$BuildDir = "build\\p11w4c2\\softmaxout_acc_single_later_token_bridge"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonRunner = Join-Path $PSScriptRoot 'run_p11w4cfamily_softmaxout_common.ps1'
$requiredPassLines = @(
    'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_BRIDGE_VISIBLE PASS',
    'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_OWNERSHIP_CHECK PASS',
    'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_EXPECTED_COMPARE PASS',
    'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_LEGACY_COMPARE PASS',
    'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_NO_SPURIOUS_TOUCH PASS',
    'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_MISMATCH_REJECT PASS',
    'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_ANTI_FALLBACK PASS',
    'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_LATER_TOKEN_CONSUME_COUNT_EXACT PASS',
    'W4C2_SOFTMAXOUT_ACC_SINGLE_LATER_TOKEN_TOKEN_SELECTOR_VISIBLE PASS',
    'PASS: tb_w4c2_softmaxout_acc_single_later_token_bridge'
)

& $commonRunner `
    -BuildDir $BuildDir `
    -Source 'tb\tb_w4c2_softmaxout_acc_single_later_token_bridge.cpp' `
    -ExeName 'tb_w4c2_softmaxout_acc_single_later_token_bridge.exe' `
    -TaskId 'P00-W4-C2-SOFTMAXOUT-ACC-SINGLE-LATER-TOKEN-BRIDGE' `
    -Banner 'PASS: run_p11w4c2_softmaxout_acc_single_later_token_bridge' `
    -RequiredPassLines $requiredPassLines

exit $LASTEXITCODE
