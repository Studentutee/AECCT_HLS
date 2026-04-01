param(
    [string]$BuildDir = "build\\p11w4c1\\softmaxout_head_token_contract_probe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonRunner = Join-Path $PSScriptRoot 'run_p11w4cfamily_softmaxout_common.ps1'
$requiredPassLines = @(
    'W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_BRIDGE_VISIBLE PASS',
    'W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_OWNERSHIP_CHECK PASS',
    'W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_DESC_VISIBLE PASS',
    'W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_CONSUME_COUNT_EXACT PASS',
    'W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_EXPECTED_COMPARE PASS',
    'W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_LEGACY_COMPARE PASS',
    'W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_NO_SPURIOUS_TOUCH PASS',
    'W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_MISMATCH_REJECT PASS',
    'W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_ANTI_FALLBACK PASS',
    'PASS: tb_w4c1_softmaxout_head_token_contract_probe'
)

& $commonRunner `
    -BuildDir $BuildDir `
    -Source 'tb\tb_w4c1_softmaxout_head_token_contract_probe.cpp' `
    -ExeName 'tb_w4c1_softmaxout_head_token_contract_probe.exe' `
    -TaskId 'P00-W4-C1-SOFTMAXOUT-HEAD-TOKEN-CONTRACT-PROBE' `
    -Banner 'PASS: run_p11w4c1_softmaxout_head_token_contract_probe' `
    -RequiredPassLines $requiredPassLines

exit $LASTEXITCODE
