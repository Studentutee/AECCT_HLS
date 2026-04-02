param(
    [string]$BuildDir = "build\\p11anb\\attnlayer0_boundary_seam_contract"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonRunner = Join-Path $PSScriptRoot 'run_p11w4cfamily_softmaxout_common.ps1'
$requiredPassLines = @(
    'P11ANB_ATTNLAYER0_QKV_DESCRIPTOR_CONSUME PASS',
    'P11ANB_ATTNLAYER0_QKV_DESCRIPTOR_SKIP PASS',
    'P11ANB_ATTNLAYER0_OUT_DESCRIPTOR_CONSUME PASS',
    'P11ANB_ATTNLAYER0_OUT_DESCRIPTOR_ANTI_FALLBACK PASS',
    'P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_CONSUME PASS',
    'P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_INVALID_FALLBACK PASS',
    'P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_DISABLED_FALLBACK PASS',
    'P11ANB_ATTNLAYER0_LEGACY_DESCRIPTOR_EQUIVALENCE PASS',
    'PASS: tb_p11anb_attnlayer0_boundary_seam_contract'
)

& $commonRunner `
    -BuildDir $BuildDir `
    -Source 'tb\tb_p11anb_attnlayer0_boundary_seam_contract.cpp' `
    -ExeName 'tb_p11anb_attnlayer0_boundary_seam_contract.exe' `
    -TaskId 'P00-011ANB-ATTNLAYER0-BOUNDARY-SEAM-CONTRACT' `
    -Banner 'PASS: run_p11anb_attnlayer0_boundary_seam_contract' `
    -RequiredPassLines $requiredPassLines

exit $LASTEXITCODE
