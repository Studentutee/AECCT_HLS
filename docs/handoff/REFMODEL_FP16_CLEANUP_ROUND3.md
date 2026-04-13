# REFMODEL FP16 CLEANUP ROUND3

## Summary
- Cleaned misleading fp32-style aliases in active RefModel path.
- Cleaned previously identified active main-path float contamination in:
  - `InvSqrtApprox.h`
  - `SoftmaxApprox.h`
  - `SoftmaxApproxLutData.h`
  - `RefE4M3Helpers.h`
  - `RefModel.cpp` FinalHead multiply / softmax exact exp / quant scale helper path
- Left debug / trace / host boundary float/double sites unchanged by request.

## Files changed
- `AECCT_ac_ref/include/InvSqrtApprox.h`
- `AECCT_ac_ref/include/SoftmaxApprox.h`
- `AECCT_ac_ref/include/SoftmaxApproxLutData.h`
- `AECCT_ac_ref/include/RefE4M3Helpers.h`
- `AECCT_ac_ref/src/RefModel.cpp`

## Evidence
- `build/refmodel_fp16_cleanup/run_refmodel_smoke.log`
- `build/refmodel_fp16_cleanup/run_refmodel_trace_compare.log`
- `build/refmodel_fp16_cleanup/check_design_purity.log`
- `build/refmodel_fp16_cleanup/refmodel_fp16_cleanup_audit.txt`

## Scope note
- This round did not clean host-only / trace-only / dump-only float and double usage.
- This round focused on active RefModel compute path cleanup only.
