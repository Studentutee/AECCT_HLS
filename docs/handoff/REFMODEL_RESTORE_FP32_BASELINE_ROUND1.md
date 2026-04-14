# RefModel restore to fp32-capable baseline (round1)

## Summary
- Restored active RefModel implementation from `archive/refmodel_mixed_precision_legacy`.
- Set active default precision mode back to `BASELINE_FP32` in `AECCT_ac_ref/include/RefModel.h`.
- Restored `AECCT_ac_ref/include/SoftmaxApproxLutData.h` from an earlier repo zip because the current repo archive copy was not present and the active file had already been converted to `ref_fp16_t` LUT entries, which made the restored mixed-precision Softmax helper fail to compile.

## Files changed
- `AECCT_ac_ref/include/InvSqrtApprox.h`
- `AECCT_ac_ref/include/RefE4M3Helpers.h`
- `AECCT_ac_ref/include/RefModel.h`
- `AECCT_ac_ref/include/SoftmaxApprox.h`
- `AECCT_ac_ref/include/SoftmaxApproxLutData.h`
- `AECCT_ac_ref/src/RefModel.cpp`
- `AECCT_ac_ref/src/ref_main.cpp`

## Validation run
- Smoke test: PASS
- Sample0 trace compare: PASS (`xpred_mismatch=0`)
- Batch16 trace compare: build succeeded, but runtime did not complete within a quick 30s check in this container, so no PASS claim is made for batch16 in this handoff.

## Notes
- This is a baseline-restore action, not a new fp16 closure.
- Active default precision now points to fp32 baseline again so existing RefModel-only tests call the legacy baseline unless explicitly overridden.
