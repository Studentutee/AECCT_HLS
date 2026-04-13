# REFMODEL_ALIAS_DOUBLE_CLEANUP_ROUND4

## Summary
- Removed the misleading active-path alias `ref_model_fp_t` from `AECCT_ac_ref/src/RefModel.cpp`.
- Cleaned the active helper headers named in review feedback:
  - `InvSqrtApprox.h`
  - `SoftmaxApprox.h`
  - `SoftmaxApproxLutData.h`
  - `RefE4M3Helpers.h`
- Switched main-path weight/bias/norm pointers in `RefModel.cpp` to fp16 cached views instead of direct double arrays.

## Files changed
- `AECCT_ac_ref/include/InvSqrtApprox.h`
- `AECCT_ac_ref/include/SoftmaxApprox.h`
- `AECCT_ac_ref/include/SoftmaxApproxLutData.h`
- `AECCT_ac_ref/include/RefE4M3Helpers.h`
- `AECCT_ac_ref/src/RefModel.cpp`
- `archive/refmodel_fp16_alias_cleanup_legacy/...` backups

## What was checked
- `tb_refmodel_purefp16_smoke.cpp`
- `tb_refmodel_purefp16_trace_compare.cpp`
- `scripts/check_design_purity.py`
- grep audit for alias / helper double-float occurrences

## Result
- Build PASS for the two RefModel-only TBs.
- Smoke runs.
- Trace compare regressed badly after the cleanup; this means the cleanup changed active numerics and is not yet closure-safe.
- `check_design_purity.py` still fails due to existing repo-tracked `weights.h` includes outside this task scope.

## Key reviewer note
- The misleading alias is gone.
- The helper headers no longer route through the previous explicit `to_double()` / raw-double main helper style.
- Remaining `double` / `float` hits in the audit are mostly boundary/debug/dump/host paths, plus a few non-mainline experiment/helper leftovers still inside `RefModel.cpp`.
