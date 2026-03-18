# P00-011AC Report - Phase-A Top-Managed K/V Migration Bring-Up (Local-Only)

## Summary
- `P00-011AC` adds a design-side bring-up helper for Top-managed Phase-A `K/V` staging with `in_ch/out_ch`.
- Internal packet contract remains provisional in AC bring-up and is intended to freeze only at `G-AC`.
- This task is local-only, not Catapult closure, and not SCVerify closure.

## Scope
- In scope:
- provisional packet header for AC bring-up
- phase-A Top-managed KV helper/wrapper path
- dedicated AC TB/checker/runner
- Out of scope:
- no Q path final integration
- no QK/score/softmax/output final integration
- no Catapult run
- no SCVerify run

## Files changed
- `include/AttnTopManagedPackets.h`
- `src/blocks/AttnPhaseATopManagedKv.h`
- `tb/tb_kv_build_stream_stage_p11ac.cpp`
- `scripts/check_p11ac_phasea_surface.ps1`
- `scripts/local/run_p11ac_phasea_top_managed.ps1`
- `docs/milestones/P00-011AC_report.md`

## Exact commands executed
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ac_phasea_surface.ps1 -OutDir build\p11ac -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ac_phasea_top_managed.ps1 -BuildDir build\p11ac\p11ac`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ac_phasea_surface.ps1 -OutDir build\p11ac -Phase post`

## Actual execution evidence excerpt
- `build\p11ac\check_p11ac_phasea_surface.log`
- `PASS: check_p11ac_phasea_surface`
- `build\p11ac\p11ac\run.log`
- `[p11ac][STREAM_ORDER][PASS] token=0 sequence=X->Wk->Wv->K->V`
- `[p11ac][MEM_ORDER][PASS] token=0 sequence=read X_WORK->read Wk->read Wv->write SCR_K->write SCR_V`
- `[p11ac][ACCESS_COUNT][PASS] token=0 X=1 Wk=1 Wv=1 SCR_K=1 SCR_V=1`
- `[p11ac][SPAN][PASS] token=0 K/V exact compare_length=32`
- `PASS: tb_kv_build_stream_stage_p11ac`
- `PASS: run_p11ac_phasea_top_managed`

## Result / verdict wording
- local-only progress is valid.
- not Catapult closure.
- not SCVerify closure.
