# P00-011AC Report - Phase-A Top-Managed K/V Design-Mainline Wiring Finalize (Local-Only)

## Summary
- `P00-011AC` now closes the AC helper/proof-only gap by wiring the landed Top-managed Phase-A `K/V` helper flow into the real design-side path.
- Completion acceptance is strengthened to require real mainline execution and anti-fallback evidence:
- `MAINLINE_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `fallback_taken = false`
- This milestone remains local-only, not Catapult closure, and not SCVerify closure.

## Scope
- In scope:
- design-side mainline call path hookup in `Top.h` (`run_transformer_layer_loop` target-layer path)
- minimal integration hooks in `TransformerLayer.h` / `AttnLayer0.h`
- helper-side integrated entrypoint for Top-managed Phase-A `K/V` execution
- p11ac acceptance evidence strengthening (path banners + explicit fallback telemetry)
- Out of scope:
- no AD-impl / AE-impl / AF-impl
- no Q final integration, no QK/score final integration, no softmax/output final integration
- no Catapult run
- no SCVerify run

## Files changed
- `src/Top.h`
- `src/blocks/TransformerLayer.h`
- `src/blocks/AttnLayer0.h`
- `src/blocks/AttnPhaseATopManagedKv.h`
- `tb/tb_kv_build_stream_stage_p11ac.cpp`
- `scripts/check_p11ac_phasea_surface.ps1`
- `scripts/local/run_p11ac_phasea_top_managed.ps1`
- `docs/process/P11_AC_AF_INTERFACE_FREEZE.md`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/milestones/P00-011AC_report.md`
- `docs/milestones/P00-011AC_AF_umbrella_report.md`

## Exact commands executed
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ac_phasea_surface.ps1 -OutDir build\p11ac_final -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ac_phasea_top_managed.ps1 -BuildDir build\p11ac_final\p11ac`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ac_phasea_surface.ps1 -OutDir build\p11ac_final -Phase post`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11ac_final\wrappers\p11r`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11ac_final\wrappers\p11s`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11z_runtime_probe.ps1 -BuildDir build\p11ac_final\wrappers\p11z`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11aa_qkv_loadw_bridge.ps1 -BuildDir build\p11ac_final\wrappers\p11aa`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ab_kv_build_stage.ps1 -BuildDir build\p11ac_final\wrappers\p11ab`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11ac_final\wrappers\p11l`

## Actual execution evidence excerpt
- `build\p11ac_final\check_p11ac_phasea_surface.log`
- `PASS: check_p11ac_phasea_surface` (pre/post)
- `build\p11ac_final\p11ac\run.log`
- `STREAM_ORDER PASS`
- `MEMORY_ORDER PASS`
- `SINGLE_READ_X_REUSE PASS`
- `EXACT_SCR_KV_COMPARE PASS`
- `NO_SPURIOUS_WRITE PASS`
- `SOURCE_PRESERVATION PASS`
- `fallback_taken = false`
- `MAINLINE_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `PASS: tb_kv_build_stream_stage_p11ac`
- `PASS: run_p11ac_phasea_top_managed`
- non-regression:
- `PASS: run_p11r_compile_prep`
- `PASS: run_p11s_compile_prep_family`
- `PASS: run_p11z_runtime_probe`
- `PASS: run_p11aa_qkv_loadw_bridge`
- `PASS: run_p11ab_kv_build_stage`
- `PASS: run_p11l_local_regression`

## Design-mainline hookup explanation
- Before this finalize step:
- AC landed helper/proof path existed (`AttnPhaseATopManagedKv` + p11ac TB), but real design mainline in `Top.h`/`AttnLayer0.h` did not execute through it.
- After this finalize step:
- `Top.h` now wires the target-layer path through:
- `run_transformer_layer_loop -> run_p11ac_layer0_top_managed_kv -> attn_phasea_top_managed_kv_mainline`
- `TransformerLayer` now propagates `kv_prebuilt_from_top_managed` (default false; backward-compatible).
- `AttnLayer0` now uses `kv_prebuilt_from_top_managed` to skip only actual `K/V` materialization work while preserving non-K/V stage side effects and unchanged Q behavior.
- The p11ac completion verdict now requires runtime proof that the new mainline path was taken and fallback was not taken.

## Scope notes
- local-only progress is valid.
- not Catapult closure.
- not SCVerify closure.

## Explicit incomplete items
- Catapult closure: not part of this task and not claimed.
- SCVerify closure: not part of this task and not claimed.
- AD/AE/AF impl: not started in this task.
