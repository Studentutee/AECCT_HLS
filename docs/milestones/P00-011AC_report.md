# P00-011AC Report - Phase-A Top-Managed K/V Design-Mainline Wiring Finalize (Local-Only)

## Acceptance Summary

### Verdict
`P00-011AC` is **accepted** as **local-only design-mainline wiring finalize**.

This acceptance is based on execution evidence showing that the AC path is no longer helper/proof-only, but is now wired into the real design-side Top mainline with explicit anti-fallback confirmation:
- `MAINLINE_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `fallback_taken = false`

### What is accepted in this round
The accepted scope of this round is:
- `Top.h` now wires the AC path through the real mainline path:
  - `run_transformer_layer_loop`
  - `run_p11ac_layer0_top_managed_kv`
  - `attn_phasea_top_managed_kv_mainline`
- `TransformerLayer` propagates `kv_prebuilt_from_top_managed` with default `false`
- `AttnLayer0` consumes that hook so that, when Top-managed K/V is already prepared, only K/V materialization work is skipped, while required Q behavior and stage flow remain on the design path

This means AC has crossed the boundary from helper/proof-only behavior into actual design-mainline hookup.

### Evidence basis
Acceptance is supported by:
- explicit file-level implementation changes
- exact command transcript for pre / main / post checks
- pre/post surface check PASS
- main test PASS with explicit anti-fallback proof
- wrapper-family non-regression PASS

Key execution evidence from `build\p11ac_final\p11ac\run.log`:
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

Key surface evidence from `build\p11ac_final\check_p11ac_phasea_surface.log`:
- `PASS: check_p11ac_phasea_surface` (pre/post)

Additional non-regression evidence:
- `PASS: run_p11r_compile_prep`
- `PASS: run_p11s_compile_prep_family`
- `PASS: run_p11z_runtime_probe`
- `PASS: run_p11aa_qkv_loadw_bridge`
- `PASS: run_p11ab_kv_build_stage`
- `PASS: run_p11l_local_regression`

### Scope boundary / non-claims
This acceptance is intentionally limited to:
- **local-only evidence**
- **design-mainline wiring finalize for AC scope**
- **non-regression within the requested local wrapper family**

This acceptance does **not** claim:
- Catapult closure
- SCVerify closure
- full runtime closure
- full numeric closure
- full algorithm closure
- AD / AE / AF implementation scope expansion

### Final acceptance statement
`P00-011AC` is accepted as a **local-only, evidence-backed design-mainline finalize step**.

The important acceptance meaning is not merely that the AC helper exists, but that the AC path is now demonstrably exercised through the real Top mainline, with explicit proof that the fallback path was not taken.

## Files changed
- `src/Top.h`
- `src/blocks/TransformerLayer.h`
- `src/blocks/AttnLayer0.h`
- `src/blocks/AttnPhaseATopManagedKv.h`
- `tb/tb_kv_build_stream_stage_p11ac.cpp`
- `scripts/check_p11ac_phasea_surface.ps1`
- `scripts/local/run_p11ac_phasea_top_managed.ps1`
- `docs/handoff/P11_AC_AF_INTERFACE_FREEZE.md`
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

## Design-mainline hookup explanation
- Before this finalize step, AC had landed helper/proof behavior (`AttnPhaseATopManagedKv` + p11ac TB), but the real design mainline in `Top.h` / `AttnLayer0.h` did not execute through it.
- After this finalize step:
  - `Top.h` wires the target-layer path through `run_transformer_layer_loop -> run_p11ac_layer0_top_managed_kv -> attn_phasea_top_managed_kv_mainline`
  - `TransformerLayer` propagates `kv_prebuilt_from_top_managed` (default `false`; backward-compatible)
  - `AttnLayer0` uses `kv_prebuilt_from_top_managed` to skip only actual K/V materialization work while preserving non-K/V stage side effects and unchanged Q behavior
- Completion acceptance now requires explicit runtime proof that the new mainline path was taken and that fallback was not taken.

