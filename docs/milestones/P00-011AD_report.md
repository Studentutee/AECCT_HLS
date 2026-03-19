# P00-011AD Report - Q Path Mainline Wiring (Local-Only)

## Acceptance Summary

### Verdict
`P00-011AD` is accepted as a **local-only design-mainline wiring step** for Q path integration.

Acceptance requires real execution through the new Top-managed mainline path with explicit anti-fallback evidence:
- `MAINLINE_Q_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `fallback_taken = false`

### What is accepted in this round
- `Top.h` wires AD path through real mainline path:
  - `run_transformer_layer_loop`
  - `run_p11ad_layer0_top_managed_q`
  - `attn_phasea_top_managed_q_mainline`
- `TransformerLayer` propagates `q_prebuilt_from_top_managed` with default `false`
- `AttnLayer0` consumes that hook so that, when Top-managed Q is already prepared, only actual Q materialization work is skipped while required non-Q bookkeeping and downstream flow remain intact
- `AttnTopManagedPacketKind` is extended minimally with `ATTN_PKT_WQ` and `ATTN_PKT_Q` only; packet fields remain unchanged

### Evidence basis
Key execution evidence from `build\p11ad_impl\p11ad\run.log`:
- `Q_PATH_MAINLINE PASS`
- `Q_EXPECTED_COMPARE PASS`
- `Q_TARGET_SPAN_WRITE PASS`
- `NO_SPURIOUS_WRITE PASS`
- `SOURCE_PRESERVATION PASS`
- `MAINLINE_Q_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `fallback_taken = false`
- `PASS: tb_q_path_impl_p11ad`
- `PASS: run_p11ad_impl_q_path`

Key surface evidence from `build\p11ad_impl\check_p11ad_impl_surface.log`:
- `PASS: check_p11ad_impl_surface` (pre/post)

Additional local non-regression evidence:
- `PASS: run_p11r_compile_prep`
- `PASS: run_p11s_compile_prep_family`
- `PASS: run_p11z_runtime_probe`
- `PASS: run_p11aa_qkv_loadw_bridge`
- `PASS: run_p11ab_kv_build_stage`
- `PASS: run_p11ac_phasea_top_managed`
- `PASS: run_p11l_local_regression`

### Scope boundary / non-claims
This acceptance is intentionally limited to:
- local-only evidence
- design-mainline Q-path wiring
- requested local non-regression chain

This acceptance does **not** claim:
- Catapult closure
- SCVerify closure
- AE / AF implementation scope expansion
- full runtime closure
- full numeric closure
- full algorithm closure

## Files changed
- `include/AttnTopManagedPackets.h`
- `src/blocks/AttnPhaseATopManagedQ.h`
- `src/Top.h`
- `src/blocks/TransformerLayer.h`
- `src/blocks/AttnLayer0.h`
- `tb/tb_q_path_impl_p11ad.cpp`
- `scripts/check_p11ad_impl_surface.ps1`
- `scripts/local/run_p11ad_impl_q_path.ps1`
- `docs/handoff/P11_AC_AF_INTERFACE_FREEZE.md`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/milestones/P00-011AD_report.md`
- `docs/milestones/P00-011AC_AF_umbrella_report.md`

## Exact commands executed
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ad_impl_surface.ps1 -OutDir build\p11ad_impl -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ad_impl_q_path.ps1 -BuildDir build\p11ad_impl\p11ad`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ad_impl_surface.ps1 -OutDir build\p11ad_impl -Phase post`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11ad_impl\wrappers\p11r`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11ad_impl\wrappers\p11s`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11z_runtime_probe.ps1 -BuildDir build\p11ad_impl\wrappers\p11z`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11aa_qkv_loadw_bridge.ps1 -BuildDir build\p11ad_impl\wrappers\p11aa`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ab_kv_build_stage.ps1 -BuildDir build\p11ad_impl\wrappers\p11ab`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ac_phasea_top_managed.ps1 -BuildDir build\p11ad_impl\wrappers\p11ac`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11ad_impl\wrappers\p11l`

## Design-mainline hookup explanation
- Before this step, AD existed as scaffold-only prep artifacts and did not execute through real Top mainline.
- After this step:
  - `Top.h` wires the target-layer path through `run_transformer_layer_loop -> run_p11ad_layer0_top_managed_q -> attn_phasea_top_managed_q_mainline`.
  - `TransformerLayer` propagates `q_prebuilt_from_top_managed` (default `false`, backward-compatible).
  - `AttnLayer0` uses `q_prebuilt_from_top_managed` to skip only actual Q materialization work, while preserving required setup/bookkeeping side effects and keeping K/V behavior unchanged.
- Completion acceptance requires proof that the new mainline path was executed and fallback was not taken.

## Scope notes
- local-only
- not Catapult closure
- not SCVerify closure

## Explicit incomplete items
- None within AD-impl scope.

