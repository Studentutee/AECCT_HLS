## Summary
- P00-011U extends the accepted P00-011T compile-time shape SSOT into a QKV payload-metadata expectation SSOT bridge.
- Scope remains local-only and additive: no algorithm change, no public signature change, no top contract change.
- This milestone is explicitly not Catapult closure and not SCVerify closure.

## Scope
- In scope:
  - Consolidate QKV payload-metadata expectations (`num_weights`, `payload_words`, `last_word_valid_count`) on a single compile-time source chain in `TernaryLiveQkvLeafKernelShapeConfig.h`.
  - Re-point QKV metadata guard compares in `TernaryLiveQkvLeafKernel.h` and `AttnLayer0.h` to `kQkvCtExpected...` constants and shared SSOT-derived values.
  - Add `scripts/check_qkv_payload_metadata_ssot.ps1` with pre/post phase checks.
  - Hook the new checker into existing `scripts/local/run_p11l_local_regression.ps1` pre/post checkpoints.
  - Sync governance docs and preserve accepted meaning for P00-011Q/R/S/T.
- Out of scope:
  - Catapult run, SCVerify run, algorithm rewrite, quant rewrite, public contract redesign, top interface redesign, generator/live-loader closure.

## Files changed
- src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h
- src/blocks/TernaryLiveQkvLeafKernel.h
- src/blocks/AttnLayer0.h
- scripts/check_qkv_shape_ssot.ps1
- scripts/check_qkv_payload_metadata_ssot.ps1
- scripts/local/run_p11l_local_regression.ps1
- docs/process/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md
- docs/process/PROJECT_STATUS_zhTW.txt
- docs/milestones/TRACEABILITY_MAP_v12.1.md
- docs/milestones/CLOSURE_MATRIX_v12.1.md
- docs/milestones/P00-011U_report.md

## Exact commands executed
1. `New-Item -ItemType Directory -Force -Path build\p11u > $null`
2. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11u -Phase pre`
3. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11u -Phase pre`
4. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11u -Phase pre`
5. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11u -Phase pre`
6. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11u -Phase pre`
7. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11u *> build\p11u\run_p11r_wrapper.log`
8. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11u *> build\p11u\run_p11s_wrapper.log`
9. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11u *> build\p11u\run_p11l_regression.log`
10. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11u -Phase post`
11. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11u -Phase post`
12. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11u -Phase post`
13. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11u -Phase post`
14. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11u -Phase post`

## Actual execution evidence excerpt
- From `build\p11u\check_qkv_payload_metadata_ssot.log`:
  - `===== check_qkv_payload_metadata_ssot phase=pre =====`
  - `PASS: check_qkv_payload_metadata_ssot`
  - `===== check_qkv_payload_metadata_ssot phase=post =====`
  - `PASS: check_qkv_payload_metadata_ssot`
- From `build\p11u\run_p11r_compile_prep.log`:
  - `PASS: tb_ternary_live_leaf_top_compile_prep_p11r`
  - `PASS: run_p11r_compile_prep`
- From `build\p11u\run_p11s_compile_prep_family.log`:
  - `PASS: tb_ternary_live_leaf_top_compile_prep_family_p11s`
  - `PASS: run_p11s_compile_prep_family`
- From `build\p11u\run_p11l_regression.log`:
  - `[p11p][PRECHECK] pass check_qkv_payload_metadata_ssot_pre`
  - `[p11p][PRECHECK] pass check_qkv_payload_metadata_ssot_post`
  - `PASS: run_p11l_local_regression`
- From summary artifacts:
  - `build\p11u\check_qkv_payload_metadata_ssot_summary.txt`: `status: PASS`, `phase: post`
  - `build\p11u\check_handoff_surface_summary.txt`: `status: PASS`, `phase: post`
  - `build\p11u\check_compile_prep_surface_summary.txt`: `status: PASS`, `phase: post`
  - `build\p11u\check_compile_prep_family_surface_summary.txt`: `status: PASS`, `phase: post`
  - `build\p11u\check_qkv_shape_ssot_summary.txt`: `status: PASS`, `phase: post`

## Result / verdict wording
- P00-011U is accepted as a local-only QKV payload-metadata SSOT bridge milestone.
- P00-011Q handoff freeze remains authoritative.
- P00-011R WQ compile-prep probe remains valid baseline.
- P00-011S WK/WV family compile-prep expansion remains valid baseline.
- P00-011T QKV shape SSOT consolidation remains valid baseline.
- This result is not Catapult closure and not SCVerify closure.

## Limitations
- local-only evidence scope only; no Catapult run; no SCVerify run.
- no runtime-variable top interface migration; no live loader / generator closure in this task.
- Authority fallback used: `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.1.txt` was not present, so `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.txt` was adopted.
- Report evidence excerpts must correspond to real logs under `build\p11u\` for final acceptance.

## Why useful for later payload-metadata SSOT bridge but not closure
- The bridge removes payload-metadata expectation drift risk by consolidating the active source chain and adding static drift checks.
- This improves handoff readiness for later Catapult-facing work while remaining local-only.
- It does not prove Catapult compile/run closure, SCVerify closure, full numeric closure, or full live migration closure.
