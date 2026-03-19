# P00-011T Report - QKV Shape SSOT Consolidation (Compile-Time SSOT + Runtime Validation Clarification)

## Summary
- `P00-011T` delivers `QKV shape SSOT consolidation` for the current QKV live-cut / compile-prep chain.
- The consolidation introduces one `compile-time shape SSOT` definition point and clarifies `runtime validation only` semantics.
- Accepted `P00-011Q`/`P00-011R`/`P00-011S` meanings are preserved.
- This task is `not Catapult closure` and `not SCVerify closure`.

## Scope
- In scope:
- add dedicated compile-time SSOT header for WQ/WK/WV rows/cols/payload words
- re-point `TernaryLiveQkvLeafKernel.h` and `TernaryLiveQkvLeafKernelCatapultPrepTop.h` to consume SSOT
- keep runtime metadata checks but clarify runtime-vs-compile-time boundary wording
- add `check_qkv_shape_ssot.ps1` with pre/post checks
- governance sync and task-local report
- Out of scope:
- no Catapult run
- no SCVerify run
- no runtime-variable top interface rewrite
- no algorithm/quant/public-contract changes

## Files changed
- `src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h`
- `src/blocks/TernaryLiveQkvLeafKernel.h`
- `src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h`
- `scripts/check_qkv_shape_ssot.ps1`
- `docs/handoff/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/milestones/P00-011T_report.md`

## Exact commands executed
- `New-Item -ItemType Directory -Force -Path build\p11t > $null`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11t -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11t -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11t -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11t -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11t *> build\p11t\run_p11r_wrapper.log`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11t *> build\p11t\run_p11s_wrapper.log`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11t *> build\p11t\run_p11p_regression.log`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11t -Phase post`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11t -Phase post`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11t -Phase post`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11t -Phase post`

## Actual execution evidence excerpt
- `build\p11t\check_handoff_surface.log`
- `===== check_handoff_surface phase=pre =====`
- `PASS: check_handoff_surface`
- `===== check_handoff_surface phase=post =====`
- `PASS: check_handoff_surface`
- `build\p11t\check_compile_prep_surface.log`
- `===== check_compile_prep_surface phase=pre =====`
- `PASS: check_compile_prep_surface`
- `===== check_compile_prep_surface phase=post =====`
- `PASS: check_compile_prep_surface`
- `build\p11t\check_compile_prep_family_surface.log`
- `===== check_compile_prep_family_surface phase=pre =====`
- `PASS: check_compile_prep_family_surface`
- `===== check_compile_prep_family_surface phase=post =====`
- `PASS: check_compile_prep_family_surface`
- `build\p11t\check_qkv_shape_ssot.log`
- `===== check_qkv_shape_ssot phase=pre =====`
- `PASS: check_qkv_shape_ssot`
- `===== check_qkv_shape_ssot phase=post =====`
- `PASS: check_qkv_shape_ssot`
- `build\p11t\run_p11r_compile_prep.log`
- `PASS: tb_ternary_live_leaf_top_compile_prep_p11r`
- `PASS: run_p11r_compile_prep`
- `build\p11t\run_p11s_compile_prep_family.log`
- `PASS: tb_ternary_live_leaf_top_compile_prep_family_p11s`
- `PASS: run_p11s_compile_prep_family`
- `build\p11t\run_p11p_regression.log`
- `PASS: check_repo_hygiene`
- `PASS: check_design_purity`
- `PASS: check_interface_lock`
- `PASS: check_macro_hygiene`
- `PASS: run_p11l_local_regression`

## Result / verdict wording
- `P00-011T` is accepted as `QKV shape SSOT consolidation` with `compile-time shape SSOT` and `runtime validation only` semantics.
- `P00-011Q handoff freeze remains authoritative`.
- `P00-011R WQ compile-prep probe remains valid baseline`.
- `P00-011S WK/WV family compile-prep expansion remains valid baseline`.
- Scope remains local compiler evidence only.
- This result is `not Catapult closure`.
- This result is `not SCVerify closure`.

## Limitations
- Catapult and SCVerify execution remain deferred.
- This task does not claim runtime-variable top interfaces, full tile/generalization implementation, full runtime closure, or full numeric closure.
- One intermediate `check_compile_prep_surface` post run failed on wording (`P00-011Q freeze boundary remains authoritative`), then passed after minimal governance wording sync.
- Authority fallback used:
- requested `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.1.txt` is not present
- adopted `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.txt` as nearest repo-available equivalent

## Why useful for later tile/generalization work but not closure
- The compile-time SSOT removes scattered local shape freezes and gives one authoritative shape definition point for current QKV compile-prep surfaces.
- The runtime-validation boundary is now explicit: runtime metadata/config validates against compile-time supported shape.
- This improves readiness for later tiling/generalization work, but it is still non-closure work and does not prove Catapult/SCVerify closure.

