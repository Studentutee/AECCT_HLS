## Summary
- `P00-011AA` adds a local-only formal runtime-path bridge probe for `L0_WQ/L0_WK/L0_WV` over `SET_W_BASE -> LOAD_W -> READ_MEM`.
- Negative coverage is split into two explicit layers:
- A. formal-path negatives (`LOAD_W` without base, incomplete length behavior, follow-up command rejection during incomplete load).
- B. probe-side semantic validation-only negatives (non-zero tail padding, illegal ternary code `10`).
- This milestone remains `local-only`, `not Catapult closure`, and `not SCVerify closure`.

## Scope
- In scope:
- Add one formal bridge TB and one local runner.
- Verify exact-word readback continuity for payload and `inv_sw` spans per matrix.
- Preserve accepted authority continuity for `P00-011Q/R/S/T/U/V/W/X/Y/Z`.
- Minimal governance sync and task-local report.
- Out of scope:
- no Catapult run.
- no SCVerify run.
- no attention math closure (`QK` score / softmax / attention_out).
- no public contract / Top opcode semantics / algorithm / quant change.
- no block-graph or dispatcher redesign.

## Files changed
- `tb/tb_qkv_formal_loadw_bridge_p11aa.cpp`
- `scripts/local/run_p11aa_qkv_loadw_bridge.ps1`
- `docs/handoff/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/milestones/P00-011AA_report.md`

## Exact commands executed
1. `New-Item -ItemType Directory -Force -Path build\p11aa\checks,build\p11aa\wrappers,build\p11aa\p11aa > $null`
2. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11aa\checks -Phase pre`
3. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11aa\checks -Phase pre`
4. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11aa\checks -Phase pre`
5. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11aa\checks -Phase pre`
6. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11aa\checks -Phase pre`
7. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11aa\checks -Phase pre`
8. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_artifact_continuity.ps1 -OutDir build\p11aa\checks -Phase pre`
9. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_consumer_semantics.ps1 -OutDir build\p11aa\checks -Phase pre`
10. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_runtime_handoff_continuity.ps1 -OutDir build\p11aa\checks -Phase pre`
11. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11aa\wrappers\p11r *> build\p11aa\wrappers\run_p11r_wrapper.log`
12. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11aa\wrappers\p11s *> build\p11aa\wrappers\run_p11s_wrapper.log`
13. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11z_runtime_probe.ps1 -BuildDir build\p11aa\wrappers\p11z *> build\p11aa\wrappers\run_p11z_wrapper.log`
14. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11aa_qkv_loadw_bridge.ps1 -BuildDir build\p11aa\p11aa *> build\p11aa\wrappers\run_p11aa_wrapper.log`
15. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11aa\wrappers\p11l *> build\p11aa\wrappers\run_p11l_regression.log`
16. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11aa\checks -Phase post`
17. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11aa\checks -Phase post`
18. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11aa\checks -Phase post`
19. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11aa\checks -Phase post`
20. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11aa\checks -Phase post`
21. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11aa\checks -Phase post`
22. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_artifact_continuity.ps1 -OutDir build\p11aa\checks -Phase post`
23. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_consumer_semantics.ps1 -OutDir build\p11aa\checks -Phase post`
24. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_runtime_handoff_continuity.ps1 -OutDir build\p11aa\checks -Phase post`
25. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11aa\wrappers\p11l *> build\p11aa\wrappers\run_p11l_regression.log`
26. `if (Test-Path docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.1.txt) { 'v3.1-present' } elseif (Test-Path docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.txt) { 'v3.1-missing-v3-present' } else { 'guide-missing' }`

## Actual execution evidence excerpt
- `build\p11aa\p11aa\run.log`
  - `[p11aa][probe_semantic] probe-side validation only; Top formal loader remains transport-only for PARAM ingest.`
  - `[p11aa][probe_semantic][PASS] non-zero tail padding rejected by probe validator (pre-LOAD_W)`
  - `[p11aa][probe_semantic][PASS] illegal ternary code 10 rejected by probe validator (pre-LOAD_W)`
  - `[p11aa][formal_negative][PASS] LOAD_W without SET_W_BASE rejected`
  - `[p11aa][formal_negative][PASS] no DONE before full expected length during incomplete LOAD_W`
  - `[p11aa][formal_negative][PASS] follow-up command rejected while incomplete LOAD_W active`
  - `[p11aa][formal_negative][PASS] reset cleanup recovered control path`
  - `[p11aa][SPAN][PASS] matrix_id=L0_WQ kind=payload target_offset=12528 compare_length=64 result=PASS`
  - `[p11aa][SPAN][PASS] matrix_id=L0_WQ kind=inv_sw target_offset=13560 compare_length=8 result=PASS`
  - `[p11aa][SPAN][PASS] matrix_id=L0_WK kind=payload target_offset=13568 compare_length=64 result=PASS`
  - `[p11aa][SPAN][PASS] matrix_id=L0_WK kind=inv_sw target_offset=14600 compare_length=8 result=PASS`
  - `[p11aa][SPAN][PASS] matrix_id=L0_WV kind=payload target_offset=14608 compare_length=64 result=PASS`
  - `[p11aa][SPAN][PASS] matrix_id=L0_WV kind=inv_sw target_offset=15640 compare_length=8 result=PASS`
  - `PASS: tb_qkv_formal_loadw_bridge_p11aa`
  - `PASS: run_p11aa_qkv_loadw_bridge`
- `build\p11aa\wrappers\run_p11aa_wrapper.log`
  - `PASS: run_p11aa_qkv_loadw_bridge`
- `build\p11aa\wrappers\run_p11r_wrapper.log`
  - `PASS: run_p11r_compile_prep`
- `build\p11aa\wrappers\run_p11s_wrapper.log`
  - `PASS: run_p11s_compile_prep_family`
- `build\p11aa\wrappers\run_p11z_wrapper.log`
  - `PASS: run_p11z_runtime_probe`
- `build\p11aa\wrappers\run_p11l_regression.log`
  - `[p11p][PRECHECK] pass check_qkv_runtime_handoff_continuity_pre`
  - `[p11p][PRECHECK] pass check_qkv_runtime_handoff_continuity_post`
  - `PASS: run_p11l_local_regression`
- `build\p11aa\checks\check_handoff_surface_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11aa\checks\check_compile_prep_surface_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11aa\checks\check_compile_prep_family_surface_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11aa\checks\check_qkv_shape_ssot_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11aa\checks\check_qkv_payload_metadata_ssot_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11aa\checks\check_qkv_weightstreamorder_continuity_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11aa\checks\check_qkv_export_artifact_continuity_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11aa\checks\check_qkv_export_consumer_semantics_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11aa\checks\check_qkv_runtime_handoff_continuity_summary.txt`
  - `status: PASS`
  - `phase: post`

## Result / verdict wording
- `P00-011AA` is accepted as a local-only formal runtime-path bridge milestone.
- Top formal-path evidence is established by externally observable responses and exact-word `READ_MEM` roundtrip checks.
- Tail-padding and illegal-code rejects are explicitly probe-side semantic validation only, not Top formal-loader reject semantics.
- This milestone is `not Catapult closure` and `not SCVerify closure`.

## Limitations
- Catapult and SCVerify remain deferred.
- This milestone does not claim full runtime execution closure or attention-path closure.
- Top formal loader remains transport-only for PARAM ingest in this scope.
- A transient file-lock failure was observed on one `run_p11l_local_regression` invocation; the same command was re-run and final evidence is PASS.
- Authority fallback occurred: `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.1.txt` is absent; adopted `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.txt`.

## Why useful for later formal runtime bridge work but not closure
- This milestone proves accepted QKV authority-chain payload/metadata can be consumed through the formal `SET_W_BASE -> LOAD_W -> READ_MEM` runtime path with exact-word roundtrip evidence.
- It does not include Catapult/SCVerify, attention compute closure, or full runtime closure, so it is not formal closure.

