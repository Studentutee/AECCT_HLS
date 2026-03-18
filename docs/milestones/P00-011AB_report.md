## Summary
- `P00-011AB` adds a local-only Phase-A K/V materialization staging proof for `X_WORK + Wk/Wv -> SCR_K/SCR_V`.
- The milestone proves strict stream order, strict memory-access order, single-read X reuse, exact span compare, `X_WORK` unchanged, and no-spurious-write behavior.
- This milestone is `local-only`, `not Catapult closure`, `not SCVerify closure`, and `Phase-A K/V materialization staging proof only`.

## Scope
- In scope:
- Add one TB-local staging proof testbench and one local runner.
- Prove `(token,d_tile)` ordering and access counts.
- Prove exact writeback to `SCR_K[token,d_tile]` and `SCR_V[token,d_tile]`.
- Preserve accepted continuity from `P00-011Q/R/S/T/U/V/W/X/Y/Z/AA`.
- Out of scope:
- no Q path.
- no QK path.
- no softmax.
- no attention output.
- no full runtime closure.
- no Catapult run.
- no SCVerify run.

## Files changed
- `tb/tb_kv_build_stream_stage_p11ab.cpp`
- `scripts/local/run_p11ab_kv_build_stage.ps1`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/milestones/P00-011AB_report.md`

## Exact commands executed
1. `New-Item -ItemType Directory -Force -Path build\p11ab\checks,build\p11ab\wrappers,build\p11ab\p11ab > $null`
2. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11ab\checks -Phase pre`
3. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11ab\checks -Phase pre`
4. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11ab\checks -Phase pre`
5. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11ab\checks -Phase pre`
6. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11ab\checks -Phase pre`
7. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11ab\checks -Phase pre`
8. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_artifact_continuity.ps1 -OutDir build\p11ab\checks -Phase pre`
9. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_consumer_semantics.ps1 -OutDir build\p11ab\checks -Phase pre`
10. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_runtime_handoff_continuity.ps1 -OutDir build\p11ab\checks -Phase pre`
11. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11ab\wrappers\p11r *> build\p11ab\wrappers\run_p11r_wrapper.log`
12. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11ab\wrappers\p11s *> build\p11ab\wrappers\run_p11s_wrapper.log`
13. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11z_runtime_probe.ps1 -BuildDir build\p11ab\wrappers\p11z *> build\p11ab\wrappers\run_p11z_wrapper.log`
14. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11aa_qkv_loadw_bridge.ps1 -BuildDir build\p11ab\wrappers\p11aa *> build\p11ab\wrappers\run_p11aa_wrapper.log`
15. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ab_kv_build_stage.ps1 -BuildDir build\p11ab\p11ab *> build\p11ab\wrappers\run_p11ab_wrapper.log`
16. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11ab\wrappers\p11l *> build\p11ab\wrappers\run_p11l_regression.log`
17. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11ab\checks -Phase post`
18. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11ab\checks -Phase post`
19. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11ab\checks -Phase post`
20. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11ab\checks -Phase post`
21. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11ab\checks -Phase post`
22. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11ab\checks -Phase post`
23. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_artifact_continuity.ps1 -OutDir build\p11ab\checks -Phase post`
24. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_consumer_semantics.ps1 -OutDir build\p11ab\checks -Phase post`
25. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_runtime_handoff_continuity.ps1 -OutDir build\p11ab\checks -Phase post`

## Actual execution evidence excerpt
- `build\p11ab\p11ab\run.log`
  - `[p11ab][FIXTURE][PASS] derived d_tile words=32 from accepted compile-time shape SSOT`
  - `[p11ab][STREAM_ORDER][PASS] token=0 d_tile=0 sequence=X->Wk->Wv->K->V`
  - `[p11ab][STREAM_ORDER][PASS] token=1 d_tile=0 sequence=X->Wk->Wv->K->V`
  - `[p11ab][MEM_ORDER][PASS] token=0 d_tile=0 sequence=read X_WORK->read W_REGION(Wk)->read W_REGION(Wv)->write SCR_K->write SCR_V`
  - `[p11ab][MEM_ORDER][PASS] token=1 d_tile=0 sequence=read X_WORK->read W_REGION(Wk)->read W_REGION(Wv)->write SCR_K->write SCR_V`
  - `[p11ab][ACCESS_COUNT][PASS] token=0 d_tile=0 X=1 Wk=1 Wv=1 SCR_K=1 SCR_V=1`
  - `[p11ab][ACCESS_COUNT][PASS] token=1 d_tile=0 X=1 Wk=1 Wv=1 SCR_K=1 SCR_V=1`
  - `[p11ab][ACCESS_TOTAL] x_reads=2 wk_reads=2 wv_reads=2 scr_k_writes=2 scr_v_writes=2`
  - `[p11ab][X_REUSE][PASS] token=0 d_tile=0 same X tile retained/reused for K and V; duplicate X_WORK read=0`
  - `[p11ab][X_REUSE][PASS] token=1 d_tile=0 same X tile retained/reused for K and V; duplicate X_WORK read=0`
  - `[p11ab][SPAN][PASS] token_index=0 d_tile_index=0 kind=K target_offset=4800 compare_length=32 result=PASS`
  - `[p11ab][SPAN][PASS] token_index=0 d_tile_index=0 kind=V target_offset=7200 compare_length=32 result=PASS`
  - `[p11ab][SPAN][PASS] token_index=1 d_tile_index=0 kind=K target_offset=4832 compare_length=32 result=PASS`
  - `[p11ab][SPAN][PASS] token_index=1 d_tile_index=0 kind=V target_offset=7232 compare_length=32 result=PASS`
  - `[p11ab][X_WORK][PASS] source span unchanged across Phase-A staging work-units`
  - `[p11ab][WRITE_GUARD][PASS] no writes to X_WORK/W_REGION/unrelated scratch; only SCR_K[token,d_tile] and SCR_V[token,d_tile] writes observed`
  - `PASS: tb_kv_build_stream_stage_p11ab`
  - `PASS: run_p11ab_kv_build_stage`
- `build\p11ab\wrappers\run_p11ab_wrapper.log`
  - `PASS: run_p11ab_kv_build_stage`
- `build\p11ab\wrappers\run_p11r_wrapper.log`
  - `PASS: run_p11r_compile_prep`
- `build\p11ab\wrappers\run_p11s_wrapper.log`
  - `PASS: run_p11s_compile_prep_family`
- `build\p11ab\wrappers\run_p11z_wrapper.log`
  - `PASS: run_p11z_runtime_probe`
- `build\p11ab\wrappers\run_p11aa_wrapper.log`
  - `PASS: run_p11aa_qkv_loadw_bridge`
- `build\p11ab\wrappers\run_p11l_regression.log`
  - `[p11p][PRECHECK] pass check_qkv_runtime_handoff_continuity_pre`
  - `[p11p][PRECHECK] pass check_qkv_runtime_handoff_continuity_post`
  - `PASS: run_p11l_local_regression`
- `build\p11ab\checks\check_handoff_surface_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11ab\checks\check_compile_prep_surface_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11ab\checks\check_compile_prep_family_surface_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11ab\checks\check_qkv_shape_ssot_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11ab\checks\check_qkv_payload_metadata_ssot_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11ab\checks\check_qkv_weightstreamorder_continuity_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11ab\checks\check_qkv_export_artifact_continuity_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11ab\checks\check_qkv_export_consumer_semantics_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11ab\checks\check_qkv_runtime_handoff_continuity_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build\p11ab\p11ab\verdict.txt`
  - `scope: local-only`
  - `closure: not Catapult closure; not SCVerify closure`

## Result / verdict wording
- `P00-011AB` is accepted as a `local-only` Phase-A K/V materialization staging proof milestone.
- This milestone is `not Catapult closure` and `not SCVerify closure`.
- Scope is limited to `Phase-A K/V materialization staging proof only`.
- This milestone explicitly does not claim Q path, QK/score path, softmax, attention output, or full runtime closure.

## Limitations
- Catapult and SCVerify are intentionally deferred.
- Top ownership/arbitration of shared SRAM remains unchanged.
- Design/runtime weight path remains `SET_W_BASE -> LOAD_W -> SRAM -> consumer`.
- TB/reference-side `weights.h + kQuantLinearMeta + kParamMeta` authority is limited to testbench/reference usage.
