## Summary
- `P00-011Z` adds a local-only, read-only runtime consume probe for `L0_WQ/L0_WK/L0_WV`.
- The probe consumes `gen/ternary_p11c_export.json` and validates continuity against authoritative `kQuantLinearMeta` in `gen/include/WeightStreamOrder.h`.
- This milestone remains `local-only`, `not Catapult closure`, and `not SCVerify closure`.

## Scope
- In scope:
- Add runtime probe TB + local runner.
- Minimal hook in existing `run_p11l_local_regression.ps1` (status/fail propagation only).
- Governance sync + task-local report.
- Out of scope:
- no Catapult run.
- no SCVerify run.
- no algorithm/public-signature/top-contract change.
- no live-loader migration.
- no generator regeneration.
- no JSON rewrite.

## Files changed
- `tb/tb_ternary_qkv_runtime_probe_p11z.cpp`
- `scripts/local/run_p11z_runtime_probe.ps1`
- `scripts/local/run_p11l_local_regression.ps1`
- `docs/handoff/P11_LOCAL_TO_CATAPULT_HANDOFF_RULES.md`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/milestones/P00-011Z_report.md`

## Exact commands executed
1. `New-Item -ItemType Directory -Force -Path build\p11z > $null`
2. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11z -Phase pre`
3. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11z -Phase pre`
4. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11z -Phase pre`
5. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11z -Phase pre`
6. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11z -Phase pre`
7. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11z -Phase pre`
8. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_artifact_continuity.ps1 -OutDir build\p11z -Phase pre`
9. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_consumer_semantics.ps1 -OutDir build\p11z -Phase pre`
10. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_runtime_handoff_continuity.ps1 -OutDir build\p11z -Phase pre`
11. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11z -Phase pre`
12. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11r_compile_prep.ps1 -BuildDir build\p11z *> build\p11z\run_p11r_wrapper.log`
13. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11s_compile_prep_family.ps1 -BuildDir build\p11z *> build\p11z\run_p11s_wrapper.log`
14. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11z_runtime_probe.ps1 -BuildDir build\p11z *> build\p11z\run_p11z_wrapper.log`
15. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11z *> build\p11z\run_p11l_regression.log`
16. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_handoff_surface.ps1 -OutDir build\p11z -Phase post`
17. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_surface.ps1 -OutDir build\p11z -Phase post`
18. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_compile_prep_family_surface.ps1 -OutDir build\p11z -Phase post`
19. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_shape_ssot.ps1 -OutDir build\p11z -Phase post`
20. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11z -Phase post`
21. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11z -Phase post`
22. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_artifact_continuity.ps1 -OutDir build\p11z -Phase post`
23. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_consumer_semantics.ps1 -OutDir build\p11z -Phase post`
24. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_runtime_handoff_continuity.ps1 -OutDir build\p11z -Phase post`
25. `if (Test-Path docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.1.txt) { 'v3.1-present' } elseif (Test-Path docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.txt) { 'v3.1-missing-v3-present' } else { 'guide-missing' }`
26. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_payload_metadata_ssot.ps1 -OutDir build\p11z -Phase post`
27. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_weightstreamorder_continuity.ps1 -OutDir build\p11z -Phase post`
28. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_artifact_continuity.ps1 -OutDir build\p11z -Phase post`
29. `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_qkv_export_consumer_semantics.ps1 -OutDir build\p11z -Phase post`

## Actual execution evidence excerpt
- `build\p11z\run_p11z_runtime_probe.log`
  - `[p11z][PASS] matrix=L0_WQ rows=32 cols=32 num_weights=1024 payload_words_2b=64 last_word_valid_count=16 weight_param_id=24 inv_sw_param_id=26`
  - `[p11z][PASS] matrix=L0_WK rows=32 cols=32 num_weights=1024 payload_words_2b=64 last_word_valid_count=16 weight_param_id=27 inv_sw_param_id=29`
  - `[p11z][PASS] matrix=L0_WV rows=32 cols=32 num_weights=1024 payload_words_2b=64 last_word_valid_count=16 weight_param_id=30 inv_sw_param_id=32`
  - `PASS: tb_ternary_qkv_runtime_probe_p11z`
  - `PASS: run_p11z_runtime_probe`
- `build\p11z\run_p11l_regression.log`
  - `[p11p][PRECHECK] pass check_qkv_runtime_handoff_continuity_pre`
  - `PASS: run_p11z_runtime_probe`
  - `[p11p][PRECHECK] pass check_qkv_runtime_handoff_continuity_post`
  - `PASS: run_p11l_local_regression`
- `build\p11z\check_qkv_runtime_handoff_continuity.log`
  - `===== check_qkv_runtime_handoff_continuity phase=pre =====`
  - `PASS: check_qkv_runtime_handoff_continuity`
  - `===== check_qkv_runtime_handoff_continuity phase=post =====`
  - `PASS: check_qkv_runtime_handoff_continuity`
- `build\p11z\check_qkv_runtime_handoff_continuity_summary.txt`
  - `status: PASS`
  - `phase: post`

## Result / verdict wording
- `P00-011Z` is accepted as a local-only runtime-consume probe milestone.
- Accepted baseline continuity for `P00-011Q/R/S/T/U/V/W/X/Y` is retained.
- This milestone is `not Catapult closure` and `not SCVerify closure`.

## Limitations
- Catapult and SCVerify remain deferred.
- No live-loader migration or runtime execution-path closure is claimed.
- No generator regeneration and no JSON rewrite are included.
- Probe supports payload-word array keys `payload_hex_words` (primary) and `payload_words_hex` (fallback); if absent, probe fails explicitly as schema limitation/mismatch and does not synthesize a second source.
- A transient post-check log file lock was observed during one full-sequence run; affected post checkers were re-run and final post summaries are PASS.
- Authority fallback occurred: `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.1.txt` is absent; adopted `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.txt`.

## Why useful for later local runtime consume work but not closure
- The probe demonstrates matrix_id-driven runtime-facing metadata consumption readiness for QKV on repo-tracked artifacts and authoritative metadata.
- It does not perform Catapult/SCVerify or full runtime execution closure, so it is not formal closure.

