# P00-011N Report — WK/WV Source-Side Integration Slice + Family Unified Dual-Binary Signature Validation (Local-Only)

## 1. Summary
- Completed `P00-011N` in local smoke scope without regressing accepted `P00-011M` behavior.
- Added WK/WV macro-gated source-side split-top path in `AttnLayer0<ATTN_STAGE_QKV>` with strict macro-on fallback order: split-top first, then existing helper path, then existing final fallback only if helper fails.
- Added family unified dual-binary TB (`p11n`) with fixed signature output format and strict compare gate on `WK_SIG/WV_SIG` only.
- Extended `run_p11l_local_regression.ps1` to default `build\p11n`, retain legacy coverage (`p11j/p11k/p11l_b/p11l_c/p11m`), and add `p11n` baseline/macro build+run+signature compare.

## 2. Files changed
- `src/blocks/AttnLayer0.h`
- `tb/tb_ternary_live_family_source_integration_smoke_p11n.cpp`
- `scripts/local/run_p11l_local_regression.ps1`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/P00-011N_report.md`

## 3. exact build commands
1. `New-Item -ItemType Directory -Force -Path build\p11n > $null`
2. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_smoke_p11j.cpp /Fe:build\p11n\tb_ternary_live_leaf_smoke_p11j.exe > build\p11n\build_p11j.log 2>&1`
3. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11k.cpp /Fe:build\p11n\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11n\build_p11k.log 2>&1`
4. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_b.cpp /Fe:build\p11n\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11n\build_p11l_b.log 2>&1`
5. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_c.cpp /Fe:build\p11n\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11n\build_p11l_c.log 2>&1`
6. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_source_integration_smoke_p11m.cpp /Fe:build\p11n\tb_ternary_live_source_integration_smoke_p11m_baseline.exe > build\p11n\build_p11m_baseline.log 2>&1`
7. `cl /nologo /std:c++14 /EHsc /utf-8 /DAECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE=1 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_source_integration_smoke_p11m.cpp /Fe:build\p11n\tb_ternary_live_source_integration_smoke_p11m_macro.exe > build\p11n\build_p11m_macro.log 2>&1`
8. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_family_source_integration_smoke_p11n.cpp /Fe:build\p11n\tb_ternary_live_family_source_integration_smoke_p11n_baseline.exe > build\p11n\build_p11n_baseline.log 2>&1`
9. `cl /nologo /std:c++14 /EHsc /utf-8 /DAECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE=1 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_family_source_integration_smoke_p11n.cpp /Fe:build\p11n\tb_ternary_live_family_source_integration_smoke_p11n_macro.exe > build\p11n\build_p11n_macro.log 2>&1`

## 4. exact run commands
1. `cmd /c build\p11n\tb_ternary_live_leaf_smoke_p11j.exe > build\p11n\run_p11j.log 2>&1`
2. `cmd /c build\p11n\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11n\run_p11k.log 2>&1`
3. `cmd /c build\p11n\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11n\run_p11l_b.log 2>&1`
4. `cmd /c build\p11n\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11n\run_p11l_c.log 2>&1`
5. `cmd /c build\p11n\tb_ternary_live_source_integration_smoke_p11m_baseline.exe > build\p11n\run_p11m_baseline.log 2>&1`
6. `cmd /c build\p11n\tb_ternary_live_source_integration_smoke_p11m_macro.exe > build\p11n\run_p11m_macro.log 2>&1`
7. `cmd /c build\p11n\tb_ternary_live_family_source_integration_smoke_p11n_baseline.exe > build\p11n\run_p11n_baseline.log 2>&1`
8. `cmd /c build\p11n\tb_ternary_live_family_source_integration_smoke_p11n_macro.exe > build\p11n\run_p11n_macro.log 2>&1`

Canonical one-shot:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11n`

## 5. actual execution evidence excerpt
- `build\p11n\run_p11j.log`
  - `PASS: tb_ternary_live_leaf_smoke_p11j`
- `build\p11n\run_p11k.log`
  - `PASS: tb_ternary_live_leaf_top_smoke_p11k`
- `build\p11n\run_p11l_b.log`
  - `L0_WK split-interface top run() exact-match equivalent to direct kernel output`
  - `L0_WV split-interface top run() exact-match equivalent to direct kernel output`
  - `PASS: tb_ternary_live_leaf_top_smoke_p11l_b`
- `build\p11n\run_p11l_c.log`
  - `L0_WQ split-interface top run() exact-match equivalent to direct kernel output`
  - `L0_WK split-interface top run() exact-match equivalent to direct kernel output`
  - `L0_WV split-interface top run() exact-match equivalent to direct kernel output`
  - `PASS: tb_ternary_live_leaf_top_smoke_p11l_c`
- `build\p11n\run_p11m_baseline.log`
  - `[p11m][KV_SIG] K=0x70D576AFA0F67AD3 V=0x70D576AFA0F67AD3`
  - `PASS: tb_ternary_live_source_integration_smoke_p11m`
- `build\p11n\run_p11m_macro.log`
  - `[p11m][KV_SIG] K=0x70D576AFA0F67AD3 V=0x70D576AFA0F67AD3`
  - `[p11m][PASS] source-side WQ integration path exact-match equivalent to split-interface local top`
  - `PASS: tb_ternary_live_source_integration_smoke_p11m`
- `build\p11n\run_p11n_baseline.log`
  - `[p11n][WK_SIG] K=0x325FD9E7650C2B6B`
  - `[p11n][WV_SIG] V=0x9F95E756718961CB`
  - `[p11n][PASS] WK/WV fallback retained under WK/WV-only integration slice`
  - `PASS: tb_ternary_live_family_source_integration_smoke_p11n`
- `build\p11n\run_p11n_macro.log`
  - `[p11n][WK_SIG] K=0x325FD9E7650C2B6B`
  - `[p11n][WV_SIG] V=0x9F95E756718961CB`
  - `[p11n][PASS] source-side WK integration path exact-match equivalent to split-interface local top`
  - `[p11n][PASS] source-side WV integration path exact-match equivalent to split-interface local top`
  - `[p11n][PASS] WK/WV fallback retained under WK/WV-only integration slice`
  - `PASS: tb_ternary_live_family_source_integration_smoke_p11n`

## 6. first blocker
- Initial manual compile probe failed because `build\p11n` did not exist yet; resolved by creating the directory (`New-Item -ItemType Directory -Force -Path build\p11n`) before compilation.

## 7. limitations
- Acceptance scope is local smoke only.
- Catapult / SCVerify is deferred and not part of this closure.
- `WQ_SIG` is emitted as auxiliary info only and is not part of baseline-vs-macro compare gate.
- Formal compare gate is strictly `WK_SIG/WV_SIG` equality between baseline and macro.

## 8. source evidence used
- Governance sources:
  - `docs/process/GOVERNANCE_ENTRYPOINT_zhTW.txt`
  - `docs/process/PROJECT_STATUS_zhTW.txt`
  - `docs/process/AECCT_PROJECT_WORKFLOW_v1_zhTW.txt`
  - `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.txt`
  - `docs/spec/AECCT_HLS_Spec_v12.1_zhTW.txt`
  - `docs/architecture/AECCT_HLS_Architecture_Guide_v12.1_zhTW.txt`
  - `docs/architecture/AECCT_module_interface_rules_v12.1_zhTW.txt`
  - `docs/architecture/AECCT_SramMap_alias_freepoint_rules_v12.1_zhTW.txt`
  - `docs/milestones/AECCT_v12_M0-M24_plan_zhTW.txt`
- Execution evidence logs:
  - `build\p11n\build_*.log`
  - `build\p11n\run_p11j.log`
  - `build\p11n\run_p11k.log`
  - `build\p11n\run_p11l_b.log`
  - `build\p11n\run_p11l_c.log`
  - `build\p11n\run_p11m_baseline.log`
  - `build\p11n\run_p11m_macro.log`
  - `build\p11n\run_p11n_baseline.log`
  - `build\p11n\run_p11n_macro.log`
