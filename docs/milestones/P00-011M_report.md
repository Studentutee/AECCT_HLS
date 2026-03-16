# P00-011M Report — WQ Source-Side Integration Slice + Dual-Binary KV Signature Validation (Local-Only)

## Goal / Scope
- 目標：在 local smoke scope 內，將 `AttnLayer0<ATTN_STAGE_QKV>` 的 `L0_WQ` source-side path 接到 split-interface local top（單一 integration slice）。
- 範圍：small production-side integration + integration smoke + regression extension + governance sync。
- 本報告記錄的是 **local smoke scope accepted**，不是 Catapult / SCVerify closure。

## Scope Guardrails
- `AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE` 只在 p11m macro build command 以 `/D` 啟用。
- `p11j/p11k/p11l_b/p11l_c` regression builds 不定義此 macro。
- 正式 acceptance evidence 只採用 `build\p11m\*.log`。

## Files Changed
- `src/blocks/AttnLayer0.h`
- `tb/tb_ternary_live_source_integration_smoke_p11m.cpp`
- `scripts/local/run_p11l_local_regression.ps1`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/P00-011M_report.md`

## Regression Baseline Not Modified
- `tb/tb_ternary_live_leaf_smoke_p11j.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11k.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11l_b.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11l_c.cpp`

## Exact Build Commands
1. `New-Item -ItemType Directory -Force -Path build\p11m > $null`
2. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_smoke_p11j.cpp /Fe:build\p11m\tb_ternary_live_leaf_smoke_p11j.exe > build\p11m\build_p11j.log 2>&1`
3. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11k.cpp /Fe:build\p11m\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11m\build_p11k.log 2>&1`
4. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_b.cpp /Fe:build\p11m\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11m\build_p11l_b.log 2>&1`
5. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_c.cpp /Fe:build\p11m\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11m\build_p11l_c.log 2>&1`
6. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_source_integration_smoke_p11m.cpp /Fe:build\p11m\tb_ternary_live_source_integration_smoke_p11m_baseline.exe > build\p11m\build_p11m_baseline.log 2>&1`
7. `cl /nologo /std:c++14 /EHsc /utf-8 /D AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE=1 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_source_integration_smoke_p11m.cpp /Fe:build\p11m\tb_ternary_live_source_integration_smoke_p11m_macro.exe > build\p11m\build_p11m_macro.log 2>&1`

## Exact Run Commands
1. `cmd /c build\p11m\tb_ternary_live_leaf_smoke_p11j.exe > build\p11m\run_p11j.log 2>&1`
2. `cmd /c build\p11m\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11m\run_p11k.log 2>&1`
3. `cmd /c build\p11m\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11m\run_p11l_b.log 2>&1`
4. `cmd /c build\p11m\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11m\run_p11l_c.log 2>&1`
5. `cmd /c build\p11m\tb_ternary_live_source_integration_smoke_p11m_baseline.exe > build\p11m\run_p11m_baseline.log 2>&1`
6. `cmd /c build\p11m\tb_ternary_live_source_integration_smoke_p11m_macro.exe > build\p11m\run_p11m_macro.log 2>&1`

## Canonical One-Shot Command
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11m`

## Execution Evidence Excerpt（from `build\p11m\*.log`）
- `build\p11m\run_p11j.log`
  - `PASS: tb_ternary_live_leaf_smoke_p11j`
- `build\p11m\run_p11k.log`
  - `PASS: tb_ternary_live_leaf_top_smoke_p11k`
- `build\p11m\run_p11l_b.log`
  - `PASS: tb_ternary_live_leaf_top_smoke_p11l_b`
- `build\p11m\run_p11l_c.log`
  - `PASS: tb_ternary_live_leaf_top_smoke_p11l_c`
- `build\p11m\run_p11m_baseline.log`
  - `[p11m][KV_SIG] K=0x70D576AFA0F67AD3 V=0x70D576AFA0F67AD3`
  - `PASS: tb_ternary_live_source_integration_smoke_p11m`
- `build\p11m\run_p11m_macro.log`
  - `[p11m][KV_SIG] K=0x70D576AFA0F67AD3 V=0x70D576AFA0F67AD3`
  - `[p11m][PASS] source-side WQ integration path exact-match equivalent to split-interface local top`
  - `[p11m][PASS] K/V fallback retained under WQ-only integration slice`
  - `PASS: tb_ternary_live_source_integration_smoke_p11m`

## Acceptance Notes
- baseline 與 macro 的 `K/V` 簽章一致（script 強制檢查）。
- `Q/Q_act_q/Q_sx` 對 `TernaryLiveL0WqRowTop` direct reference 的 exact-match 只在 macro build 驗證。

## Governance Sync Notes
- `PROJECT_STATUS / CLOSURE_MATRIX / TRACEABILITY_MAP` 已同步補入 P00-011M。
- touched docs 的 governance entry 引用維持 `docs/process/GOVERNANCE_ENTRYPOINT_zhTW.txt`。

## Deferred
- Catapult / SCVerify flow：deferred（不在本輪範圍）。
