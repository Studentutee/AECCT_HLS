# P00-011L-D Report — QKV Local-Smoke Consolidation Super Bundle

## Goal / Scope
- 目標：在 local smoke scope 內完成 QKV split-interface family 的中型整包收斂。
- 實作路線：Script+Docs only（不改 TB code / 不改 production）。
- 本報告記錄的是 **local smoke scope accepted**，不是 Catapult / SCVerify closure。

## Scope Guardrails
- `build\p11l_d_probe` 僅作 preflight / baseline probe，不作為 P00-011L-D 正式 acceptance evidence。
- P00-011L-D 正式 evidence 僅採用 `build\p11l_d\*.log`。

## Files Changed
- `scripts/local/run_p11l_local_regression.ps1`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/P00-011L-D_report.md`

## Regression Baseline Not Modified
- `tb/tb_ternary_live_leaf_smoke_p11j.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11k.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11l_b.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11l_c.cpp`

## Exact Build Commands
1. `New-Item -ItemType Directory -Force -Path build\p11l_d > $null`
2. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_smoke_p11j.cpp /Fe:build\p11l_d\tb_ternary_live_leaf_smoke_p11j.exe > build\p11l_d\build_p11j.log 2>&1`
3. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11k.cpp /Fe:build\p11l_d\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11l_d\build_p11k.log 2>&1`
4. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_b.cpp /Fe:build\p11l_d\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11l_d\build_p11l_b.log 2>&1`
5. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_c.cpp /Fe:build\p11l_d\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11l_d\build_p11l_c.log 2>&1`

## Exact Run Commands
1. `cmd /c build\p11l_d\tb_ternary_live_leaf_smoke_p11j.exe > build\p11l_d\run_p11j.log 2>&1`
2. `cmd /c build\p11l_d\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11l_d\run_p11k.log 2>&1`
3. `cmd /c build\p11l_d\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11l_d\run_p11l_b.log 2>&1`
4. `cmd /c build\p11l_d\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11l_d\run_p11l_c.log 2>&1`

## Execution Evidence Excerpt（from `build\p11l_d\*.log`）
- `build\p11l_d\run_p11j.log`
  - `[p11j][PASS] kernel call succeeded for ternary_live_l0_wq_materialize_row_kernel`
  - `PASS: tb_ternary_live_leaf_smoke_p11j`
- `build\p11l_d\run_p11k.log`
  - `[p11k][PASS] split-interface top run() exact-match equivalent to direct P11J kernel output`
  - `PASS: tb_ternary_live_leaf_top_smoke_p11k`
- `build\p11l_d\run_p11l_b.log`
  - `L0_WK split-interface top run() exact-match equivalent to direct kernel output`
  - `L0_WV split-interface top run() exact-match equivalent to direct kernel output`
  - `PASS: tb_ternary_live_leaf_top_smoke_p11l_b`
- `build\p11l_d\run_p11l_c.log`
  - `L0_WQ split-interface top run() exact-match equivalent to direct kernel output`
  - `L0_WK split-interface top run() exact-match equivalent to direct kernel output`
  - `L0_WV split-interface top run() exact-match equivalent to direct kernel output`
  - `PASS: tb_ternary_live_leaf_top_smoke_p11l_c`

## Governance Sync Notes
- 本輪 touched docs 的 governance entry 引用已統一為 `docs/process/GOVERNANCE_ENTRYPOINT_zhTW.txt`。
- `PROJECT_STATUS / CLOSURE_MATRIX / TRACEABILITY_MAP` 已同步補入 P00-011L-D（local smoke accepted, Catapult / SCVerify deferred）。

## Deferred
- Catapult / SCVerify flow：deferred（不在本輪範圍內）。
