# P00-007 Report

## Goal / Scope
- 任務目標: 將 Step 2 skeleton/contract bring-up 收斂為可重現的 `tb_top_m0` smoke baseline，作為 M0 closure 前置準備。
- 本輪範圍: 證據收斂與治理交付。
- 本輪不做: 演算法擴寫、Top/block contract 改造、SramMap 大改、repo-wide hygiene cleanup。

## Current Baseline
- Top contract SSOT: `src/Top.h`。
- `design/AecctTop.h` 為 wrapper/adapter-only。
- 外部 Top contract 維持 4 channels（`ctrl_cmd/ctrl_rsp/data_in/data_out`）。
- `in_fifo` 仍以 `ac_channel` 表示，未引入 RTL-style FIFO。

## What Changed
- 技術程式碼變更: none（no-code-change）。
- 新增 P00-007 交付:
  - `docs/milestones/P00-007_report.md`
  - `docs/milestones/P00-007_artifacts/build.log`
  - `docs/milestones/P00-007_artifacts/run_tb.log`
  - `docs/milestones/P00-007_artifacts/verdict.txt`
  - `docs/milestones/P00-007_artifacts/file_manifest.txt`
  - `docs/milestones/P00-007_artifacts/diff.patch`

## Why Changed
- 以最小風險完成 m0 smoke baseline 穩定化證據，避免將 Step 2 任務擴大成新一輪技術重構。
- 先建立可審核、可回滾、可重現的 M0 pre-closure 基線。

## Validation Scope
- In scope:
  - `tb_top_m0` compile。
  - `tb_top_m0` smoke run 連續 3 次（reproducibility）。
  - `tb_top_m1` compile-only。
  - `tb_top_m2` compile-only。
  - 靜態檢查:
    - SSOT/wrapper 一致性。
    - 無 `ac_channel.size()/available()` 依賴。
- Out of scope:
  - `tb_top_m1/m2` runtime。
  - 完整數值正確性與 full datapath 驗證。

## m0 Success Criteria
- `tb_top_m0` compile PASS。
- `tb_top_m0` smoke run PASS，且連續 3 次均 PASS。
- reset/idle 最小 command flow 正常（由 `tb_top_m0` NOOP/SOFT_RESET/unknown-op ERR 路徑驗證）。
- 無明顯 interface mismatch。
- 不依賴 `size()/available()` 作為功能正確性前提。

## Build / Run Results
- Build:
  - `tb_top_m0`: PASS
  - `tb_top_m1`: PASS (compile-only)
  - `tb_top_m2`: PASS (compile-only)
- Run:
  - `tb_top_m0` run #1: PASS
  - `tb_top_m0` run #2: PASS
  - `tb_top_m0` run #3: PASS
  - `tb_top_m1` runtime: SKIPPED (by scope)
  - `tb_top_m2` runtime: SKIPPED (by scope)
- 詳細 stdout/stderr 與 ExitCode 請見 artifacts logs。

## Verified / Not Verified
- Verified:
  - m0 compile + 3/3 smoke reproducibility。
  - m1/m2 compile 相容未破壞。
  - SSOT/wrapper 關係一致。
  - 未發現 `size()/available()` 依賴。
- Not verified:
  - m1/m2 runtime。
  - full algorithm correctness / golden compare。

## Known Limitations
- 仍有既有編譯 warning（HLS pragma 與 third_party/ac_types）但不影響本輪 smoke 成功條件。
- 本輪僅針對 m0 baseline 穩定化，非 M0 formal closure 最終判定。

## Governance / Milestone Interpretation
- 本輪定位: P00-007「M0 closure prep」證據收斂。
- 結論不等同 full milestone closeout，僅代表 m0 smoke baseline 已可重現且可交付。

## README Auto Handling
- 本輪未更新 `README.md` Auto。
- 理由: 本任務為 docs/evidence-only，未變更 `include/design/tb/scripts` 技術內容或執行流程契約。

## Recommended Next Step
1. 以 P00-007 基線進入 M0 formal closure 檢查清單（保持 no-scope-creep）。
2. 將 m1/m2 runtime 與更高層 datapath 驗證獨立成後續任務，不回灌本輪範圍。
