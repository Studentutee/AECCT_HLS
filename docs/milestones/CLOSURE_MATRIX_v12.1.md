# CLOSURE_MATRIX_v12.1

## 1. Purpose
本文件是 v12.1 baseline 的治理收斂索引，用來快速回答「目前做到哪」。
它不是技術規格文件（spec），也不是逐 patch 的完整變更紀錄。
它的定位是把 closure 狀態、最新證據與主要檔案關聯到同一個視圖。
正式規格與細節判定仍以既有 spec/rules 與 milestone artifacts 為準。

## 2. Status Legend
- `PASS`: 該區塊已有可對照證據，且目前判定為已收斂。
- `PARTIAL`: 已完成主要方向，但仍有明確未關閉項。
- `OPEN`: 尚未完成或尚未進入正式 closure。
- `FROZEN-DOCS`: 文件規格已凍結，但不代表實作已 fully live closure。
- `NON-LIVE-VALIDATED`: 已有非 live 路徑的驗證/工件，live migration 尚未完成。

## 3. Closure Matrix
| Area | Scope | Current Status | Latest Evidence / Patch | Key Files / Artifacts | Notes |
|---|---|---|---|---|---|
| Governance Entrypoint | 治理入口與 authority 順序 | PASS | P00-012（工作項名稱） | `GOVERNANCE_ENTRYPOINT.txt` | repo 已存在治理入口文件。 |
| Project Status Summary | 專案主線狀態摘要 | PASS | `P00-009_report.md` + 2026-03-15 status refresh | `docs/process/PROJECT_STATUS_zhTW.txt` | 已補記 P00-011F~P00-011L-A 的目前 accepted / deferred 狀態。 |
| v12.1 Docs Baseline | v12.1 文件基線收斂（docs-only） | FROZEN-DOCS | `P00-008_report.md` + `P00-008_artifacts/verdict.txt` | `docs/spec/AECCT_HLS_Spec_v12.1_zhTW.txt`<br>`docs/architecture/AECCT_HLS_Architecture_Guide_v12.1_zhTW.txt` | 文件凍結不等於 live implementation closure。 |
| M0 Formal Closure | M0 skeleton/contract/smoke baseline | PASS | `M0_report.md` + `M0_artifacts/verdict.txt` | `docs/milestones/M0_report.md`<br>`docs/milestones/M0_artifacts/closure_checklist.txt` | M0 範圍明確，global open items 已分離追蹤。 |
| Pragma Hygiene | project code pragma 清理 | PASS | P00-010（工作項名稱） | `design/AecctTop.h` | 現況檢查僅保留合法 `#pragma hls_design top/interface`。 |
| Ternary Phase A Semantic Alignment | `_S_W` 語意對齊 `inv_s_w` carrier | PASS | P00-011A（工作項名稱） | `include/WeightStreamOrder.h`<br>`tb/weights_streamer.h` | 以語意/metadata 對齊為主，未切 live packed payload。 |
| Ternary Phase B Preview / Validation | ternary pack/decode 預演驗證（TB-only） | NON-LIVE-VALIDATED | P00-011B（工作項名稱） | `tb/tb_ternary_pack_p11b.cpp`<br>`tb/weights_streamer.h` | 非 live 路徑驗證，不改 live layout。 |
| Ternary Phase C Offline Export / Artifact | offline exporter 與機讀 artifact 對齊 | NON-LIVE-VALIDATED | P00-011C（工作項名稱） | `tb/tb_ternary_export_p11c.cpp`<br>`gen/ternary_p11c_export.json` | 已有輸出工件，仍屬 non-live。 |
| Ternary Live Source-Side Cut Chain | design-side helper + source-side call-site live cut through `ATTN_STAGE_QKV` for L0_WQ/WK/WV | PARTIAL | accepted progress through P00-011F~P00-011I | `src/blocks/TernaryLiveQkvLeafKernel.h`<br>`src/blocks/AttnLayer0.h`<br>`tb/tb_ternary_live_cut_p11f.cpp`<br>`tb/tb_ternary_live_cut_p11i.cpp` | 已接受 task-local 進度，但尚未等同 Catapult / full runtime closure。 |
| Ternary Leaf / Top Local Smoke | tiny leaf-kernel、local wrapper、repo-tracked split-interface P11K smoke repair | NON-LIVE-VALIDATED | `P00-011L-A_report.md` | `tb/tb_ternary_live_leaf_smoke_p11j.cpp`<br>`tb/tb_ternary_live_leaf_top_smoke_p11k.cpp`<br>`src/blocks/TernaryLiveQkvLeafKernelTop.h` | P00-011J / P00-011K / P00-011L-A 已接受本地 smoke scope；`mc_scverify.h` 不再是本地硬依賴；Catapult/SCVerify deferred。 |
| m1/m2 Runtime Closure | m1/m2 runtime 正式收斂 | OPEN | `PROJECT_STATUS_zhTW.txt` | `docs/process/PROJECT_STATUS_zhTW.txt`<br>`docs/milestones/M0_report.md` | 目前主要是 compile-compatible evidence；runtime closure 未完成。 |
| Full Numeric Closure | 端到端完整數值正確性 closure | OPEN | `PROJECT_STATUS_zhTW.txt` | `docs/process/PROJECT_STATUS_zhTW.txt` | 尚未達成 full numeric closure。 |
| Live Ternary Packed Payload Migration | ternary packed payload live migration | OPEN | P00-011A/B/C + P00-011F~P00-011L-A（階段性進展） | `include/WeightStreamOrder.h`<br>`src/blocks/TernaryLiveQkvLeafKernel.h`<br>`src/blocks/TernaryLiveQkvLeafKernelTop.h`<br>`tb/tb_ternary_live_leaf_top_smoke_p11k.cpp` | 目前已有 non-live 與 local smoke 進展，但尚未完成 full live migration / Catapult closure。 |

## 4. Current Readout
v12.1 docs baseline 與 M0 formal closure 已具備可追溯證據，專案主線仍維持「基線明確、後續擴展待完成」。治理入口與狀態摘要可支援新對話快速接手。

ternary 主線已完成 Phase A/B/C 的 non-live 收斂，並進一步把 L0_WQ/WK/WV 的 source-side live cut 推進到 `ATTN_STAGE_QKV`。其後 P00-011J / P00-011K / P00-011L-A 已把 leaf-kernel、local top wrapper 與 repo-tracked split-interface P11K 本地 smoke 路徑收斂到可執行狀態，但目前接受範圍仍限 local smoke，不等同 Catapult / SCVerify 正式 closure。

## 5. Next Recommended Focus
1. 沿同 family 與同驗證入口繼續推進下一批 ternary live-cut / local-top 項目，維持 repo-tracked source + TB 為正式成果。  
2. 等累積到一個值得一起驗的批次後，再一次性做 Catapult / SCVerify bring-up，避免為單一小修反覆切環境。  
3. 補齊 task-local report / evidence 索引，讓 local acceptance 與 deferred closure 在 repo 內可追溯。  
