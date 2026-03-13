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
| Project Status Summary | 專案主線狀態摘要 | PASS | `P00-009_report.md` | `docs/process/PROJECT_STATUS_zhTW.txt` | 後續里程碑變化仍需定期更新。 |
| v12.1 Docs Baseline | v12.1 文件基線收斂（docs-only） | FROZEN-DOCS | `P00-008_report.md` + `P00-008_artifacts/verdict.txt` | `docs/spec/AECCT_HLS_Spec_v12.1_zhTW.txt`<br>`docs/architecture/AECCT_HLS_Architecture_Guide_v12.1_zhTW.txt` | 文件凍結不等於 live implementation closure。 |
| M0 Formal Closure | M0 skeleton/contract/smoke baseline | PASS | `M0_report.md` + `M0_artifacts/verdict.txt` | `docs/milestones/M0_report.md`<br>`docs/milestones/M0_artifacts/closure_checklist.txt` | M0 範圍明確，global open items 已分離追蹤。 |
| Pragma Hygiene | project code pragma 清理 | PASS | P00-010（工作項名稱） | `design/AecctTop.h` | 現況檢查僅保留合法 `#pragma hls_design top/interface`。 |
| Ternary Phase A Semantic Alignment | `_S_W` 語意對齊 `inv_s_w` carrier | PASS | P00-011A（工作項名稱） | `include/WeightStreamOrder.h`<br>`tb/weights_streamer.h` | 以語意/metadata 對齊為主，未切 live packed payload。 |
| Ternary Phase B Preview / Validation | ternary pack/decode 預演驗證（TB-only） | NON-LIVE-VALIDATED | P00-011B（工作項名稱） | `tb/tb_ternary_pack_p11b.cpp`<br>`tb/weights_streamer.h` | 非 live 路徑驗證，不改 live layout。 |
| Ternary Phase C Offline Export / Artifact | offline exporter 與機讀 artifact 對齊 | NON-LIVE-VALIDATED | P00-011C（工作項名稱） | `tb/tb_ternary_export_p11c.cpp`<br>`gen/ternary_p11c_export.json` | 已有輸出工件，仍屬 non-live。 |
| m1/m2 Runtime Closure | m1/m2 runtime 正式收斂 | OPEN | `PROJECT_STATUS_zhTW.txt` | `docs/process/PROJECT_STATUS_zhTW.txt`<br>`docs/milestones/M0_report.md` | 目前主要是 compile-compatible evidence；runtime closure 未完成。 |
| Full Numeric Closure | 端到端完整數值正確性 closure | OPEN | `PROJECT_STATUS_zhTW.txt` | `docs/process/PROJECT_STATUS_zhTW.txt` | 尚未達成 full numeric closure。 |
| Live Ternary Packed Payload Migration | ternary packed payload live migration | OPEN | P00-011A/B/C（工作項名稱） | `include/WeightStreamOrder.h`<br>`tb/tb_ternary_pack_p11b.cpp`<br>`tb/tb_ternary_export_p11c.cpp` | 目前到 non-live 驗證，live path 尚未切換。 |

## 4. Current Readout
v12.1 docs baseline 與 M0 formal closure 已具備可追溯證據，專案主線目前可視為「基線明確、後續擴展待完成」。治理入口與狀態摘要也已建立，可供新對話快速接手。

ternary 主線已完成 Phase A/B/C 的 non-live 收斂：Phase A 完成語意與 metadata 對齊，Phase B 完成 TB pack/decode preview，Phase C 完成 offline export artifact。live ternary migration、m1/m2 runtime closure 與 full numeric closure 仍為開放項。

## 5. Next Recommended Focus
1. 先做 loader-side parser/checker preview（non-live），把 ternary format 消費端驗證邏輯釘清楚。  
2. 規劃並切分 live migration design cut，將 non-live A/B/C 結果映射到可控的 live 轉換步驟。  
3. 等下一個明確里程碑完成後，再同步更新 `docs/process/PROJECT_STATUS_zhTW.txt`。  
