# TRACEABILITY_MAP_v12.1

## 1. Purpose
本文件用來把 v12.1 baseline、目前狀態、工作項與可追溯證據串成同一張對照圖。
目的是讓新對話或新接手者能快速回答「這個結論從哪裡來」。
它不是完整技術規格，也不取代既有 milestone report / artifacts。
若本文件與原始證據有差異，應以原始檔案路徑所指內容為準。

## 2. Traceability Principles
- 只採用 repo 內目前存在的文件、程式碼與工件作為證據來源。
- 不把不存在的 milestone report / artifact 當作已完成證據。
- 明確區分 `docs freeze`、`non-live validation`、`live closure` 三種不同成熟度。
- 每個結論都必須可回指到具體檔案路徑，避免模糊敘述。

## 3. Traceability Map
| Topic | Current Claim | Source of Truth / Evidence | Main Files | Evidence Type | Notes |
|---|---|---|---|---|---|
| Governance entry / authority | `docs/process/GOVERNANCE_ENTRYPOINT_zhTW.txt` 已作為 repo 治理入口並定義 authority order。 | `docs/process/GOVERNANCE_ENTRYPOINT_zhTW.txt` | `docs/process/GOVERNANCE_ENTRYPOINT_zhTW.txt` | governance entry | P00-012 以工作項名稱追蹤；repo 無同名正式 report。 |
| Current project progress | 專案目前狀態已有單點摘要，可快速判讀主線進度與 open items。 | `docs/process/PROJECT_STATUS_zhTW.txt` | `docs/process/PROJECT_STATUS_zhTW.txt` | status summary | 2026-03-16 已補記 P00-011F~P00-011L-D local acceptance / deferred closure。 |
| v12.1 docs baseline freeze | v12.1 文件基線已完成 docs-only freeze。 | `docs/milestones/P00-008_report.md` + `docs/milestones/P00-008_artifacts/verdict.txt` | `docs/spec/AECCT_HLS_Spec_v12.1_zhTW.txt`<br>`docs/architecture/AECCT_HLS_Architecture_Guide_v12.1_zhTW.txt` | milestone report | 屬 `FROZEN-DOCS`，非 implementation closure。 |
| M0 closure | M0 formal closure 已完成，且明確區分 M0 與全域治理 open items。 | `docs/milestones/M0_report.md` + `docs/milestones/M0_artifacts/verdict.txt` | `docs/milestones/M0_report.md`<br>`docs/milestones/M0_artifacts/closure_checklist.txt` | milestone report | M0 scope 已關閉。 |
| Pragma hygiene cleanup | project pragma 已收斂為合法 top/interface 形態。 | 現況程式碼檢查（pragma scan） | `design/AecctTop.h` | source implementation | P00-010 以工作項名稱追蹤；無同名正式 report。 |
| Ternary Phase A (`_S_W` semantic -> `inv_s_w`) | quantized-linear `_S_W` 在 stream semantic 上已對齊為 `inv_s_w` carrier，並有 SSOT helper。 | `include/WeightStreamOrder.h` + `tb/weights_streamer.h` | `include/WeightStreamOrder.h`<br>`tb/weights_streamer.h` | source implementation | P00-011A 為工作項名稱引用。 |
| Ternary Phase B (pack/decode preview) | 已有 TB-only pack/decode preview 與自檢程式，屬 non-live 驗證。 | `tb/tb_ternary_pack_p11b.cpp` + pack/decode helper | `tb/tb_ternary_pack_p11b.cpp`<br>`tb/weights_streamer.h` | self-check TB | P00-011B 為工作項名稱引用。 |
| Ternary Phase C (offline export artifact) | 已有 standalone exporter，並產生 machine-readable artifact。 | `tb/tb_ternary_export_p11c.cpp` + `gen/ternary_p11c_export.json` | `tb/tb_ternary_export_p11c.cpp`<br>`gen/ternary_p11c_export.json` | generated artifact | P00-011C 為工作項名稱引用。 |
| Ternary live source-side cut chain (F~I) | L0_WQ/WK/WV 的 source-side live cut 已在目前 codebase 推進到 `ATTN_STAGE_QKV`。 | 現況程式碼檢查（leaf helper + `AttnLayer0` call-site + `p11f..p11i` TB） | `src/blocks/TernaryLiveQkvLeafKernel.h`<br>`src/blocks/AttnLayer0.h`<br>`tb/tb_ternary_live_cut_p11f.cpp`<br>`tb/tb_ternary_live_cut_p11i.cpp` | source implementation | repo 內尚未補齊對應 milestone report；應搭配 `PROJECT_STATUS_zhTW.txt` 閱讀其 accepted/deferred 口徑。 |
| Ternary local smoke path (J / K / L-A) | repo-tracked leaf/local-top/split-interface P11K 本地 smoke 已可執行，且本地不再硬依賴 `mc_scverify.h`；接受範圍限 local smoke。 | `docs/milestones/P00-011L-A_report.md` + `src/blocks/TernaryLiveQkvLeafKernelTop.h` + `tb/tb_ternary_live_leaf_top_smoke_p11k.cpp` | `docs/milestones/P00-011L-A_report.md`<br>`src/blocks/TernaryLiveQkvLeafKernelTop.h`<br>`tb/tb_ternary_live_leaf_smoke_p11j.cpp`<br>`tb/tb_ternary_live_leaf_top_smoke_p11k.cpp` | task-local report + source implementation | Catapult / SCVerify validation deferred；不應誤讀成 full closure。 |
| Ternary split-interface same-family extension (L-B) | L0_WK / L0_WV split-interface local top 已以同族擴展方式接受，接受範圍限 local smoke。 | `docs/milestones/P00-011L-B_report.md` + repo-tracked source/TB | `docs/milestones/P00-011L-B_report.md`<br>`src/blocks/TernaryLiveQkvLeafKernel.h`<br>`src/blocks/TernaryLiveQkvLeafKernelTop.h`<br>`tb/tb_ternary_live_leaf_top_smoke_p11l_b.cpp` | local smoke evidence / task-local report | implementation/local smoke accepted；Catapult / SCVerify deferred。 |
| Ternary TB-local family common smoke (L-C) | WQ/WK/WV split-interface top 以 TB-local common driver 收斂，接受範圍限 local smoke。 | `docs/milestones/P00-011L-C_report.md` + repo-tracked TB | `docs/milestones/P00-011L-C_report.md`<br>`tb/tb_ternary_live_leaf_top_smoke_p11l_c.cpp` | local smoke evidence / task-local report | commonization 僅在 TB 內；Catapult / SCVerify deferred。 |
| QKV local regression packaging (L-D) | 以 repo-tracked Windows script 將 p11j/p11k/p11l_b/p11l_c 進行同輪 local regression，並將治理同步到 L-D。 | `docs/milestones/P00-011L-D_report.md` + `build\p11l_d\*.log` | `docs/milestones/P00-011L-D_report.md`<br>`scripts/local/run_p11l_local_regression.ps1`<br>`tb/tb_ternary_live_leaf_smoke_p11j.cpp`<br>`tb/tb_ternary_live_leaf_top_smoke_p11k.cpp`<br>`tb/tb_ternary_live_leaf_top_smoke_p11l_b.cpp`<br>`tb/tb_ternary_live_leaf_top_smoke_p11l_c.cpp` | local smoke evidence / task-local report | `build\p11l_d_probe` 僅 preflight；正式 acceptance evidence 只採 `build\p11l_d\*.log`；Catapult / SCVerify deferred。 |
| Open items: m1/m2 runtime closure | m1/m2 runtime closure 仍未正式完成。 | `docs/process/PROJECT_STATUS_zhTW.txt` | `docs/process/PROJECT_STATUS_zhTW.txt` | status summary | 目前為 open item。 |
| Open items: full numeric closure | full numeric correctness closure 尚未達成。 | `docs/process/PROJECT_STATUS_zhTW.txt` | `docs/process/PROJECT_STATUS_zhTW.txt` | status summary | 目前為 open item。 |
| Open items: live ternary migration | live ternary packed payload migration 尚未完成。 | Phase A/B/C + F~L-D 現況 | `include/WeightStreamOrder.h`<br>`src/blocks/TernaryLiveQkvLeafKernel.h`<br>`src/blocks/TernaryLiveQkvLeafKernelTop.h`<br>`tb/tb_ternary_live_leaf_top_smoke_p11k.cpp`<br>`tb/tb_ternary_live_leaf_top_smoke_p11l_b.cpp`<br>`tb/tb_ternary_live_leaf_top_smoke_p11l_c.cpp`<br>`scripts/local/run_p11l_local_regression.ps1` | source implementation + task-local report | 已有 non-live 與 local smoke 進展，但尚未 full live closure。 |

## 4. Fast Path for New Conversations
1. `docs/process/GOVERNANCE_ENTRYPOINT_zhTW.txt`  
2. `docs/process/PROJECT_STATUS_zhTW.txt`  
3. `docs/milestones/CLOSURE_MATRIX_v12.1.md`  
4. `docs/milestones/TRACEABILITY_MAP_v12.1.md`  
5. 若任務與 ternary live-cut / local smoke 直接相關，再讀 `docs/milestones/P00-011L-A_report.md`、`docs/milestones/P00-011L-B_report.md`、`docs/milestones/P00-011L-C_report.md`、`docs/milestones/P00-011L-D_report.md` 與對應 `src/blocks` / `tb` / `scripts` 正式檔。  

## 5. Known Gaps
- m1/m2 runtime closure 尚未正式完成。  
- full numeric closure 尚未完成。  
- live ternary packed payload migration 尚未完成。  
- P00-011J / P00-011K / P00-011L-A / P00-011L-B / P00-011L-C / P00-011L-D 目前僅接受 local smoke scope，Catapult / SCVerify 驗證仍待後續批次執行。  
- global governance open items 仍需後續 cleanup/waiver resolution。  
