# REVIEWER_GUIDE_QKV_MAINLINE_zhTW
Date: 2026-03-19

## 1. 文件定位（累積版）
- 本文件是 **累積式 reviewer-facing guide**，對目前 accepted local-only Q/KV mainline 路徑做導讀。
- 本文件不是單一 round completion report；內容應持續 merge-forward，不覆蓋既有有效導讀。
- 本版明確以以下內容為合併來源：
  - Round 1 guide baseline（historical pre-relayout path）：`docs/process/REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md@0001a68`
  - Round 2 guide baseline（historical pre-relayout path）：`docs/process/REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md@b2079e3`
- Round 1 低層導讀保留原則：
  - 以既有 reviewer-guide/既有回報可追溯內容為 source of truth。
  - 不用當前 code 註解重新推導或重寫 Round 1 低層語意。

## 1.1 Review by Question（Quick Entry Index）
| 問題 | 先看（主 guide） | 再看（companion doc） |
| --- | --- | --- |
| 我想先確認 SRAM 決策到底誰擁有？ | [5. Ownership / Boundary 摘要](#ownership-boundary) | [REVIEW_CHECKLIST：Write-back/Fallback/Ownership](./REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md#writeback-fallback-ownership) |
| 我想知道 fallback meaning 是在哪裡被鎖定？ | [5. Ownership / Boundary 摘要](#ownership-boundary) | [REVIEW_CHECKLIST：Top / integration boundary](./REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md#top-integration-boundary) |
| 我想知道 Q/K/V handoff 主要在哪裡發生？ | [6.3 AttnLayer0](#attnlayer0-role) | [ATTNLAYER0_STAGE_CROSSCHECK：QKV stage](./ATTNLAYER0_STAGE_CROSSCHECK_zhTW.md#stage-qkv) |
| 我想快速知道 Catapult GUI 應優先看哪些 loop family？ | [7. Loop-Role Notes](#loop-role-notes) | [TERNARY_LEAF_ROLEMAP：Loop family quick map](./TERNARY_LEAF_ROLEMAP_zhTW.md#loop-family-quick-map) |
| 我想先做 10 分鐘快檢，不重看整份 guide | [3. First-Pass Review Order](#first-pass-review-order) | [REVIEW_CHECKLIST：Fast 10-minute pass](./REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md#fast-10-minute-pass) |
| 我要做 30 分鐘深檢，想要可執行清單 | [8. Most Important Code Regions](#most-important-code-regions) | [REVIEW_CHECKLIST：Deeper 30-minute pass](./REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md#deeper-30-minute-pass) |
| 我要快速檢查 Top/Transformer（15 分鐘） | [6.1 Top](#top-role) + [6.2 TransformerLayer](#transformerlayer-role) | [TOP_TRANSFORMER_QUICKCHECK](./TOP_TRANSFORMER_QUICKCHECK_zhTW.md#15-minute-fast-scan-order) |
| 我要快速產出 reviewer 結論文字 | [9. PASS 代表什麼 / 不代表什麼](#pass-semantics) | [REVIEW_VERDICT_TEMPLATE](./REVIEW_VERDICT_TEMPLATE_QKV_MAINLINE_zhTW.md#ultra-short-note-template) |

<a id="review-kit-map"></a>
## 1.2 Review Kit Map（Closeout Bundle）
| 文件 | 主要用途 | 建議使用時機 |
| --- | --- | --- |
| `REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md` | 累積式主導讀與邊界語意 | 第一次進入或對齊全局語意 |
| `REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md` | 可執行 checklist（10 分鐘/30 分鐘） | 要快速確認「有無漏檢」 |
| `ATTNLAYER0_STAGE_CROSSCHECK_zhTW.md` | AttnLayer0 三 stage cross-check | 要釐清 QKV/SCORES/OUT 分層與邊界 |
| `TERNARY_LEAF_ROLEMAP_zhTW.md` | ternary family 權責與常見誤讀 | 要釐清 kernel/wrapper/shape-config 角色 |
| `REVIEW_VERDICT_TEMPLATE_QKV_MAINLINE_zhTW.md` | reviewer 結論模板（短版/完整版） | 要輸出可接受/需補件等結論 |
| `TOP_TRANSFORMER_QUICKCHECK_zhTW.md` | Top/Transformer 15 分鐘快檢 | 要先判斷高層 dispatch/delegation 邊界 |

## 2. Covered Files
- `src/Top.h`
- `src/blocks/TransformerLayer.h`
- `src/blocks/AttnLayer0.h`
- `src/blocks/TernaryLiveQkvLeafKernel.h`
- `src/blocks/TernaryLiveQkvLeafKernelTop.h`
- `src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h`
- `src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h`
- Companion docs（stable review kit）：
  - `docs/review/REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md`
  - `docs/review/ATTNLAYER0_STAGE_CROSSCHECK_zhTW.md`
  - `docs/review/TERNARY_LEAF_ROLEMAP_zhTW.md`
  - `docs/review/REVIEW_VERDICT_TEMPLATE_QKV_MAINLINE_zhTW.md`
  - `docs/review/TOP_TRANSFORMER_QUICKCHECK_zhTW.md`

<a id="first-pass-review-order"></a>
## 3. First-Pass Review Order
1. `Top.h`（外部 contract、FSM dispatch、layer orchestration）
2. `TransformerLayer.h`（Top -> Attn/FFN/LN integration boundary）
3. `AttnLayer0.h`（QKV/score/out 主計算路徑）
4. `TernaryLiveQkvLeafKernel*.h`（leaf row materialization 與 shape/metadata guard）

### 3.1 Lower-Level First-Pass Review Order（stable kit section）
1. `AttnLayer0.h` 先看 `ATTN_STAGE_QKV`，確認 Q/K/V materialization 與 fallback/bypass 邊界。
2. `AttnLayer0.h` 再看 `ATTN_STAGE_SCORES`，確認 score/softmax/pre-concat 的資料流。
3. `AttnLayer0.h` 最後看 `ATTN_STAGE_OUT`，確認 final write-back 邊界。
4. `TernaryLiveQkvLeafKernel.h` 看 core row kernel metadata guard + decode/MAC 角色。
5. `TernaryLiveQkvLeafKernelTop.h` / `...CatapultPrepTop.h` 看 wrapper surface 角色，不看成 policy owner。

<a id="end-to-end-dataflow"></a>
## 4. End-to-End Dataflow（3~8 步）
1. Top ingest `SET_W_BASE/LOAD_W/INFER`，建立可用 `W_REGION` 與 input payload。
2. Top 執行 preproc + pre-layernorm，產生 layer 輸入 `X_WORK` 映射區。
3. Top 在 layer0 先嘗試 Top-managed Q/KV mainline hook，建立 prebuilt/fallback 結果。
4. Top 將 base/range 與 prebuilt flags 傳入 `TransformerLayer(...)`。
5. `TransformerLayer` 委派 `AttnLayer0` 與 `FFNLayer0`，並完成 residual + LN glue。
6. `AttnLayer0` 在 QKV/score/out path 內呼叫 ternary leaf row materialization 並完成 write-back。
7. Top 完成 end LN + FinalHead，依 outmode 從對應 base 將結果寫到 `data_out`。

<a id="ownership-boundary"></a>
## 5. Ownership / Boundary 摘要
- SRAM 決策 owner：Top（唯一 shared SRAM owner）。
- Dispatch/Delegate：
  - Top dispatch：state machine、layer loop、readback/writeback 邊界。
  - TransformerLayer delegate：Attn/FFN/LN layer-local integration。
- 主要 buffer 生產/消費：
  - `W_REGION`：Top 管理載入與可見範圍；blocks/leaf 只讀指定區段。
  - `X_WORK`：Top 管理頁切換與 phase-safe overwrite；layer blocks 在指定 base 下讀寫。
  - `SCR_K/SCR_V`：attention scratch，由 Top 指派邊界與生命週期。
- fallback 語意位置：由 Top 在 `run_transformer_layer_loop(...)` 的 layer0 hook 區段建立並鎖定旗標。
- write-back/readback 邊界：
  - write-back：Top `infer_emit_outmode_payload(...)`
  - readback：Top `handle_read_mem(...)`

<a id="lower-level-ownership"></a>
### 5.1 Lower-Level File Ownership / Non-Ownership（stable kit section）
- `AttnLayer0.h`
  - owns: stage-scoped attention compute 與 caller-provided windows 內的寫入行為。
  - does not own: Top FSM、外部 command/rsp 協定、全域 SRAM policy。
- `TernaryLiveQkvLeafKernel.h`
  - owns: row-kernel guard/decode/MAC 與 matrix-specific row materialization。
  - does not own: W_REGION policy、runtime scheduling、runtime-variable shape negotiation。
- `TernaryLiveQkvLeafKernelTop.h`
  - owns: fixed-shape local wrapper surface（split-interface run surface）。
  - does not own: SRAM policy、compile-prep policy、global orchestration。
- `TernaryLiveQkvLeafKernelCatapultPrepTop.h`
  - owns: compile-prep-facing wrapper surface。
  - does not own: runtime behavior contract 擴張、SRAM policy。
- `TernaryLiveQkvLeafKernelShapeConfig.h`
  - owns: compile-time shape/payload SSOT constants。
  - does not own: materialization logic、wrapper interface behavior。

## 6. File Roles
<a id="top-role"></a>
### 6.1 `Top.h`
- 唯一外部 4-channel contract (`ctrl_cmd/ctrl_rsp/data_in/data_out`) 的 dispatch owner。
- 管理 `ST_CFG_RX/ST_PARAM_RX/ST_INFER_RX/ST_HALTED` 與 payload ingest。
- 管理 layer orchestration（含 layer0 mainline hook、mid/end LN 插入、FinalHead 接續）。

<a id="transformerlayer-role"></a>
### 6.2 `TransformerLayer.h`
- 接收 Top 指派的 `x/scratch/param` 邊界與 prebuilt flags。
- 委派 `AttnLayer0<ATTN_STAGE_FULL>` 與 `FFNLayer0<FFN_STAGE_FULL>`。
- 完成 layer-local residual add + sublayer1 norm param load + LN 呼叫。
- 不擁有 Top-FSM、共享 SRAM policy、或 fallback 政策定義權。

<a id="attnlayer0-role"></a>
### 6.3 `AttnLayer0.h`（Round 1 核心低層導讀保留）
- `ATTN_STAGE_QKV`：Q/K/V materialization（含 live path 與 fallback/bypass）。
- `ATTN_STAGE_SCORES`：QK score、softmax、V 加權輸出到 pre/post concat。
- `ATTN_STAGE_OUT`：將 `post_concat` 寫回 attention output base。
- handoff-in（from `TransformerLayer`）：
  - `x_in_base_word` / `sc.*_base_word` 形成當前 stage 讀寫邊界。
  - `q_prebuilt_from_top_managed` / `kv_prebuilt_from_top_managed` 只控制對應 materialization 是否略過。
- handoff-out（to downstream layer glue）：
  - `attn_out_base_word` 是本 block 對後續 FFN/residual glue 的輸出交接點。
- write-back 邊界：
  - pre/post concat copy 在 attention scratch 內。
  - OUT stage 才把最終 tensor 寫到 caller-provided `attn_out_base_word`。
- 不負責 Top state machine、外部 command/rsp 協定、全域 SRAM ownership 仲裁。

<a id="ternary-role-section"></a>
### 6.4 `TernaryLiveQkvLeafKernel*.h`（Round 1 核心低層導讀保留）
- `ShapeConfig`：QKV compile-time shape/payload expectation SSOT。
- `LeafKernel`：row-kernel（metadata guard + ternary decode + MAC）。
- `LeafKernelTop`：local split-interface top wrapper。
- `LeafKernelCatapultPrepTop`：compile-prep wrapper/surface adapter。
- handoff 關係：
  - `AttnLayer0` 呼叫 `LeafKernelTop` split interface 完成 row materialization。
  - `LeafKernel` 本體可由 SRAM-backed generic entry 或 split-interface wrappers 委派進入。
- role 分層：
  - core kernel = 計算/guard
  - local top wrapper = fixed-shape runtime-facing local adapter
  - compile-prep top wrapper = Catapult compile-prep adapter
  - shape config = constants/SSOT only
- 不負責 runtime-variable shape negotiation、Top SRAM policy、closure 宣告。

<a id="loop-role-notes"></a>
## 7. Loop-Role Notes（代表家族）
<a id="top-loop-family"></a>
### 7.1 `TOP_*`（Round 2 新增）
- `TOP_LAYER_ORCHESTRATION_LOOP`：layer 主排程迴圈。
- `TOP_OUTMODE_XPRED_WRITEBACK_LOOP` / `TOP_OUTMODE_LOGITS_WRITEBACK_LOOP`：Top 最終輸出寫回。
- `TOP_READ_MEM_STREAM_LOOP`：READ_MEM readback stream。
- `TOP_COPY_X_WORDS_LOOP` / `TOP_NORM_PARAM_COPY_LOOP`：中間快照與參數搬移輔助迴圈。

<a id="transformer-loop-family"></a>
### 7.2 `TRANSFORMER_*`（Round 2 新增）
- `TRANSFORMER_LAYER_SUBLAYER1_NORM_PARAM_COPY_LOOP`：LN gamma/beta 載入。
- `TRANSFORMER_LAYER_FFN_RESIDUAL_ADD_LOOP`：FFN2 輸出與 residual 加總。

<a id="attn-loop-family"></a>
### 7.3 `ATTN_*`（Round 1 低層重點保留）
- `ATTN_QKV_KV_PRIME_COPY_LOOP`
- `ATTN_Q_SPLIT_TOKEN_LOOP` / `ATTN_Q_SPLIT_INPUT_COL_LOOP` / `ATTN_Q_SPLIT_OUTPUT_COL_LOOP`
- `ATTN_K_*` / `ATTN_V_*`
- `ATTN_SCORE_TOKEN_LOOP` / `ATTN_SCORE_HEAD_LOOP` / `ATTN_SCORE_KEY_TOKEN_LOOP` / `ATTN_SCORE_DOT_COL_LOOP`
- `ATTN_PRECONCAT_HEAD_COL_LOOP` / `ATTN_PRECONCAT_KEY_TOKEN_ACC_LOOP`
- `ATTN_POSTCONCAT_COPY_LOOP` / `ATTN_OUT_WRITEBACK_LOOP`
- map（本輪補強）：
  - `ATTN_Q*` = Q materialization / fallback / bypass family
  - `ATTN_K*` = K materialization / fallback / bypass family
  - `ATTN_V*` = V materialization / fallback / bypass family
  - `ATTN_SCORE*` = score + softmax 前計算
  - `ATTN_PRECONCAT*` = V 加權累積
  - `ATTN_POSTCONCAT*` / `ATTN_OUT*` = concat 收斂與最終寫回

<a id="ternary-loop-family"></a>
### 7.4 `TERNARY_*`（Round 1 低層重點保留）
- `TERNARY_QKV_IMPL_OUT_ROW_LOOP`
- `TERNARY_WQ_SPLIT_OUT_ROW_LOOP` + `TERNARY_WQ_SPLIT_IN_COL_LOOP`
- `TERNARY_WK_SPLIT_OUT_ROW_LOOP` + `TERNARY_WK_SPLIT_IN_COL_LOOP`
- `TERNARY_WV_SPLIT_OUT_ROW_LOOP` + `TERNARY_WV_SPLIT_IN_COL_LOOP`
- map（本輪補強）：
  - `TERNARY_QKV_IMPL_*` = generic SRAM-backed row kernel path
  - `TERNARY_WQ_SPLIT_*` = WQ fixed-shape split-interface row path
  - `TERNARY_WK_SPLIT_*` = WK fixed-shape split-interface row path
  - `TERNARY_WV_SPLIT_*` = WV fixed-shape split-interface row path

<a id="most-important-code-regions"></a>
## 8. Most Important Code Regions（7 個）
1. `src/Top.h` `top(...)`（state + command dispatch）
2. `src/Top.h` `run_transformer_layer_loop(...)`（layer0 hook + fallback latch）
3. `src/Top.h` `infer_emit_outmode_payload(...)`（Top write-back boundary）
4. `src/Top.h` `handle_read_mem(...)`（Top readback boundary）
5. `src/blocks/TransformerLayer.h` `TransformerLayer(...)`（integration boundary）
6. `src/blocks/AttnLayer0.h` `AttnLayer0(...)`（QKV/score/out 主體）
7. `src/blocks/TernaryLiveQkvLeafKernel.h` `ternary_live_qkv_materialize_row_kernel_impl(...)`（leaf materialization/guard）

<a id="pass-semantics"></a>
## 9. PASS 代表什麼 / 不代表什麼
- 目前 PASS 代表：
  - local-only accepted progress 可重現
  - 指定 regression/compile-prep/probe script 在本地可通過
  - handoff boundary 內主線行為符合目前 acceptance 條件
- 目前 PASS 不代表：
  - Catapult closure
  - SCVerify closure
  - full runtime/numeric/global closure

## 10. Historical Notes（Convergence record）
### 10.1 Round 1 additions（保留）
- 建立 Attn/ternary leaf 低層導讀與 loop-role 家族說明。
- 建立 buffer/ownership/fallback 檢查視角。

### 10.2 Round 2 additions（合併）
- 補 Top/Transformer integration boundary 專章與 `TOP_*`/`TRANSFORMER_*` loop 可視性。
- 補 Top write-back/readback 邊界導讀與高層 review order。

### 10.3 Remaining debt / optional enhancement（post-closeout）
- `Top.h`/`TransformerLayer.h` 之外的 design-side 歷史註解一致性仍可能有零星議題（optional，non-blocking）。
- 以下 reviewer-facing 項目已完成（debt-paydown closeout）：
  - `AttnLayer0.h` stage cross-check checklist 化（`ATTNLAYER0_STAGE_CROSSCHECK_zhTW.md`）
  - ternary leaf family / compile-prep wrapper 權責圖示化（`TERNARY_LEAF_ROLEMAP_zhTW.md`）
  - cumulative checklist 與 quick-entry index（`REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md` + 本文件 1.1）
- 目前剩餘項目屬 optional enhancement（非 blocking debt）：
  - 針對特定 review 場景（例如 Catapult GUI 專項）補更細的「問答式範本」
  - 視後續任務需要，新增針對新 mainline 工作的增量 sidecar（docs-only）

