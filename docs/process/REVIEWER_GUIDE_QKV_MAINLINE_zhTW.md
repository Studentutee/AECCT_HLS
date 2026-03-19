# REVIEWER_GUIDE_QKV_MAINLINE_zhTW
Date: 2026-03-19

## 1. 文件定位（累積版）
- 本文件是 **累積式 reviewer-facing guide**，對目前 accepted local-only Q/KV mainline 路徑做導讀。
- 本文件不是單一 round completion report；內容應持續 merge-forward，不覆蓋既有有效導讀。
- 本版明確以以下內容為合併來源：
  - Round 1 guide baseline：`docs/process/REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md@0001a68`
  - Round 2 guide baseline：`docs/process/REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md@b2079e3`

## 2. Covered Files
- `src/Top.h`
- `src/blocks/TransformerLayer.h`
- `src/blocks/AttnLayer0.h`
- `src/blocks/TernaryLiveQkvLeafKernel.h`
- `src/blocks/TernaryLiveQkvLeafKernelTop.h`
- `src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h`
- `src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h`

## 3. First-Pass Review Order
1. `Top.h`（外部 contract、FSM dispatch、layer orchestration）
2. `TransformerLayer.h`（Top -> Attn/FFN/LN integration boundary）
3. `AttnLayer0.h`（QKV/score/out 主計算路徑）
4. `TernaryLiveQkvLeafKernel*.h`（leaf row materialization 與 shape/metadata guard）

## 4. End-to-End Dataflow（3~8 步）
1. Top ingest `SET_W_BASE/LOAD_W/INFER`，建立可用 `W_REGION` 與 input payload。
2. Top 執行 preproc + pre-layernorm，產生 layer 輸入 `X_WORK` 映射區。
3. Top 在 layer0 先嘗試 Top-managed Q/KV mainline hook，建立 prebuilt/fallback 結果。
4. Top 將 base/range 與 prebuilt flags 傳入 `TransformerLayer(...)`。
5. `TransformerLayer` 委派 `AttnLayer0` 與 `FFNLayer0`，並完成 residual + LN glue。
6. `AttnLayer0` 在 QKV/score/out path 內呼叫 ternary leaf row materialization 並完成 write-back。
7. Top 完成 end LN + FinalHead，依 outmode 從對應 base 將結果寫到 `data_out`。

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

## 6. File Roles
### 6.1 `Top.h`
- 唯一外部 4-channel contract (`ctrl_cmd/ctrl_rsp/data_in/data_out`) 的 dispatch owner。
- 管理 `ST_CFG_RX/ST_PARAM_RX/ST_INFER_RX/ST_HALTED` 與 payload ingest。
- 管理 layer orchestration（含 layer0 mainline hook、mid/end LN 插入、FinalHead 接續）。

### 6.2 `TransformerLayer.h`
- 接收 Top 指派的 `x/scratch/param` 邊界與 prebuilt flags。
- 委派 `AttnLayer0<ATTN_STAGE_FULL>` 與 `FFNLayer0<FFN_STAGE_FULL>`。
- 完成 layer-local residual add + sublayer1 norm param load + LN 呼叫。
- 不擁有 Top-FSM、共享 SRAM policy、或 fallback 政策定義權。

### 6.3 `AttnLayer0.h`（Round 1 核心低層導讀保留）
- `ATTN_STAGE_QKV`：Q/K/V materialization（含 live path 與 fallback/bypass）。
- `ATTN_STAGE_SCORES`：QK score、softmax、V 加權輸出到 pre/post concat。
- `ATTN_STAGE_OUT`：將 `post_concat` 寫回 attention output base。
- 不負責 Top state machine、外部 command/rsp 協定、全域 SRAM ownership 仲裁。

### 6.4 `TernaryLiveQkvLeafKernel*.h`（Round 1 核心低層導讀保留）
- `ShapeConfig`：QKV compile-time shape/payload expectation SSOT。
- `LeafKernel`：row-kernel（metadata guard + ternary decode + MAC）。
- `LeafKernelTop`：local split-interface top wrapper。
- `LeafKernelCatapultPrepTop`：compile-prep wrapper/surface adapter。
- 不負責 runtime-variable shape negotiation、Top SRAM policy、closure 宣告。

## 7. Loop-Role Notes（代表家族）
### 7.1 `TOP_*`（Round 2 新增）
- `TOP_LAYER_ORCHESTRATION_LOOP`：layer 主排程迴圈。
- `TOP_OUTMODE_XPRED_WRITEBACK_LOOP` / `TOP_OUTMODE_LOGITS_WRITEBACK_LOOP`：Top 最終輸出寫回。
- `TOP_READ_MEM_STREAM_LOOP`：READ_MEM readback stream。
- `TOP_COPY_X_WORDS_LOOP` / `TOP_NORM_PARAM_COPY_LOOP`：中間快照與參數搬移輔助迴圈。

### 7.2 `TRANSFORMER_*`（Round 2 新增）
- `TRANSFORMER_LAYER_SUBLAYER1_NORM_PARAM_COPY_LOOP`：LN gamma/beta 載入。
- `TRANSFORMER_LAYER_FFN_RESIDUAL_ADD_LOOP`：FFN2 輸出與 residual 加總。

### 7.3 `ATTN_*`（Round 1 低層重點保留）
- `ATTN_QKV_KV_PRIME_COPY_LOOP`
- `ATTN_Q_SPLIT_TOKEN_LOOP` / `ATTN_Q_SPLIT_INPUT_COL_LOOP` / `ATTN_Q_SPLIT_OUTPUT_COL_LOOP`
- `ATTN_K_*` / `ATTN_V_*`
- `ATTN_SCORE_TOKEN_LOOP` / `ATTN_SCORE_HEAD_LOOP` / `ATTN_SCORE_KEY_TOKEN_LOOP` / `ATTN_SCORE_DOT_COL_LOOP`
- `ATTN_PRECONCAT_HEAD_COL_LOOP` / `ATTN_PRECONCAT_KEY_TOKEN_ACC_LOOP`
- `ATTN_POSTCONCAT_COPY_LOOP` / `ATTN_OUT_WRITEBACK_LOOP`

### 7.4 `TERNARY_*`（Round 1 低層重點保留）
- `TERNARY_QKV_IMPL_OUT_ROW_LOOP`
- `TERNARY_WQ_SPLIT_OUT_ROW_LOOP` + `TERNARY_WQ_SPLIT_IN_COL_LOOP`
- `TERNARY_WK_SPLIT_OUT_ROW_LOOP` + `TERNARY_WK_SPLIT_IN_COL_LOOP`
- `TERNARY_WV_SPLIT_OUT_ROW_LOOP` + `TERNARY_WV_SPLIT_IN_COL_LOOP`

## 8. Most Important Code Regions（7 個）
1. `src/Top.h` `top(...)`（state + command dispatch）
2. `src/Top.h` `run_transformer_layer_loop(...)`（layer0 hook + fallback latch）
3. `src/Top.h` `infer_emit_outmode_payload(...)`（Top write-back boundary）
4. `src/Top.h` `handle_read_mem(...)`（Top readback boundary）
5. `src/blocks/TransformerLayer.h` `TransformerLayer(...)`（integration boundary）
6. `src/blocks/AttnLayer0.h` `AttnLayer0(...)`（QKV/score/out 主體）
7. `src/blocks/TernaryLiveQkvLeafKernel.h` `ternary_live_qkv_materialize_row_kernel_impl(...)`（leaf materialization/guard）

## 9. PASS 代表什麼 / 不代表什麼
- 目前 PASS 代表：
  - local-only accepted progress 可重現
  - 指定 regression/compile-prep/probe script 在本地可通過
  - handoff boundary 內主線行為符合目前 acceptance 條件
- 目前 PASS 不代表：
  - Catapult closure
  - SCVerify closure
  - full runtime/numeric/global closure

## 10. Round Notes（Convergence）
### 10.1 Round 1 additions（保留）
- 建立 Attn/ternary leaf 低層導讀與 loop-role 家族說明。
- 建立 buffer/ownership/fallback 檢查視角。

### 10.2 Round 2 additions（合併）
- 補 Top/Transformer integration boundary 專章與 `TOP_*`/`TRANSFORMER_*` loop 可視性。
- 補 Top write-back/readback 邊界導讀與高層 review order。

### 10.3 Remaining debt / next likely review target
- `Top.h`/`TransformerLayer.h` 之外的 design-side 歷史註解一致性仍可能有零星欠債。
- 下一個高價值目標可放在 `AttnLayer0.h` 與 leaf family 的 comment consistency 與 reviewer導讀深度對齊（不改行為前提）。
