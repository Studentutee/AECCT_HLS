# REVIEWER_GUIDE_QKV_MAINLINE_zhTW
Date: 2026-03-19

## 1. 文件定位
- 本文件是 reviewer-facing sidecar，目標是降低目前 local-only QKV mainline 的理解成本。
- 本文件聚焦已接受路徑的「可讀性與邊界說明」，不代表新增功能或架構改版。
- 本輪（Debt Sprint 1 / Round 2）重點是 `Top.h` 與 `TransformerLayer.h` 的 reviewer-facing understanding debt backfill。

## 2. 本輪 Scope 與 Non-Goals
### Scope In
- `src/Top.h`
- `src/blocks/TransformerLayer.h`
- 本文件（reviewer guide）

### Scope Out
- 不做 AE/AF extension
- 不做新功能、演算法、shape、quant policy 變更
- 不改 Top 外部 public contract
- 不做 block graph redesign
- 不做 Catapult pragma / scheduler retuning

## 3. Files Covered（本路徑全景）
- `src/Top.h`
- `src/blocks/TransformerLayer.h`
- `src/blocks/AttnLayer0.h`
- `src/blocks/TernaryLiveQkvLeafKernel.h`
- `src/blocks/TernaryLiveQkvLeafKernelTop.h`
- `src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h`
- `src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h`

## 4. Top.h 專章（Round 2 主焦點）
### 4.1 Top 擁有什麼
- 唯一外部 4-channel contract dispatch（`ctrl_cmd/ctrl_rsp/data_in/data_out`）。
- Top-FSM / receiver state（`ST_IDLE/ST_CFG_RX/ST_PARAM_RX/ST_INFER_RX/ST_HALTED`）。
- 共享 SRAM 的 ownership、lifetime、readback/dispatch 邊界。
- Layer loop orchestration（layer 迴圈、X page alternation、mid/end LN 插入）。

### 4.2 Top dispatch 與 sub-block ownership 分界
- Top 決定：`X_WORK` / `SCRATCH` / `W_REGION` 的 base 與呼叫時機。
- sub-block（`TransformerLayer/AttnLayer0/...`）只消費 Top 指派邊界，不宣告共享 SRAM ownership。

### 4.3 Fallback / not-fallback 語意建立點
- Layer0 的 Top-managed Q/KV hook 由 `run_transformer_layer_loop(...)` 內 `lid == 0` 分支建立。
- `p11ad_mainline_q_path_taken` / `p11ad_q_fallback_taken` 與 `p11ac_mainline_path_taken` / `p11ac_fallback_taken` 皆在 Top 端被鎖定與回報。

### 4.4 Round 2 新增的 Top loop labels（Catapult GUI 可視）
- `TOP_LAYER_ORCHESTRATION_LOOP`
- `TOP_OUTMODE_XPRED_WRITEBACK_LOOP`
- `TOP_OUTMODE_LOGITS_WRITEBACK_LOOP`
- `TOP_READ_MEM_STREAM_LOOP`
- `TOP_COPY_X_WORDS_LOOP`
- `TOP_NORM_PARAM_COPY_LOOP`

## 5. TransformerLayer.h 專章（Round 2 主焦點）
### 5.1 TransformerLayer 接受什麼
- 由 Top 傳入的 layer 執行邊界：
  - `x_in_base_word`, `x_out_base_word`
  - `LayerScratch sc`
  - `LayerParamBase pb`
  - `kv_prebuilt_from_top_managed`, `q_prebuilt_from_top_managed`

### 5.2 TransformerLayer 委派給誰
- Attention 主體委派給 `AttnLayer0<ATTN_STAGE_FULL>(...)`。
- FFN 主體委派給 `FFNLayer0<FFN_STAGE_FULL>(...)`。
- 本檔負責 layer integration glue：residual add + layer sublayer1 norm parameter load + LayerNormBlock 呼叫。

### 5.3 TransformerLayer 不擁有什麼
- 不擁有 Top-FSM。
- 不擁有共享 SRAM arbitration/lifetime policy。
- 不定義 fallback 政策本身，只消費 Top 傳入的 prebuilt/fallback 結果。

### 5.4 Round 2 新增的 TransformerLayer loop labels
- `TRANSFORMER_LAYER_SUBLAYER1_NORM_PARAM_COPY_LOOP`
- `TRANSFORMER_LAYER_FFN_RESIDUAL_ADD_LOOP`

## 6. First-Pass Review Order（建議）
1. `src/Top.h` `top(...)`：外部 contract dispatch 與 state gating。
2. `src/Top.h` `run_transformer_layer_loop(...)`：layer0 Q/KV mainline hook 與 fallback latch。
3. `src/blocks/TransformerLayer.h` `TransformerLayer(...)`：Top -> Attn/FFN/LN handoff。
4. `src/blocks/AttnLayer0.h` `AttnLayer0(...)`：QKV/score/out 實際計算路徑。
5. `src/blocks/TernaryLiveQkvLeafKernel*.h`：row-kernel materialization 與 metadata/shape guard。

## 7. 本輪更新後最重要的 5 個 code regions
1. `src/Top.h` `run_transformer_layer_loop(...)`（layer orchestration + fallback meaning）
2. `src/Top.h` `infer_emit_outmode_payload(...)`（Top write-back boundary）
3. `src/Top.h` `handle_read_mem(...)`（debug readback boundary）
4. `src/Top.h` `top(...)`（external contract dispatch）
5. `src/blocks/TransformerLayer.h` `TransformerLayer(...)`（integration boundary）

## 8. Dataflow 摘要（本輪 reviewer 角度）
1. Top ingest `SET_W_BASE/LOAD_W/INFER`，建立 `W_REGION` 與 input payload。
2. Top 跑 preproc + pre-layernorm，準備 layer input（`X_WORK` 對應區）。
3. Top 在 layer0 先嘗試 Top-managed Q/KV path，建立 prebuilt/fallback 結果。
4. Top 把 base/buffer 邊界與 prebuilt flags 交給 `TransformerLayer`。
5. `TransformerLayer` 委派 `AttnLayer0`；`AttnLayer0` 依 flags 進入既有 QKV path。
6. `AttnLayer0` 內部使用 ternary leaf path 完成 row materialization（Q/K/V）與後續 score/out 流程。
7. 回到 Top，完成 end LN + FinalHead，再依 outmode 做 write-back 到 `data_out`。

## 9. Ownership Boundary 摘要
- SRAM 決策 owner：Top（唯一共享 SRAM owner）。
- `W_REGION`：由 Top 管理載入與可見範圍；block 只讀指定區段。
- `X_WORK`：Top 管理 phase-safe overwrite 與 layer 間交接；block 在指定 base 下讀寫。
- `SCR_K/SCR_V`：attention scratch，由 Top 指派落點與生命週期邊界。
- Fallback 語意 owner：Top（在 layer0 hook 區塊定義並鎖定標誌）；TransformerLayer/AttnLayer0 消費該語意。

## 10. PASS 代表什麼 / 不代表什麼（本輪）
- 代表：
  - local-only accepted progress 仍可重現
  - reviewer-facing 可讀性與邊界說明改善
  - Catapult GUI loop 可辨識性提升（透過穩定 label）
- 不代表：
  - Catapult closure
  - SCVerify closure
  - full runtime/numeric/global closure

## 11. Catapult / SCVerify 狀態宣告
- 若本輪未執行 Catapult 或 SCVerify，則 closure 明確維持 deferred。
- 本文件中的 PASS 用語僅對應 local-only acceptance 與本輪指定腳本證據範圍。
