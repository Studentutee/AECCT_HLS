# REVIEWER_GUIDE_QKV_MAINLINE_zhTW
Date: 2026-03-19

## 1. 文件定位
- 本文件是 reviewer-facing sidecar，針對目前已接受的 local-only ternary/QKV mainline 路徑做快速導讀。
- 本文件重點是「看得懂目前路徑」，不是新增功能或宣稱 Catapult/SCVerify closure。

## 2. Files Covered
- `src/Top.h`
- `src/blocks/TransformerLayer.h`
- `src/blocks/AttnLayer0.h`
- `src/blocks/TernaryLiveQkvLeafKernel.h`
- `src/blocks/TernaryLiveQkvLeafKernelTop.h`
- `src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h`
- `src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h`

## 3. 為什麼這些檔案重要
- `Top.h`：唯一 Top contract 與 SRAM owner，負責 CFG/PARAM/INFER/READ_MEM state 與 block 排程。
- `TransformerLayer.h`：layer 內 attention + ffn + residual + layernorm 的整合入口。
- `AttnLayer0.h`：目前 accepted ternary/QKV local path 的核心執行點（Q/K/V materialization、score、write-back）。
- `TernaryLiveQkvLeafKernel*.h`：Q/K/V row materialization 的固定 shape kernel 與 Catapult/compile-prep wrapper 面。

## 4. 架構角色摘要
### 4.1 `Top.h` 目前角色
- 接收 `ctrl_cmd/ctrl_rsp/data_in/data_out` 四通道外部 contract。
- 維持 Top 狀態機與 ingest（`ST_CFG_RX/ST_PARAM_RX/ST_INFER_RX/ST_HALTED`）。
- 透過 `run_transformer_layer_loop` 在 layer 0 將 Top-managed Q 與 Top-managed K/V mainline hook 接入 `TransformerLayer`。
- Top 決定 base/range/phase，block 僅消費這些已給定邊界。

### 4.2 `TransformerLayer.h` 目前角色
- 用 `TransformerLayer(...)` 串接：
  - `AttnLayer0<ATTN_STAGE_FULL>`
  - `FFNLayer0<FFN_STAGE_FULL>`
  - residual add
  - layer sublayer1 norm
- 接受 `kv_prebuilt_from_top_managed` 與 `q_prebuilt_from_top_managed` 旗標，把 layer0 mainline hook 傳入 `AttnLayer0`，避免重做已在 Top 管理路徑完成的 materialization。

### 4.3 `AttnLayer0.h` 目前角色
- 在 `ATTN_STAGE_QKV`：做 Q/K/V materialization（live path + fallback/bypass path）。
- 在 `ATTN_STAGE_SCORES`：做 QK score、softmax、與 V 加權輸出到 `pre/post concat`。
- 在 `ATTN_STAGE_OUT`：把 `post_concat` 寫回 attention output base。
- 不負責：
  - Top 狀態機管理
  - 外部協定回應
  - SRAM ownership 仲裁

### 4.4 `TernaryLiveQkvLeafKernel*.h` 目前角色
- `TernaryLiveQkvLeafKernelShapeConfig.h`：QKV compile-time shape/payload expectation SSOT 常數。
- `TernaryLiveQkvLeafKernel.h`：row-kernel 實作（metadata guard + ternary decode + MAC）。
- `TernaryLiveQkvLeafKernelTop.h`：local split-interface top wrapper。
- `TernaryLiveQkvLeafKernelCatapultPrepTop.h`：compile-prep top wrapper（surface adapter）。
- 不負責：
  - runtime-variable shape negotiation
  - Top-level SRAM policy/ownership 定義
  - Catapult closure 宣告

## 5. 3~8 步資料流摘要（本路徑 7 步）
1. `Top.h` ingest `SET_W_BASE/LOAD_W/INFER`，建立可用的 W_REGION 與 input payload。
2. `run_infer_pipeline` 先做 preproc + pre-layernorm，產生 layer 輸入 X。
3. `run_transformer_layer_loop` 在 layer0 先跑 Top-managed Q/KV mainline hooks（若成功則設 prebuilt flags）。
4. `TransformerLayer` 呼叫 `AttnLayer0`，依 prebuilt flags 決定 materialization/bypass 邊界。
5. `AttnLayer0` 在 QKV stage 生成（或 fallback copy）Q/K/V 與 act_q；score stage 生成注意力輸出到 pre/post concat。
6. `AttnLayer0` OUT stage 把結果寫到 layer 後續消費的 attention output base；`TransformerLayer` 再進 FFN + residual + LN。
7. Top 完成 end LN + FinalHead，依 outmode 從對應 base 輸出 data_out。

## 6. Touched Buffers / SRAM / Ownership Boundary
- W_REGION（參數/payload/metadata）：由 Top 管理生命週期；kernel/blocks 只讀取指定範圍。
- X_WORK（主工作區）：layer input/output 的主映射區；覆寫必須符合 phase-safe overwrite。
- SCRATCH（含 SCR_K/SCR_V 等）：attention 中介資料區；依 phase/liveness 管理。
- `AttnLayer0` 主要 touch：
  - `x_in_base`, `q_base/k_base/v_base`
  - `score_base`, `softmax_base`
  - `pre_base`, `post_base`, `out_base`
- Ownership 邊界：
  - Top = 唯一共享 SRAM owner
  - `TransformerLayer/AttnLayer0/LeafKernel` = Top 指派範圍的 consumer/producer，不宣告共享 SRAM ownership

## 7. 最先看的 5 個 code regions
1. `Top` state + command dispatch: `src/Top.h` `top(...)`（約 L1010 起）
2. Layer mainline hook wiring: `src/Top.h` `run_transformer_layer_loop(...)`（約 L802 起）
3. Layer integration boundary: `src/blocks/TransformerLayer.h` `TransformerLayer(...)`（約 L72 起）
4. QKV/score/write-back 主體: `src/blocks/AttnLayer0.h` `AttnLayer0(...)`（約 L75 起）
5. Row-kernel materialization/guard: `src/blocks/TernaryLiveQkvLeafKernel.h` `ternary_live_qkv_materialize_row_kernel_impl(...)` 與 `ternary_live_l0_w{q,k,v}_materialize_row_kernel_split(...)`

## 8. Loop-role notes（優先檢查）
- `AttnLayer0.h`
  - `ATTN_QKV_KV_PRIME_COPY_LOOP`: QKV stage baseline priming copy
  - `ATTN_Q_SPLIT_TOKEN_LOOP` / `ATTN_Q_SPLIT_INPUT_COL_LOOP` / `ATTN_Q_SPLIT_OUTPUT_COL_LOOP`: Q split-top token/col materialization
  - `ATTN_K_*` / `ATTN_V_*` 對應 K/V split-top 與 fallback token/col loops
  - `ATTN_SCORE_TOKEN_LOOP` / `ATTN_SCORE_HEAD_LOOP` / `ATTN_SCORE_KEY_TOKEN_LOOP` / `ATTN_SCORE_DOT_COL_LOOP`: QK score核心計算
  - `ATTN_PRECONCAT_HEAD_COL_LOOP` / `ATTN_PRECONCAT_KEY_TOKEN_ACC_LOOP`: V 加權輸出累積
  - `ATTN_POSTCONCAT_COPY_LOOP` / `ATTN_OUT_WRITEBACK_LOOP`: concat 與最終寫回
- `TernaryLiveQkvLeafKernel.h`
  - `TERNARY_QKV_IMPL_OUT_ROW_LOOP`: generic SRAM row kernel output row loop
  - `TERNARY_WQ_SPLIT_OUT_ROW_LOOP` + `TERNARY_WQ_SPLIT_IN_COL_LOOP`
  - `TERNARY_WK_SPLIT_OUT_ROW_LOOP` + `TERNARY_WK_SPLIT_IN_COL_LOOP`
  - `TERNARY_WV_SPLIT_OUT_ROW_LOOP` + `TERNARY_WV_SPLIT_IN_COL_LOOP`

## 9. PASS 代表什麼 / 不代表什麼
- 目前 PASS（本家族）代表：
  - local-only accepted progress
  - 指定 local regression / compile-prep / probe 路徑可重現
  - handoff boundary 內的 mainline 行為符合當前 acceptance 條件
- 目前 PASS 不代表：
  - Catapult closure
  - SCVerify closure
  - full runtime/numeric/global closure

## 10. Reviewer 快速檢查建議
- 先看 Top layer0 hook 是否只在目標 layer 啟用（避免 scope 漂移）。
- 再看 `TransformerLayer -> AttnLayer0` flags 傳遞是否只控制 materialization bypass，不改其他 side effects。
- 再看 `AttnLayer0` 的 fallback/bypass 邊界是否維持原本寫回語意。
- 最後看 leaf kernel metadata guards 是否仍與 shape/payload SSOT 鏈一致。
