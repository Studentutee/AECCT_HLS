# PREPROC_CHANNELIZATION_PILOT_CONTRACT_v0

狀態：draft v0  
定位：PreprocEmbedSPE 的 **transport contract / dataflow contract** 草案  
治理口徑：local-only / compile-first / evidence-first / not Catapult closure / not SCVerify closure

---

## 0. 一句話目標

把 **PreprocEmbedSPE** 定成第一個正式的 **Top memory-to-stream + block stream-compute** pilot：

- **Top** 負責 shared SRAM / SCRATCH 的 ownership、tile read/write transport、base/len/region bookkeeping
- **PreprocEmbedSPE** 只 consume `ac_channel` payload、維持 block-local / tile-local state、輸出完整 `preproc_x` token
- sub-block 不自行長出 shared SRAM request semantics
- 外部 Top 4-channel contract 不變

---

## 1. 文件目的與範圍

### 1.1 本文件要回答什麼
本文件固定以下內容：

1. PreprocEmbedSPE 的 block 邊界
2. Preproc pilot 的正式 channel inventory
3. 每條 channel 的 producer / consumer 與 payload 語意
4. variable-major ingest 的 consume 順序
5. `check_parity` 的 backing-store-first 規則
6. `preproc_x` 的輸出順序
7. pilot-local checker / acceptance 方向
8. 未來 FP16 migration 的保留接口

### 1.2 In Scope
- `PreprocEmbedSPE` 的 channelized transport contract
- `input_y` 的 variable-major ingest
- `H_by_var` / adjacency-by-variable consume
- `check_parity` backing-store-first 規則
- `preproc_x` token stream output
- task-local checker / compile-first 驗收

### 1.3 Out of Scope
- 不改 Top 外部 4-channel contract
- 不改 TransformerLayer 主鏈
- 不做 broad refactor
- 不做 Catapult closure
- 不做 SCVerify closure
- 不把 FP32 -> FP16 綁進本輪
- 不讓 sub-block 直接發 shared SRAM request

---

## 2. Block 邊界定義

## 2.1 Block 名稱
- `PreprocEmbedSPE`

## 2.2 邏輯輸入
- `input_y`
- variable-side embedding / parameter payload
- `lpe_token`
- `H_by_var` / adjacency-by-variable
- `check_parity` backing tile（由 Top 搬運）

## 2.3 中間值
- `abs_y[v]`
- `hard_bit[v]`
- `var_feature[v]`
- `check_parity[c]`
- `check_feature[c]`
- `node_feature[token]`

## 2.4 正式輸出
- `preproc_x[token]`

### 2.5 關於矩陣形狀的說明
`preproc_x` 在邏輯上會形成一面 `tokens × d_model` 的矩陣，其中：

- token 是 **列(Row)**
- `d_model` 是 **行(Colume)**

但在 transport 介面上，**不是一次傳整面矩陣**，而是：
- 逐 token 輸出
- 每次 `preproc_x_out_ch` 輸出一個完整 token 向量

### 2.6 名詞澄清
- `node_feature` 是 block 內部邏輯中介值
- `preproc_x` 才是對下游的正式輸出 payload
- `embedding` 與 `lpe_token` 在輸入側分開傳
- 輸出側不拆兩條，直接輸出完整 `preproc_x[token]`

---

## 3. 正式 transport channels

本 pilot 採 **7 條正式 channel**。

### 3.1 Input channels
1. `y_in_ch`
2. `h_by_var_adj_ch`
3. `embed_param_ch`
4. `lpe_token_ch`
5. `check_acc_rd_ch`

### 3.2 Output channels
6. `check_acc_wr_ch`
7. `preproc_x_out_ch`

---

## 4. 每條 channel 的角色

## 4.1 `y_in_ch`
- Producer：Top
- Consumer：PreprocEmbedSPE
- Payload：`y[v]`
- 順序：variable-major
- 第 `k` 筆對應 `var_idx = k`

## 4.2 `h_by_var_adj_ch`
- Producer：Top
- Consumer：PreprocEmbedSPE
- Payload：當前 variable 對應的 adjacency list
- 注意：
  - 這不是 full H matrix stream
  - 也不是 full row-scan
  - transport 只傳該 `y[v]` 需要更新的 check adjacency

## 4.3 `embed_param_ch`
- Producer：Top
- Consumer：PreprocEmbedSPE
- Payload：該 token 對應的 embedding / parameter payload
- 第一版 baseline precision：FP32

## 4.4 `lpe_token_ch`
- Producer：Top
- Consumer：PreprocEmbedSPE
- Payload：`lpe_token`
- 第一版 baseline precision：FP32
- LPE 不在線上計算，這裡只傳已準備好的 `lpe_token`

## 4.5 `check_acc_rd_ch`
- Producer：Top
- Consumer：PreprocEmbedSPE
- Payload：當前 `CHECK_PARITY_TILE` 的 backing data
- 角色：
  - Top 從 backing store / SCRATCH 讀 tile
  - 再經由 channel 送進 block
- 這不是 block 自己讀 SRAM

## 4.6 `check_acc_wr_ch`
- Producer：PreprocEmbedSPE
- Consumer：Top
- Payload：更新後的 `CHECK_PARITY_TILE`
- 角色：
  - block 更新 tile
  - 再交回 Top 寫回 backing store / SCRATCH

## 4.7 `preproc_x_out_ch`
- Producer：PreprocEmbedSPE
- Consumer：Top / downstream
- Payload：完整 `preproc_x[token]`
- 順序：
  1. 先 variable-side tokens
  2. 再 check-side tokens

---

## 5. Packet / Payload 預設欄位（pilot 預設全部帶 formal metadata）

### 5.0 預設政策
本 pilot 預設：**所有正式 channel packet 都攜帶 formal metadata**。

目的不是把所有 block-local loop counter 都搬進 transport，而是避免：
- Top 與 block 各自重建正式順序 / 身分語意
- 把 payload 解讀綁死在隱含 consume 次數
- 讓 token / var / tile / boundary 的正式語意只存在於某一側的 local counter

### 5.0.1 本文件中的 formal metadata 是什麼
本文件中的 formal metadata，指的是會影響 payload 正式語意的欄位，例如：
- `var_idx`
- `check_idx` / `check_idx_list[]`
- `token_kind`
- `token_idx`
- `tile_id`
- `adj_count`
- `word_count`
- `is_last_y` / `end_of_*` 類 boundary 欄位

### 5.0.2 不屬 formal metadata 的東西
下列項目屬 block-local / tile-local implementation detail，不進正式 transport packet：
- 向量內 `dim` 掃描 index
- MAC / reduce loop 的 local index
- local FIFO / local buffer 的內部步數
- 純 block 內部暫存器輪轉次序

### 5.0.3 後續優化原則
本版先採「全部 packet 都帶 formal metadata」，之後若要省欄位，必須先證明：
- channel identity + phase/window + fixed sequence order 已足以唯一推出 payload 語意
- 不會讓 Top 與 block 各自維護第二套正式 counter / boundary 規則
- checker / trace compare / reviewer 導讀不會因此退化

本版先不做此類省欄位優化。

## 5.1 `y_in_ch` payload
pilot 預設欄位：
- `var_idx`
- `is_last_y`
- `y_value`

說明：
- 雖然 variable-major 順序理論上可由 consume 次數推出 `var_idx`，但本 pilot 不依賴 block 端 local counter 來重建正式 variable identity。
- `is_last_y` 作為 boundary metadata，供 block / checker / Top hook 共用。

## 5.2 `h_by_var_adj_ch` payload
pilot 預設欄位：
- `var_idx`
- `adj_count`
- `check_idx_list[]`

說明：
- 本版以 `adj_count + check_idx_list[]` 表示有效 adjacency 範圍。
- `adj_count` 表示有效 adjacency 項數。
- 對齊而多出的 padding / 無效位元，由 Top 在送入 channel 前丟掉。
- Preproc 只 consume `adj_count` 內有效項。
- 若後續要改為逐 adjacency item packet，再另行定義 `adj_idx` / `is_last_adj_for_var`；本版先不展開。

## 5.3 `embed_param_ch` payload
pilot 預設欄位：
- `token_kind`
- `token_idx`
- `embed_word_count`
- `embed_words[]`

說明：
- `token_kind` 可用：
  - `VAR_TOKEN`
  - `CHECK_TOKEN`

## 5.4 `lpe_token_ch` payload
pilot 預設欄位：
- `token_kind`
- `token_idx`
- `lpe_word_count`
- `lpe_words[]`

說明：
- 即使 `lpe_token_ch` 在固定順序下理論上可由 Top / block 雙邊 counter 推出 `token_idx`，本 pilot 仍預設顯式攜帶，避免雙邊各算一份正式 token counter。

## 5.5 `check_acc_rd_ch` payload
pilot 預設欄位：
- `tile_id`
- `word_count`
- `acc_words[]`

## 5.6 `check_acc_wr_ch` payload
pilot 預設欄位：
- `tile_id`
- `word_count`
- `acc_words[]`

## 5.7 `preproc_x_out_ch` payload
pilot 預設欄位：
- `token_kind`
- `token_idx`
- `word_count`
- `x_words[]`

說明：
- 雖然輸出順序固定為「先 variable-side、再 check-side」，本 pilot 仍保留 `token_kind + token_idx`，避免 Top / downstream / checker 依賴隱含 consume 次數來重建輸出 identity。

---

## 6. `check_parity` backing 規則

## 6.1 正式方向
本 pilot 採 **backing-store-first**：

- `check_parity[c]` 不預設 full-register 化
- 主體放在 backing store
- 只有當前 `CHECK_PARITY_TILE` 進 regs / 小型 local state
- 不做每筆 `y` 的 shared SRAM rolling read-modify-write

## 6.2 本 pilot 的 backing 模式
本 pilot 已拍板採：

- **Top-managed SCRATCH backing**

也就是：
- backing store 在 Top 管理的 SCRATCH / backing region
- `check_acc_rd_ch` / `check_acc_wr_ch` 是正式 transport
- Top 仍是唯一 shared-SRAM owner

## 6.3 Sub-block 禁止事項
PreprocEmbedSPE 不得：
- 自行配置 shared SRAM
- 自行決定 shared SRAM arbitration
- 自行定義第二套 shared-memory contract
- 以 hidden request path 偷渡 SRAM access semantics

---

## 7. 資料流順序（正式版本）

## 7.1 variable-side ingest 順序
對每個 `y[v]`，固定順序如下：

1. 從 `y_in_ch` 取得 `y[v]`
2. 從 `embed_param_ch` 取得對應 variable token 的 embedding payload
3. 從 `h_by_var_adj_ch` 取得該 `y[v]` 對應的 adjacency list
4. 依 adjacency list 決定需要更新哪些 `check_parity` entries
5. 必要時從 `check_acc_rd_ch` 取入對應 `CHECK_PARITY_TILE`
6. 完成 tile-local 更新
7. 必要時透過 `check_acc_wr_ch` 回寫更新後 tile
8. 形成 variable-side token 所需內容

## 7.2 check-side finalize 順序
- 必須等全部 `input_y` consume 完成後，才可開始
- 對所有 check nodes：
  1. 從 backing store 取 `check_parity`
  2. 產生 `check_feature`
  3. 形成 check-side token

## 7.3 Output 順序
`preproc_x_out_ch` 的正式順序固定為：

1. variable-side tokens：
   - `var[0]`
   - `var[1]`
   - ...
   - `var[n-1]`

2. check-side tokens：
   - `check[0]`
   - `check[1]`
   - ...
   - `check[m-1]`

---

## 8. strict group-ready 規則

第一版同步策略採 **strict group-ready**。

### 定義
當前工作單位所需 payload 未齊備前，PreprocEmbedSPE 不開始 consume 該工作單位。

### variable-side 需要齊備的輸入
- `y_in_ch`
- `embed_param_ch`
- `h_by_var_adj_ch`
- 必要時對應的 `check_acc_rd_ch`

### check-side 需要齊備的輸入
- 對應 check token 的 `embed_param_ch`
- 對應 check token 的 `lpe_token_ch`
- 必要的 `check_acc_rd_ch`

---

## 9. adjacency list 規則

## 9.1 採用形式
本 pilot 採 **adjacency list**，不採 bitmask / full packed matrix。

## 9.2 理由
- 與 variable-major ingest 對齊
- 與 `H_by_var` / adjacency-by-variable 方向一致
- 不需要傳 full-row padding
- 比較適合 stream consume

## 9.3 對齊 / padding 規則
- 若 Top 端為了 memory layout / word alignment 需要 padding
- **padding cleanup 由 Top 負責**
- `h_by_var_adj_ch` 只送有效 adjacency items
- PreprocEmbedSPE 不負責解釋 padding

---

## 10. Tile 規則

## 10.1 正式固定的內容
- `DATA_W = 32 bits`
- streams 以 word 為基本單位

## 10.2 pilot-local tile 設定
本 pilot 先採：

- `CHECK_PARITY_TILE_WORDS = 4`

也就是：
- 一個 tile 預設是 4 words
- 等價 128 bits payload

## 10.3 這一條現在不能 overclaim 的地方
- `CHECK_PARITY_TILE_WORDS = 4` 是 **pilot-local choice**
- 不是全專案 frozen global SRAM port width
- 後續可依 compile / schedule / checker 結果調整

---

## 11. Precision 規則（保留 FP16 hook）

## 11.1 本版 baseline
本版 baseline 先採：
- `embed_param_ch` = FP32 payload
- `lpe_token_ch` = FP32 payload
- `preproc_x_out_ch` = FP32 payload

## 11.2 未來保留
本文件保留未來獨立 bounded migration：
- `PREPROC_FP16_PAYLOAD_MIGRATION`

## 11.3 規則
- 本輪不把 FP32 -> FP16 綁進 channelization
- precision 應設計成 compile-time contract parameter
- transport contract 本身保持 precision-agnostic

---

## 12. Done / writeback / free-point 邊界

## 12.1 variable-side tile done
當某個 `CHECK_PARITY_TILE`：
- 所有本輪應用於該 tile 的 adjacency updates 已完成
- 且更新後資料已送出 `check_acc_wr_ch`
才視為該 tile 本輪 writeback 完成

## 12.2 check-side finalize done
當所有 check nodes 的：
- `check_feature`
- check-side token 組裝
都完成時，check-side finalize 才算完成

## 12.3 Preproc 整體 done
當：
1. 所有 variable-side tokens 已輸出
2. 所有 check-side tokens 已輸出
3. 所有必要 accumulator tiles 已完成 writeback
才視為本次 PreprocEmbedSPE inference done

---

## 13. 驗收 / checker 計畫（第一版）

本版只要求：
- local-only
- compile-first
- evidence-first

### 建議 PASS banners
- `PREPROC_VAR_STREAM_ORDER PASS`
- `PREPROC_ADJ_LIST_CONSUME PASS`
- `PREPROC_ACC_TILE_RMW PASS`
- `PREPROC_CHECK_FINALIZE_ORDER PASS`
- `PREPROC_OUTPUT_ORDER PASS`
- `NO_DIRECT_SRAM_REQUEST_IN_PREPROC PASS`

### 驗收重點
- sequence 正確
- producer / consumer 正確
- `check_acc_rd_ch` / `check_acc_wr_ch` 邊界正確
- 無 hidden SRAM request
- 輸出順序正確
- 不 overclaim 成 Catapult closure / SCVerify closure

---

## 14. 實作提示（給後續 Codex / reviewer）

## 14.1 channel 命名原則
- 以 payload class 命名
- 不以 phase 名稱假裝單一混合通道
- 例如：
  - `y_in_ch`
  - `h_by_var_adj_ch`
  - `embed_param_ch`
  - `lpe_token_ch`
  - `check_acc_rd_ch`
  - `check_acc_wr_ch`
  - `preproc_x_out_ch`

## 14.2 design-side 註解原則
- 註解以 ASCII / English 為主
- 中文導讀放 repo-tracked docs / sidecar
- loop 建議加穩定 label
- 要註明 input -> intermediate -> output 資料流
- 要註明 ownership / writeback boundary

## 14.3 不要做的事
- 不要把 sequence order 當成唯一語意來源
- 不要把 H matrix 當成 full dense stream 傳
- 不要把 `check_parity` 做成每筆 `y` 的 shared SRAM rolling RMW
- 不要讓 block 自己決定 shared SRAM address / arbitration
- 不要現在就把 FP16 和 transport cut 綁一起

---

## 15. 本文件目前的決策摘要

### 已拍板
- 完整版 pilot
- Top-managed SCRATCH backing
- adjacency list
- padding 由 Top 丟掉
- 輸出是完整 `preproc_x[token]`
- 先 variable-side tokens，再 check-side tokens
- baseline precision 先 FP32
- 保留未來 FP16 hook

### 尚未凍結成全域規格
- `CHECK_PARITY_TILE_WORDS = 4` 只屬 pilot-local
- packet 欄位仍可在不破壞 contract 下微調
- checker banner 名稱可依 task-local runner 調整
