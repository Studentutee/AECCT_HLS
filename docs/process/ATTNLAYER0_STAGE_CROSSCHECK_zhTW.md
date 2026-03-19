# ATTNLAYER0_STAGE_CROSSCHECK_zhTW
Date: 2026-03-19

用途：提供 `AttnLayer0` 的 stage-by-stage reviewer cross-check。  
邊界：此文件只做 reviewer 導讀，不重述演算法細節或逐行 code 解釋。

## 1. 使用方式（先看）
1. 先用 `REVIEWER_GUIDE` 的 [6.3 AttnLayer0](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attnlayer0-role) 對齊 ownership/handoff 邊界。
2. 再依本文件 `QKV -> SCORES -> OUT` 順序檢查，避免跨 stage 誤讀。
3. 每 stage 完成後，回到 checklist 對應條目勾選。

<a id="stage-qkv"></a>
## 2. Stage: QKV
- Role:
  - 完成 Q/K/V materialization 主路徑。
  - 在既有旗標與 gate 條件下，處理 fallback/bypass 分流。
- Main inputs:
  - `x_in_base_word`
  - `sc.q_* / sc.k_* / sc.v_*` base windows
  - `q_prebuilt_from_top_managed` / `kv_prebuilt_from_top_managed`
  - live metadata（Q/K/V 對應權重與縮放資訊）
- Intermediates:
  - Q/K/V scratch windows
  - `act_q` mirror 區段
  - live gate 判斷結果（是否走 split-top 或 fallback/bypass）
- Outputs / write-back:
  - Q/K/V 與相關 `act_q` 寫入 attention scratch。
  - 本 stage 不做最終 `attn_out` write-back。
- Fallback / bypass relevance:
  - fallback meaning 的「政策鎖定」在 Top，QKV stage 只消費旗標執行對應路徑。
  - bypass copy 是顯式路徑，不是隱式副作用。
- Lower-level helper interaction:
  - 透過 ternary leaf split-interface wrappers 進行 row materialization。
- What to inspect first:
  - 先看 gating 與 prebuilt flags 對路徑選擇的影響。
  - 再看 Q/K/V 三路是否都可追蹤到一致的 fallback/bypass 邏輯。
- Primary pointer:
  - Code: `src/blocks/AttnLayer0.h` 的 `ATTN_STAGE_QKV`、`ATTN_Q*`/`ATTN_K*`/`ATTN_V*`
  - Guide: [6.3 AttnLayer0](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attnlayer0-role), [7.3 ATTN_*](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attn-loop-family)

<a id="stage-scores"></a>
## 3. Stage: SCORES
- Role:
  - 用 Q/K 計算 score，執行 softmax 前後處理，並進行 pre-concat 累積。
- Main inputs:
  - Q/K/V scratch windows
  - token/head 迴圈邊界
- Intermediates:
  - `score_row`
  - `prob_row`
  - pre/post concat scratch
- Outputs / write-back:
  - 產生 pre/post concat 相關中間結果。
  - 不直接對 `attn_out_base_word` 寫回。
- Fallback / bypass relevance:
  - SCORES stage 主要消費 QKV stage 已建立的 tensor；不新增 fallback policy。
- Lower-level helper interaction:
  - 直接依賴前一 stage 的 materialization 結果。
- What to inspect first:
  - 先確認 score/softmax/pre-concat 邊界是否與 OUT stage 分層。
  - 再檢查 loop family 是否支援 GUI trace。
- Primary pointer:
  - Code: `src/blocks/AttnLayer0.h` 的 `ATTN_STAGE_SCORES`、`ATTN_SCORE*`、`ATTN_PRECONCAT*`
  - Guide: [6.3 AttnLayer0](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attnlayer0-role), [7.3 ATTN_*](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attn-loop-family)

<a id="stage-out"></a>
## 4. Stage: OUT
- Role:
  - 完成 attention block 內最終輸出寫回到 `attn_out_base_word`。
- Main inputs:
  - `post_concat` scratch tensor
  - `attn_out_base_word`
- Intermediates:
  - 本 stage 以 copy/write-back 為主，無額外大型中間數值結構。
- Outputs / write-back:
  - `post_concat -> attn_out_base_word`。
  - 注意：這是 Attn block 邊界，不是 Top 最終 outmode write-back。
- Fallback / bypass relevance:
  - OUT stage 不定義 fallback meaning，只消費前段結果。
- Lower-level helper interaction:
  - 不再直接呼叫 ternary leaf；主要是收斂後寫回。
- What to inspect first:
  - 先確認 write-back boundary 與 Top 的最終 write-back boundary 不混淆。
  - 再確認 loop label 對 GUI 的可追蹤性。
- Primary pointer:
  - Code: `src/blocks/AttnLayer0.h` 的 `ATTN_STAGE_OUT`、`ATTN_OUT_WRITEBACK_LOOP`
  - Guide: [6.3 AttnLayer0](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attnlayer0-role), [5. Ownership / Boundary](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ownership-boundary)

## 5. Reviewer quick pitfalls（常見誤讀）
- 把 `q_prebuilt_from_top_managed` / `kv_prebuilt_from_top_managed` 誤讀成 Attn 自主政策決策。
- 把 SCORES stage 誤讀成最終 output write-back stage。
- 把 Attn OUT write-back 誤讀成 Top outmode write-back。
- Primary pointer:
  - [REVIEW_CHECKLIST：Write-back/Fallback/Ownership](./REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md#writeback-fallback-ownership)
