# TOP_TRANSFORMER_QUICKCHECK_zhTW
Date: 2026-03-19

用途：提供 `Top.h` + `TransformerLayer.h` 的 reviewer-facing 15 分鐘快檢路徑。

<a id="15-minute-fast-scan-order"></a>
## 1. 15-minute fast scan order
1. 先看 `Top` ownership 與外部 contract 邊界。
2. 再看 `Top` dispatch/delegation 與 fallback meaning 鎖定位置。
3. 再看 `Top` write-back / readback 邊界。
4. 最後看 `TransformerLayer` 是否只做 layer-local integration（不越權）。

Primary pointer:
- `REVIEWER_GUIDE` [6.1 Top](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#top-role)
- `REVIEWER_GUIDE` [6.2 TransformerLayer](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#transformerlayer-role)
- `REVIEWER_GUIDE` [5. Ownership / Boundary](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ownership-boundary)

## 2. Key questions（快速提問）
- Ownership:
  - Top 是否仍是唯一 shared SRAM owner？
  - TransformerLayer 是否沒有接管 Top policy 決策？
- Dispatch / Delegation:
  - Top 是否只 dispatch，TransformerLayer 是否只 delegate Attn/FFN/LN？
- Fallback meaning:
  - fallback flag 的語意是否在 Top hook 鎖定，而非在下游臨時定義？
- Write-back / readback:
  - Attn OUT write-back 與 Top outmode write-back 是否仍清楚分層？
  - `handle_read_mem(...)` 邊界是否與 active flow 不混淆？

Primary pointer:
- `src/Top.h::run_transformer_layer_loop(...)`
- `src/Top.h::infer_emit_outmode_payload(...)`
- `src/Top.h::handle_read_mem(...)`
- `src/blocks/TransformerLayer.h::TransformerLayer(...)`

## 3. First loops/regions to inspect（Catapult GUI or code）
- `TOP_LAYER_ORCHESTRATION_LOOP`
- `TOP_OUTMODE_XPRED_WRITEBACK_LOOP`
- `TOP_OUTMODE_LOGITS_WRITEBACK_LOOP`
- `TOP_READ_MEM_STREAM_LOOP`
- `TRANSFORMER_LAYER_SUBLAYER1_NORM_PARAM_COPY_LOOP`
- `TRANSFORMER_LAYER_FFN_RESIDUAL_ADD_LOOP`

Primary pointer:
- `REVIEWER_GUIDE` [7.1 TOP_*](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#top-loop-family)
- `REVIEWER_GUIDE` [7.2 TRANSFORMER_*](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#transformer-loop-family)

## 4. Common misreadings（常見誤讀）
- 誤讀 1: TransformerLayer 是全域 policy owner。  
  正確: 它是 layer-local integration boundary，不擁有 Top policy。
- 誤讀 2: fallback meaning 在 AttnLayer0 或 leaf 才決定。  
  正確: fallback meaning 在 Top hook 建立與鎖定，下游只消費旗標。
- 誤讀 3: Attn OUT write-back = Top 最終 outmode write-back。  
  正確: 這是兩層不同邊界。

## 5. PASS does / does-not prove（快檢版）
- PASS currently proves:
  - local-only reviewer-facing path 可重現且邊界可追蹤。
- PASS currently does not prove:
  - Catapult closure
  - SCVerify closure
  - full runtime/global closure

Primary pointer:
- `REVIEWER_GUIDE` [9. PASS semantics](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#pass-semantics)
- `REVIEW_CHECKLIST` [Write-back/Fallback/Ownership](./REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md#writeback-fallback-ownership)
