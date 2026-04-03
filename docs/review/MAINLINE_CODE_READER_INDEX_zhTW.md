# MAINLINE_CODE_READER_INDEX_zhTW
Date: 2026-04-03

用途：這是一份 **主線閱讀索引**。
給現在想把 AECCT_HLS 主路看懂的人，一個比較不痛苦的閱讀順序。

邊界：
- 這份文件是 reader index，不是 closure report。
- 它不主張 Catapult closure，也不主張 SCVerify closure。
- 它的目的只有一個：告訴你現在先看哪幾份 reader guide，才不會直接被原始碼淹沒。

---

## 1. 先講一句話結論
現在不要直接從 `Top.h` 第一行一路讀到最後一行。

比較好的順序是：
1. 先建立整體地圖
2. 再看 layer glue
3. 最後進 block 本體

---

## 2. 建議閱讀順序

### 第一輪：先建立全圖
1. `ATTN_CODE_READER_GUIDE_zhTW.md`
2. `TOP_READER_GUIDE_zhTW.md`
3. `MAINLINE_CODE_READER_INDEX_zhTW.md`（本檔）

這一輪的目標只有一個：
知道誰是 owner、誰是接線板、誰是 block 主體。

---

### 第二輪：看中間最厚的整合層
4. `TRANSFORMERLAYER_READER_GUIDE_zhTW.md`
5. `ATTNLAYER0_READER_GUIDE_zhTW.md`
6. `ATTNLAYER0_STAGE_CROSSCHECK_zhTW.md`

這一輪的目標是：
把 attention 主線和 layer glue 的關係看清楚。

---

### 第三輪：把其他主要 block 補齊
7. `PREPROCEMBEDSPE_READER_GUIDE_zhTW.md`
8. `FFNLAYER0_READER_GUIDE_zhTW.md`
9. `LAYERNORMBLOCK_READER_GUIDE_zhTW.md`
10. `FINALHEAD_READER_GUIDE_zhTW.md`

這一輪的目標是：
把 attention 以外的主線 block 也看懂。

---

## 3. 如果你現在只想快速知道「各檔案是什麼」

### `Top.h`
系統 owner / dispatcher / observability board。

### `TransformerLayer.h`
layer 級接線板。

### `AttnLayer0.h`
attention block 入口與 migration 過渡邊界。

### `FFNLayer0.h`
FFN 的 W1 / RELU / W2 三段主體，帶 top-fed 與 fallback 共存。

### `LayerNormBlock.h`
比較乾淨的 two-pass token-wise block。

### `PreprocEmbedSPE.h`
較早收斂為 Top-managed window 風格的 block。

### `FinalHead.h`
Pass A / Pass B 分離的尾端輸出 block。

---

## 4. 你現在最值得記住的 5 句話
1. Top 是唯一 shared SRAM owner。
2. `TransformerLayer` 是接線板，不是 policy owner。
3. `AttnLayer0` 仍是 migration 過渡區，不要誤判成 fully migrated。
4. `FFNLayer0` 已有 top-fed handoff，但仍保留 SRAM fallback。
5. `LayerNorm` 和 `Preproc` 比較適合拿來建立閱讀信心。

---

## 5. 如果你現在閱讀負擔很重，先只看哪 3 份
1. `TOP_READER_GUIDE_zhTW.md`
2. `TRANSFORMERLAYER_READER_GUIDE_zhTW.md`
3. `ATTNLAYER0_READER_GUIDE_zhTW.md`

先把這三份吃掉，再回去看原始碼，會輕鬆很多。
