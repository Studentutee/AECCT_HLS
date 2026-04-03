# TRANSFORMERLAYER_READER_GUIDE_zhTW
Date: 2026-04-03

用途：給「現在要真的讀 `src/blocks/TransformerLayer.h`」的人看的中文導讀。

邊界：
- 這份文件是 reader guide，不是 closure report。
- 它不主張 Catapult closure，也不主張 SCVerify closure。
- 它的目的只有一個：讓你知道 `TransformerLayer.h` 現在到底在做哪幾段 glue、哪些 ownership 已往 Top 收、哪些地方還留 direct SRAM preload。

---

## 1. 先講一句話結論
`TransformerLayer.h` 現在不是「單純呼叫 attention 和 FFN 的薄 wrapper」；它其實是 **layer 級整合層**，同時負責：
- attention compatibility shell 的入口
- FFN payload 選路與 preload bridge
- residual add
- sublayer1 LayerNorm 的參數 fallback 與呼叫

白話講：
**如果 `Top.h` 是總控，`AttnLayer0.h` 是 attention block 入口，那 `TransformerLayer.h` 就是把一整層黏起來的接線板。**

---

## 2. 你先不要從第一行硬讀，先用這個順序看

### Step 1：先看檔頭那 3 句註解
檔頭其實已經有很重要的英文 intent：
- inputs 的 base word / scratch layout / prebuilt flag 都由 Top 擁有
- attention 內部運算委派給 `AttnLayer0`
- shared SRAM 的 lifetime / arbitration ownership 仍留在 Top

這三句其實已經把責任分界講完一半了。

### Step 2：先看 `TransformerLayerFfnTopfedHandoffDesc`
這段不是演算法。
它是在定義：
- W1 要用的 x / weight / bias
- W2 要用的 input / weight / bias
有沒有可能從 Top 先餵進來。

白話：
這是一張 **FFN 交接單**，不是 FFN 本體。

### Step 3：再看 `transformer_layer_select_topfed_words(...)`
這個小 helper 很值得看。
它在做的事很簡單：
- 如果 Top 真的有餵資料，而且 valid > 0，就優先選 Top 餵的 payload
- 不然就退回本地 preload 的那份資料

白話：
這就是 `TransformerLayer` 裡很典型的 **新路徑 / 舊路徑二選一**。

### Step 4：最後再看兩個主入口
- `TransformerLayerTopManagedAttnBridge(...)`
- `TransformerLayer(...)`

你可以先把它們理解成：
- 前者：array-shaped bridge 版本，偏 Catapult-facing
- 後者：一般 pointer 版本，現在主路比較常看到的入口

它們大方向幾乎一樣：
先處理 attention，再處理 FFN，再做 residual，再接 LayerNorm。

---

## 3. 這個檔案實際在做什麼

## 3.1 它不是 policy owner，它是 integration boundary
### 這是什麼
`TransformerLayer(...)` 本身不決定全域 policy。
像這些事情不是它擁有的：
- 哪一層是 target layer
- 哪一層可以關掉 attention compatibility shell
- shared SRAM 的正式仲裁規則
- global fallback policy

### 為什麼要這樣做
因為專案口徑是：
**Top 才是 shared SRAM 唯一 owner。**

所以 `TransformerLayer` 的角色比較像：
- 接收 Top 已決定好的 boundary 條件
- 依這些條件把 attention / FFN / residual / LN 串起來

### 對目前專案的影響
你看到很多 flag 從參數傳進來，不要把它誤會成這份檔自己在定 policy。
很多時候它只是「吃下 Top 的決定」。

### 你現在最需要注意什麼
最重要的一個例子就是：
- `attn_compat_shell_enable`

這個 flag 不是在 `TransformerLayer` 內部自己計算出來的。
它是 Top 先算，再傳進來。

---

## 3.2 attention 段：先看 shell gate，不要先看數學
### 這是什麼
attention 這段的關鍵不是 Q/K/V 數學，而是：
- shell 會不會跑
- 哪些子段已經 prebuilt
- 有沒有額外 top-fed out payload

### 為什麼要這樣做
因為目前正處在 bounded migration 階段。
專案不是一次把 `AttnLayer0` 全部改完，而是先讓 Top 能控制：
- 哪些工作已在上游做好
- 哪些情況還要留 compatibility shell

### 對目前專案的影響
你在 `TransformerLayer` 會看到：
- `kv_prebuilt_from_top_managed`
- `q_prebuilt_from_top_managed`
- `score_prebuilt_from_top_managed`
- `out_prebuilt_from_top_managed`
- `attn_out_topfed_payload_enable`

這些不是在重寫 attention 演算法，
而是在描述 **attention 哪些段落已先做好，後面需不需要再進舊殼層。**

### 你現在最需要注意什麼
這個判斷：
- `attn_fully_prebuilt_from_top_managed`
- `attn_shell_must_run`

白話：
- 四段都 prebuilt 好，而且沒有額外 top-fed out payload，才有機會完全不跑 shell
- 只要條件不滿，還是得進 `AttnLayer0` 那個 compatibility shell

所以你可以把這段看成：
**不是 attention 已經完全搬走，而是 shell 的進場條件已經被明確寫出來。**

---

## 3.3 FFN 段：名字叫 topfed，不代表已經完全脫離 SRAM
### 這是什麼
FFN 這段最長，也最容易誤判。
你會看到很多 loop 名字像：
- `TRANSFORMER_LAYER_FFN_TOPFED_X_PRELOAD_BRIDGE_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W1_PRELOAD_BRIDGE_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_PRELOAD_BRIDGE_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_PRELOAD_BRIDGE_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_PRELOAD_BRIDGE_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_PRELOAD_BRIDGE_LOOP`

### 為什麼要這樣做
因為目前這段還在過渡期。
設計希望支援 Top 餵進來的 payload，
但同時又要保留舊的本地 preload fallback。

### 對目前專案的影響
這代表：
- 介面語意已經開始往 Top-fed / handoff 描述子靠
- 但真正的資料搬運，很多時候還是 `TransformerLayer` 自己從 `sram[...]` 搬一份 local buffer 出來

### 你現在最需要注意什麼
`transformer_layer_select_topfed_words(...)` 是這整段的閱讀鑰匙。

白話：
- Top 有給資料，就選 Top 的
- Top 沒給，或 valid 為 0，就選本地 preload 的

所以目前 FFN 的現狀不是「已經 fully top-managed」，
而是：
**Top-fed seam 已經開了，但舊 preload 骨架還留著。**

這也是你為什麼會覺得 code 很難懂：
因為兩個時代的東西現在是疊在一起的。

---

## 3.4 FFN stage dispatch：W1 / RELU / W2 是分段叫的
### 這是什麼
這份檔不是一次叫完整 FFN，而是分 stage 去叫：
- `FFNLayer0TopManagedWindowBridge<FFN_STAGE_W1>`
- `FFNLayer0TopManagedWindowBridge<FFN_STAGE_RELU>`
- `FFNLayer0TopManagedWindowBridge<FFN_STAGE_W2>`

### 為什麼要這樣做
因為這樣比較容易把 payload ownership 拆開看。
例如：
- W1 需要哪一批輸入/權重/偏置
- RELU 只做中間結果轉換
- W2 再吃另一批輸入/權重/偏置

### 對目前專案的影響
這種分段呼叫方式很適合 bounded migration，
因為你可以一次只推進一段 handoff，不一定要整個 FFN 一次重寫。

### 你現在最需要注意什麼
先不要被 template 參數嚇到。
你可以把它讀成：
- 先跑 W1
- 再跑 RELU
- 再跑 W2

而且每一段都可以有自己的 topfed / fallback payload 選路。

---

## 3.5 residual add：這段目前還是明顯 direct SRAM 風格
### 這是什麼
這段 loop：
- `TRANSFORMER_LAYER_FFN_RESIDUAL_ADD_BRIDGE_LOOP`

做的就是把：
- attention 輸出
- FFN W2 輸出
相加後，寫到 `add2_base`

### 為什麼要這樣做
這就是 Transformer sublayer 後半段很標準的 residual add。

### 對目前專案的影響
這一段非常值得注意，因為它很直接：
- 從 `sram_window[residual_base + i]` 讀
- 從 `sram_window[w2_base + i]` 讀
- 再寫回 `sram_window[add2_base + i]`

白話：
**這段目前還很明顯是 block 內直接碰 shared SRAM 的風格。**

### 你現在最需要注意什麼
不要看到前面一堆 `topfed` 就以為整個 `TransformerLayer` 已經不碰 SRAM。
至少 residual add 這段就還是很直接。

---

## 3.6 LayerNorm 段：有 Top preload，也留 legacy fallback
### 這是什麼
LayerNorm 這段分成兩部分：
1. gamma / beta 參數要不要由 block 內自己抓
2. 真正呼叫 `LayerNormBlockTopManagedWindowBridge(...)`

### 為什麼要這樣做
因為這條線也在 migration 中。
如果 Top 已經 preload 好 sublayer1 norm 參數，
那 `TransformerLayer` 就不用再自己去 param region 抓一次。

### 對目前專案的影響
你會看到：
- `sublayer1_norm_preloaded_by_top`
- `load_layer_sublayer1_norm_params(...)`

而且程式自己也寫了很直白的註解：
> Legacy fallback: keep in-block param fetch only when Top did not preload.

白話：
這段其實已經很明確承認：
**block 內參數抓取現在只是 legacy fallback。**

### 你現在最需要注意什麼
這是一個很好的閱讀示範。
因為它把 migration 狀態寫得很清楚：
- Top 有 preload：走新路
- Top 沒 preload：才留舊路

如果之後其他段也能做到這麼直白，就會好讀很多。

---

## 4. 這份檔最值得你先看的 6 段 code

### 1. 檔頭 intent comment
用途：先建立責任分界，不然你會以為這份檔自己在管全世界。

### 2. `TransformerLayerFfnTopfedHandoffDesc`
用途：看懂 FFN handoff descriptor 長什麼樣。

### 3. `transformer_layer_select_topfed_words(...)`
用途：看懂 Top-fed 與 local preload 到底怎麼選。

### 4. `attn_shell_must_run` 那段判斷
用途：看懂 attention shell 什麼時候還會進場。

### 5. 三段 `FFNLayer0TopManagedWindowBridge<...>` 呼叫
用途：看懂 FFN 是 W1 / RELU / W2 分段 dispatch。

### 6. `TRANSFORMER_LAYER_FFN_RESIDUAL_ADD_BRIDGE_LOOP` 與 `load_layer_sublayer1_norm_params(...)`
用途：看懂哪裡還是 direct SRAM 風格，哪裡已有 Top preload / fallback 分流。

---

## 5. 這份檔裡最容易誤解的 5 件事

### 誤解 1：看到 `topfed` 就代表完全不碰 SRAM
不是。
現在很多 `topfed_*` 陣列，仍然是先從 `sram_window[...]` preload 出來，再跟 Top handoff 做選路。

### 誤解 2：`TransformerLayer` 是 policy owner
不是。
大部分 policy 決定，例如 `attn_compat_shell_enable`，是 Top 先算好再傳進來。

### 誤解 3：attention 已經 fully bypass
不是。
只有在 prebuilt 條件與 payload 條件都滿足時，shell 才可能不跑。

### 誤解 4：FFN 已經 fully top-managed
還沒有。
目前比較像是 handoff seam 已出現，但 preload fallback 還在。

### 誤解 5：LayerNorm 已經完全不從 block 內抓參數
也沒有。
若 `sublayer1_norm_preloaded_by_top == false`，仍會走 legacy fallback 去抓參數。

---

## 6. 如果你現在只想先搞懂 1 條主線，請這樣記

`TransformerLayer` 目前的主線可以先記成：

1. Top 決定這層 attention shell 要不要開
2. `TransformerLayer` 視情況呼叫 `AttnLayer0`
3. `TransformerLayer` 準備 FFN 要的 W1 / RELU / W2 payload
4. Top-fed 有資料就優先吃，沒有就本地 preload fallback
5. 跑完 FFN 後做 residual add
6. LayerNorm 參數若 Top 已 preload 就直接用，否則 block 內補抓
7. 最後呼叫 `LayerNormBlockTopManagedWindowBridge(...)`

白話版：
**這份檔就是把「attention 後半層 + FFN + residual + LN」這整條鏈，在過渡時期先接起來。**

---

## 7. 你現在最需要知道的閱讀結論
- 這份檔最難的地方，不是數學，而是 ownership 過渡。
- `attn_compat_shell_enable`、`topfed_*`、`*_preloaded_by_top` 這些旗標，都是在描述「新路和舊路現在怎麼共存」。
- 如果你想判斷 direct SRAM 問題還剩多少，`TransformerLayer.h` 是很關鍵的一份，因為它把：
  - 已往 Top 收的東西
  - 還留在 block 內的 preload / residual / param fetch
  同時都攤在你眼前。

---

## 8. 建議下一份要補什麼
如果這份你看完有幫助，下一份最值得補的是：
- `ATTNLAYER0_COREWINDOW_READER_GUIDE_zhTW.md`

原因很簡單：
`TransformerLayer.h` 難，是因為它是整合層；
但真正 attention 演算法本體，還是在 `AttnLayer0CoreWindow(...)`。
