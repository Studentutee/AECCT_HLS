# FFNLAYER0_READER_GUIDE_zhTW
Date: 2026-04-03

用途：給「現在要讀 `src/blocks/FFNLayer0.h`」的人看的中文導讀。

邊界：
- 這份文件是 reader guide，不是 closure report。
- 它不主張 Catapult closure，也不主張 SCVerify closure。
- 它的目標是幫你把 FFN 這份檔案從「一大坨 W1/RELU/W2 + fallback」拆成幾段可理解流程。

---

## 1. 先講一句話結論
`FFNLayer0.h` 不是單純的全新 channelized FFN；
它比較像是 **Top-fed tile path 和 legacy SRAM fallback 共存** 的過渡版本。

白話講：
- 它已經會吃 Top 預先送來的 tile payload
- 但 payload 不齊時，還是會退回去讀 SRAM
- 而且還有 reject / fallback observability，確保你知道自己是不是又滑回舊路徑

---

## 2. 你要怎麼讀這個檔案

### Step 1：先把 FFN 分成 3 段
不要從頭到尾硬讀。
先在腦中切成：
1. W1
2. RELU
3. W2

這份檔案本身也是沿這個順序排的。

### Step 2：先看 top-fed 介面，不要先看乘加細節
最值得先看的是 `FFNLayer0CoreWindow(...)` 參數列。
你會看到很多：
- `topfed_x_words`
- `topfed_w1_weight_words`
- `topfed_w2_input_words`
- `topfed_w2_weight_words`
- `topfed_*_bias_words`

白話：
這是在告訴你 FFN 的哪些 payload 已經可以由上游先餵好。

### Step 3：再看 fallback/reject 相關旗標
- `fallback_policy_flags`
- `fallback_policy_reject_flag`
- `fallback_legacy_touch_counter`
- `fallback_policy_reject_stage`

這些是最能回答「這輪到底有沒有真的走新路徑」的地方。

### Step 4：最後才看 tile kernel
像：
- `ffn_block_mac_tile(...)`
- `ffn_block_relu_tile(...)`

這些比較像局部運算核心，
不是最難懂的地方。

---

## 3. 這個檔案實際在做什麼

## 3.1 先把 `STAGE_MODE` 看懂
### 這是什麼
`FFNLayer0CoreWindow` 是模板函式，
會依 `STAGE_MODE` 選擇只跑：
- W1
- RELU
- W2
- 或 FULL

### 為什麼要這樣做
因為 migration 或局部驗證時，
常常只想切一段出來看，
不一定每次都要整個 FFN 全跑。

### 對目前專案的影響
這代表 FFN 本體已經被拆成幾個可局部觀察的子段。
這對 bounded migration 很重要。

### 你現在最需要注意什麼
讀這份檔案時，先確認你看的那段到底屬於：
- W1
- RELU
- W2
哪一段。
不然很容易把 top-fed payload 的語意混掉。

---

## 3.2 top-fed payload：這才是現在真正的主題
### 這是什麼
檔案裡有很多 `topfed_*` 參數。
它們代表上游已經先把 FFN 某些資料 tile 準備好了。

### 為什麼要這樣做
因為專案目標不是把 `sram` 變數刪掉，
而是讓 block 不再直接持有 production shared SRAM ownership。

Top 先送 payload，
就代表資料搬運決策開始往上收。

### 對目前專案的影響
這讓 FFN 不再只剩「自己去 SRAM 讀」這一條路。
它已經有新的 handoff 入口。

### 你現在最需要注意什麼
看 top-fed 參數時，先分清楚 6 類 payload：
- x
- W1 weight
- W1 bias
- W2 input
- W2 weight
- W2 bias

不要把它們全部混成「有餵資料就好」。
FFN 每一段要吃的 payload 不一樣。

---

## 3.3 fallback / reject：這是證據面，不是數學面
### 這是什麼
這份檔案保留了很完整的 fallback / reject 機制。

### 為什麼要這樣做
因為 migration 最怕表面上把 top-fed 參數接上了，
實際上卻一直默默退回去走 SRAM。

### 對目前專案的影響
所以你會看到：
- 可以要求某段必須 top-fed
- 如果條件不夠，直接 reject
- 或至少記一次 legacy touch

### 你現在最需要注意什麼
這裡最重要的不是 reject 本身，
而是它讓你能分辨：
**這輪是新路徑真的跑了，還是只是在 fallback。**

---

## 3.4 W1 段：最像「讀 x、讀 weight、做 MAC」
### 這是什麼
W1 段會對每個 token、每個輸出位置，
用 x 與 W1 weight 做乘加，最後加 bias。

### 為什麼要這樣做
這就是 FFN 前半段的本質。
只是現在被寫成 tile-driven 版本。

### 對目前專案的影響
W1 段很能看出 top-fed 與 fallback 的雙路共存：
- x tile 可以 top-fed
- W1 weight tile 可以 top-fed
- W1 bias 也可以 top-fed
- 缺哪塊就可能回 SRAM

### 你現在最需要注意什麼
看 W1 loop 時，先看「tile load」而不是先看 MAC。
因為現在的閱讀難點不在公式，
而在資料是從哪裡來。

---

## 3.5 RELU 段：比較乾淨
### 這是什麼
RELU 段主要是在 W1 結果上做逐元素的非線性。

### 為什麼要這樣做
因為這段本來就不太需要外部參數。

### 對目前專案的影響
RELU 段通常比 W1 / W2 容易讀，
也更像單純的 block-local compute。

### 你現在最需要注意什麼
把這段當成中場休息就好。
真正的 ownership 複雜度主要還是在 W1 / W2 的 payload 選路。

---

## 3.6 W2 段：再做一次 tile-driven accumulate
### 這是什麼
W2 段會吃 RELU 輸出，
再乘上 W2 weight，最後加 bias，回到 d_model。

### 為什麼要這樣做
這是 FFN 後半段的標準結構。

### 對目前專案的影響
W2 跟 W1 很像，也保留了：
- top-fed input
- top-fed weight
- top-fed bias
- fallback policy

所以它也是 ownership 觀察重點。

### 你現在最需要注意什麼
如果你只能先看一段 W2，
先看：
- tile load
- top-fed / SRAM 二選一
- reject / fallback counter

不要先被最內層 MAC 迴圈吸走注意力。

---

## 3.7 `ffn_block_mac_tile(...)` / `ffn_block_relu_tile(...)`
### 這是什麼
這兩個 helper 是比較局部的 tile kernel。

### 為什麼要這樣做
這可以把主流程的資料搬運與內部算子分開。

### 對目前專案的影響
它們比較接近「可合成小核心」，
不是 migration policy 本體。

### 你現在最需要注意什麼
這兩個 helper 可以晚一點看。
先把 `CoreWindow` 裡資料從哪裡來搞清楚，閱讀收益比較高。

---

## 4. 你現在最值得看的 5 個位置
1. `FFNLayer0CoreWindow(...)` 的參數列
2. `fallback_policy_*` 相關欄位
3. W1 段的 tile load / select
4. RELU 段
5. W2 段的 tile load / select

---

## 5. 最後一個提醒
FFN 這份檔案很容易讓人誤以為「都已經 top-fed 了」。
其實現在更精確的說法應該是：
**FFN 已有 Top-fed handoff 能力，但仍保留 SRAM fallback。**

白話講：
看到 `topfed`，不要直接把它等同於 fully migrated。
