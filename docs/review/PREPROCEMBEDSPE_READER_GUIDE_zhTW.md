# PREPROCEMBEDSPE_READER_GUIDE_zhTW
Date: 2026-04-03

用途：給「現在要讀 `src/blocks/PreprocEmbedSPE.h`」的人看的中文導讀。

邊界：
- 這份文件是 reader guide，不是 closure report。
- 它不主張 Catapult closure，也不主張 SCVerify closure。
- 它主要解決一件事：讓你知道 preproc 這塊現在是怎麼被 Top 派工、怎麼吃 token/tile window。

---

## 1. 先講一句話結論
`PreprocEmbedSPE.h` 是目前比較乾淨的 Top-managed block 之一。

白話講：
它的重點不是複雜演算法，
而是把 preproc 這件事寫成「Top 指定 token 範圍 / tile 範圍，block 只負責在這塊窗口內做事」的樣子。

---

## 2. 你要怎麼讀這個檔案

### Step 1：先看 `PreprocBlockContract`
這就是 preproc 的工作單。
它告訴 block：
- 現在是不是 start
- token_range 是哪一段
- tile_range 是哪一段
- x_work / scratch / w_base 在哪裡

白話：
這張表比演算法本體還重要，
因為它直接告訴你 ownership 在誰手上。

### Step 2：再看 `PreprocTopManagedWindowMeta`
這不是 payload，
它是每個 token/tile window 對應的 metadata。

你可以把它想成：
**這塊窗口的標籤。**

### Step 3：最後看 4 個入口
- `PreprocEmbedSPECoreWindow(...)`
- `PreprocEmbedSPECoreWindowDirect(...)`
- `PreprocEmbedSPE(...)`
- `PreprocEmbedSPETopManagedWindowBridge(...)`

這 4 個入口其實是在回答：
- 真正主體在哪裡
- 舊 direct 路徑還留不留
- 對不同 `SramView` 型別怎麼接橋

---

## 3. 這個檔案實際在做什麼

## 3.1 `PreprocBlockContract`：先看這張工作單
### 這是什麼
它定義了 preproc block 的基本 contract。

### 為什麼要這樣做
因為專案不希望 block 自己去決定共享記憶體政策。
所以 Top 會先把：
- phase
- token_range
- tile_range
- base word
打包好再往下傳。

### 對目前專案的影響
這表示 preproc 這條線已經不是「block 自己想讀哪就讀哪」的風格。

### 你現在最需要注意什麼
看 contract 時，先抓：
- `token_range`
- `tile_range`
- `x_work_base_word`

這三個就是 preproc 在 shared SRAM 上的工作邊界。

---

## 3.2 `PreprocTopManagedWindowMeta`：這是窗口標籤，不是資料本體
### 這是什麼
這份 meta 包含：
- token_begin / token_end
- token_idx
- tile_begin / tile_end
- tile_idx
- tile_valid_words

### 為什麼要這樣做
因為 Top-managed window 流程不只要送資料，
還要能驗證：
「現在吃到的這塊資料，真的就是我以為的那一塊嗎？」

### 對目前專案的影響
這讓 block 可以做最基本的防呆，
避免 token / tile 對不上時還默默往下算。

### 你現在最需要注意什麼
不要把 `*_meta_ok(...)` 當成無聊小 helper。
它其實是在守一個很重要的東西：
**窗口與 payload 的一致性。**

---

## 3.3 `PreprocEmbedSPECoreWindow(...)`：真正主體在這裡
### 這是什麼
這是 preproc 的 mainline 核心入口。
它收：
- `SramView`
- cfg
- input base / x output base
- contract
- optional top-fed input words

### 為什麼要這樣做
這個介面形狀很能代表專案現在的風格：
- shared SRAM 還在
- 但 block 已經不是自己定義正式全域 policy
- Top 可以額外送 top-fed payload 進來

### 對目前專案的影響
這表示 preproc 已經在用「窗口化」的方式思考資料流，
而不是整塊平鋪式亂讀。

### 你現在最需要注意什麼
你進 `CoreWindow` 時，先看 4 件事：
1. token_count 怎麼算
2. tile_count 怎麼算
3. token_range / tile_range 怎麼被 clamp
4. top-fed input 和 SRAM fallback 怎麼二選一

---

## 3.4 `CoreWindowDirect(...)`：為什麼還留著 direct 版本？
### 這是什麼
這是 preproc 的 direct 版本入口。

### 為什麼要這樣做
因為 migration 還沒到「所有地方都只留新路徑」。
保留 direct 版本可以讓：
- 舊 call site 先不爆
- 新 call site 慢慢改成 window 風格

### 對目前專案的影響
所以不要看到 direct 版本還在，就判定這份檔案沒有前進。
真正該看的，是 default 入口最後是不是已經走向 `CoreWindow`。

### 你現在最需要注意什麼
直接去看 `PreprocEmbedSPE(...)`，
確認 default mainline 到底叫哪一個。

---

## 3.5 `PreprocEmbedSPE(...)`：這個最值得看
### 這是什麼
這是大多數 call site 會碰到的 block 入口。

### 為什麼要這樣做
因為這裡最能回答一個問題：
**現在 preproc 的 default mainline 到底站在哪一邊？**

### 對目前專案的影響
如果 default 入口已經轉去 `CoreWindow`，
那就代表 preproc 這條線在 ownership 思想上已經比較乾淨。

### 你現在最需要注意什麼
檔內其實已經有一句很重要的英文註解：
`Mainline migration: default Preproc entry now consumes Top-managed token/tile windows.`

這句話值很高。
因為它直接告訴你：
**主路已經改成 Top-managed window。**

---

## 3.6 `TopManagedWindowBridge(...)`：這是 Catapult/型別橋，不是新演算法
### 這是什麼
bridge 版本主要是在接不同型別的 `SramView`，
例如固定大小 array 版本。

### 為什麼要這樣做
因為 HLS / Catapult 常常會對 top-facing array 形狀比較敏感，
bridge 可以讓主體維持同一套邏輯。

### 對目前專案的影響
它的價值主要在 interface friendliness，
不是新功能。

### 你現在最需要注意什麼
讀 bridge 時不要花太多力氣。
先確認它有沒有偷偷改 policy；
如果沒有，那它就只是接線橋。

---

## 4. 你現在最值得看的 4 個位置
1. `PreprocBlockContract`
2. `PreprocTopManagedWindowMeta`
3. `PreprocEmbedSPECoreWindow(...)`
4. `PreprocEmbedSPE(...)`

---

## 5. 最後一個提醒
preproc 這塊比較適合拿來建立信心。
因為它比 attention / transformer layer 乾淨很多。

白話講：
**如果你想先看一個比較「像 Top-managed block」的例子，先看 `PreprocEmbedSPE.h` 是對的。**
