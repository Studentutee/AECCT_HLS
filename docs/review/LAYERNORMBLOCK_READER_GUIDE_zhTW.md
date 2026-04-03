# LAYERNORMBLOCK_READER_GUIDE_zhTW
Date: 2026-04-03

用途：給「現在要讀 `src/blocks/LayerNormBlock.h`」的人看的中文導讀。

邊界：
- 這份文件是 reader guide，不是 closure report。
- 它不主張 Catapult closure，也不主張 SCVerify closure。
- 它的目標是把 LayerNorm 這份檔案從「兩段 pass + 一堆 base word」變成可追的資料流。

---

## 1. 先講一句話結論
`LayerNormBlock.h` 是一個 **token-wise two-pass block**。

白話講：
它的重點不是複雜控制，
而是：
1. 先做統計
2. 再做 normalize + affine

而且目前 default 入口已經是 Top-managed window 風格。

---

## 2. 你要怎麼讀這個檔案

### Step 1：先看 `LayerNormBlockContract`
這是 LayerNorm 的工作單：
- token_range
- tile_range
- x_in / x_out
- gamma / beta

這張表先看懂，後面就不容易迷路。

### Step 2：再看 `LayerNormTopManagedTileMeta`
這跟 preproc 那份 meta 很像。
它是在描述每個 tile window 的標籤。

### Step 3：最後看三個入口
- `LayerNormBlockCoreWindow(...)`
- `LayerNormBlockCoreWindowDirect(...)`
- `LayerNormBlock(...)`

你真正最該看的，是 `LayerNormBlock(...)` 最後 default mainline 呼叫了誰。

---

## 3. 這個檔案實際在做什麼

## 3.1 `LayerNormBlockContract`：工作單先行
### 這是什麼
LayerNorm block 的 contract，描述本輪的 token / tile / base 邊界。

### 為什麼要這樣做
因為 Top 才是 owner，
block 只負責處理被派來的範圍。

### 對目前專案的影響
這表示 LN 已經不是「自己定義要掃哪段 SRAM」的風格。

### 你現在最需要注意什麼
先看：
- `token_range`
- `tile_range`
- `x_base_word`
- `gamma_base_word`
- `beta_base_word`

這幾個欄位會把資料流的邊界直接講清楚。

---

## 3.2 `LayerNormTopManagedTileMeta`：窗口一致性檢查
### 這是什麼
這是每塊 tile window 的 metadata。

### 為什麼要這樣做
因為 Top-managed 流程不是只看 payload，
還要驗證 payload 跟窗口標籤有沒有對上。

### 對目前專案的影響
這跟 attention / preproc 的思路是一致的：
先把窗口語意固定，再談計算。

### 你現在最需要注意什麼
跟 preproc 一樣，
不要小看 `*_meta_ok(...)`。
這些 helper 是在保護 window 邏輯。

---

## 3.3 `LayerNormBlockCoreWindow(...)`：真正主體
### 這是什麼
這是 Top-managed 的 LN 主體。

### 為什麼要這樣做
因為 LN 最自然的切法就是：
- 對每個 token 做統計
- 再對每個 token 做 normalize + affine

tile window 只是把 d 維度切塊。

### 對目前專案的影響
這份主體已經很像比較成熟的 block：
- contract 明確
- token/tile 邊界明確
- two-pass 結構清楚

### 你現在最需要注意什麼
進 `CoreWindow` 後，先把它分成兩段：
1. Pass-1：mean / variance 所需統計
2. Pass-2：normalize + gamma/beta + writeback

不要試圖一次讀完整個函式。

---

## 3.4 Pass-1：先做統計
### 這是什麼
第一段會以 token 為中心，掃過所有 tile，累積統計量。

### 為什麼要這樣做
LayerNorm 本來就需要整個 token 向量的 mean / variance。

### 對目前專案的影響
這段通常是 LN 的核心數學來源。
但從架構角度，它比較像：
**為了第二段準備參數。**

### 你現在最需要注意什麼
先看它怎麼掃 token、怎麼掃 tile，
不要先糾結每一個 FP32 小細節。

---

## 3.5 Pass-2：normalize + affine + writeback
### 這是什麼
第二段會把第一段的統計結果拿來做 normalize，
再乘 gamma、加 beta，最後寫回。

### 為什麼要這樣做
這就是 LN 的標準兩段式做法。

### 對目前專案的影響
這裡很能看出 x/gamma/beta 的資料路徑是否清楚。

### 你現在最需要注意什麼
看這段時，先抓：
- x 是從哪裡讀
- gamma/beta 是從哪裡讀
- x_out 是寫到哪裡

這樣你比較不會被公式本身蓋過去。

---

## 3.6 `CoreWindowDirect(...)`：為什麼還有 direct 版本
### 這是什麼
這是舊風格 direct 版本的 LN 主體。

### 為什麼要這樣做
這跟 preproc 一樣，是 migration 過渡需要。

### 對目前專案的影響
所以看 LN 時，真正該問的不是「direct 版本還在不在」，
而是「default mainline 現在站在哪邊」。

### 你現在最需要注意什麼
直接去看 `LayerNormBlock(...)`。

---

## 3.7 `LayerNormBlock(...)`：這個很重要
### 這是什麼
這是多數 call site 看到的 block 入口。

### 為什麼要這樣做
因為這裡最能回答：
現在 LN default 到底走 `CoreWindow` 還是 `Direct`。

### 對目前專案的影響
如果 default mainline 已經走 `CoreWindow`，
那就代表 LN 在 ownership 方向上其實已經相對乾淨。

### 你現在最需要注意什麼
看入口最後呼叫哪個主體，
這件事比讀一堆 pass 公式還重要。

---

## 4. 你現在最值得看的 4 個位置
1. `LayerNormBlockContract`
2. `LayerNormTopManagedTileMeta`
3. `LayerNormBlockCoreWindow(...)`
4. `LayerNormBlock(...)`

---

## 5. 最後一個提醒
LayerNorm 這塊的控制複雜度沒有 attention / transformer layer 那麼高。

白話講：
**如果你想看一個比較容易建立直覺的 two-pass block，先看 `LayerNormBlock.h` 很適合。**
