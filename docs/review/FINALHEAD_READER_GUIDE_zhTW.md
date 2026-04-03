# FINALHEAD_READER_GUIDE_zhTW
Date: 2026-04-03

用途：給「現在要讀 `src/blocks/FinalHead.h`」的人看的中文導讀。

邊界：
- 這份文件是 reader guide，不是 closure report。
- 它不主張 Catapult closure，也不主張 SCVerify closure。
- 它主要是把 FinalHead 的 Pass A / Pass B 拆開，讓你看懂它在做什麼。

---

## 1. 先講一句話結論
`FinalHead.h` 現在最重要的閱讀方式，不是把它當單一黑盒，
而是把它拆成：
- **Pass A：產生 token-wise scalar (`s_t`)**
- **Pass B：對 `s_t` 做 OUT_FC reduction，最後依 out_mode 決定輸出**

白話講：
FinalHead 不是「最後多做一層 FC」而已，
它已經被明確拆成兩段正式 phase。

---

## 2. 你要怎麼讀這個檔案

### Step 1：先看 `FinalHeadContract`
這是 FinalHead 的工作單。
最重要的欄位是：
- `x_work_base_word`
- `scratch_base_word`
- `final_scalar_base_word`
- `w_base_word`

這幾個欄位幾乎把 FinalHead 的資料落點講完了。

### Step 2：再看 `HeadParamBase`
這不是演算法，
它是在幫你把 head 相關權重的 base address 整理好。

### Step 3：最後看 `FinalHeadCorePassABTopManaged(...)`
真正主體在這裡。

---

## 3. 這個檔案實際在做什麼

## 3.1 `FinalHeadContract`：先看工作單
### 這是什麼
FinalHead block 的 contract。

### 為什麼要這樣做
因為 FinalHead 雖然在 pipeline 尾端，
但它一樣不能自己宣稱 shared SRAM ownership。

### 對目前專案的影響
所以 contract 會把：
- x_work
- scratch
- final scalar buffer
- weight base
全部先定好。

### 你現在最需要注意什麼
先把 `final_scalar_base_word` 記住。
這是讀懂 Pass A / Pass B 的關鍵。

---

## 3.2 `HeadParamBase`：不要把它當成雜項 helper
### 這是什麼
它是把 FinalHead 會用到的幾塊 parameter base address 先算好。

### 為什麼要這樣做
因為 head 這段會碰到：
- ffn1_w
- ffn1_b
- out_fc_w
- out_fc_b

如果每次都在主流程裡自己加 offset，很難讀。

### 對目前專案的影響
這讓主體流程可以比較聚焦在 Pass A / Pass B。

### 你現在最需要注意什麼
看到 `make_head_param_base(...)`，
就把它理解成「權重位址整理器」。

---

## 3.3 Pass A：先產生 `s_t`
### 這是什麼
Pass A 會根據 token 的表示，產生 token-wise scalar，
並寫到 `FINAL_SCALAR_BUF`。

### 為什麼要這樣做
這是 v12.1 明確拆出的邏輯邊界：
先得到 `s_t`，再做後面的 OUT_FC reduction。

### 對目前專案的影響
這讓 FinalHead 不再是一大坨最後一段運算，
而是能清楚分成兩段 phase。

### 你現在最需要注意什麼
看 Pass A 時，先抓兩件事：
1. scalar 是從哪裡來
2. scalar 是寫到哪裡去

尤其第二點很重要：
它會寫到 `FINAL_SCALAR_BUF`。

---

## 3.4 top-fed scalar path：這是新的 handoff 能力
### 這是什麼
`FinalHeadCorePassABTopManaged(...)` 有 `topfed_final_scalar_words` 參數。

### 為什麼要這樣做
這表示 Top 可以先把 final scalar 餵下來，
FinalHead 不一定每次都只能從自己 local 流程產生。

### 對目前專案的影響
這跟 attention / FFN 的 top-fed 思路一致：
資料搬運逐步往 Top 收。

### 你現在最需要注意什麼
看到 top-fed scalar 時，不要直接解讀成 fully migrated。
比較精確的說法是：
**FinalHead 已經有 top-fed scalar handoff 能力。**

---

## 3.5 Pass B：對 `s_t` 做 OUT_FC reduction
### 這是什麼
第二段會針對 `FINAL_SCALAR_BUF` 做 reduction，
再依 `out_mode` 輸出。

### 為什麼要這樣做
這就是 FinalHead 尾端的正式輸出路徑。

### 對目前專案的影響
所以你在這段要看的是：
- scalar buffer 怎麼被讀
- OUT_FC weight/bias 怎麼被用
- 最後是 stream xpred、stream logits，還是 compatibility writeback

### 你現在最需要注意什麼
別只盯 `data_out`。
先看 reduction 是如何建立出 logits / xpred 的。

---

## 3.6 out_mode：這裡決定最後怎麼吐結果
### 這是什麼
檔案裡有：
- `FINAL_HEAD_OUTMODE_XPRED`
- `FINAL_HEAD_OUTMODE_LOGITS`
- `FINAL_HEAD_OUTMODE_NONE`

### 為什麼要這樣做
因為最尾端輸出不一定每次都同一種格式。

### 對目前專案的影響
這也是 FinalHead 跟前面 block 比較不一樣的地方：
它同時牽涉到最終對外輸出 sequencing。

### 你現在最需要注意什麼
先把 out_mode 看成「最後封包格式選擇」。
不要一開始就掉進所有細節。

---

## 3.7 `FinalHead(...)`：default 入口
### 這是什麼
這是大多數 call site 會碰到的 block 入口。

### 為什麼要這樣做
你真正想知道的是：
現在 FinalHead default mainline 是否已經站在 PassABTopManaged 這邊。

### 對目前專案的影響
如果 default 已經走 `FinalHeadCorePassABTopManaged(...)`，
那代表 FinalHead 的主路已經收斂到現在的兩段式架構。

### 你現在最需要注意什麼
去看 `FinalHead(...)` 最後真正呼叫哪一個主體。

---

## 4. 你現在最值得看的 4 個位置
1. `FinalHeadContract`
2. `HeadParamBase`
3. `FinalHeadCorePassABTopManaged(...)`
4. `FinalHead(...)`

---

## 5. 最後一個提醒
看 FinalHead 時，不要把它當「最後一個普通 block」。

白話講：
**它一半像運算 block，一半像輸出整形 block。**
這就是為什麼它需要拆成 Pass A / Pass B 來讀。
