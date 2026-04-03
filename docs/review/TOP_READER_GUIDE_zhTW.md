# TOP_READER_GUIDE_zhTW
Date: 2026-04-03

用途：給「現在真的要讀 `src/Top.h`」的人看的中文導讀。

邊界：
- 這份文件是 reader guide，不是 closure report。
- 它不主張 Catapult closure，也不主張 SCVerify closure。
- 它的目的只有一個：幫你把 `Top.h` 從「超大雜湊檔」拆成幾個可以理解的責任區。

---

## 1. 先講一句話結論
`Top.h` 不是單純的 top wrapper；它其實是 **全系統 owner + dispatcher + scoreboard**。

白話講：
- 它決定誰能碰 shared SRAM
- 它決定這輪要跑哪個 block
- 它也保留大量 migration 期間的觀測點，幫你看出新路徑到底有沒有真的被走到

所以你在 `Top.h` 看到很多 flag / counter / base_word，不要先把它們當雜訊。
很多其實都是 **ownership 邊界** 的痕跡。

---

## 2. 你要怎麼讀 `Top.h`

### Step 1：先只看檔頭 30~40 行
檔頭已經直接講了 3 個核心責任：
- external 4-channel contract
- shared SRAM lifetime / arbitration ownership
- block call dispatch with explicit base/range boundaries

白話：
`Top.h` 的存在目的，就是把「誰擁有記憶體、誰只是被叫去算」這件事釘死。

### Step 2：把 `TopRegs` 當成儀表板，不要當演算法
`TopRegs` 很大，第一次看很容易頭痛。
比較好的讀法是先分區：
- FSM / RX / HALT 這些是控制面
- cfg / metadata / debug 這些是管理面
- `p11a* / p11b*` 這些多半是 migration observability

白話：
`TopRegs` 不是在做數學運算，
它比較像是整個系統的「狀態機暫存 + 測試探針板」。

### Step 3：再看 3 類 helper
你可以先抓這 3 類 helper：
1. **ingest / metadata**：誰送了多少資料、資料是不是合法
2. **dispatch**：現在要去叫哪個 block
3. **handoff / shell gate**：新路徑和舊路徑怎麼選

### Step 4：最後才看 block call chain
建議順序：
- preproc
- transformer layer loop
- final head

原因很簡單：
attention / FFN 都是透過 `TransformerLayer` 接起來的，
如果你一開始就衝到中間最複雜的段落，很容易迷路。

---

## 3. 這個檔案到底在做什麼

## 3.1 它是 shared SRAM 的正式 owner
### 這是什麼
專案口徑很固定：
**Top 是唯一 production shared-SRAM owner。**

`Top.h` 就是這個口徑在程式上的落點。

### 為什麼要這樣做
因為只要 block 自己也開始定義 memory ownership，
整個專案就會出現第二套 arbitration 語意，後面很難收斂。

### 對目前專案的影響
所以你會看到 `Top.h` 裡充滿：
- `*_BASE_WORD`
- `TokenRange`
- `TileRange`
- `scratch / x_work / w_base`

這些不是單純地址常數，
它們其實是在描述：
**這次 block 被分配到哪一段工作範圍。**

### 你現在最需要注意什麼
先把 `base / range / phase` 看成「工作單」就好。
`Top.h` 的工作是派工，
不是自己去實作每一個 block 的演算法。

---

## 3.2 `TopRegs`：大，但不要怕
### 這是什麼
`TopRegs` 是 top-level 的主狀態集合。
裡面塞了很多東西：
- RX state
- halt/debug state
- local-only handoff flags
- fallback counters
- target-layer controls

### 為什麼會這麼大
因為 migration 現在還在進行中。
新舊路徑要共存時，Top 常常要同時記住：
- 這輪有沒有嘗試新路徑
- descriptor 有沒有 valid
- fallback 有沒有發生
- 是不是只有 lid0 / target layer 開

### 對目前專案的影響
這表示你不能把 `TopRegs` 當成「雜項垃圾桶」。
它其實是這個專案目前最重要的觀測面。

### 你現在最需要注意什麼
先抓這 4 類欄位：
- `*_enable`
- `*_descriptor_valid`
- `*_gate_taken_count`
- `*_fallback_seen_count`

只要看到這幾類名字，通常都在回答一件事：
**新路徑到底有沒有真的吃到資料，而不是只有把 flag 接上去。**

---

## 3.3 為什麼一堆 `p11ax / p11bd` 這種名字？
### 這是什麼
這些多半是 task-local 的 observability 命名。

### 為什麼要這樣做
因為 bounded migration 不是一次大改。
每一輪只切一小刀時，需要留一個「這一刀真的有前進」的證據點。

### 對目前專案的影響
所以你會在 `Top.h` 看到像：
- `p11bd_attn_compat_shell_enabled_count`
- `p11av_ffn_handoff_gate_taken_count`
- `p11ax_attn_out_payload_fallback_seen_count`

這些名字雖然不漂亮，
但它們的角色很明確：
**不是做功能，而是做證據。**

### 你現在最需要注意什麼
讀這些名字時，不用先記 task 編號。
先只看後半段語意：
- shell enabled / disabled
- handoff gate taken
- fallback seen
- non_empty

後半段才是你理解資料流的關鍵。

---

## 3.4 `top_should_enable_attn_compat_shell(...)`
### 這是什麼
這個 helper 是 Top 端對 attention 舊殼層的啟閉判斷。

### 為什麼要這樣做
因為 attention 現在不是全新路徑完全取代舊路徑，
而是「有些子段已 prebuilt，有些情況還得留 shell」。

### 對目前專案的影響
這代表 shell 的 policy 已經開始往 Top 收，
而不是在 `TransformerLayer` 或 `AttnLayer0` 裡臨時各自決定。

### 你現在最需要注意什麼
看這個 helper 時，先只問兩件事：
1. 哪些 prebuilt flag 需要同時為真？
2. 有沒有 top-fed OUT payload 讓 shell 仍必須保留？

不要一開始就陷進所有局部條件。

---

## 3.5 `top_dispatch_transformer_layer(...)`
### 這是什麼
這是 Top 把 layer 級工作單丟給 `TransformerLayer(...)` 的邊界。

### 為什麼要這樣做
因為 `TransformerLayer` 不是 policy owner，
它吃的是 Top 已經決定好的條件。

### 對目前專案的影響
這個函式很值得看，因為它會把：
- shell enable
- prebuilt descriptors
- top-fed payload enable
- layer_id / base / scratch
一起往下傳。

### 你現在最需要注意什麼
你把它看成一張「layer dispatch 單」就對了。
這裡最能看出 ownership 到底有沒有往 Top 收。

---

## 3.6 `run_transformer_layer_loop(...)`
### 這是什麼
這是 Top 在多層迴圈中調用 `TransformerLayer` 的地方。

### 為什麼要這樣做
因為同一個硬體 layer 會 time-multiplex 跑多個 runtime layers。

### 對目前專案的影響
這裡通常也是：
- target layer 控制
- counter 更新
- local-only gate 生效
最容易聚集的地方。

### 你現在最需要注意什麼
不要先看每一行。
先看：
- 這裡有沒有算 `attn_compat_shell_enable`
- 這裡有沒有更新觀測 counter
- 這裡有沒有只對某個 layer 開 handoff

---

## 3.7 Top 裡還有另一條線：preproc / LN / final head
### 這是什麼
很多人看 `Top.h` 只盯 attention，
但其實 Top 還有：
- PreprocEmbedSPE 派工
- LayerNorm block 派工
- FinalHead 派工

### 為什麼要這樣做
因為 ownership 收斂不是只改 attention。
真正的終局是：
所有 block 都由 Top 派工、Top 管 shared SRAM。 

### 對目前專案的影響
所以你在 `Top.h` 看到 attention 以外的 block contract，
不是離題；
它們其實是在一起拼成 v12.1 的主線。

### 你現在最需要注意什麼
如果你閱讀負擔很大，建議先分三輪看：
1. preproc + top FSM
2. transformer layer loop
3. final head / output sequencing

---

## 4. 你現在最值得看的 5 個位置
1. 檔頭對 Top ownership 的英文註解
2. `TopRegs`
3. `top_should_enable_attn_compat_shell(...)`
4. `top_dispatch_transformer_layer(...)`
5. `run_transformer_layer_loop(...)`

---

## 5. 最後一個提醒
`Top.h` 大，不代表它在做所有計算。
很多時候它真正做的是：
- 定義邊界
- 決定派工
- 保留證據

白話講：
**`Top.h` 最重要的不是數學，而是 ownership。**
