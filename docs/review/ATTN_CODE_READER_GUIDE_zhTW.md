# ATTN_CODE_READER_GUIDE_zhTW
Date: 2026-04-03

用途：給「現在真的要讀 `Top.h` / `TransformerLayer.h` / `AttnLayer0.h` 的人」看的中文導讀。

邊界：
- 這份文件是 reader guide，不是 closure report。
- 它不主張 Catapult closure，也不主張 SCVerify closure。
- 它的目的只有一個：讓你知道現在這三個檔案到底各自負責什麼、資料怎麼流、哪些地方還沒遷完。

---

## 1. 先講結論
目前 attention 這條線已經不是「完全沒接 Top-managed」；真正的閱讀重點是：
- `Top.h` 已經開始決定哪些 layer 可以關掉 attention compatibility shell。
- `TransformerLayer.h` 還是中間整合層，裡面同時保留 attention glue、FFN preload、residual/LN glue。
- `AttnLayer0.h` 目前看到的外層 wrapper 很薄，真正的 compute 本體還在 `AttnLayer0CoreWindow(...)` 那一層。

白話講：
**現在最難讀，不是因為一切都亂，而是因為控制決策已經往 Top 收，但資料搬運本體還留很多在中間層。**

---

## 2. 5 分鐘讀法（先看這裡就好）

### Step 1：先看 `Top.h`
先看這 3 個名字：
- `top_should_enable_attn_compat_shell(...)`
- `top_dispatch_transformer_layer(...)`
- `run_transformer_layer_loop(...)`

你先把它想成：
- 第一個：**Top 先做判斷**
- 第二個：**Top 把判斷結果丟給 TransformerLayer**
- 第三個：**Top 在 layer loop 裡面持續累積觀測證據**

### Step 2：再看 `TransformerLayer.h`
先看 `TransformerLayer(...)` 這個主函式。
你先不要一行一行讀。
先只抓 3 件事：
- attention 有沒有走 compatibility shell
- FFN payload 是不是還在這裡 preload
- residual / LayerNorm 是不是還在這裡 glue

### Step 3：最後看 `AttnLayer0.h`
先只看外層 `AttnLayer0(...)` wrapper 與 `AttnLayer0TopManagedWindowBridge(...)`。
先知道：
- 這兩層大多是在**轉接參數形狀與 handoff 描述子**
- 真正 attention 主體仍是 `AttnLayer0CoreWindow(...)`

如果你一開始直接衝進 `AttnLayer0CoreWindow(...)`，閱讀負擔會很大。

---

## 3. 這三個檔案各自是什麼

## 3.1 `Top.h`
### 這是什麼
`Top.h` 是整個設計的總控制層。
它不是單純呼叫別人而已，它同時負責：
- 外部 4-channel contract
- state machine
- layer loop
- shared SRAM ownership
- 哪些 local-only handoff 本輪要開、哪些不開

### 為什麼要這樣做
因為這個專案的固定口徑是：
**Top 是唯一 production shared-SRAM owner。**
所以只要是「共享記憶體該不該由 block 直接碰」這種政策問題，最後都應該往 Top 收。

### 對目前專案的影響
這代表你在 `Top.h` 看到很多奇怪的 gate、counter、target-layer 變數，不是雜訊，
而是 migration 過程中的**邊界證據**。

### 你現在最需要注意什麼
先看下面這 4 類東西：

#### A. shell 啟閉判斷
- `top_should_enable_attn_compat_shell(...)`

白話：
如果 attention 四段都已經由 Top-managed prebuild 好，而且沒有額外 top-fed OUT payload，
那這一層就可以不跑舊的 compatibility shell。

#### B. dispatch 邊界
- `top_dispatch_transformer_layer(...)`

白話：
Top 不是只說「去跑 layer」。
它會把這層該不該開 shell、哪些 payload 已經 prebuilt、哪些 handoff descriptor 有效，
一併送進 `TransformerLayer(...)`。

#### C. layer-loop 裡的觀測寄存器
你會看到這類名字：
- `p11bc_managed_attention_target_layer_id`
- `p11bd_attn_compat_shell_enabled_count`
- `p11bd_attn_compat_shell_disabled_count`
- `p11av_ffn_handoff_gate_taken_count`
- `p11av_ffn_handoff_fallback_seen_count`

白話：
這不是演算法本體，這是**觀測點**。
目的是回答：
- 哪一層被當成這輪的 target layer
- 哪幾層還在跑舊 shell
- FFN handoff 這輪到底有沒有真的被吃到

#### D. local-only handoff enable
例如：
- `lid0_local_only_ffn_handoff_enable`
- `lid0_local_only_attn_out_payload_enable`
- `lid0_local_only_qkscore_*_handoff_enable`

白話：
這些通常都不是正式長期 contract，
而是 bounded migration 過程中的 local-only 測試切口。

---

## 3.2 `TransformerLayer.h`
### 這是什麼
`TransformerLayer.h` 是 layer-local integration boundary。
也就是說，它不是最高層政策 owner，
但它負責把 attention、FFN、residual、LayerNorm 接成「一個 layer 看起來會動」。

### 為什麼要這樣做
因為 Top 不適合直接塞滿每個 sub-block 的細節。
所以需要一層中間整合層，把：
- Attention
- FFN
- residual add
- norm parameter load
串起來。

### 對目前專案的影響
這個檔案之所以難讀，是因為它現在同時處在兩個時代中間：
- 一邊已經開始接受 Top 傳下來的 managed/prebuilt/handoff 描述子
- 一邊還保留不少直接從 SRAM preload 的舊骨架

### 你現在最需要注意什麼

#### A. `attn_compat_shell_enable`
這個旗標很重要。
它不是在 `TransformerLayer` 內自己想出來的，
而是 Top 已經先算好，再傳進來。

白話：
`TransformerLayer` 在這裡主要是**消費這個決策**，不是擁有這個決策。

#### B. FFN preload 仍然很多
你會看到一大串像這樣的 loop：
- `TRANSFORMER_LAYER_FFN_TOPFED_X_PRELOAD_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W1_PRELOAD_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W1_BIAS_PRELOAD_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W2_INPUT_PRELOAD_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W2_WEIGHT_PRELOAD_LOOP`
- `TRANSFORMER_LAYER_FFN_TOPFED_W2_BIAS_PRELOAD_LOOP`

白話：
雖然名字叫 topfed / handoff，
但目前很多資料還是在 `TransformerLayer` 內自己先從 `sram[...]` 搬到 local array。

這就是為什麼不能因為 helper 名字叫 Top-managed，
就直接以為 block 已經完全脫離 direct SRAM。

#### C. residual / LN glue 仍在這裡
這份檔案不是單純「呼叫 Attn 再回傳」。
它還會：
- 做 FFN dispatch
- 做 residual add
- 做 LayerNorm 參數 preload 與呼叫

所以你讀這份檔時，要把它看成「layer glue」，不是看成單一 kernel。

---

## 3.3 `AttnLayer0.h`
### 這是什麼
`AttnLayer0.h` 是 attention block 的入口層。
但你目前最先看到的，不是整個 attention 本體，
而是一些 wrapper 與 bridge。

### 為什麼要這樣做
因為 migration 過程中，
同一個 attention 核心需要同時支援：
- 一般 pointer 形式
- array-shaped bridge（給 Catapult-facing 路徑）
- prebuilt/handoff 描述子版本

所以你會看到外面有幾層薄薄的包裝。

### 對目前專案的影響
這讓 `AttnLayer0.h` 看起來像很簡單，
但實際上真正複雜的部分被推到 `AttnLayer0CoreWindow(...)` 裡。

### 你現在最需要注意什麼

#### A. wrapper 不等於本體
你現在先看到的：
- `AttnLayer0(...)`
- `AttnLayer0TopManagedWindowBridge(...)`

多半只是：
- 收參數
- 包 handoff descriptor
- 轉成 `AttnLayer0CoreWindow(...)` 要的格式

#### B. prebuilt flag 的意思
像這些：
- `kv_prebuilt_from_top_managed`
- `q_prebuilt_from_top_managed`
- `score_prebuilt_from_top_managed`
- `out_prebuilt_from_top_managed`

白話：
它們是在說「這段工作是不是已經在上游做好」。
不是在說 attention block 變成 SRAM owner。

#### C. TopManagedWindowBridge 的意思
`AttnLayer0TopManagedWindowBridge(...)` 的重點不是改演算法，
而是把呼叫邊界變成 Catapult 比較容易接受、比較好分析的 array-shaped surface。

白話：
這比較像是**邊界整形**，不是 attention 數學被重寫。

---

## 4. 現在最容易看不懂的詞，我幫你翻成白話

### `compatibility shell`
白話：
舊路徑外面包的一層相容殼。
當新的 Top-managed 四段都 ready 時，就希望不要再進這層。

### `prebuilt`
白話：
這段資料或結果，前面已經先準備好了。
後面不一定還要重算。

### `topfed`
白話：
不是 block 自己去 shared SRAM 抓，而是上游把 payload 或描述子先餵進來。

### `handoff descriptor`
白話：
不是完整資料本體，而是描述「有沒有資料、資料在哪、有效多少」的交接資訊。

### `fallback`
白話：
本輪想走的新路徑沒成功或沒資料時，退回舊路徑。

### `write-back boundary`
白話：
哪一層負責把結果正式寫回某個共享落點。

---

## 5. 為什麼 code 幾乎沒有你期待的那種註解
先講白話版：
**因為這個 repo 現在的治理規則，優先保護 Catapult 與 compile log，不是優先保護第一次閱讀的人。**

所以它的傾向是：
- design code 註解要少而精
- 儘量 ASCII/English
- 中文導讀放 `docs/review/` 或 `docs/handoff/`
- loop label 要能給 GUI 看

這做法對工具友善，
但如果沒有 sidecar，你就會很痛苦。

所以這份文件存在的目的，就是把這個缺口補起來。

---

## 6. 我現在只建議你先看哪 3 段 code

### 第一段：`Top.h`
先看：
- `top_should_enable_attn_compat_shell(...)`
- `top_dispatch_transformer_layer(...)`
- `run_transformer_layer_loop(...)`

你要回答的問題只有一個：
**這層要不要跑舊 shell，到底是誰決定的？**

答案：Top。

### 第二段：`TransformerLayer.h`
先看：
- `TransformerLayer(...)`
- FFN topfed preload 那整段 loop

你要回答的問題只有一個：
**這裡到底只是 glue，還是真的已經完全變成 channel-only？**

答案：目前還沒有完全變成 channel-only。

### 第三段：`AttnLayer0.h`
先看：
- `AttnLayer0(...)`
- `AttnLayer0TopManagedWindowBridge(...)`

你要回答的問題只有一個：
**我現在看到的是本體，還是只是 wrapper？**

答案：先看到的多半只是 wrapper。

---

## 7. 目前這條線到底前進到哪裡
### 已經前進的地方
- Top 已能決定 attention compatibility shell 的啟閉。
- target layer 的 managed path 已有專門觀測點。
- FFN handoff 也已有 gate/non-empty/fallback 類計數器。
- AttnLayer0 已有 TopManagedWindowBridge 這種邊界整形入口。

### 還沒前進完的地方
- `TransformerLayer` 內仍有大量 preload / glue / residual / LN 邏輯。
- FFN payload 很多地方仍是先從 `sram[...]` 搬進 local array。
- attention 真正 compute 主體仍不是你現在第一眼看到的 wrapper，而是在 deeper core。

### 這代表什麼
白話：
**ownership 的口徑已經開始往 Top 收，但 data mover 還沒有完全收乾淨。**

---

## 8. 如果你下一輪要繼續補檔，我建議怎麼補
下一輪不要貪多。
先補下面其中一個就好：

### 方案 A：補 `TransformerLayer` 專屬 reader guide
專門講：
- attention shell
- FFN preload
- residual/LN glue
- 哪些 direct SRAM 還留著

### 方案 B：補 `AttnLayer0CoreWindow` stage guide
專門講：
- QKV
- score/softmax
- out write-back
- 哪些 flag 只是 handoff，哪些真會改路徑

### 方案 C：補「變數名稱白話對照表」
把最常見的 30 個名字翻譯成人話。
例如：
- prebuilt
- topfed
- handoff
- compat shell
- fallback
- write-back
- target layer
- non-empty

如果你的目標是先看懂 code，
我最推薦先做 **方案 A**。

---

## 9. 這份文件不保證什麼
這份文件只保證：
- 幫你降低第一次讀 code 的負擔
- 幫你把角色、資料流、ownership 邊界講清楚

這份文件不保證：
- Catapult closure
- SCVerify closure
- 全部 direct SRAM 已移除
- attention 演算法本體已重構完成

