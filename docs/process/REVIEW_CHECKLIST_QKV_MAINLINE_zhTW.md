# REVIEW_CHECKLIST_QKV_MAINLINE_zhTW
Date: 2026-03-19

用途：給 reviewer 的可執行檢查清單。  
原則：每條都提供 `Primary pointer`，讓 reviewer 可直接跳到下一個檢查位置。

## 1. Evidence posture（先對齊）
- 目前可引用的 PASS 屬於 local-only evidence。
- Catapult closure 與 SCVerify closure 仍屬 deferred（未執行即不宣稱）。
- 主要參考：`REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md` 的「PASS 代表什麼 / 不代表什麼」。

## 1.1 Companion quick tools
- Top/Transformer 15 分鐘快檢：[`TOP_TRANSFORMER_QUICKCHECK_zhTW.md`](./TOP_TRANSFORMER_QUICKCHECK_zhTW.md#15-minute-fast-scan-order)
- Reviewer 結論模板（短版/完整版）：[`REVIEW_VERDICT_TEMPLATE_QKV_MAINLINE_zhTW.md`](./REVIEW_VERDICT_TEMPLATE_QKV_MAINLINE_zhTW.md#ultra-short-note-template)

<a id="fast-10-minute-pass"></a>
## 2. Fast 10-minute pass（10 項）
- [ ] Check: Top 是唯一 SRAM policy owner。  
  Owns / Not owns: owns shared SRAM 決策；not owns layer block 內部數學。  
  Inputs / Outputs: command/payload 進入，邊界/可見區段決策輸出到下游。  
  Evidence posture: local-only。  
  Primary pointer: `src/Top.h::run_transformer_layer_loop(...)` + `REVIEWER_GUIDE` [5. Ownership / Boundary](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ownership-boundary)

- [ ] Check: fallback meaning 在 Top hook 被鎖定，而非在 Attn/leaf 臨時決策。  
  Owns / Not owns: owns fallback latch policy；not owns leaf kernel math。  
  Inputs / Outputs: layer0 hook 觀測狀態，輸出 prebuilt/fallback flags。  
  Evidence posture: local-only。  
  Primary pointer: `src/Top.h::run_transformer_layer_loop(...)` + `REVIEWER_GUIDE` [5. Ownership / Boundary](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ownership-boundary)

- [ ] Check: TransformerLayer 是 integration delegate，不是全域 policy owner。  
  Owns / Not owns: owns Attn/FFN/LN layer-local glue；not owns Top FSM/SRAM policy。  
  Inputs / Outputs: consume x/scratch/param base 與 flags，輸出 layer-local結果。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/TransformerLayer.h::TransformerLayer(...)` + `REVIEWER_GUIDE` [6.2 TransformerLayer](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#transformerlayer-role)

- [ ] Check: AttnLayer0 的 handoff-in/handoff-out 邊界清楚。  
  Owns / Not owns: owns stage-scoped attention compute；not owns external contract。  
  Inputs / Outputs: consume `x_in_base_word`/`sc.*_base_word`/prebuilt flags；produce `attn_out_base_word`。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h::AttnLayer0(...)` + `REVIEWER_GUIDE` [6.3 AttnLayer0](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attnlayer0-role)

- [ ] Check: QKV stage 的 fallback/bypass 是顯式路徑而非隱式副作用。  
  Owns / Not owns: owns stage內 materialization/bypass；not owns fallback policy lock。  
  Inputs / Outputs: X/W metadata 輸入；Q/K/V scratch 輸出。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h` 的 `ATTN_STAGE_QKV` + [ATTNLAYER0_STAGE_CROSSCHECK QKV](./ATTNLAYER0_STAGE_CROSSCHECK_zhTW.md#stage-qkv)

- [ ] Check: SCORES stage 負責 score/softmax/pre-concat，不提前做最終 write-back。  
  Owns / Not owns: owns score/probability/pre-concat；not owns final OUT boundary。  
  Inputs / Outputs: consume Q/K/V；produce pre/post-concat scratch。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h` 的 `ATTN_STAGE_SCORES` + [ATTNLAYER0_STAGE_CROSSCHECK SCORES](./ATTNLAYER0_STAGE_CROSSCHECK_zhTW.md#stage-scores)

- [ ] Check: OUT stage 才是 attention 最終寫回點。  
  Owns / Not owns: owns attention output write-back；not owns Top 最終 outmode write-back。  
  Inputs / Outputs: consume `post_concat`；write `attn_out_base_word`。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h` 的 `ATTN_STAGE_OUT` + `REVIEWER_GUIDE` [6.3 AttnLayer0](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attnlayer0-role)

- [ ] Check: ternary core kernel 與 wrappers 的角色分層正確。  
  Owns / Not owns: core owns guard/decode/MAC；wrappers not own SRAM policy。  
  Inputs / Outputs: core consume row/payload/meta 並輸出 materialized row；wrappers提供固定介面。  
  Evidence posture: local-only + compile-prep family regression。  
  Primary pointer: `src/blocks/TernaryLiveQkvLeafKernel*.h` + [TERNARY_LEAF_ROLEMAP](./TERNARY_LEAF_ROLEMAP_zhTW.md#file-role-comparison)

- [ ] Check: loop family 解讀使用語意群組，不逐行猜測。  
  Owns / Not owns: loop labels owns GUI 可視語義；not owns policy decision本身。  
  Inputs / Outputs: reviewer trace input->compute->writeback。  
  Evidence posture: local-only。  
  Primary pointer: `REVIEWER_GUIDE` [7. Loop-Role Notes](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#loop-role-notes) + [TERNARY loop map](./TERNARY_LEAF_ROLEMAP_zhTW.md#loop-family-quick-map)

- [ ] Check: PASS 解讀不越權到 closure。  
  Owns / Not owns: owns local acceptance meaning；not owns Catapult/SCVerify formal closure。  
  Inputs / Outputs: consume test logs；output review verdict wording。  
  Evidence posture: local-only。  
  Primary pointer: `REVIEWER_GUIDE` [9. PASS 代表什麼 / 不代表什麼](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#pass-semantics)

- [ ] Check: 輸出 reviewer verdict 時，使用統一模板避免過度宣稱。  
  Owns / Not owns: owns review wording hygiene；not owns technical closure itself。  
  Inputs / Outputs: consume本輪檢查結果；輸出可接受/保留/需補件 verdict。  
  Evidence posture: local-only。  
  Primary pointer: [REVIEW_VERDICT_TEMPLATE（ultra-short）](./REVIEW_VERDICT_TEMPLATE_QKV_MAINLINE_zhTW.md#ultra-short-note-template) + [REVIEW_VERDICT_TEMPLATE（full）](./REVIEW_VERDICT_TEMPLATE_QKV_MAINLINE_zhTW.md#full-review-note-template)

<a id="deeper-30-minute-pass"></a>
## 3. Deeper 30-minute pass（18 項）
<a id="top-integration-boundary"></a>
### 3.1 Top / integration boundary
- [ ] Check: command ingest (`SET_W_BASE/LOAD_W/INFER`) 與 `W_REGION` 可見範圍建立一致。  
  Owns / Not owns: Top owns region visibility；not owns leaf compute。  
  Inputs / Outputs: command/data_in -> SRAM window readiness。  
  Evidence posture: local-only。  
  Primary pointer: `src/Top.h::top(...)` + `REVIEWER_GUIDE` [4. End-to-End Dataflow](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#end-to-end-dataflow)

- [ ] Check: layer orchestration 只派工，不重定義 leaf/attn 內部職責。  
  Owns / Not owns: owns orchestration；not owns stage-local algorithm。  
  Inputs / Outputs: per-layer base/range + flags -> delegate call chain。  
  Evidence posture: local-only。  
  Primary pointer: `src/Top.h::run_transformer_layer_loop(...)` + `REVIEWER_GUIDE` [6.1 Top](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#top-role)

- [ ] Check: Top write-back/readback 邊界不被 layer block 混用。  
  Owns / Not owns: owns outmode writeback/readback boundary；not owns Attn internal scratch behavior。  
  Inputs / Outputs: final tensor -> `data_out` / read_mem stream。  
  Evidence posture: local-only。  
  Primary pointer: `src/Top.h::infer_emit_outmode_payload(...)` / `handle_read_mem(...)` + `REVIEWER_GUIDE` [5. Ownership / Boundary](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ownership-boundary)

<a id="transformerlayer-boundary"></a>
### 3.2 TransformerLayer boundary
- [ ] Check: TransformerLayer 只整合 Attn/FFN/LN，不重新定義 Top 決策。  
  Owns / Not owns: owns layer-local integration；not owns Top fallback lock。  
  Inputs / Outputs: consume Top-provided boundaries/flags -> produce layer-local outputs。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/TransformerLayer.h::TransformerLayer(...)` + `REVIEWER_GUIDE` [6.2 TransformerLayer](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#transformerlayer-role)

- [ ] Check: residual + norm glue 的位置與責任不外溢到 Top/Attn。  
  Owns / Not owns: owns layer residual/norm glue；not owns global state machine。  
  Inputs / Outputs: Attn/FFN outputs + norm params -> layer output。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/TransformerLayer.h` + `REVIEWER_GUIDE` [6.2 TransformerLayer](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#transformerlayer-role)

<a id="attnlayer0-lower-level-flow"></a>
### 3.3 AttnLayer0 lower-level flow
- [ ] Check: QKV stage 讀寫邊界符合 `x_in_base_word` 與 `sc.*_base_word` 設計。  
  Owns / Not owns: owns stage compute；not owns global SRAM arbitration。  
  Inputs / Outputs: X + W metadata -> Q/K/V windows。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h` (`ATTN_STAGE_QKV`) + [ATTNLAYER0 QKV](./ATTNLAYER0_STAGE_CROSSCHECK_zhTW.md#stage-qkv)

- [ ] Check: `q_prebuilt_from_top_managed` / `kv_prebuilt_from_top_managed` 只控制 materialization skip。  
  Owns / Not owns: owns skip gating usage；not owns flag definition policy。  
  Inputs / Outputs: consume prebuilt flags -> select live/fallback/bypass path。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h::AttnLayer0(...)` + `REVIEWER_GUIDE` [6.3 AttnLayer0](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attnlayer0-role)

- [ ] Check: score/softmax/pre-concat 與 OUT stage write-back 分層清楚。  
  Owns / Not owns: scores owns pre-output accumulation；OUT owns final attention writeback。  
  Inputs / Outputs: Q/K/V -> pre/post concat -> attn_out。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h` (`ATTN_STAGE_SCORES`, `ATTN_STAGE_OUT`) + [ATTNLAYER0 SCORES](./ATTNLAYER0_STAGE_CROSSCHECK_zhTW.md#stage-scores)

- [ ] Check: fallback/bypass 對 Q/K/V 三路都可追蹤到明確意圖。  
  Owns / Not owns: owns stage內 fallback/bypass execution；not owns fallback policy ownership。  
  Inputs / Outputs: live gate result -> bypass copy or live materialization result。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h` (`ATTN_Q*`, `ATTN_K*`, `ATTN_V*`) + `REVIEWER_GUIDE` [7.3 ATTN_*](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attn-loop-family)

- [ ] Check: Attn handoff-out 僅透過 `attn_out_base_word` 對接下游。  
  Owns / Not owns: owns attention output region write；not owns downstream FFN semantics。  
  Inputs / Outputs: post-concat -> attn_out window。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h` (`ATTN_OUT_WRITEBACK_LOOP`) + `REVIEWER_GUIDE` [6.3 AttnLayer0](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#attnlayer0-role)

<a id="ternary-leaf-family"></a>
### 3.4 Ternary leaf family
- [ ] Check: `LeafKernel` 是 core compute/guard owner。  
  Owns / Not owns: owns row kernel guard/decode/MAC；not owns orchestration/policy。  
  Inputs / Outputs: x row + payload/meta -> quantized row + act_q。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/TernaryLiveQkvLeafKernel.h` + [TERNARY role map](./TERNARY_LEAF_ROLEMAP_zhTW.md#file-role-comparison)

- [ ] Check: `LeafKernelTop` 是 fixed-shape local wrapper，不是 policy owner。  
  Owns / Not owns: owns local split-interface surface；not owns SRAM policy。  
  Inputs / Outputs: fixed-shape wrapper inputs -> delegate to split row kernels。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/TernaryLiveQkvLeafKernelTop.h` + [TERNARY role map](./TERNARY_LEAF_ROLEMAP_zhTW.md#file-role-comparison)

- [ ] Check: `LeafKernelCatapultPrepTop` 僅 compile-prep adapter。  
  Owns / Not owns: owns compile-prep wrapper surface；not owns runtime policy。  
  Inputs / Outputs: compile-prep wrapper inputs -> delegate kernel path。  
  Evidence posture: compile-prep family local regression。  
  Primary pointer: `src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h` + [TERNARY role map](./TERNARY_LEAF_ROLEMAP_zhTW.md#file-role-comparison)

- [ ] Check: `ShapeConfig` 是 constants SSOT，不含 materialization 行為。  
  Owns / Not owns: owns constants；not owns runtime compute。  
  Inputs / Outputs: compile-time config constants -> consumed by wrappers/kernels。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h` + [TERNARY role map](./TERNARY_LEAF_ROLEMAP_zhTW.md#file-role-comparison)

- [ ] Check: loop label 家族用於 GUI trace，不應被誤解成 policy owner。  
  Owns / Not owns: owns observability naming；not owns semantic ownership decision。  
  Inputs / Outputs: loop trace -> reviewer evidence chain。  
  Evidence posture: local-only。  
  Primary pointer: `REVIEWER_GUIDE` [7.4 TERNARY_*](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ternary-loop-family) + [Loop family quick map](./TERNARY_LEAF_ROLEMAP_zhTW.md#loop-family-quick-map)

<a id="writeback-fallback-ownership"></a>
### 3.5 Write-back / fallback / ownership
- [ ] Check: Attn OUT write-back 與 Top outmode write-back 是兩層不同邊界。  
  Owns / Not owns: Attn owns attn_out boundary；Top owns external outmode boundary。  
  Inputs / Outputs: post_concat -> attn_out -> data_out。  
  Evidence posture: local-only。  
  Primary pointer: `src/blocks/AttnLayer0.h::ATTN_OUT_WRITEBACK_LOOP` + `src/Top.h::infer_emit_outmode_payload(...)`

- [ ] Check: fallback lock 發生在 Top，Attn/leaf 僅消費旗標。  
  Owns / Not owns: Top owns meaning lock；Attn/leaf not own policy origin。  
  Inputs / Outputs: lock flags -> delegated execution choices。  
  Evidence posture: local-only。  
  Primary pointer: `src/Top.h::run_transformer_layer_loop(...)` + `src/blocks/AttnLayer0.h::AttnLayer0(...)`

- [ ] Check: reviewer 結論必須保留 deferred 限制。  
  Owns / Not owns: owns current acceptance wording；not owns formal closure claim。  
  Inputs / Outputs: local logs -> bounded review verdict。  
  Evidence posture: local-only。  
  Primary pointer: `REVIEWER_GUIDE` [9. PASS 代表什麼 / 不代表什麼](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#pass-semantics)
