# SRAM Precision Policy Draft (2026-03-30)

## 1. 文件定位
- 這是 task-local 的 architecture/policy draft，服務於 ref 探索與後續設計評估。
- 目標：在「容量優先」前提下，提出可落地的 precision 分級方向，並同時考慮計算便利與平行化。
- 這不是 frozen spec，不是 HLS mainline closure，不是 Catapult/SCVerify closure。

## 2. 前提與邊界
- v12.1 baseline 前提不改：
  - native linear quant 主集合：`Wq/Wk/Wv/Wo/Wff1/Wff2`
  - FinalHead (`FinalEmbedding/OUT_FC/logits/x_pred`) 獨立於普通 linear kernel
- evaluator 邊界不改：
  - `target_x = trace_output_x_pred_step0_tensor`
  - BER/FER 是 trace-reference-aligned
- fragility evidence 鏈結（本草案依據）：
  - full stress 早期 RED
  - `GENERIC_E4M3_EXCEPT_G5` 在 4/16 patterns 無 RED
  - `G5_sub × G2` 中 `embed_only` 在 `4~15` 已 RED（delta BER/FER > 0）
  - `preproc_assembly`、`spe_only` 在 `16~31` 出現 baseline-relative flip，但 BER/FER 未變差

## 3. 核心原則
1. 大容量物件優先壓 storage bitwidth；小量高敏感 scalar/state 可保留較高精度。  
2. `W_REGION` 的 ternary payload 已小於 8-bit，不應被誤化為「是否降到 8-bit」問題。  
3. 明確分離：
   - SRAM storage bitwidth
   - on-read widen/unpack
   - compute bitwidth
   - accumulator/scalar-state bitwidth
   - write-back/requant bitwidth  
4. 不為了「全 8-bit 口號」把 LN/softmax/final scalar 等小量敏感狀態硬降精度。  
5. 若降 storage 造成 control/unpack/rescale/divider 額外成本，要一併計入 area/parallelism trade-off。  

## 4. Precision Policy Table (Draft)

| Storage / object | Role / semantic meaning | Capacity driver | Current storage form / precision (in repo/spec) | Proposed storage bitwidth | Proposed compute bitwidth | Proposed accumulator / scalar-state bitwidth | Allow widen-on-read? | Why not lower? / Why can lower? | Fragility evidence linkage | Hardware convenience / area / parallelism comment | Recommended priority | Validation next step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `W_REGION` ternary payload | native linear quant 權重 payload | persistent capacity 大戶 | 2-bit ternary packed（16 weights/32-bit word） | **維持 2-bit packed** | decode to `{-1,0,+1}` integer domain | INT16（native linear accum） | Yes（tile decode/unpack） | 已經比 8-bit 更省；再降意義有限且增加 codec 風險 | full stress failure 不是由 W payload/INT16 ovf 引起（ovf=0） | 以 tile unpack 換取 MAC 內迴圈簡化，可提升可平行化程度 | P0 keep | 維持現行；量測 unpack 帶寬與局部 SRAM burst 效率 |
| `W_REGION` quant metadata (`inv_s_w`, section metadata, scale params) | decode/dequant 參數與索引 | persistent，但容量小於 payload | 多數以 FP32/U32 metadata 存放 | 16-bit 為優先（可保留少量 32-bit 例外） | FP16/FP32 | FP32（scale chain） | Yes | metadata 容量小但全域可見；盲降到 8-bit 會放大 scale 誤差 | `G5_sub×G2` 顯示敏感區在 activation side 互動，非 metadata 單點 | 16-bit metadata 有助減少 SRAM 容量，對 datapath 控制成本可控 | P1 | 先做 metadata 類型盤點，再做逐欄位 sensitivity 掃描 |
| `X_WORK` main token×d working set | preproc/layer outputs/LN outputs 主工作矩陣 | **active working-set 大戶** | v12.1 邏輯語意 FP32（single X_WORK） | 目標候選：8-bit storage（block scale + layout-aware） | on-read widen 到 FP16/FP32 進入非線性區 | FP32（LN/softmax/residual/final sensitive path） | Yes（必要） | 可降是因容量壓力最大；不可盲降是因 interaction fragility 明顯 | full stress RED；`EXCEPT_G5` pass；`G5_sub×G2` 顯示 preproc/residual 交互敏感 | 8-bit storage 可顯著減 SRAM；但 widen/rescale datapath 必須可流水化 | P0 candidate | 先做 `G5_sub×G2` 進一步定位，再定義「高精度例外槽位」最小集合 |
| `SCR_K` | K cache / layer-scoped scratch | active working-set 大戶 | spec 為 SCRATCH 子區，語意多為 FP32 | 候選 8-bit storage（含 block scale） | dot/softmax neighborhood 用 FP16/FP32 | FP32 for score/online state feed | Yes | 可降容量大；但不宜把 score/online state 一起低精度 | G4/G3 組合在 `EXCEPT_G5` 下可存活，顯示可探索但需守 mask/online stability | K/V storage 低精度可降面積與記憶體，若 on-read widen 單元簡潔可換平行度 | P1 candidate | 以不破壞 masked prob=0 語意為前提做局部驗證 |
| `SCR_V` | V cache / layer-scoped scratch | active working-set 大戶 | spec 為 SCRATCH 子區，語意多為 FP32 | 候選 8-bit storage（含 block scale） | context accumulation 用 FP16/FP32 | FP32 for context accumulation | Yes | 與 `SCR_K` 類似；不宜把 accumulation 也壓到 8-bit | pairwise 未顯示單獨 V catastrophic，但 full stress 仍顯示累積互動風險 | 若 V storage 降精度且 accumulation 保高精度，可兼顧容量與穩定 | P1 candidate | 與 `SCR_K` 聯合做 tile 粒度 sensitivity |
| `FINAL_SCALAR_BUF` | FinalHead Pass A/B 之間 `s_t` staging | 小容量 scalar buffer | spec 要求合法 scratch staging，語意 FP32 | 建議先保守 `>=16-bit`（FP16/BF16/FP32） | FP16/FP32 | FP32 for OUT_FC reduction accum | Optional | 量小但高敏感，不值得優先壓到 8-bit | FinalHead partial 可行，但 full stress 與 G5×G2 顯示邊界附近敏感 | 保高精度對容量影響小，能降低 readout 邊界風險 | P0 protect | 先維持高精度；後續若要降，先做小範圍 local-island A/B test |
| LN local states (`mean/var/invstd`, token-local reductions) | normalization scalar states | 小容量、強敏感 | 目前多為 local FP32 狀態 | 建議維持 FP32（可研究 BF16） | FP32 | FP32 | No（原則上 local state 直接用） | 這類狀態數量小，硬降會直接放大 normalization 誤差 | fragility 證據指向 interaction，不支持先動 LN 演算法/精度 | 保持高精度可減少反覆 debug，對 SRAM 容量影響有限 | P0 protect | 保持現狀，先完成非 LN 區塊 attribution |
| softmax local states (`max/sumexp/reciprocal`, acc vec tile) | online softmax 核心狀態 | 小容量、強敏感、高更新頻率 | spec 建議 regs/local buffers，不應回主 SRAM | 建議 FP32 local（不以 8-bit 為目標） | FP32 | FP32 | No（盡量不落 SRAM） | masked/online state 對穩定性敏感，硬降容易造成 boundary drift | full stress 早期 RED 但非 NaN/Inf，符合「數值邊界漂移」特徵 | 本地高精度狀態通常比頻繁 SRAM RMW 更省時省功耗 | P0 protect | 維持 online softmax 高精度狀態，優先優化資料搬運而非降精度 |
| token-local scratch / local tile buffers / local FIFOs | block-local tiles、decode、短生命週期 accum | 容量中小，取決於 tile 參數 | 多為 local array/reg | 8~16-bit 視用途；acc tile 建議 >=16/FP32 | compute 依 kernel（int/FP mixed） | accum 視 kernel：INT16 或 FP32 | Yes（依 tile） | 局部可降但不可犧牲 accumulator 精度 | `G5_sub×G2` 顯示 interaction 更重要，先保留 accum 安全邊界 | 較小運算單元可換取更高並行，但需平衡 unpack/rescale 負擔 | P1 | 先對 decode/tile buffer 做定向 profiling，再決定 bitwidth |
| IO staging (`data_in/data_out` path, optional IO_REGION) | 協定/搬運 staging | 協定驅動，不是主要模型容量 | 32-bit words（u32 raw protocol） | 維持 32-bit word | N/A（協定層） | N/A | No | 協定固定；壓位寬會增加封包/控制複雜度 | 與 fragility 無直接關係 | 維持簡單協定有助驗證與除錯 | P0 keep | 不改協定位寬；僅優化輸出節流與搬運時序 |

## 5. 簡短結論（草案）
- 最值得先做 8-bit storage 的類別：
  - `X_WORK`, `SCR_K`, `SCR_V`（但要搭配 on-read widen 與高精度 accumulator/state）
- 應先保守保留高精度的類別：
  - `FINAL_SCALAR_BUF`
  - LN scalar states
  - softmax online states
  - token-local 高頻 accumulator
- 已經比 8-bit 更省的類別：
  - `W_REGION` ternary packed payload（2-bit）
- 最需要更多 ref evidence 才能定案的類別：
  - `X_WORK` 中哪些 boundary 必須列為高精度例外（特別是 `G5_sub × G2` 交互區）
- 若要用「最小高精度例外集合」達成 SRAM 大幅下降，初步建議優先保護：
  - `G5 embed/preproc` 與 `G2 residual merge` 交界
  - FinalHead scalar/readout path
  - LN/softmax 核心 scalar state

## 6. Governance Posture
- ref-only exploration input
- local-only evidence
- not HLS mainline closure
- not Catapult closure
- not SCVerify closure
- this policy is a draft, not frozen spec
