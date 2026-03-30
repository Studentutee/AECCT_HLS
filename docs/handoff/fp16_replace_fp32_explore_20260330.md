# FP16 Replace FP32 Explore Note (2026-03-30)

## 1. Purpose / Scope
- 本文件記錄 `AECCT_ac_ref` 的 ref-only sanity check：
  - 保持 native linear quant 主集合不變
  - 測試將原 FP32 islands 以 FP16 roundtrip 取代
- 不是 HLS mainline closure，不是 Catapult closure，不是 SCVerify closure。

## 2. Mode Definition
- 新模式：`FP16_REPLACE_FP32_GLOBAL`
- 定義：
  - native linear quant 主集合（Wq/Wk/Wv/Wo/Wff1/Wff2）維持原樣
  - 其餘原本 FP32 intermediate/island 走 FP16 roundtrip
  - FinalHead 仍維持特殊 Pass A/Pass B 定位，不重分類成普通 linear kernel

## 3. What Stayed Unchanged
- native linear quant 主集合語義不變（INT8 activation + ternary weight + INT16 accum 路徑不改）
- evaluator 定義不變：`target_x = trace_output_x_pred_step0_tensor`
- LN algorithm 不變
- FinalHead 特殊定位不變

## 4. FP16 Roundtrip Semantics
- 使用 `ref_fp16_t = ac_std_float<16,5>`
- 實作：`FP32/current -> FP16 -> FP32`
- 計數器：
  - `fp16 roundtrip count`
  - `fp16 nan in/out`
  - `fp16 inf in/out`
  - `fp16 underflow->zero`
  - `fp16 first nonfinite block`

## 5. Test Windows and Commands
- quick: `begin=0,count=4`
- follow-up: `begin=4,count=12`
- extended: `begin=16,count=16`
- 每個視窗皆跑 compare + eval-compare。

## 6. Result Summary (FP16 global)

| window | x_pred flip patterns | sign flip patterns | total flipped bits | delta BER | delta FER | experiment BER | experiment FER | worst min margin | worst logits MSE | worst max abs diff | fp16 roundtrip | fp16 nonfinite/underflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0~3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.041406250e+01 | 2.189558404e+00 | 5.062484741e+00 | 830280 | nan/inf=0/0, underflow=0 |
| 4~15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.753906250e-01 | 2.333995270e+00 | 6.264122009e+00 | 2490840 | nan/inf=0/0, underflow=0 |
| 16~31 | 0 | 0 | 0 | 0 | 0 | 1.984126984e-03 | 1.250000000e-01 | 5.882697552e-02 | 2.912346392e+00 | 7.025508881e+00 | 3321120 | nan/inf=0/0, underflow=0 |

備註：
- `16~31` baseline 本身即有 bit/frame errors（BER=1.984126984e-03, FER=1.25e-01），FP16 與 baseline 完全同值。

## 7. Comparison to Existing Sensitive-Path Evidence (boundary-aware)
- `generic_e4m3_g2_embed_only`（局部敏感路徑模式）在 `4~15`：
  - `delta BER=+1.322751323e-03`, `delta FER=+8.333333333e-02`, 且有 x_pred/sign flip
- `int8_fixedexp_zone3_embed_g2`（局部敏感路徑模式）在 `4~15`：
  - `delta BER=0`, `delta FER=0`, 無 x_pred/sign flip
- 本輪 `FP16_REPLACE_FP32_GLOBAL`（全域 FP32 islands 替換模式）在 `4~15`：
  - `delta BER=0`, `delta FER=0`, 無 x_pred/sign flip

比較邊界：
- 上述三者 footprint 不同，不能當作完全 apples-to-apples。
- 但可用於判斷「高精度例外層」的穩定性方向：FP16 global 在目前 coverage 下非常穩。

## 8. Interpretation Boundary
- 能回答：
  - 在本輪 coverage，FP16 全域替換原 FP32 islands 對 evaluator 幾乎無影響（delta BER/FER 全為 0）。
  - FP16 可以作為高敏感區保險層候選。
- 不能回答：
  - FP16 已是最終量產策略。
  - 可直接取代所有低精度探索。
  - 已完成全模型/全資料集 closure。

## 9. SRAM Policy Impact (draft-level)
- 支持分層策略：
  - 大頁面主策略：8-bit storage（例如 shared-exp INT8）
  - 小量高敏感例外：FP16 保護層（尤其 LN/softmax local states、FinalHead scalar/readout neighborhood）
- `W_REGION` ternary packed payload 維持 2-bit 類型，不應混淆為改 8-bit。

## 10. Next Recommended Step
- 先不動 LN 演算法，維持現有 evaluator 口徑。
- 以 FP16 作為高敏感例外 baseline，繼續針對大容量頁面做 shared-exp INT8 局部擴展與 attribution。

## 11. Evidence Sources
- `build/ref_eval/compare_summary_begin0_count4_fp16_replace_fp32_global.txt`
- `build/ref_eval/eval_compare_begin0_count4_fp16_replace_fp32_global.txt`
- `build/ref_eval/compare_summary_begin4_count12_fp16_replace_fp32_global.txt`
- `build/ref_eval/eval_compare_begin4_count12_fp16_replace_fp32_global.txt`
- `build/ref_eval/compare_summary_begin16_count16_fp16_replace_fp32_global.txt`
- `build/ref_eval/eval_compare_begin16_count16_fp16_replace_fp32_global.txt`
- 既有對照：`*_g2_embed_only*.txt`, `*_int8fx_z3_embed_g2*.txt`

## 12. Governance Posture
- ref-only exploration
- local-only evidence
- not HLS mainline closure
- not Catapult closure
- not SCVerify closure
- trace-reference-aligned BER/FER interpretation
