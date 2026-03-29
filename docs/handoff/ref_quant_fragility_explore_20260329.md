# AECCT_ac_ref Ref-Only Quant Fragility Explore Note (2026-03-29)

## 1. Purpose / Scope
- 本文件是 `AECCT_ac_ref` 的 **ref-only quant fragility exploration** 任務摘要。
- 本文件用途是交接與後續探索參考，不是 closure 報告。
- 明確非以下範疇：
- 非 HLS mainline closure
- 非 Catapult closure
- 非 SCVerify closure
- 非 milestone acceptance

## 2. Evaluator Interpretation Boundary
- `target_x` 來源：`trace_output_x_pred_step0_tensor`（`output_x_pred_step0.h`）。
- 這裡的 BER/FER 是 **trace-reference-aligned BER/FER**。
- `delta BER = BER_experiment - BER_baseline`。
- `delta FER = FER_experiment - FER_baseline`。
- 這不是 transmitted all-zero ground-truth BER/FER，解讀時不可混用口徑。

## 3. Experiment Mode Definitions
- 正式 v12.1 baseline（不在此任務改動）：
- native linear quant 主集合為 `Wq/Wk/Wv/Wo/Wff1/Wff2`，其餘 major datapath 原則上維持 FP32。
- `FinalHead` 必須獨立看待：
- Pass A: `FinalEmbedding -> s_t`
- Pass B: `OUT_FC` 對 token 維度做 reduction，輸出 `logits/x_pred`
- `FinalEmbedding/OUT_FC/logits/x_pred` 不是普通 linear kernel。
- `FULL_E4M3_NONLINEAR_STRESS`：
- 除 native linear 主集合外，幾乎所有 major FP32 datapath 盡量 E4M3，已知會早期 RED。
- `GENERIC_E4M3_FRAG_BISECT` + `--frag-group G1..G5/C1..C4`：
- 單組或小組合脆弱性定位模式。
- `GENERIC_E4M3_EXCEPT_G5`：
- 保留 G5（preproc/embedding bundle）高精度，量化 G1~G4；保留 native linear path 與 FinalHead partial(S0)。
- 本輪新增 pairwise mode（保留 native linear + FinalHead S0）：
- `GENERIC_E4M3_G5_G4`
- `GENERIC_E4M3_G5_G1`
- `GENERIC_E4M3_G5_G3`
- `GENERIC_E4M3_G5_G2`

## 4. Existing Evidence Sources
- 本 note 的數據直接來自本機 `build/ref_eval` 真實 summary/txt，非口頭重述。
- 本輪必用來源：
- `build/ref_eval/compare_summary_begin0_count4_except_g5.txt`
- `build/ref_eval/eval_compare_begin0_count4_except_g5.txt`
- `build/ref_eval/compare_summary_begin0_count16_except_g5.txt`
- `build/ref_eval/eval_compare_begin0_count16_except_g5.txt`
- full stress RED 來源（既有檔案）：
- `build/ref_eval/compare_summary_begin0_count4.txt`
- `build/ref_eval/eval_compare_begin0_count4.txt`
- 本輪 pairwise 來源（新增）：
- `build/ref_eval/compare_summary_begin0_count4_g5_g4.txt`
- `build/ref_eval/eval_compare_begin0_count4_g5_g4.txt`
- `build/ref_eval/compare_summary_begin4_count12_g5_g4.txt`
- `build/ref_eval/eval_compare_begin4_count12_g5_g4.txt`
- `build/ref_eval/compare_summary_begin0_count4_g5_g1.txt`
- `build/ref_eval/eval_compare_begin0_count4_g5_g1.txt`
- `build/ref_eval/compare_summary_begin4_count12_g5_g1.txt`
- `build/ref_eval/eval_compare_begin4_count12_g5_g1.txt`
- `build/ref_eval/compare_summary_begin0_count4_g5_g3.txt`
- `build/ref_eval/eval_compare_begin0_count4_g5_g3.txt`
- `build/ref_eval/compare_summary_begin4_count12_g5_g3.txt`
- `build/ref_eval/eval_compare_begin4_count12_g5_g3.txt`
- `build/ref_eval/compare_summary_begin0_count4_g5_g2.txt`
- `build/ref_eval/eval_compare_begin0_count4_g5_g2.txt`
- `build/ref_eval/compare_summary_begin4_count12_g5_g2.txt`
- `build/ref_eval/eval_compare_begin4_count12_g5_g2.txt`
- `build/ref_eval/compare_summary_begin16_count16_g5_g2.txt`
- `build/ref_eval/eval_compare_begin16_count16_g5_g2.txt`
- 本輪使用的是本地 `build/ref_eval` 直接檔案，不是從 `ref_eval.zip` 解壓取用。

## 5. Result Table

### 5.1 Baseline reference rows (full stress vs except_g5)
| mode | begin | count | x_pred flip patterns | sign flip patterns | total flipped bits | delta BER | delta FER | exp BER | exp FER | NaN/Inf (exp) | INT16 ovf | worst min margin | worst logits MSE | worst max abs diff | group roundtrip counters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| FULL_E4M3_NONLINEAR_STRESS | 0 | 4 | 4 | 4 | 43 | 1.706349206e-01 | 1.000000000e+00 | 1.706349206e-01 | 1.000000000e+00 | 0/0 | 0 | 6.250000000e-02 | 1.375847792e+03 | 7.740373230e+01 | roundtrip=811080 (legacy summary, no per-group split) |
| GENERIC_E4M3_EXCEPT_G5 | 0 | 4 | 0 | 0 | 0 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 | 0/0 | 0 | 1.045813084e+01 | 2.299253162e+01 | 2.576817322e+01 | g1/g2/g3/g4/g5=57600/38400/57600/397728/0 |
| GENERIC_E4M3_EXCEPT_G5 | 0 | 16 | 0 | 0 | 0 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 | 0/0 | 0 | 3.022854030e-01 | 2.468346968e+01 | 2.643944740e+01 | g1/g2/g3/g4/g5=230400/153600/230400/1590912/0 |

### 5.2 Pairwise (`G5 x Gi`) rows (this round)
| mode | begin | count | x_pred flip patterns | sign flip patterns | total flipped bits | delta BER | delta FER | exp BER | exp FER | NaN/Inf (exp) | INT16 ovf | worst min margin | worst logits MSE | worst max abs diff | group roundtrip counters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| GENERIC_E4M3_G5_G4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 8.321762085e+00 | 7.513259396e+00 | 8.377410889e+00 | g1/g2/g3/g4/g5=0/0/0/397728/9600 |
| GENERIC_E4M3_G5_G4 | 4 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 1.283529997e-01 | 4.353309963e+01 | 2.278879166e+01 | g1/g2/g3/g4/g5=0/0/0/1193184/28800 |
| GENERIC_E4M3_G5_G1 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 8.651923180e+00 | 2.088236799e+01 | 1.356169701e+01 | g1/g2/g3/g4/g5=57600/0/0/0/9600 |
| GENERIC_E4M3_G5_G1 | 4 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 3.995540738e-02 | 3.757275084e+01 | 1.989690781e+01 | g1/g2/g3/g4/g5=172800/0/0/0/28800 |
| GENERIC_E4M3_G5_G3 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 1.071681023e+01 | 9.903135029e+00 | 1.283599854e+01 | g1/g2/g3/g4/g5=0/0/57600/0/9600 |
| GENERIC_E4M3_G5_G3 | 4 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 1.373367310e-01 | 4.582488066e+01 | 2.281652451e+01 | g1/g2/g3/g4/g5=0/0/172800/0/28800 |
| GENERIC_E4M3_G5_G2 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 8.532430649e+00 | 1.588582504e+01 | 1.341138077e+01 | g1/g2/g3/g4/g5=0/38400/0/0/9600 |
| GENERIC_E4M3_G5_G2 | 4 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 1.466605067e-02 | 5.581344921e+01 | 2.625559616e+01 | g1/g2/g3/g4/g5=0/115200/0/0/28800 |
| GENERIC_E4M3_G5_G2 | 16 | 16 | 2 | 2 | 2 | 0 | 0 | 1.984126984e-03 | 1.250000000e-01 | 0/0 | 0 | 5.882697552e-02 | 9.191372299e+01 | 2.759650803e+01 | g1/g2/g3/g4/g5=0/153600/0/0/38400 |

## 6. Interpretation Boundary
- `GENERIC_E4M3_EXCEPT_G5` 目前支持以下敘述：
- G5 bundle 很可能是 major fragility source 之一。
- 但這**不等於**已證明 SPE alone 是 root cause。
- 目前更像是 interaction failure，不是單點已定罪。
- G5 單獨不是目前已知充分條件。
- 在缺少 G5（`EXCEPT_G5`）情況下，至少在目前 coverage 內，G1~G4 尚不足以造成 decision drift。
- 本輪 pairwise 顯示：
- `G5+G2` 在 `16~31` 出現 baseline-relative x_pred/sign flips（早期 decision-risk signal）。
- 但同窗口 evaluator 的 `delta BER/delta FER = 0`，表示對 trace target 的 frame/bit error 總量尚未惡化，需保守解讀。

## 7. Runtime Policy Update
- quick screen 通過後，後續 coverage 預設採 **disjoint windows**。
- 不再無理由重跑已覆蓋 patterns。
- 同一新模式第一輪擴充先補齊 `4~15`，不直接跳 `16~31`。
- `16~31` 僅在已有資訊顯示值得加測時才做（本輪僅對 `G5+G2` 加測）。

## 8. Next Recommended Action
- 下一輪優先繼續 `G5 × Gi` 精細化（以資訊增益排序）：
1. `G5 + G4`
2. `G5 + G1`
3. `G5 + G3`
4. `G5 + G2`
- 補充：雖然目前 `G5+G2` 在 `16~31` 已出現 baseline-relative flip，仍應維持「不過度宣稱」原則；建議下一輪在 `G5+G2` 內做更細粒度 boundary refine（例如拆 residual-attn vs residual-ffn）。

