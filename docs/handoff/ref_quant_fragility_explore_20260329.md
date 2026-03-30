# AECCT_ac_ref Ref-Only Quant Fragility Explore Note (2026-03-29, updated 2026-03-30)

## 1. Purpose / Scope
- 這份文件是 `AECCT_ac_ref` 的 **ref-only quant fragility exploration** 摘要紀錄。
- 定位是 task-local supporting note，不是正式里程碑 closure。
- 明確不是：
  - HLS mainline closure
  - Catapult closure
  - SCVerify closure
  - milestone acceptance

## 2. Evaluator Interpretation Boundary
- `target_x` 來源是 `trace_output_x_pred_step0_tensor`（`data/trace/output_x_pred_step0.h`）。
- BER/FER 為 **trace-reference-aligned BER/FER**。
- `delta BER = BER_experiment - BER_baseline`。
- `delta FER = FER_experiment - FER_baseline`。
- 這不是 transmitted all-zero ground-truth BER/FER。

## 3. Experiment Mode Definitions
- 正式 v12.1 baseline（不在此 note 內改寫）：
  - native linear quant 主集合：`Wq/Wk/Wv/Wo/Wff1/Wff2`
  - 其他 major datapath 原生為 FP32 語意
- FinalHead 必須獨立看待，不等於普通 linear kernel：
  - Pass A: `FinalEmbedding -> s_t`
  - Pass B: `OUT_FC` 對 `s_t` 作 token 維度 reduction，輸出 `logits/x_pred`
- 主要實驗模式：
  - `FULL_E4M3_NONLINEAR_STRESS`
  - `GENERIC_E4M3_EXCEPT_G5`
  - `GENERIC_E4M3_G5_G4 / G5_G1 / G5_G3 / G5_G2`
  - 本輪新增 `G5_sub × G2`：
    - `GENERIC_E4M3_G2_EMBED_ONLY`
    - `GENERIC_E4M3_G2_SPE_ONLY`
    - `GENERIC_E4M3_G2_PREPROC_ASSEMBLY`
    - `GENERIC_E4M3_G2_PRELAYER_HANDOFF`

## 4. Evidence Sources
- 本 note 所有數字直接來自 `build/ref_eval/*.txt` summary 檔，不採口頭重述。
- 錨點 evidence：
  - `build/ref_eval/compare_summary_begin0_count4.txt`
  - `build/ref_eval/eval_compare_begin0_count4.txt`
  - `build/ref_eval/compare_summary_begin0_count4_except_g5.txt`
  - `build/ref_eval/eval_compare_begin0_count4_except_g5.txt`
  - `build/ref_eval/compare_summary_begin0_count16_except_g5.txt`
  - `build/ref_eval/eval_compare_begin0_count16_except_g5.txt`
  - `build/ref_eval/compare_summary_begin16_count16_g5_g2.txt`
  - `build/ref_eval/eval_compare_begin16_count16_g5_g2.txt`
- 本輪 `G5_sub × G2` evidence：
  - `compare/eval_compare_begin0_count4_g2_*.txt`
  - `compare/eval_compare_begin4_count12_g2_*.txt`
  - `compare/eval_compare_begin16_count16_g2_preproc_assembly.txt`
  - `compare/eval_compare_begin16_count16_g2_spe_only.txt`

## 5. Result Tables

### 5.1 Anchor rows (full stress / except_g5 / pairwise anchor)
| mode | begin | count | x_pred flip patterns | sign flip patterns | total flipped bits | delta BER | delta FER | exp BER | exp FER | exp NaN/Inf | INT16 ovf | worst min margin | worst logits MSE | worst max abs diff | roundtrip footprint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| FULL_E4M3_NONLINEAR_STRESS | 0 | 4 | 4 | 4 | 43 | +1.706349206e-01 | +1.000000000e+00 | 1.706349206e-01 | 1.000000000e+00 | 0/0 | 0 | 6.250000000e-02 | 1.375847792e+03 | 7.740373230e+01 | roundtrip=811080 |
| GENERIC_E4M3_EXCEPT_G5 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 1.045813084e+01 | 2.299253162e+01 | 2.576817322e+01 | g1/g2/g3/g4/g5=57600/38400/57600/397728/0 |
| GENERIC_E4M3_EXCEPT_G5 | 0 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 3.022854030e-01 | 2.468346968e+01 | 2.643944740e+01 | g1/g2/g3/g4/g5=230400/153600/230400/1590912/0 |
| GENERIC_E4M3_G5_G2 | 16 | 16 | 2 | 2 | 2 | 0 | 0 | 1.984126984e-03 | 1.250000000e-01 | 0/0 | 0 | 5.882697552e-02 | 9.191372299e+01 | 2.759650803e+01 | g1/g2/g3/g4/g5=0/153600/0/0/38400 |

### 5.2 This round: `G5_sub × G2` rows
| mode | begin | count | x_pred flip patterns | sign flip patterns | total flipped bits | delta BER | delta FER | exp BER | exp FER | exp NaN/Inf | INT16 ovf | worst min margin | worst logits MSE | worst max abs diff | roundtrip g1/g2/g3/g4/g5 | g5 sub (embed/spe/assembly/prelayer) | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|
| G2 + embed_only | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 1.071681023e+01 | 8.327341248e+00 | 6.923957825e+00 | 0/38400/0/0/7200 | 7200/0/0/0 | quick pass |
| G2 + embed_only | 4 | 12 | 1 | 1 | 1 | +1.322751323e-03 | +8.333333333e-02 | 1.322751323e-03 | 8.333333333e-02 | 0/0 | 0 | 2.782515287e-01 | 1.010924838e+01 | 1.508080864e+01 | 0/115200/0/0/21600 | 21600/0/0/0 | **RED (stop)** |
| G2 + spe_only | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 9.419803619e+00 | 9.679619364e+00 | 1.302730560e+01 | 0/38400/0/0/2400 | 0/2400/0/0 | quick pass |
| G2 + spe_only | 4 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 2.546133399e-01 | 7.497227491e+01 | 2.969159698e+01 | 0/115200/0/0/7200 | 0/7200/0/0 | pass (0~15) |
| G2 + spe_only | 16 | 16 | 1 | 1 | 1 | -9.920634921e-04 | -6.250000000e-02 | 9.920634921e-04 | 6.250000000e-02 | 0/0 | 0 | 5.882697552e-02 | 7.915161571e+01 | 2.183360100e+01 | 0/153600/0/0/9600 | 0/9600/0/0 | compare risk, eval non-degrade |
| G2 + preproc_assembly | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 8.532430649e+00 | 1.588582504e+01 | 1.341138077e+01 | 0/38400/0/0/9600 | 0/0/9600/0 | quick pass |
| G2 + preproc_assembly | 4 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 1.466605067e-02 | 5.581344921e+01 | 2.625559616e+01 | 0/115200/0/0/28800 | 0/0/28800/0 | pass (0~15, near-boundary) |
| G2 + preproc_assembly | 16 | 16 | 2 | 2 | 2 | 0 | 0 | 1.984126984e-03 | 1.250000000e-01 | 0/0 | 0 | 5.882697552e-02 | 9.191372299e+01 | 2.759650803e+01 | 0/153600/0/0/38400 | 0/0/38400/0 | compare risk, eval parity |
| G2 + prelayer_handoff | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 8.532430649e+00 | 1.588582504e+01 | 1.341138077e+01 | 0/38400/0/0/9600 | 0/0/0/9600 | quick pass |
| G2 + prelayer_handoff | 4 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 1.466605067e-02 | 5.581344921e+01 | 2.625559616e+01 | 0/115200/0/0/28800 | 0/0/0/28800 | pass (0~15) |

## 6. Interpretation Boundary
- 這輪最重要新訊號：
  - `G2 + embed_only` 在 `4~15` 已出現 **compare flip + evaluator delta BER/FER > 0**，屬早期明確 RED。
  - `G2 + preproc_assembly` 與 `G2 + spe_only` 在 `16~31` 出現 baseline-relative flip，但 evaluator 沒有 BER/FER 惡化（`preproc_assembly` 為 delta 0，`spe_only` 為負 delta）。
- 因此目前可以說：
  - `G5` 不是單一均質區塊，`embed_only × G2` 的交互最值得優先追。
  - 目前證據支持 **interaction failure**，不是單點 root cause 定罪。
- 目前不能說：
  - 已證明 `SPE` 單獨是主犯。
  - 已證明只要保住單一 G5_sub 就可全域安全。
- compare flip 與 trace-reference BER/FER delta 不是同一概念：
  - compare flip：baseline-relative decision-risk signal。
  - BER/FER delta：相對 trace target 的 error-correction 指標變化。

## 7. Runtime Policy Update
- 新模式一律先跑 quick `0~3`。
- quick 通過才補 `4~15`（`begin=4,count=12`），不重跑 `0~3`。
- `16~31` 只對最脆弱的 1~2 模式加測，不做全面 brute-force 擴張。
- 本輪 `16~31` 選擇了：
  - `G2 + preproc_assembly`
  - `G2 + spe_only`
  - `G2 + embed_only` 因 `4~15` 已 RED，依規則停止擴張。

## 8. Next Recommended Action
- 下一輪優先建議：在 `G2 + embed_only` 上做更細 attribution（例如 residual-attn vs residual-ffn、embed path 內更細 boundary）。
- 不建議直接切換 LN 演算法，先維持目前證據鏈可比性。
- 保留這份結論為 ref-only 探索，不升格成 spec closure。
