# Ref Quant Fragility Explore Note (2026-03-29, updated 2026-03-30)

## 1. Purpose / Scope
- 此文件是 `AECCT_ac_ref` 的 ref-only fragility exploration supporting note。
- 不是 HLS mainline closure，不是 Catapult closure，不是 SCVerify closure，不是 milestone acceptance。

## 2. Evaluator Interpretation Boundary
- `target_x = trace_output_x_pred_step0_tensor`（`data/trace/output_x_pred_step0.h`）。
- BER/FER 為 trace-reference-aligned 指標。
- `delta BER = BER_experiment - BER_baseline`。
- `delta FER = FER_experiment - FER_baseline`。
- 不是 transmitted all-zero ground-truth BER/FER。

## 3. Mode Definitions (high level)
- `FULL_E4M3_NONLINEAR_STRESS`：除了 native linear quant 主集合外，幾乎所有 major FP32 datapath 壓到 generic E4M3。
- `GENERIC_E4M3_EXCEPT_G5`：G1~G4 注入 generic E4M3，G5 凍結高精度。
- Pairwise：`GENERIC_E4M3_G5_G4/G1/G3/G2`。
- `G5_sub × G2`：
  - `GENERIC_E4M3_G2_EMBED_ONLY`
  - `GENERIC_E4M3_G2_SPE_ONLY`
  - `GENERIC_E4M3_G2_PREPROC_ASSEMBLY`
  - `GENERIC_E4M3_G2_PRELAYER_HANDOFF`

## 4. Evidence Sources
- 主要依據 `build/ref_eval/*.txt`：
  - `compare_summary_begin0_count4.txt`
  - `eval_compare_begin0_count4.txt`
  - `compare_summary_begin0_count4_except_g5.txt`
  - `eval_compare_begin0_count4_except_g5.txt`
  - `compare_summary_begin0_count16_except_g5.txt`
  - `eval_compare_begin0_count16_except_g5.txt`
  - `compare_summary_begin16_count16_g5_g2.txt`
  - `eval_compare_begin16_count16_g5_g2.txt`
  - `compare_summary_begin4_count12_g2_embed_only.txt`
  - `eval_compare_begin4_count12_g2_embed_only.txt`

## 5. Core Results (condensed)

### 5.1 Full stress vs except_g5
- `FULL_E4M3_NONLINEAR_STRESS` 在 quick screen (`0~3`) 即 RED：
  - x_pred/sign flips 出現
  - delta BER > 0，delta FER > 0
  - 無 NaN/Inf，無 INT16 overflow
- `GENERIC_E4M3_EXCEPT_G5` 在 `0~3` 與 `0~15` 未 RED。

### 5.2 G5_sub × G2
- `G2 + embed_only` 在 `4~15` 已 RED：
  - compare 出現 x_pred/sign flip
  - eval-compare 出現 `delta BER=+1.322751323e-03`, `delta FER=+8.333333333e-02`
- `G2 + spe_only` 與 `G2 + preproc_assembly` 在較後窗口可見 baseline-relative flip，
  但 evaluator 不一定惡化（可能維持或改善）。

### 5.3 Interpretation
- 可支持：`G5 bundle` 是主要 fragility source 之一，且 `embed_only × G2` 目前最危險。
- 不能支持：`SPE` 單獨已被定罪為 root cause。
- 較符合 interaction failure：多群組交互造成 decision drift，而非單點充分條件。

## 6. Runtime Policy
- 新模式先跑 quick `0~3`。
- quick 通過後先補 `4~15`（`begin=4,count=12`），避免重跑 `0~3`。
- 只對最有資訊增益的模式擴到 `16~31`，不做無差別 brute-force。

## 7. Follow-up Link (2026-03-30)
- 已新增 `docs/handoff/int8_fixedexp_zone_explore_20260330.md`。
- 在目前最敏感路徑 `G2 + embed_only`，INT8 shared-exp 候選已完成與 E4M3 正面對照。

## 8. Governance Posture
- ref-only exploration
- local-only evidence
- not HLS mainline closure
- not Catapult closure
- not SCVerify closure
- trace-reference-aligned BER/FER interpretation

## 9. FP16 Global Follow-up (2026-03-30)
- 新模式：`FP16_REPLACE_FP32_GLOBAL`（native linear quant 主集合不變，原 FP32 islands 走 FP16 roundtrip）。
- 視窗結果：
  - `0~3`：delta BER/FER = 0，x_pred/sign flips = 0
  - `4~15`：delta BER/FER = 0，x_pred/sign flips = 0
  - `16~31`：delta BER/FER = 0，x_pred/sign flips = 0
- 解讀：
  - FP16 在目前 coverage 下可作為高敏感例外層候選。
  - 仍不可直接外推為全模型最終 policy closure。
