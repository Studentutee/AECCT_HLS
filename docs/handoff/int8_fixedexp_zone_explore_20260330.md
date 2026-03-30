# INT8 Fixed-Exponent Zone Exploration Note (2026-03-30)

## 1. Purpose
- 這份 note 是 `AECCT_ac_ref` 的 **ref-only quant exploration** 紀錄。
- 本輪主題：驗證「少量 shared-exponent zones 的 INT8 fixed-exp」在目前最敏感路徑 `G2 + embed_only` 上，是否比 generic E4M3 更穩定。
- 這不是 HLS mainline closure，不是 Catapult closure，也不是 SCVerify closure。

## 2. Evaluator Boundary (must keep)
- `target_x` 來源：`trace_output_x_pred_step0_tensor`（`data/trace/output_x_pred_step0.h`）。
- BER/FER 解讀：**trace-reference-aligned BER/FER**。
- `delta BER = BER_experiment - BER_baseline`。
- 不是 transmitted all-zero GT BER/FER。

## 3. Input Evidence Used for Zone Proposal
- 檔案：`c:\Users\Peter\Downloads\best_int_exp8_mse_trace_step0_100_200_compare.txt`
- 重要觀察：
  - 報告設定是 `BEST_METRIC=MSE`、`ROUNDING=floor`、`OVERFLOW=wrap`（僅作線索，不直接當硬體候選語義）。
  - `same_best_exp_in_all_3_steps=139/155 (0.8968)`，best exponent 穩定度高。
  - 指數分布集中在少數值：`-4/-5/-8/-6/-3/0`。
  - `embed_node_embed`、`embed_plus_SPE` 在 step0/100/200 皆為 `exp=-5`。

## 4. Zone Proposals

### 4.1 Zone-3 proposal (coarse)
- Z1: embed / preproc family，shared exp = `-5`
- Z2: residual / main representation family，shared exp = `-4`
- Z3: softmax/context family，shared exp = `-7`

### 4.2 Zone-4 proposal (coarse)
- Z1: embed / preproc family，shared exp = `-5`
- Z2: residual / main representation family，shared exp = `-4`
- Z3: softmax/context family，shared exp = `-7`
- Z4: output/special scalar family，shared exp = `-2`

### 4.3 Exception policy (not forced into shared-exp INT8 in this round)
- FinalHead special readout scalars (`s_t`, OUT_FC/logits path) 仍維持既有 partial strategy；不把 FinalHead 當普通 linear kernel。
- LN / softmax local scalar states 仍先保守高精度。

## 5. Implementation Assumptions in Ref
- 新增 precision modes：
  - `INT8_FIXEDEXP_ZONE3_EMBED_G2`
  - `INT8_FIXEDEXP_ZONE4_EMBED_G2`
- 量化語義（本輪實作）：
  - storage model: INT8
  - quant formula: `q = round_to_nearest(x / 2^exp)`
  - overflow policy: saturate/clamp to `[-127, 127]`
  - dequant: `x_hat = q * 2^exp`
- 這輪只注入在 `G2 + embed_only` footprint：
  - G2 residual merge path
  - G5 sub-island: `embed_only`
- 其餘不在這輪 footprint 的 path 維持原先設定。

## 6. Commands Run (this round)
```powershell
msbuild AECCT_HLS.vcxproj /p:Configuration=Debug /p:Platform=x64 /m

build\bin\x64\Debug\ref_sim.exe --mode compare --precision-exp generic_e4m3_g2_embed_only --pattern-begin 0 --pattern-count 4 --summary-only
build\bin\x64\Debug\ref_sim.exe --mode eval-compare --precision-exp generic_e4m3_g2_embed_only --pattern-begin 0 --pattern-count 4 --summary-only
build\bin\x64\Debug\ref_sim.exe --mode compare --precision-exp generic_e4m3_g2_embed_only --pattern-begin 4 --pattern-count 12 --summary-only
build\bin\x64\Debug\ref_sim.exe --mode eval-compare --precision-exp generic_e4m3_g2_embed_only --pattern-begin 4 --pattern-count 12 --summary-only

build\bin\x64\Debug\ref_sim.exe --mode compare --precision-exp int8_fixedexp_zone3_embed_g2 --pattern-begin 0 --pattern-count 4 --summary-only
build\bin\x64\Debug\ref_sim.exe --mode eval-compare --precision-exp int8_fixedexp_zone3_embed_g2 --pattern-begin 0 --pattern-count 4 --summary-only
build\bin\x64\Debug\ref_sim.exe --mode compare --precision-exp int8_fixedexp_zone3_embed_g2 --pattern-begin 4 --pattern-count 12 --summary-only
build\bin\x64\Debug\ref_sim.exe --mode eval-compare --precision-exp int8_fixedexp_zone3_embed_g2 --pattern-begin 4 --pattern-count 12 --summary-only
build\bin\x64\Debug\ref_sim.exe --mode compare --precision-exp int8_fixedexp_zone3_embed_g2 --pattern-begin 16 --pattern-count 16 --summary-only
build\bin\x64\Debug\ref_sim.exe --mode eval-compare --precision-exp int8_fixedexp_zone3_embed_g2 --pattern-begin 16 --pattern-count 16 --summary-only

build\bin\x64\Debug\ref_sim.exe --mode compare --precision-exp int8_fixedexp_zone4_embed_g2 --pattern-begin 0 --pattern-count 4 --summary-only
build\bin\x64\Debug\ref_sim.exe --mode eval-compare --precision-exp int8_fixedexp_zone4_embed_g2 --pattern-begin 0 --pattern-count 4 --summary-only
```

## 7. Key Results

### 7.1 `G2 + embed_only`, window 4~15 (main comparison)
| mode | x_pred flip patterns | sign flip patterns | total flipped bits | delta BER | delta FER | experiment BER | experiment FER | worst min margin | worst logits MSE | worst max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generic E4M3 | 1 | 1 | 1 | +1.322751323e-03 | +8.333333333e-02 | 1.322751323e-03 | 8.333333333e-02 | 2.782515287e-01 | 1.010924838e+01 | 1.508080864e+01 |
| INT8 fixed-exp zone3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3.022854030e-01 | 3.361773101e+02 | 6.157894897e+01 |

解讀：
- 在 `4~15`，zone3 在 evaluator（BER/FER）與 decision flips 上優於 E4M3。
- 但 zone3 的 logits drift 指標（MSE/maxabs）更大，表示「decision quality 與 logit distance」不是同一件事。

### 7.2 zone3 extended window 16~31
- compare: `x_pred flip=1`, `sign flip=1`, `total flipped bits=1`
- eval-compare: `delta BER=-9.920634921e-04`, `delta FER=-6.250000000e-02`

解讀：
- 這是 baseline-relative early risk signal（compare flip），但 trace-reference evaluator 反而略優於 baseline。
- 不能過度宣稱 zone3 全面安全；只能說在目前 coverage 下相對 E4M3 更有希望。

### 7.3 zone4 quick screen
- `0~3` 下，zone4 與 zone3數字相同，且 counter 顯示 `zone4_count=0`。
- 代表目前實際注入 footprint 尚未使用到 zone4 額外區域。
- 因此本輪不再擴 zone4 coverage，避免重複成本。

## 8. Counter Evidence (footprint)
- zone3 `4~15`：
  - `int8 fixedexp roundtrip = 136800`
  - `zone z1/z2/z3/z4 = 21600 / 115200 / 0 / 0`
  - `footprint g2/g5_embed = 115200 / 21600`
  - `clamp_count = 0`
- e4m3 `4~15`：
  - `e4m3 roundtrip = 136800`
  - `g1/g2/g3/g4/g5 = 0 / 115200 / 0 / 0 / 21600`

## 9. Interpretation Boundary
- 本輪可以回答：
  - 在目前最敏感的 `G2 + embed_only`，少量 shared-exp zones 的 INT8 fixed-exp（zone3）在 evaluator 上明顯不差於、且優於 E4M3。
  - 需要的 zone 數目前不多，至少在此 footprint 下 2 個 active zones 已可工作。
- 本輪不能回答：
  - 全模型所有群組都適合 shared-exp INT8。
  - `SPE` 單獨已被證明 root cause。
  - 已達到硬體 closure / mainline quant policy freeze。

## 10. Next Recommended Step
- 先保留 zone3 配置，針對 `G2 + embed_only` 做有限擴窗（非 brute-force）與少量 exponent 微調（例如 Z1 -5/-6、Z2 -4/-5）再比較 evaluator。
- 同時維持 LN/softmax scalar states 高精度，不在這輪先做演算法切換。

## 11. Governance Posture
- ref-only exploration
- local-only evidence
- not HLS mainline closure
- not Catapult closure
- not SCVerify closure
- trace-reference-aligned BER/FER interpretation
