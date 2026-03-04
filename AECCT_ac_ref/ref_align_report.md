# Ref Align Report (step0)

## Scope
- Model reference source: `algorithm_ref.ipynb` (step0 flow)
- Edited area: `AECCT_ac_ref/`, `tools/`, `AECCT_HLS.vcxproj`
- Design RTL (`design/`) untouched

## Root Cause (First Divergence)
Pre-fix (old C++ ref behavior), the first major divergence starts at `layer0_q`.

- First diverge checkpoint: `layer0_q`
- Symptom (from pre-fix reproduction against trace):
  - `layer0_q` maxabs ~= `1.2109e+01`
  - `layer0_k` maxabs ~= `8.8599e+00`
  - `layer0_v` maxabs ~= `1.4753e+01`
- Root cause:
  1. Q/K/V used plain `x @ W^T + b`, but notebook uses quant-dequant linear:
     - `q = round(x * s_x)`
     - `out = q @ (W^T / (s_x * s_w)) + b`

After fixing Q/K/V, the next large divergence was in attention probability.

- Diverge checkpoint: `layer0_attn_probs`
- Root cause:
  1. Mask/head assumption mismatch:
     - Notebook uses `one_ring` for head `0..3`, `second_ring` for head `4..7`
     - `masked_fill(mask, -inf)` where mask `True` means blocked
  2. Old C++ added `lpe/lpe_proj` bias to score, but notebook step0 path does not.

Other high-impact mismatches fixed:
- Missing `decoder.norm2` between layer0 and layer1
- `x_pred` decision rule mismatch:
  - fixed to `logits[n] * sign(input_y[n]) < 0`
- FP32 bring-up input path:
  - added `input_y_fp32` in `RefModelIO` to avoid unnecessary act quantization during correctness stage

## Implemented Fixes
1. `RefModel.cpp`
- Full FP32 internal path
- Preproc = `|y| + syndrome_pm1` + `lpe_token`
- Quant-dequant linear for Q/K/V, Wo, FFN w1/w2
- Attention mask/head routing aligned to notebook
- No LPE score-bias injection in attention score
- Layer order aligned: `layer0 -> norm2 -> layer1 -> norm`
- Added `.npy` checkpoint dump (single-pattern run)

2. `RefModel.h` / `ref_main.cpp`
- Added optional `input_y_fp32` in `RefModelIO`
- Added dump config (`RefDumpConfig`) and single-pattern dump enable
- Kept CLI: `ref_sim.exe [pattern_index]`

3. Build
- `AECCT_HLS.vcxproj` post-build copies `AECCT_HLS.exe` to `ref_sim.exe`

4. Python tools
- `tools/run_algorithm_ref_step0.py`
  - Executes notebook-equivalent step0 path
  - Verifies `.pt` vs `data/trace/*.h` consistency
  - Dumps required checkpoints to `.npy`
- `tools/compare_checkpoints.py`
  - Ordered checkpoint compare, reports first diverge with maxabs/mae/rmse

## Validation Results
### MSVC build
- Command:
  - `msbuild AECCT_HLS.vcxproj /p:Configuration=Debug /p:Platform=x64 /m`
- Result: `0 error`

### C++ run (`ref_sim.exe 0`)
- MSE: `1.831116e-10`
- RMSE: `1.353187e-05`
- MAE: `8.749583e-06`
- MaxAbs: `3.814697e-05`
- x_pred match: `100.00% (63/63)`

- logits[0:8] (golden vs ref)
  - golden: `[24.814947, 25.889328, 20.760557, 24.449009, 22.794401, 46.781998, 22.262341, 23.891241]`
  - ref   : `[24.814939, 25.889328, 20.760553, 24.449009, 22.794405, 46.781990, 22.262341, 23.891243]`

### Python vs C++ checkpoint compare
- Command:
  - `python tools/compare_checkpoints.py --python-dir logs/ref_py/pattern_0 --cpp-dir logs/ref_cpp/pattern_0 --threshold 1e-4`
- Result:
  - `first diverge: none (all <= 1e-4)`
  - `final_logits maxabs: 3.814697e-05`

## Output Paths
- C++ dumps: `logs/ref_cpp/pattern_0/*.npy`
- Python dumps: `logs/ref_py/pattern_0/*.npy`
## v11.9 ApproxMath Update (2026-03-04)

### Invariants kept (no algorithm drift)
- mask/head routing unchanged (`h0~h3 -> one_ring`, `h4~h7 -> second_ring`)
- no LPE score-bias injection
- layer order unchanged (`layer0 -> norm2 -> layer1 -> norm`)
- quant-dequant linear path unchanged (`round(x*s_x)` with `W^T/(s_x*s_w)`)
- x_pred rule unchanged (`logit * sign(y) < 0`)

### Numeric path changes
- Core tensors use `ac_ieee_float<binary32>` (`fp32_ref_t`)
- Softmax replaced by LUT exp + LUT reciprocal (+ 1-step Newton refinement)
- LayerNorm denominator replaced by `inv_sqrt` LUT approximation (no `std::sqrt`, no `/` in LN/softmax/activation blocks)

### Build / run
- Build: `msbuild AECCT_HLS.vcxproj /p:Configuration=Debug /p:Platform=x64 /m`
  - compile/link: success
  - known intermittent post-build copy lock on `ref_sim.exe` (manual copy workaround used)
- Run: `build/bin/x64/Debug/ref_sim.exe 0`
  - MSE: `6.545970e-01`
  - RMSE: `8.090717e-01`
  - MAE: `5.842421e-01`
  - MaxAbs: `2.854652e+00`
  - x_pred match: `100.00% (63/63)`

### Checkpoint divergence summary (python exact vs C++ approx)
- First over-threshold tensor (`1e-4`): `layer0_ln_in` (already includes softmax approximation effect)
- Largest attention-prob gap in layer0: `layer0_attn_probs maxabs ~= 1.556e-02`
- Final logits gap: `final_logits maxabs ~= 2.855`, `mae ~= 5.842e-01`

### Error attribution
Primary error source is approximation (Softmax + LayerNorm denominator approximation), not algorithm mismatch.
The previous algorithm-drift failure mode (`layer0_ln_out` exploding to `~1e5`) was fixed; current checkpoints remain finite and follow the same computational topology as notebook step0.
