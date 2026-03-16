# PREF-001 Report

Date: 2026-03-10  
Patch ID: PREF-001  
Scope mode: algorithm-ref alignment (synth-first, minimal-scope)

## Files changed

- `AECCT_ac_ref/synth/RefStep0Synth.cpp`
- `AECCT_ac_ref/src/RefModel.cpp`

## What was aligned in PREF-001

1. Synth-path softmax alignment in `RefStep0Synth.cpp`:
- Replaced two-pass attention softmax accumulation with single-pass online state update in both attention regions (`run_layer_writeback`, `run_final_layer_pass_a`).
- Kept LUT source unchanged: `ref_softmax_exp_lut`, `ref_softmax_rcp_lut`.
- Removed ad-hoc direct division from the datapath; normalization remains LUT reciprocal based.

2. FinalHead semantic guardrails kept:
- No `PAGE_NEXT`/`TEMP_PAGE` staging was introduced for normal FinalHead readout.
- `final_scalar_buf` remains the staging of logical `s_t` semantics.
- Direct streaming `logits` / `x_pred` behavior remains unchanged.

3. Native linear quant boundary alignment (v1 minimal):
- Added symmetric INT8 saturation helper (`[-127, +127]`) and applied only at activation quant entry boundaries of native linear paths.
- Applied in synth quant-linear entry path and ref-model quant-linear helper entry paths.
- No quantization extension into LayerNorm, softmax, residual, or FinalHead.

4. Naming/trace alignment kept conservative:
- Added lightweight logical-name comments (`endLN_out`, `s_t`) without broad trace schema churn.

## Intentionally deferred / known limitations

1. RefModel softmax is not declared fully end-to-end single-pass aligned:
- Main attention accumulation now follows online-softmax state update.
- A trace-only probability materialization loop is still retained in `RefModel` for existing trace usefulness.
- This is an intentional minimal-scope compromise and remains a mismatch versus strict v12.1 online-softmax direction.

2. `SoftmaxApprox.h` was not broadly refactored:
- No framework-level refactor was done.
- Existing LUT helper interfaces were sufficient for this patch.

3. This patch does not attempt Top/contract bring-up:
- No 4-channel Top/FSM/CFG/PARAM/INFER contract work was added.
- No opcode/bitfield/memory-model changes were made.

## Patch-local verification

Commands executed:

1. `msbuild AECCT_HLS.vcxproj /p:Configuration=Debug /p:Platform=x64 /m`
2. `python tools/check_ref_approx_rules.py`
3. `.\build\bin\x64\Debug\ref_sim.exe 0`

Observed results:

- Build: PASS
- Approx rule check: PASS
- `ref_sim.exe 0`: PASS (exit code 0)
- Runtime metrics snapshot:
  - `MSE = 7.073902e-01`
  - `RMSE = 8.410649e-01`
  - `MAE = 5.954486e-01`
  - `MaxAbs = 2.549500e+00`
  - `x_pred match = 100.00% (63/63)`

Artifacts:

- `docs/milestones/PREF-001_artifacts/diff.patch`
- `docs/milestones/PREF-001_artifacts/build.log`
- `docs/milestones/PREF-001_artifacts/run_tb.log`
- `docs/milestones/PREF-001_artifacts/verdict.txt`
- `docs/milestones/PREF-001_artifacts/file_manifest.txt`

## Environment limitations (pre-existing repo-wide gate/hygiene issues)

From `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gates.ps1`:

- `check_design_purity`: PASS
- `check_interface_lock`: PASS
- `check_repo_hygiene`: FAIL
  - `.gitignore missing reports/`
  - `utf8-bom found: .vs/AECCT_HLS.slnx/v18/HierarchyCache.v1.txt`
  - `archive in repo: AECCT_ac_ref/AECCT_ac_ref.zip`
  - `utf8-bom found: AECCT_ac_ref/include/RefModel.h`
  - `utf8-bom found: AECCT_ac_ref/include/SoftmaxApprox.h`
  - `utf8-bom found: AECCT_ac_ref/src/RefModel.cpp`
  - `utf8-bom found: tools/check_ref_approx_rules.py`
  - `utf8-bom found: tools/compare_checkpoints.py`
  - `utf8-bom found: tools/gen_ref_lut.py`
  - `utf8-bom found: tools/run_algorithm_ref_step0.py`

## Verdict

PREF-001 is accepted at patch scope: synth-first algorithm-ref alignment objectives were met with minimal, auditable diffs.  
This verdict is patch-local only and is not a repo-wide full-gates pass.
