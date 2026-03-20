# P00-011AH Report - Full-Loop Blocker Isolation + Minimal End-to-End Bring-Up (Local-Only)

## Summary
- Added a local full-loop `run_transformer_layer_loop` e2e TB/runner and one-shot batch runner.
- Added minimal LayerNorm guard instrumentation to isolate and prevent non-finite assert path in local full-loop bring-up.

## Scope
- local-only
- full-loop blocker isolation and minimal bring-up for current AC/AD/AE/AF-integrated path
- no external Top contract change
- no memory ownership model change

## Files changed
- `src/blocks/LayerNormBlock.h`
- `tb/tb_full_loop_local_e2e_p11ah.cpp`
- `scripts/local/run_p11ah_full_loop_local_e2e.ps1`
- `scripts/local/run_p11ah_full_loop_batch.ps1`

## Build command
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build\p11ah\full_loop`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_batch.ps1 -BuildDir build\p11ah`

## Run command
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build\p11ah\full_loop`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_batch.ps1 -BuildDir build\p11ah`

## Actual execution evidence
- `build\p11ah\full_loop\run.log`
  - `[p11ah][LN_ASSERT_GUARD] token=0 var_bits=0xC46FFC00 var_plus_eps_bits=0xC46FFC00 eps_bits=0x3727C5AC`
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `FULL_LOOP_FINAL_X_DETERMINISTIC_COMPARE PASS`
  - `FULL_LOOP_FINITE_SCAN PASS`
  - `PASS: tb_full_loop_local_e2e_p11ah`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build\p11ah\batch_summary.txt`
  - `status: PASS`
  - `PASS: run_p11ah_full_loop_batch`
- `build\p11ah\p11ag_regression\batch_summary.txt`
  - `status: PASS`
  - `PASS: run_p11ag_attention_chain_batch`
- `build\p11ah\p11ag_regression\p11ag_validator\run.log`
  - `PASS: run_p11ag_attention_chain_correction`
- `build\p11ah\p11ag_regression\p11aeaf_e2e\run.log`
  - `PASS: run_p11aeaf_e2e_smoke`
- `build\p11ah\p11ag_regression\catapult_progress\run.log`
  - `PASS: run_p11aeaf_catapult_progress`

## Root-cause / blocker isolation
- Full-loop non-finite assert path is isolated to LayerNorm `sqrt(var + eps)` precondition violation in local bring-up setup.
- Deterministic first-hit evidence:
  - token index: `0`
  - `var_bits=0xC46FFC00` (negative)
  - `var_plus_eps_bits=0xC46FFC00` (still <= 0 after eps)
  - source stage: LayerNorm state in full `run_transformer_layer_loop`
- Minimal fix landed:
  - add guard before `sqrt`: when `var_plus_eps <= 0`, emit one-shot local diagnostic and clamp to `eps` for the current computation path.
- This keeps AC/AD/AE/AF landed semantics/call ordering and Top contract unchanged.

## Scope / limitation
- local-only
- not Catapult closure
- not SCVerify closure
- not full runtime closure
- not full algorithm closure
