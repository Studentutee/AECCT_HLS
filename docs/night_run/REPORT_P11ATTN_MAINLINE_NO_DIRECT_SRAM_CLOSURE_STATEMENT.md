# REPORT_P11ATTN_MAINLINE_NO_DIRECT_SRAM_CLOSURE_STATEMENT

Date: 2026-04-05  
Scope: attention mainline closure statement (local-only)

## What is closed in this round
- Closure target: attention mainline no-direct-SRAM-fallback posture.
- This is a local closure statement for current mainline scope.
- This is not Catapult closure.
- This is not SCVerify closure.

## Evidence that was actually run
- `scripts/check_design_purity.ps1` -> `PASS: check_design_purity`
- `scripts/check_repo_hygiene.ps1 -Phase pre` -> `PASS: check_repo_hygiene`
- `scripts/local/run_p11aj_top_managed_sram_provenance.ps1` -> `PASS: run_p11aj_top_managed_sram_provenance`
- `scripts/local/run_p11anb_attnlayer0_boundary_seam_contract.ps1` -> `PASS: run_p11anb_attnlayer0_boundary_seam_contract`
- `build/p11aj/p11aj/run.log` includes:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `fallback_taken = false`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `FULLY_PREBUILT_NO_PAYLOAD_DISABLED PASS`
  - `FULLY_PREBUILT_PAYLOAD_OUT_ONLY PASS`
  - `OTHER_PARTIAL_BUCKETS_REMAIN_FULL PASS`
- `build/p11anb/attnlayer0_boundary_seam_contract/run.log` includes:
  - `P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_FULLY_PREBUILT_OUT_ONLY PASS`
  - `P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_FULLY_PREBUILT_NO_PAYLOAD_DISABLED PASS`
  - `P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_OTHER_PARTIAL_STILL_FULL PASS`

## Code-path reasoning (inference from source)
- `Top.h` only enters AE/AF mainline when `q_prebuilt_from_top_managed && kv_prebuilt_from_top_managed`.
- `score_prebuilt_from_top_managed` is assigned from `ae_mainline_score_path_taken`.
- `out_prebuilt_from_top_managed` is assigned from `af_mainline_softmax_output_path_taken`.
- Therefore current mainline implies:
  - `out_prebuilt=true` => `score_prebuilt=true`
  - `out_prebuilt=true` => `q_prebuilt=true && kv_prebuilt=true`

## Why remaining out=1 buckets are not a mainline gap
Bit order: `kv, q, score, out, payload`

- Class 2 (reachable as selector-input space, but fallback/safety-net in current runloop):
  - `10110`, `10111`, `01110`, `01111`, `00110`, `00111`
- Class 3 (likely unreachable for current mainline due to AE/AF derivation constraints):
  - `11010`, `11011`, `10010`, `10011`, `01010`, `01011`, `00010`, `00011`

These residual FULL buckets are intentionally treated as defensive/fallback surface, not evidence of an open mainline path.

## What evidence is required to re-open out=1 shrink
- A new compile-backed run must show at least one current `out=1` unresolved bucket is produced by the real Top-managed mainline path (not just selector-input reachability).
- The run must preserve:
  - Top-only shared-SRAM ownership semantics
  - unchanged 4-channel external contract
  - no FFN/LayerNorm scope expansion
- The same run should include corresponding p11aj/p11anb acceptance lines proving no regression on already-converged buckets.

## Posture
- local-only
- compile-first
- evidence-first
- not Catapult closure
- not SCVerify closure
