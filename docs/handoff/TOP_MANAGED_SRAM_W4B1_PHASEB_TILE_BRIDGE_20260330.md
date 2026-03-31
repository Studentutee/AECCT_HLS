# TOP_MANAGED_SRAM_W4B1_PHASEB_TILE_BRIDGE_20260330

## Executive Summary
- Task: W4-B1 bounded Phase-B tile bridge (single-cut, local-only).
- Primary cut landed on `AttnPhaseBTopManagedSoftmaxOut`:
  - from phase-entry probe-only to one bounded caller-fed/top-fed V-tile bridge.
- Scope remains bounded:
  - no external formal contract change
  - no broad rewrite
  - no inner compute/reduction/writeback loop rewrite
  - not Catapult closure; not SCVerify closure

## Why This Cut
- Wave4 already had M1/M2/M3 phase-entry probe templates.
- This cut proves the next step is feasible:
  - not just probe visibility, but bounded tile payload consume at phase-B entry.

## Landed Anchors
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
  - bridge inputs:
    - `phase_tile_bridge_v_base_word`
    - `phase_tile_bridge_v_words`
    - `phase_tile_bridge_v_words_valid`
    - `phase_tile_bridge_d_tile_idx`
  - bridge observability:
    - `phase_tile_bridge_visible`
    - `phase_tile_bridge_owner_ok`
    - `phase_tile_bridge_consumed`
    - `phase_tile_bridge_compare_ok`
  - bridge compare loop label:
    - `ATTN_P11AF_TILE_BRIDGE_COMPARE_LOOP`
- `src/Top.h`
  - `run_p11af_layer0_top_managed_softmax_out(...)` passthrough now includes W4-B1 bridge fields.
- `scripts/check_top_managed_sram_boundary_regression.ps1`
  - W4-B1 passthrough/signature/anchor guards added.

## Targeted Validation
- `tb/tb_w4b1_phaseb_tile_bridge.cpp`
- `scripts/local/run_p11w4b1_phaseb_tile_bridge.ps1`
- PASS banners:
  - `W4B1_PHASEB_TILE_BRIDGE_VISIBLE PASS`
  - `W4B1_PHASEB_OWNERSHIP_CHECK PASS`
  - `W4B1_PHASEB_NO_SPURIOUS_TOUCH PASS`
  - `W4B1_PHASEB_EXPECTED_COMPARE PASS`
  - `W4B1_PHASEB_BRIDGE_MISMATCH_REJECT PASS`

## Deferred Boundaries
- This cut is one bounded tile bridge only.
- Wave4 full payload migration is still deferred.
- `AttnLayer0` and phase inner compute/writeback main loops remain SRAM-centric in bounded scope.
