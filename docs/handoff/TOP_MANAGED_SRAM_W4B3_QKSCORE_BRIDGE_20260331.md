# TOP_MANAGED_SRAM_W4B3_QKSCORE_BRIDGE_20260331

## Executive Summary
- Task: W4-B3 bounded Phase-B QkScore bridge (single-cut, local-only).
- Primary cut landed on `AttnPhaseBTopManagedQkScore`:
  - extends W4-B2 from head0-fixed bridge to selectable-head bounded bridge.
- This run validated non-head0 + secondary key-range consume.
- Scope remains bounded:
  - no external formal contract change
  - no broad rewrite
  - no inner compute/reduction/writeback loop rewrite
  - not Catapult closure; not SCVerify closure

## Why This Cut
- W4-B2 already proved one bounded score bridge on constrained head0 range.
- W4-B3 proves the bridge pattern scales one step further:
  - caller/Top can dispatch bounded score tile for selectable head, not only head0.

## Landed Anchors
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
  - bridge selector:
    - `score_tile_bridge_head_idx`
    - `score_tile_bridge_head_idx_u32`
  - bounded select remains at:
    - `ATTN_P11AE_KEY_TOKEN_LOOP`
  - observability remains:
    - `score_tile_bridge_visible`
    - `score_tile_bridge_owner_ok`
    - `score_tile_bridge_consumed`
    - `score_tile_bridge_compare_ok`
- `src/Top.h`
  - `run_p11ae_layer0_top_managed_qk_score(...)` passthrough now includes bridge head selector.
- `scripts/check_top_managed_sram_boundary_regression.ps1`
  - W4-B3 selectable-head passthrough/signature/anchor guards added.

## Targeted Validation
- `tb/tb_w4b3_qkscore_bridge.cpp`
- `scripts/local/run_p11w4b3_qkscore_bridge.ps1`
- PASS banners:
  - `W4B3_QKSCORE_BRIDGE_VISIBLE PASS`
  - `W4B3_QKSCORE_OWNERSHIP_CHECK PASS`
  - `W4B3_QKSCORE_NON_HEAD1_PATH PASS`
  - `W4B3_QKSCORE_NO_SPURIOUS_TOUCH PASS`
  - `W4B3_QKSCORE_EXPECTED_COMPARE PASS`
  - `W4B3_QKSCORE_BRIDGE_MISMATCH_REJECT PASS`

## Deferred Boundaries
- This cut is still one bounded score-tile bridge window.
- Wave4 full payload migration is still deferred.
- QkScore inner compute/reduction/writeback loops remain SRAM-centric in bounded scope.
