# TOP_MANAGED_SRAM_W4B2_QKSCORE_TILE_BRIDGE_20260331

## Executive Summary
- Task: W4-B2 bounded Phase-B QkScore score-tile bridge (single-cut, local-only).
- Primary cut landed on `AttnPhaseBTopManagedQkScore`:
  - from phase-entry probe-only to one bounded caller-fed/top-fed score-tile bridge.
- Scope remains bounded:
  - no external formal contract change
  - no broad rewrite
  - no inner compute/reduction/writeback loop rewrite
  - not Catapult closure; not SCVerify closure

## Why This Cut
- W4-M1 proved QkScore phase-entry descriptor probe visibility.
- W4-B2 proves a next-step bridge is feasible:
  - not only probe visibility, but bounded score-tile payload consume at token-write boundary.

## Landed Anchors
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
  - bridge inputs:
    - `score_tile_bridge_base_word`
    - `score_tile_bridge_words`
    - `score_tile_bridge_words_valid`
    - `score_tile_bridge_key_begin`
  - bridge observability:
    - `score_tile_bridge_visible`
    - `score_tile_bridge_owner_ok`
    - `score_tile_bridge_consumed`
    - `score_tile_bridge_compare_ok`
  - bridge consume gate:
    - bounded select at `ATTN_P11AE_KEY_TOKEN_LOOP`
- `src/Top.h`
  - `run_p11ae_layer0_top_managed_qk_score(...)` passthrough now includes W4-B2 bridge fields.
- `scripts/check_top_managed_sram_boundary_regression.ps1`
  - W4-B2 passthrough/signature/anchor guards added.

## Targeted Validation
- `tb/tb_w4b2_qkscore_tile_bridge.cpp`
- `scripts/local/run_p11w4b2_qkscore_tile_bridge.ps1`
- PASS banners:
  - `W4B2_QKSCORE_TILE_BRIDGE_VISIBLE PASS`
  - `W4B2_QKSCORE_OWNERSHIP_CHECK PASS`
  - `W4B2_QKSCORE_NO_SPURIOUS_TOUCH PASS`
  - `W4B2_QKSCORE_EXPECTED_COMPARE PASS`
  - `W4B2_QKSCORE_BRIDGE_MISMATCH_REJECT PASS`

## Deferred Boundaries
- This cut is one bounded score-tile bridge only.
- Wave4 full payload migration is still deferred.
- `AttnLayer0` and phase inner compute/writeback loops remain SRAM-centric in bounded scope.
