# TOP_MANAGED_SRAM_W4_QKSCORE_B5_FAMILY_BRIDGE_20260331

## Executive Summary
- Task: W4-B5 bounded QkScore family generalization (local-only).
- Primary cut landed on `AttnPhaseBTopManagedQkScore`:
  - extends single selected score-tile bridge into bounded multi-case family bridge (2~3 selected head/window cases per invocation).
- This run validates family-level ownership, anti-fallback, and mismatch-reject without broad rewrite.
- Scope remains bounded:
  - no external formal contract change
  - no broad rewrite
  - no compute/reduction/writeback major-loop rewrite
  - not Catapult closure; not SCVerify closure

## Why This Cut
- W4-B3 proved one selectable-head bounded bridge case.
- W4-B5 proves the same bridge contract scales to a small selected-family set in one call:
  - selected cases use bounded bridge payload
  - non-selected cases remain on legacy path deterministically

## Landed Anchors
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
  - family descriptors:
    - `score_tile_bridge_family_case_count`
    - `score_tile_bridge_family_*`
  - family precheck/select loops:
    - `ATTN_P11AE_SCORE_TILE_FAMILY_PRECHECK_LOOP`
    - `ATTN_P11AE_SCORE_TILE_FAMILY_CASE_LOOP`
  - observability:
    - `score_tile_bridge_family_visible_count`
    - `score_tile_bridge_family_owner_ok`
    - `score_tile_bridge_family_consumed_count`
    - `score_tile_bridge_family_compare_ok`
    - `score_tile_bridge_family_case_mask`
- `src/Top.h`
  - `run_p11ae_layer0_top_managed_qk_score(...)` passthrough includes family bridge args.
- `scripts/check_top_managed_sram_boundary_regression.ps1`
  - W4-B5 passthrough/signature/anchor guards added.

## Targeted Validation
- `tb/tb_w4b5_qkscore_family_bridge.cpp`
- `scripts/local/run_p11w4b5_qkscore_family_bridge.ps1`
- PASS banners:
  - `W4B5_QKSCORE_FAMILY_BRIDGE_VISIBLE PASS`
  - `W4B5_QKSCORE_FAMILY_OWNERSHIP_CHECK PASS`
  - `W4B5_QKSCORE_FAMILY_EXPECTED_COMPARE PASS`
  - `W4B5_QKSCORE_FAMILY_LEGACY_COMPARE PASS`
  - `W4B5_QKSCORE_FAMILY_NO_SPURIOUS_TOUCH PASS`
  - `W4B5_QKSCORE_FAMILY_MULTI_CASE_ANTI_FALLBACK PASS`
  - `W4B5_QKSCORE_FAMILY_MISMATCH_REJECT PASS`

## Deferred Boundaries
- This cut generalizes bounded score-tile bridge family only.
- Full Phase-B payload ownership migration is still deferred.
- QkScore inner compute/reduction/writeback loops remain SRAM-centric.
