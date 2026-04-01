# TOP_MANAGED_SRAM_W4_C4_WRITEBACK_SINGLE_SELECTED_20260401

## Scope
- Round: W4-C4
- Goal: SoftmaxOut WRITEBACK-path single selected `(head, later-token, d_tile)` probe / contract-first.
- Posture:
  - local-only
  - compile-first / evidence-first
  - not Catapult closure
  - not SCVerify closure

## Hook Point
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- WRITEBACK branch:
  - `ATTN_P11AF_MAINLINE_WRITEBACK_TILE_LOOP`
  - `ATTN_P11AF_MAINLINE_WRITEBACK_LOOP`
- C4 selector/probe label:
  - `ATTN_P11AF_TILE_BRIDGE_FAMILY_WRITEBACK_CASE_LOOP`

## Contract Decision
- Decision: YES, clean bounded cut.
- Why:
  - change is selector/descriptor observability-first for one selected later-token family case.
  - no external Top 4-channel contract change.
  - no second ownership/arbitration semantics.
  - no broad rewrite of ACC/RENORM/WRITEBACK skeleton.

## Minimal Selector / Descriptor / Observability
- Selector/descriptor reused:
  - `phase_tile_bridge_family_head_idx`
  - `phase_tile_bridge_family_d_tile_idx`
  - `phase_tile_bridge_family_key_token_begin`
  - `phase_tile_bridge_family_key_token_count`
- Existing ownership/compare anchors reused:
  - `phase_tile_bridge_family_owner_ok`
  - `phase_tile_bridge_family_compare_ok`
- C4 WRITEBACK observability (internal helper only):
  - `phase_tile_bridge_family_writeback_selected_count`
  - `phase_tile_bridge_family_writeback_case_mask`
  - `phase_tile_bridge_family_writeback_touch_count`

## Kept Out
- No external Top interface drift.
- No shared-SRAM owner semantics drift.
- No WRITEBACK full migration.
- No claim of full direct-SRAM closure.
