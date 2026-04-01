# TOP_MANAGED_SRAM_W4_C3_RENORM_SINGLE_SELECTED_20260401

## Scope
- Round: W4-C3
- Goal: RENORM-path single selected probe / contract-first with bounded consume.
- Posture:
  - local-only
  - compile-first / evidence-first
  - not Catapult closure
  - not SCVerify closure

## Hook Point
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- RENORM branch:
  - `ATTN_P11AF_MAINLINE_RENORM_TILE_LOOP`
  - `ATTN_P11AF_MAINLINE_RENORM_LOOP`
- Bounded renorm selectors:
  - `ATTN_P11AF_TILE_BRIDGE_FAMILY_RENORM_CASE_LOOP`
  - `ATTN_P11AF_TILE_BRIDGE_FAMILY_RENORM_COMPARE_LOOP`

## Contract Decision
- Decision: YES, clean bounded cut.
- Why:
  - one selected head + one selected later-token + one selected d_tile only.
  - WRITEBACK path untouched.
  - external Top contract unchanged.
  - no second ownership/arbitration semantics.

## Minimal Descriptor / Observability
- Selector/descriptor:
  - `phase_tile_bridge_family_head_idx`
  - `phase_tile_bridge_family_d_tile_idx`
  - `phase_tile_bridge_family_key_token_begin`
  - `phase_tile_bridge_family_key_token_count`
- Existing observability reused:
  - visible/owner/compare/consumed/case_mask
  - descriptor visibility mask
- C3 renorm path observability (internal helper only):
  - `phase_tile_bridge_family_renorm_selected_count`
  - `phase_tile_bridge_family_renorm_case_mask`

## Kept Out
- No WRITEBACK implementation change.
- No broad online core rewrite.
- No full SoftmaxOut migration claim.
