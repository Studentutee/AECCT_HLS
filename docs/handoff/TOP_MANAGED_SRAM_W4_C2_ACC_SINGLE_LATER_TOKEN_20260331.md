# TOP_MANAGED_SRAM_W4_C2_ACC_SINGLE_LATER_TOKEN_20260331

## Scope
- Round: W4-C2
- Goal: SoftmaxOut ACC-path single selected later-token bounded bridge.
- Posture:
  - local-only
  - compile-first / evidence-first
  - not Catapult closure
  - not SCVerify closure

## Exact Hook Point
- Function:
  - `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- ACC cut:
  - `ATTN_P11AF_MAINLINE_ACC_TILE_LOOP`
  - `ATTN_P11AF_MAINLINE_ACC_LOOP`
- Added bounded family selectors in ACC branch only:
  - `ATTN_P11AF_TILE_BRIDGE_FAMILY_ACC_CASE_LOOP`
  - `ATTN_P11AF_TILE_BRIDGE_FAMILY_ACC_COMPARE_LOOP`

## Contract Decision (C2)
- Decision: YES, clean bounded cut.
- Why:
  - change is ACC-path only, no RENORM edit, no WRITEBACK edit.
  - external Top 4-channel contract unchanged.
  - no second ownership/arbitration contract introduced.

## Minimal Bridge Shape (C2)
- Selected fields:
  - head selector: `phase_tile_bridge_family_head_idx[c]`
  - d-tile selector: `phase_tile_bridge_family_d_tile_idx[c]`
  - later-token selector: `phase_tile_bridge_family_key_token_begin[c]`
  - token span: `phase_tile_bridge_family_key_token_count[c]`
- Payload:
  - `phase_tile_bridge_family_v_words`
  - `phase_tile_bridge_family_v_words_valid`
  - `phase_tile_bridge_family_v_base_words`
- Observability:
  - `phase_tile_bridge_family_visible_count`
  - `phase_tile_bridge_family_owner_ok`
  - `phase_tile_bridge_family_compare_ok`
  - `phase_tile_bridge_family_consumed_count`
  - `phase_tile_bridge_family_case_mask`
  - `phase_tile_bridge_family_desc_visible_count`
  - `phase_tile_bridge_family_desc_case_mask`

## Bounded Guards (C2)
- Single-token selector only:
  - `case_key_token_count == 1`
- Later-token bounded singleton:
  - `phase_tile_bridge_family_later_case_count_u32 <= 1`
  - if later-token exists, `phase_tile_bridge_family_case_count_u32 == 1`

## Non-Goals Kept
- No RENORM path bridge implementation.
- No WRITEBACK skeleton change.
- No full SoftmaxOut migration claim.
- No global `SramView& sram` removal claim.
