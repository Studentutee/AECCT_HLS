# DIRECT_SRAM_ENDGAME_INVENTORY_20260331

## Session Posture
- local-only
- compile-first / evidence-first
- not Catapult closure
- not SCVerify closure

## A. Mature Bridge Line (Executed)

### Hotspot A1: Phase-B QkScore bridge generalization line
- Hook point: `src/blocks/AttnPhaseBTopManagedQkScore.h::attn_phaseb_top_managed_qk_score_mainline`
- Direct SRAM role:
  - Q/K dot-product source read remains SRAM-resident in mainline loops.
  - Score row writeback remains SRAM write path.
- Adjacent bridge baseline:
  - W4-B2 / W4-B3 / W4-B5 / W4-B6 / W4-B7.
- Landed bounded cuts:
  - W4-B8: family max cases `4 -> 8`.
  - W4-B9: family flatten span promoted to token-domain.
- Status: advanced to practical stop boundary for this mature line.

## B. SoftmaxOut Contract-First Mini-Campaign (Executed)

### Hotspot B1: Phase-B SoftmaxOut bridge familyization candidate (W4-C0)
- Hook point: `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- Direct SRAM role:
  - Score read: `score_head_base + j` during online loop.
  - V read: `v_row_base + tile_offset + i` across init/renorm/acc.
  - pre/post/out writeback: `pre_row_base/post_row_base/out_row_base`.
- Contract-first decision:
  - Clean bounded cut exists at `INIT_ACC` consume neighborhood only.
  - Keep renorm/writeback skeleton untouched.
  - Keep external Top 4-channel contract untouched.
- Minimal contract delta landed:
  - Single `phase_tile_bridge` remains.
  - Added optional family descriptor path for init-acc consume only:
    - `phase_tile_bridge_family_case_count`
    - `phase_tile_bridge_family_v_base_words`
    - `phase_tile_bridge_family_v_words`
    - `phase_tile_bridge_family_v_words_valid`
    - `phase_tile_bridge_family_d_tile_idx`
    - family observability outputs (visible/owner/consumed/compare/mask)
  - `src/Top.h` updated for internal helper passthrough only.
- Topology note:
  - Current model topology can be `d_tile_count == 1`; TB covers this with family-only mode.
  - When `d_tile_count > 1`, TB validates disjoint single+family selector coexistence.
- Status: advanced (contract-validated bounded implementation).

## C. Near Skeleton Risk Zone (Do Not Force)

### Hotspot C1: AttnLayer0 score/reduction/writeback core loops
- Hook point: `src/blocks/AttnLayer0.h`
  - `ATTN_SCORE_*`
  - `ATTN_PRECONCAT_*`
  - `ATTN_OUT_WRITEBACK_LOOP`
- Risk:
  - Requires compute/reduction/writeback skeleton edits beyond bounded bridge migration.
- Status: unchanged high-risk zone.

## D. Blocked / Out of Bounded Scope

### Hotspot D1: Global `SramView& sram` ownership removal across AttnLayer0/TransformerLayer
- Hook point:
  - `src/blocks/AttnLayer0.h`
  - `src/blocks/TransformerLayer.h`
- Blocker:
  - Cross-module ownership contract redesign required.
  - Not a bounded migration step.
- Status: blocked (unchanged).

## Priority Evolution (Current)
1. W4-B8/W4-B9 QkScore mature line: done.
2. W4-C0 contract-first mini-campaign: done with bounded contract delta.
3. Remaining deferred: AttnLayer0/TransformerLayer skeleton-level ownership migration.
