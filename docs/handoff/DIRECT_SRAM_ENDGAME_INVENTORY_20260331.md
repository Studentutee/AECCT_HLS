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

### Hotspot B2: SoftmaxOut head/token family descriptor contract-only (W4-C1)
- Hook point: `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- C0 landed contract location:
  - `ATTN_P11AF_MAINLINE_INIT_ACC_TILE_LOOP` family selector/consume neighborhood.
- C1 bounded insertion points:
  - family descriptor precheck block (head/token descriptor validity + bounded guard)
  - descriptor visibility probe loop:
    - `ATTN_P11AF_TILE_BRIDGE_FAMILY_DESC_PROBE_LOOP`
  - family selector condition refinement in init-acc tile consume loop.
- Minimal descriptor set (contract-only):
  - selector:
    - `phase_tile_bridge_family_head_idx[c]`
    - `phase_tile_bridge_family_d_tile_idx[c]`
  - payload span:
    - `phase_tile_bridge_family_key_token_begin[c]`
    - `phase_tile_bridge_family_key_token_count[c]`
  - observability only:
    - `phase_tile_bridge_family_desc_visible_count`
    - `phase_tile_bridge_family_desc_case_mask`
  - helper-only passthrough (must stay internal Top helper):
    - all C1 arrays/counters above (no external Top 4-channel contract exposure)
- Bounded guard (current C1 scope):
  - descriptor token span is constrained to single-token selector:
    - `case_key_token_count == 1`
- Why still bounded:
  - no renorm/acc/writeback skeleton reordering
  - no external Top interface drift
  - no second ownership/arbitration semantics
- Status: advanced (contract-only + probe validated).

### Hotspot B3: SoftmaxOut ACC-path single selected later-token bounded bridge (W4-C2)
- Hook point: `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- Exact cut location:
  - `ATTN_P11AF_MAINLINE_ACC_TILE_LOOP` / `ATTN_P11AF_MAINLINE_ACC_LOOP`
  - bounded family select/compare labels:
    - `ATTN_P11AF_TILE_BRIDGE_FAMILY_ACC_CASE_LOOP`
    - `ATTN_P11AF_TILE_BRIDGE_FAMILY_ACC_COMPARE_LOOP`
- Direct SRAM role advanced in this round:
  - selected `(head, key_token, d_tile)` in ACC path can consume caller-fed bridge payload instead of SRAM read.
  - non-selected ACC and all RENORM/WRITEBACK paths remain SRAM reads/writes.
- Bounded selector shape:
  - one exact key token (`key_token_count == 1`)
  - bounded later-token path limit (`phase_tile_bridge_family_later_case_count_u32 <= 1`)
  - if later-token case exists, this round requires one-case descriptor (`family_case_count == 1`)
- Observability/evidence fields used:
  - `phase_tile_bridge_family_visible_count`
  - `phase_tile_bridge_family_owner_ok`
  - `phase_tile_bridge_family_compare_ok`
  - `phase_tile_bridge_family_consumed_count`
  - `phase_tile_bridge_family_case_mask`
  - `phase_tile_bridge_family_desc_visible_count`
  - `phase_tile_bridge_family_desc_case_mask`
- Why still bounded:
  - RENORM path untouched.
  - WRITEBACK skeleton untouched.
  - external Top 4-channel contract untouched.
  - no second ownership/arbitration semantics introduced.
- Status: advanced (ACC-only later-token consume proof landed).

### Hotspot B4: SoftmaxOut C-family harness consolidation (W4-CFamily)
- Hook points:
  - shared TB bootstrap harness:
    - `tb/tb_w4cfamily_softmaxout_harness_common.h`
  - shared runner strategy:
    - `scripts/local/run_p11w4cfamily_softmaxout_common.ps1`
  - thin wrappers:
    - `run_p11w4c0_softmaxout_contract_probe.ps1`
    - `run_p11w4c1_softmaxout_head_token_contract_probe.ps1`
    - `run_p11w4c2_softmaxout_acc_single_later_token_bridge.ps1`
- Chosen consolidation shape:
  - shared harness + shared runner, legacy per-round entries kept as thin wrappers.
- Why bounded:
  - no external contract change
  - no design semantic rewrite under consolidation-only scope
  - PASS line contracts preserved per round runner wrapper
- Status: advanced (family harness/runner landed; C0/C1/C2 baseline retained).

### Hotspot B5: SoftmaxOut RENORM-path single selected probe (W4-C3)
- Hook point: `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- Exact cut location:
  - `ATTN_P11AF_MAINLINE_RENORM_TILE_LOOP` / `ATTN_P11AF_MAINLINE_RENORM_LOOP`
  - bounded renorm family select/compare labels:
    - `ATTN_P11AF_TILE_BRIDGE_FAMILY_RENORM_CASE_LOOP`
    - `ATTN_P11AF_TILE_BRIDGE_FAMILY_RENORM_COMPARE_LOOP`
- Direct SRAM role advanced in this round:
  - selected `(head, later-token, d_tile)` can consume caller-fed bridge payload in RENORM path.
  - non-selected RENORM, ACC non-selected, and WRITEBACK remain existing SRAM behavior.
- C3 observability (internal helper only):
  - `phase_tile_bridge_family_renorm_selected_count`
  - `phase_tile_bridge_family_renorm_case_mask`
- Why still bounded:
  - WRITEBACK untouched.
  - no external Top contract drift.
  - no second ownership/arbitration semantics introduced.
- Status: advanced (RENORM single selected probe/consume proof landed).

### Hotspot B6: SoftmaxOut WRITEBACK-path single selected probe (W4-C4)
- Hook point: `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- Exact cut location:
  - `ATTN_P11AF_MAINLINE_WRITEBACK_TILE_LOOP` / `ATTN_P11AF_MAINLINE_WRITEBACK_LOOP`
  - bounded writeback selector label:
    - `ATTN_P11AF_TILE_BRIDGE_FAMILY_WRITEBACK_CASE_LOOP`
- Direct SRAM role advanced in this round:
  - selected later-token family descriptor now has explicit WRITEBACK-path visibility on `(head, d_tile)` touch.
  - writeback skeleton behavior is preserved for all paths; selected case is probe/observability-first.
- C4 observability (internal helper only):
  - `phase_tile_bridge_family_writeback_selected_count`
  - `phase_tile_bridge_family_writeback_case_mask`
  - `phase_tile_bridge_family_writeback_touch_count`
- Why still bounded:
  - no external Top 4-channel contract drift.
  - no second ownership/arbitration semantics introduced.
  - no broad rewrite of online core loops.
  - selected case remains one later-token + one head + one d-tile bounded probe.
- Status: advanced (WRITEBACK single selected probe landed).

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
3. W4-C1 SoftmaxOut head/token contract-only groundwork: done with probe validation.
4. W4-C2 SoftmaxOut ACC-path single selected later-token bounded bridge: done.
5. W4-CFamily harness consolidation: done (shared harness + shared runner wrappers).
6. W4-C3 SoftmaxOut RENORM single selected probe: done.
7. W4-C4 SoftmaxOut WRITEBACK single selected probe: done.
8. Remaining deferred: broader WRITEBACK-side migration + AttnLayer0/TransformerLayer skeleton-level ownership migration.
