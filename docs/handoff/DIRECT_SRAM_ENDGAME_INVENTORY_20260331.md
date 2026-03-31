# DIRECT_SRAM_ENDGAME_INVENTORY_20260331

## Session Posture
- local-only
- compile-first / evidence-first
- not Catapult closure
- not SCVerify closure

## A. Mature Bridge Line (Ready and Executed)

### Hotspot A1: Phase-B QkScore bridge generalization line
- Hook point: `src/blocks/AttnPhaseBTopManagedQkScore.h::attn_phaseb_top_managed_qk_score_mainline`
- Direct SRAM role:
  - Q/K dot-product source read still SRAM-resident in mainline loops.
  - Score row writeback still writes `score_head_base + j` in SRAM.
- Adjacent bridge baseline:
  - W4-B2 / W4-B3 / W4-B5 / W4-B6 / W4-B7.
- This session cuts:
  - W4-B8 (done): family max cases `4 -> 8` for full-head mixed bounded coverage.
  - W4-B9 (done): family payload span from tile-domain flatten to token-domain flatten.
- Ownership-forward rationale:
  - Expanded top-fed bridge payload ownership without adding a second ownership/arbitration contract.
- Risk:
  - B8 medium (capacity-only extension).
  - B9 medium (span extension with compile-time guard retained).
- Status: `advanced`.
- Priority: P1 completed for this session.

## B. Bounded-Cut Candidate (Evaluated, Deferred)

### Hotspot B1: Phase-B SoftmaxOut bridge familyization candidate (W4-C0)
- Hook point: `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- Direct SRAM role:
  - Score/V consume path reads SRAM across online softmax (`init/renorm/acc`).
  - Pre/post/out writeback remains SRAM write path.
- Adjacent bridge baseline:
  - W4-B1 single phase tile bridge.
- Candidate next cut:
  - W4-C0 single+family selector around init-acc consume boundary.
- Re-inventory result after B9:
  - Not selected in this session.
  - Requires touching selector semantics inside online softmax core loops and Top helper passthrough shape together.
  - Risk is now near skeleton semantics (`init/renorm/acc/writeback` adjacency), no longer as clean as B8/B9 coverage/span cuts.
- Risk: medium-high to high.
- Status: deferred as practical stop boundary.
- Priority: moved from P3 candidate to `deferred`.

## C. Near Skeleton Risk Zone (Do Not Force)

### Hotspot C1: AttnLayer0 score/reduction/writeback core loops
- Hook point: `src/blocks/AttnLayer0.h`
  - `ATTN_SCORE_*`
  - `ATTN_PRECONCAT_*`
  - `ATTN_OUT_WRITEBACK_LOOP`
- Risk:
  - Requires compute/reduction/writeback skeleton edits beyond bounded bridge migration.
- Status: unchanged high-risk zone.
- Priority: deferred / stop boundary.

## D. Blocked / Out of Bounded Scope

### Hotspot D1: Global `SramView& sram` ownership removal across AttnLayer0/TransformerLayer
- Hook point:
  - `src/blocks/AttnLayer0.h`
  - `src/blocks/TransformerLayer.h`
- Blocker:
  - Cross-module ownership contract redesign required.
  - Not a bounded migration step.
- Status: blocked (unchanged).

## Priority Evolution (This Session)
1. W4-B8 mandatory: completed and gated PASS.
2. W4-B9 conditional: completed and gated PASS after B8 clean PASS.
3. W4-C0 optional: evaluated after B9; deferred due skeleton-adjacent risk and diminishing bounded safety.
