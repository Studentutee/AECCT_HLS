# TOP_MANAGED_SRAM_W4_QKSCORE_BRIDGE_CAMPAIGN_COMPLETION_20260331

## Campaign snapshot
- W4-M1 (QkScore phase-entry descriptor probe): landed and PASS (local-only).
- W4-B2 (QkScore bounded score-tile bridge, head0-constrained): landed and PASS (local-only).
- W4-B3 (this round, QkScore selectable-head bounded score-tile bridge): landed and PASS (local-only).

## What W4-B3 adds beyond W4-B2
- W4-B2 proved one bounded bridge window on a constrained head0 path.
- W4-B3 proves the same bounded bridge model can be dispatched to non-head0 (validated head1) with secondary key-range selection.

## Boundaries kept in this campaign
- no external formal contract change
- no remote simulator/PLI/site-local diagnostics
- no broad rewrite
- no inner compute/reduction/writeback major-loop rewrite
- not Catapult closure; not SCVerify closure

## Residual Wave4 risk
- Main QkScore payload ownership migration for inner compute/writeback remains open.
- `AttnLayer0` and phase cores remain major SRAM-centric hotspots outside bounded bridge scope.

## Suggested next cut
- W4-B4 bounded secondary bridge window (token/head slice expansion) or stricter bridge-ready gating, while keeping current bounded no-spurious/mismatch-reject harness.
