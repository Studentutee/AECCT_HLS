# TOP_MANAGED_SRAM_W4_QKSCORE_CAMPAIGN_COMPLETION_20260331

## Campaign snapshot
- W4-M1 (QK-score phase-entry descriptor probe): landed and PASS (local-only).
- W4-B2 (this round): bounded QkScore score-tile bridge landed and PASS (local-only).

## What W4-B2 adds beyond W4-M1
- W4-M1 proved phase-entry descriptor visibility and ownership/reject observability.
- W4-B2 proves one bounded score tile can be caller-fed/top-fed and consumed at token-write boundary.

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
- W4-B3 bounded secondary score-range bridge or strict bridge-ready gating expansion with the same guard and targeted reject/no-spurious harness.
