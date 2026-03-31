# TOP_MANAGED_SRAM_W4_PHASEB_CAMPAIGN_COMPLETION_20260330

## Campaign snapshot
- W4-M1 (QK-score phase-entry probe): landed and PASS (local-only).
- W4-M2 (SoftmaxOut phase-entry V probe): landed and PASS (local-only).
- W4-M3 (Phase-A Q phase-entry x-row probe): landed and PASS (local-only).
- W4-M3 KV dedicated pass (Phase-A KV x-row probe): landed and PASS (local-only).
- W4-B1 (this round): bounded Phase-B V-tile bridge landed and PASS (local-only).

## What W4-B1 adds beyond prior probes
- Prior wave4 cuts proved descriptor probe visibility at phase entry.
- W4-B1 proves a bounded payload bridge can be consumed at tile entry while preserving bounded scope.

## Boundaries kept in this campaign
- no external formal contract change
- no remote simulator/PLI/site-local diagnostics
- no broad rewrite
- no inner compute/writeback major loop rewrite
- not Catapult closure; not SCVerify closure

## Residual Wave4 risk
- Main payload ownership migration for inner compute/writeback remains open.
- `AttnLayer0` and phase cores remain major SRAM-centric hotspots outside bounded bridge scope.

## Suggested next cut
- W4-B2 bounded QK-score tile bridge with the same guard and targeted reject/no-spurious harness.
