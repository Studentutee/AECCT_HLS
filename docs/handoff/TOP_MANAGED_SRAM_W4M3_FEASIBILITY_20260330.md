# TOP MANAGED SRAM W4-M3 FEASIBILITY (2026-03-30)

## Scope
- Refinement-only feasibility for next Wave4 probe after W4-M2.
- Focus paths:
  - `AttnPhaseATopManagedQ`
  - `AttnPhaseATopManagedKv`
  - `AttnLayer0` boundary interactions

## Inventory summary
- Remaining phase-entry payload reads are still SRAM-centric in Phase-A Q/KV mainline entry:
  - Q entry x-row tile consume from SRAM.
  - KV entry x-row tile consume from SRAM.
- Existing TopManaged packetization exists, but entry payload ownership is still implicitly SRAM-derived in these paths.

## Candidate ranking
1. W4-M3 primary (next recommended): Phase-A Q x-row phase-entry caller-fed descriptor probe.
2. W4-M3 backup: Phase-A KV x-row phase-entry probe.
3. Optional later: AttnLayer0 wrapper-level probe exposure.

## Blocker map
- Architectural blockers:
  - Q/KV entry and downstream writeback are tightly coupled with current SRAM entry assumptions.
- Implementation blockers:
  - missing dedicated Phase-A ownership/no-spurious targeted TB for probe mismatch reject behavior.
  - optional micro-cut this round would increase runner+guard fanout and dilute W4-M2 bounded focus.

## Optional micro-cut decision in this run
- Not attempted.
- Reason: bounded campaign focus kept on W4-M2 landing + full rerun chain stability.

## Suggested next cut
- Dedicated W4-M3 task:
  - add one Phase-A Q x-row caller-fed descriptor probe anchor,
  - add one targeted ownership/no-spurious/mismatch-reject TB,
  - keep MAC/writeback loops unchanged.
