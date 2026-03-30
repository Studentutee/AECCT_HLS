# TOP MANAGED SRAM G6 WAVE4 FEASIBILITY (2026-03-30)

## Scope
- Feasibility-only survey for Wave4 paths:
  - `AttnLayer0`
  - `TransformerLayer`
  - `AttnPhaseATopManagedKv`
  - `AttnPhaseATopManagedQ`
  - `AttnPhaseBTopManagedQkScore`
  - `AttnPhaseBTopManagedSoftmaxOut`
- No Wave4 production patch landed in this round.

## Inventory summary
- Multiple direct-SRAM payload paths still dominate Wave4 internals:
  - q/k/v tile reads,
  - score/softmax intermediate reads and writes,
  - phase-local output writes to multiple SRAM targets.
- Several blocks are TopManaged in naming and dispatch shape, but payload data movement is still SRAM-centric inside core loops.

## Candidate ranking
1. W4-M1 (recommended next): single phase-entry caller-fed descriptor probe on QK-score path.
2. W4-M2: single softmax-out V-tile caller-fed probe.
3. W4-M3: single x-row caller-fed descriptor anchor for Q/KV work path.

## Blocker map
- Architectural blockers:
  - tight coupling between payload decode, compute, and writeback across phase loops.
- Implementation blockers:
  - missing narrow ownership-focused TB probes for per-phase no-spurious-touch assertions.
  - micro-cut validation still needs new phase-focused negative tests.

## Why no micro-cut landed this round
- Candidate fanout and validation cost exceeded bounded low-risk threshold after completing FFN Subwave A/B and mandatory rerun chain.
- To avoid broad rewrite drift, this round stops at ranking + blocker capture.

## Recommended next cut
- Execute dedicated W4-M1 task:
  - one caller-fed descriptor probe anchor,
  - one targeted ownership/no-spurious-touch TB,
  - keep inner loop rewiring out of scope.
