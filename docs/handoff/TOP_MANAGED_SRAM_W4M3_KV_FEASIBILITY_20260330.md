# TOP MANAGED SRAM W4-M3 KV FEASIBILITY (2026-03-30)

## Scope
- Secondary track refinement only for `AttnPhaseATopManagedKv` in W4-M3 campaign.
- No KV code patch in this run.

## Inventory summary
- Phase-A KV phase-entry remains SRAM-centric at x-row consume anchor:
  - `ATTN_P11AC_MAINLINE_XROW_LOAD_LOOP` consumes `sram[row_x_base + tile_offset + i]`.
- Existing Top-managed packet/metadata naming exists, but entry payload ownership remains implicitly SRAM-derived.

## Candidate ranking
1. Primary next KV cut:
   - `attn_phasea_top_managed_kv_mainline(...)` single x-row caller-fed descriptor probe.
2. Backup:
   - split K-first then V probe staging (two-step bounded sequence).

## Blocker map
- Architectural:
  - KV computes K and V together, so one probe touches dual branches and increases coupling risk.
- Implementation debt:
  - missing dedicated KV probe TB/runner with owner mismatch + no-spurious reject checks.
  - missing KV-specific guard anchors for probe observability.

## Optional KV micro-cut decision
- Not attempted in this run.
- Reason:
  - keep W4-M3 primary (Phase-A Q) bounded and stable,
  - avoid fanout growth in same batch,
  - preserve deterministic rerun cost.

## Suggested next cut
- Dedicated W4-M3 KV micro-cut run:
  - add one KV phase-entry x-row probe anchor,
  - add one focused TB and runner,
  - keep K/V inner compute and writeback loops untouched.
