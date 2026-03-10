# M0~M16 Closeout Backlog (v12.1 single-X_WORK)

## Priority A (Foundational / bookkeeping blockers)

| Item | Why prioritized | Files likely to create/repair | Work type |
|---|---|---|---|
| A1. Re-baseline M2 + M12 to single-X_WORK formal baseline | Blocks trustworthy milestone bookkeeping for downstream scheduler/final-head items; current top/sram semantics still dual-page centric | `include/SramMap.h`, `src/Top.h`, `docs/milestones/M02_report.md`, `docs/milestones/M02_artifacts/*`, `docs/milestones/M12_report.md`, `docs/milestones/M12_artifacts/*` | code+evidence |
| A2. Establish missing early formal closure packs (M0, M1, M6) | These milestones are foundational and already have implementation/TB signal but no formal closure artifacts | `docs/milestones/M00_report.md`, `docs/milestones/M00_artifacts/*`, `docs/milestones/M01_report.md`, `docs/milestones/M01_artifacts/*`, `docs/milestones/M06_report.md`, `docs/milestones/M06_artifacts/*` | documentation/evidence only |
| A3. Close M13 as real FinalHead milestone | Current M13 status is not done; blocks clean interpretation of M14 end-to-end status | `src/blocks/FinalHead.h`, `src/Top.h`, `tb/tb_top_end2end_m13.cpp`, `docs/milestones/M13_report.md`, `docs/milestones/M13_artifacts/*` | code+evidence |
| A4. Formalize M14 evidence pack with latest expected coverage | M14 TB exists but no formal milestone closure; outmode/overlap evidence is currently unbookkept | `docs/milestones/M14_report.md`, `docs/milestones/M14_artifacts/*`, `tb/tb_regress_m14.cpp` (only if evidence instrumentation is needed) | documentation/evidence only (likely) |

## Priority B (Spec drift / re-baseline work)

| Item | Why prioritized | Files likely to create/repair | Work type |
|---|---|---|---|
| B1. Resolve M16 scope/name drift (generator integration vs compliance closeout) | Current `M16_report.md` evidences compliance convergence, not latest-plan generator-integration closure | `docs/milestones/M16_report.md` (or new remap report), `docs/milestones/M16_artifacts/*`, `tools/gen_headers.py`, potential `scripts/gen_sram_map.py`, potential `scripts/gen_weight_stream_order.py` | code+evidence |
| B2. Add explicit drift traceability for single-X_WORK migration | Multiple milestones are impacted by renamed scope; bookkeeping needs explicit mapping note to avoid rework loops | `docs/milestones/M0_M16_audit_v12.1.md` follow-up annex, milestone reports for `M2/M12/M13` | documentation/evidence only |
| B3. Confirm M10 softmax direction closure policy | Current code and milestone naming can be interpreted differently; explicit closure policy avoids ambiguous claims | `docs/milestones/M10_report.md`, `docs/milestones/M10_artifacts/*`, `include/SoftmaxApprox.h`, `src/blocks/AttnLayer0.h` (if behavior alignment is required) | code+evidence (or docs/evidence if declared deferred) |

## Priority C (Implementation likely exists, but mapping/evidence is messy)

| Item | Why prioritized | Files likely to create/repair | Work type |
|---|---|---|---|
| C1. Backfill milestone packs for M7~M11 | Code/TB exists but milestone closure artifacts are absent; this is mostly bookkeeping debt | `docs/milestones/M07_report.md`~`M11_report.md`, `docs/milestones/M07_artifacts/*`~`M11_artifacts/*` | documentation/evidence only |
| C2. Close M15 freeze milestone with explicit boundary table | Quant-related pieces exist but no coherent milestone-grade freeze evidence | `docs/milestones/M15_report.md`, `docs/milestones/M15_artifacts/*`, supporting docs in `docs/spec`/`docs/process` only if needed | documentation/evidence only (possibly small code touch if mismatch found) |
| C3. Add cross-reference map from patch packs (P00/PREF) to milestone closure claims | Reduces ambiguity when implementation arrived via patch IDs instead of milestone-named reports | `docs/milestones/*_report.md` (cross-reference sections), optional `docs/milestones/traceability_map.md` | documentation/evidence only |

## Notes

- This backlog is evidence-first and does not assume hypothesis outcomes.
- Formal PASS claims remain gated by governance-required milestone report + complete artifact pack aligned to latest scope.
