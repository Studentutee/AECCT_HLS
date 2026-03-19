# P11 AC~AF Execution Plan (Wave-Driven, Local-Only)

## Purpose
- Execute AC as critical path while AD/AE/AF prep runs in parallel.
- Keep evidence-first local-only posture.
- Keep workstreams mergeable/reviewable with explicit gates.

## Wave Plan
- Wave-0 parallel: `P00-011AC`, `P00-011AD-prep`, `P00-011AE-prep`, `P00-011AF-prep`.
- Phase-2 ordered: `P00-011AD-impl` (after `G-AC`) -> `P00-011AE-impl` (after `G-AC` + `G-AD-IF`) -> `P00-011AF-impl` (after `G-AE-IF`).

## Locked Direction
- Top remains sole shared-SRAM owner/arbiter.
- High-level dataflow uses `in_ch/out_ch`.
- Leaf kernels keep fixed-size arrays/local buffers.
- local-only evidence is valid.
- local progress is not Catapult closure and not SCVerify closure.

## Prep Merge Policy
- Merge prep streams early only if compile-isolated.
- If any prep stream depends on non-landed AC headers/symbols, keep ready-for-merge and land immediately after `G-AC`.

## Gate Names
- `G-AC`: AC evidence landed and provisional AC internals frozen for downstream use.
- `G-AD-IF`: AD interface freeze entry landed.
- `G-AE-IF`: AE interface freeze entry landed.
