# TOP MANAGED SRAM PROGRESS (2026-03-29)

## Executive Summary
- Overnight focus shifted from helper-only cleanup to architecture-forward boundary convergence.
- Completed two minimal cuts that push ownership semantics toward "Top allocates and dispatches, blocks consume":
  1) Top-owned contract dispatch for preproc / initial layernorm / final-head active infer path.
  2) Top-owned preload of Transformer sublayer1 norm params with block-side guarded compatibility fallback.
- Added a dedicated regression guard for this boundary to prevent silent rollback.

## Completed Architecture-Forward Tasks
1. Built and published architecture gap inventory with fixability ranking.
   - `docs/handoff/TOP_MANAGED_SRAM_ARCH_GAP_INVENTORY_20260329.md`
2. Implemented C1 Top-owned contract dispatch in active infer path.
   - `src/Top.h`
3. Implemented C2 Top preload of sublayer1 norm params and guarded block fallback.
   - `src/Top.h`
   - `src/blocks/TransformerLayer.h`
4. Added and executed Top-managed SRAM boundary regression checker.
   - `scripts/check_top_managed_sram_boundary_regression.ps1`
   - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
5. Re-ran local evidence chain and hygiene checks.
   - `build/p11ah/top_managed_sram_push/run.log`
   - `build/p11aj/top_managed_sram_push/run.log`

## Tasks Attempted But Blocked
- No final blocker.
- One transient compile mismatch occurred during first `p11aj` attempt after function signature tightening:
  - symptom: TB callsite expected legacy `run_mid_or_end_layernorm` signature
  - action: added compatibility overload path via optional contract pointer in `src/Top.h`
  - rerun result: PASS

## Validation Summary
- `run_p11ah_full_loop_local_e2e`: PASS
- `run_p11aj_top_managed_sram_provenance`: PASS
- `check_top_managed_sram_boundary_regression`: PASS
- `check_helper_channel_split_regression`: PASS
- `check_p11ap_active_chain_residual_rawptr`: PASS
- `check_design_purity`: PASS
- `check_repo_hygiene -Phase pre`: PASS
- `check_agent_tooling`: PASS

## Exact Artifact Index
- Code:
  - `src/Top.h`
  - `src/blocks/TransformerLayer.h`
  - `scripts/check_top_managed_sram_boundary_regression.ps1`
- Handoff:
  - `docs/handoff/TOP_MANAGED_SRAM_ARCH_GAP_INVENTORY_20260329.md`
  - `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
  - `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
  - `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`
- Evidence:
  - `build/p11ah/top_managed_sram_push/build.log`
  - `build/p11ah/top_managed_sram_push/run.log`
  - `build/p11aj/top_managed_sram_push/build.log`
  - `build/p11aj/top_managed_sram_push/run.log`
  - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
  - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression_summary.txt`

## Governance Posture
- Local-only progress and evidence.
- Top remains sole production shared-SRAM owner at active boundaries touched in this round.
- not Catapult closure.
- not SCVerify closure.

## Night-Batch Extension: G4 Ingest/Base-Shadow Push
- Added Top-managed infer ingest contractization in active path:
  - `InferIngestContract` in `src/Top.h`
  - `run_preproc_block` consumes ingest contract base/valid-length/window range
  - `run_infer_pipeline` FinalHead label source routed to Top-managed SRAM ingest view
  - `OP_INFER` now performs contract span preflight validation before RX state entry
- Expanded boundary regression guard:
  - `scripts/check_top_managed_sram_boundary_regression.ps1` now enforces G4 ingest anchors.

### Additional local evidence (this batch)
- `build/p11ah/g4_night_batch/run.log`: PASS
- `build/p11aj/g4_night_batch/run.log`: PASS
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS

### Deferred boundary
- Full cross-command unified ingest rearchitecture (CFG/PARAM/INFER metadata harmonization) remains deferred.

## G4 Hardening Closeout (hygiene/evidence/consistency)
- This closeout round does not add a broad architecture rewrite.
- Focus is acceptance hardening for already-landed G4 mincut:
  - completion report hygiene,
  - handoff consistency,
  - evidence bundle completeness,
  - targeted preflight negative validation.

### Exact files changed (closeout round)
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g4_infer_ingest_preflight_negative.ps1`
- `tb/tb_infer_ingest_preflight_negative_p11g4.cpp`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_INGEST_MINCUT_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_HARDENING_EVIDENCE_INDEX_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_HARDENING_COMPLETION_20260329.md`

### Additional local-only evidence (closeout round)
- `build/evidence/g4_hardening_20260329/evidence_manifest.txt`
- `build/p11g4/infer_ingest_preflight_negative/run.log`: PASS
  - `PREFLIGHT_INVALID_BASE_REJECT PASS`
  - `PREFLIGHT_INVALID_SPAN_REJECT PASS`
  - `PREFLIGHT_ERR_MEM_RANGE_GUARD_BEHAVIOR PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: OP_INFER preflight reject path maps invalid span to ERR_MEM_RANGE`
- `check_repo_hygiene -Phase post`: PASS (captured in evidence bundle)

### Side-change disposition
- `AECCT_ac_ref/include/RefPrecisionMode.h`
- `AECCT_ac_ref/src/RefModel.cpp`
- `AECCT_ac_ref/src/ref_main.cpp`
- Current repo reality check found no active diff on these paths in this closeout round; they are excluded from G4 acceptance payload.

### Posture
- local-only evidence
- not Catapult closure
- not SCVerify closure
- remote PLI / site-local simulator line not touched
