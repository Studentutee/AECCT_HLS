# TOP MANAGED SRAM W4 PHASE-A CAMPAIGN COMPLETION (2026-03-30)

## 1. Summary
- Completed W4-M3 bounded campaign run:
  - W4-M3 primary landed: Phase-A Q single phase-entry caller-fed x-row probe.
  - Secondary KV track completed as feasibility/ranking/blocker refinement only.
- Scope remained bounded:
  - no external formal contract change,
  - no broad rewrite,
  - no inner compute/reduction/writeback loop rewrite.

## 2. W4-M3 primary completed
- Added Top helper passthrough for Phase-A Q probe payload/metadata.
- Added Phase-A Q phase-entry probe visibility + ownership + compare/reject anchors.
- Added targeted TB + local runner.
- Expanded boundary regression guard with W4-M3 anchors.

## 3. KV secondary track completed
- Captured inventory/ranking/blocker map for `AttnPhaseATopManagedKv`.
- Optional KV micro-cut intentionally not attempted in this bounded run.

## 4. Exact files changed
- `src/blocks/AttnPhaseATopManagedQ.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_w4m3_phasea_q_phase_entry_probe.cpp`
- `scripts/local/run_p11w4m3_phasea_q_phase_entry_probe.ps1`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_PHASEAQ_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_FEASIBILITY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_PHASEA_CAMPAIGN_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`

## 5. Governance posture
- local-only evidence.
- not Catapult closure.
- not SCVerify closure.
- remote simulator/site-local PLI line untouched.
- Top sole production shared-SRAM owner posture preserved.

## 6. Residual risks
- Phase-A payload compute/writeback loops remain SRAM-centric outside phase-entry probe range.
- KV probe patch is still not landed.
- Wave4 full payload migration remains deferred.

## 7. Recommended next step
- Execute dedicated W4-M3 KV bounded pass:
  - one phase-entry x-row probe anchor,
  - one focused owner mismatch/no-spurious/mismatch-reject TB,
  - keep compute/writeback loops unchanged.
