# TOP MANAGED SRAM W4 CAMPAIGN COMPLETION (2026-03-30)

## 1. Summary
- Completed W4 phase-entry probe campaign bounded run:
  - W4-M2 landed: SoftmaxOut single phase-entry caller-fed V-tile probe.
  - W4-M3 landed as feasibility/ranking/blocker refinement only (no code patch).
- Preserved bounded scope:
  - no external formal contract change,
  - no broad rewrite,
  - no inner compute/reduction/writeback loop rewrite.

## 2. W4-M2 work completed
- Added Top helper passthrough for phase-entry probe payload/metadata:
  - `run_p11af_layer0_top_managed_softmax_out(...)`.
- Added SoftmaxOut phase-entry probe anchors:
  - visibility,
  - ownership check,
  - compare/reject observability.
- Added targeted TB and local runner.
- Expanded boundary regression guard with W4-M2 anchors.

## 3. W4-M3 feasibility completed
- Ranked next entry candidate as Phase-A Q x-row probe first.
- Captured implementation blockers and test-capability gaps.
- Optional W4-M3 micro-cut intentionally not attempted in this bounded run.

## 4. Exact files changed
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_w4m2_softmaxout_phase_entry_probe.cpp`
- `scripts/local/run_p11w4m2_softmaxout_phase_entry_probe.ps1`
- `docs/handoff/TOP_MANAGED_SRAM_W4M2_SOFTMAXOUT_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M2_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M2_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_FEASIBILITY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_CAMPAIGN_COMPLETION_20260330.md`
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
- Wave4 full payload migration is still open.
- SoftmaxOut inner compute/writeback loops remain SRAM-centric in this bounded scope.
- W4-M3 remains feasibility-only and needs dedicated targeted TB/runner in next cut.

## 7. Recommended next step
- Run dedicated W4-M3 bounded pass:
  - Phase-A Q x-row caller-fed descriptor probe,
  - ownership/no-spurious/mismatch-reject targeted validation,
  - no inner compute/writeback rewrite.
