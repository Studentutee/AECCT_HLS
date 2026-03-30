# TOP MANAGED SRAM W4 PHASE-A KV CAMPAIGN COMPLETION (2026-03-30)

## 1. Summary
- Completed W4-M3 KV dedicated bounded pass:
  - landed Phase-A KV single phase-entry caller-fed x-row probe,
  - added KV targeted TB/runner and guard anchors,
  - retained required regression chain PASS.
- Scope remained bounded:
  - no external formal contract change,
  - no broad rewrite,
  - no inner compute/reduction/writeback loop rewrite.

## 2. Exact files changed
- `src/blocks/AttnPhaseATopManagedKv.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m3_kv_phase_entry_probe.ps1`
- `tb/tb_w4m3_kv_phase_entry_probe.cpp`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_PHASEA_KV_CAMPAIGN_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`

## 3. Governance posture
- local-only evidence.
- not Catapult closure.
- not SCVerify closure.
- remote simulator/site-local PLI line untouched.
- Top sole production shared-SRAM owner posture preserved.

## 4. Residual risks
- Wave4 full payload migration is still deferred.
- KV compute/writeback loops remain SRAM-centric outside phase-entry probe range.

## 5. Recommended next step
- Keep Wave4 progression in bounded slices: one entrypoint, one targeted TB, one guard extension per run.
