# TOP MANAGED SRAM W4-M3 PHASE-A Q PROBE (2026-03-30)

## Scope
- Task: W4-M3 single phase-entry caller-fed x-row descriptor probe (Phase-A Q path).
- Bounded micro-cut only:
  - keep inner Q compute/writeback loops unchanged,
  - keep external formal contract unchanged,
  - keep block graph unchanged.
- Local-only validation/evidence.

## What landed
1. Added optional caller-fed x-row probe passthrough at Top helper:
   - `run_p11ad_layer0_top_managed_q(...)`.
2. Added Phase-A Q phase-entry probe visibility/ownership/compare anchors:
   - `attn_phasea_top_managed_q_mainline(...)`.
3. Added W4-M3 targeted TB + runner:
   - positive visibility/ownership/expected-compare/no-spurious checks,
   - negative owner-mismatch reject and compare-mismatch reject checks.
4. Extended boundary guard with W4-M3 anchors.

## Exact files changed
- `src/blocks/AttnPhaseATopManagedQ.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_w4m3_phasea_q_phase_entry_probe.cpp`
- `scripts/local/run_p11w4m3_phasea_q_phase_entry_probe.ps1`

## Validation/evidence highlights (local-only)
- `build/p11w4m3/phasea_q_phase_entry_probe/run.log`:
  - `W4M3_PHASEAQ_CALLER_FED_XROW_VISIBLE PASS`
  - `W4M3_PHASEAQ_OWNERSHIP_CHECK PASS`
  - `W4M3_PHASEAQ_NO_SPURIOUS_TOUCH PASS`
  - `W4M3_PHASEAQ_EXPECTED_COMPARE PASS`
  - `W4M3_PHASEAQ_PROBE_MISMATCH_REJECT PASS`
  - `PASS: run_p11w4m3_phasea_q_phase_entry_probe`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: W4-M3 Phase-A Q phase-entry caller-fed x-row probe anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

## Governance posture
- local-only bounded micro-cut.
- not Catapult closure.
- not SCVerify closure.
- remote simulator/site-local PLI line untouched.

## Deferred boundaries
- This task does **not** migrate full Phase-A Q payload path.
- Q inner compute/writeback loops remain SRAM-centric in bounded scope.
- KV remains feasibility/refinement-only in this run.
