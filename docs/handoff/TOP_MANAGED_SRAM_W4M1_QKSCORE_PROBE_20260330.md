# TOP MANAGED SRAM W4-M1 QKSCORE PROBE (2026-03-30)

## Scope
- Task: W4-M1 single phase-entry caller-fed descriptor probe (QK-score path).
- Bounded micro-cut only:
  - keep inner dot/compute/writeback loops unchanged,
  - keep external formal contract unchanged.
- Local-only validation/evidence.

## What landed
1. Added optional caller-fed phase-entry probe descriptor passthrough on Top helper:
   - `run_p11ae_layer0_top_managed_qk_score(...)`.
2. Added phase-entry probe visibility/ownership/compare anchors in:
   - `attn_phaseb_top_managed_qk_score_mainline(...)`.
3. Added W4-M1 targeted TB + runner:
   - verifies descriptor visibility,
   - verifies ownership/base alignment,
   - verifies no-spurious-touch in the bounded probe scope,
   - verifies expected compare remains deterministic.
4. Extended top-managed SRAM boundary guard with W4-M1 anchors.

## Exact files changed
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_w4m1_qkscore_phase_entry_probe.cpp`
- `scripts/local/run_p11w4m1_qkscore_phase_entry_probe.ps1`

## Validation/evidence highlights (local-only)
- `build/p11w4m1/qkscore_phase_entry_probe/run.log`:
  - `W4M1_QKSCORE_CALLER_FED_DESCRIPTOR_VISIBLE PASS`
  - `W4M1_QKSCORE_OWNERSHIP_CHECK PASS`
  - `W4M1_QKSCORE_NO_SPURIOUS_TOUCH PASS`
  - `W4M1_QKSCORE_EXPECTED_COMPARE PASS`
  - `W4M1_QKSCORE_PROBE_MISMATCH_REJECT PASS`
  - `PASS: run_p11w4m1_qkscore_phase_entry_probe`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: W4-M1 QK-score phase-entry caller-fed descriptor probe anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

## Governance posture
- local-only bounded micro-cut.
- not Catapult closure.
- not SCVerify closure.
- remote simulator/site-local PLI line untouched.

## Deferred boundaries
- This task does **not** migrate full Wave4 payload path.
- Inner QK-score compute/writeback loops remain SRAM-centric by design in this bounded pass.
- Wave4 broader migration remains deferred.
