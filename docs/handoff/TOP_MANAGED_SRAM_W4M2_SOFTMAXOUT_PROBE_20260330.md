# TOP MANAGED SRAM W4-M2 SOFTMAXOUT PROBE (2026-03-30)

## Scope
- Task: W4-M2 single phase-entry caller-fed V-tile probe (SoftmaxOut path).
- Bounded micro-cut only:
  - keep inner online softmax/reduction loops unchanged,
  - keep writeback loops unchanged,
  - keep external formal contract unchanged.
- Local-only validation/evidence.

## What landed
1. Added optional caller-fed V-tile probe passthrough at Top helper:
   - `run_p11af_layer0_top_managed_softmax_out(...)`.
2. Added SoftmaxOut phase-entry probe visibility/ownership/compare anchors:
   - `attn_phaseb_top_managed_softmax_out_mainline(...)`.
3. Added W4-M2 targeted TB + runner:
   - positive visibility/ownership/expected-compare/no-spurious checks,
   - negative mismatch-reject check.
4. Extended boundary guard with W4-M2 anchors.

## Exact files changed
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_w4m2_softmaxout_phase_entry_probe.cpp`
- `scripts/local/run_p11w4m2_softmaxout_phase_entry_probe.ps1`

## Validation/evidence highlights (local-only)
- `build/p11w4m2/softmaxout_phase_entry_probe/run.log`:
  - `W4M2_SOFTMAXOUT_CALLER_FED_VTILE_VISIBLE PASS`
  - `W4M2_SOFTMAXOUT_OWNERSHIP_CHECK PASS`
  - `W4M2_SOFTMAXOUT_NO_SPURIOUS_TOUCH PASS`
  - `W4M2_SOFTMAXOUT_EXPECTED_COMPARE PASS`
  - `W4M2_SOFTMAXOUT_PROBE_MISMATCH_REJECT PASS`
  - `PASS: run_p11w4m2_softmaxout_phase_entry_probe`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: W4-M2 SoftmaxOut phase-entry caller-fed V-tile probe anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

## Governance posture
- local-only bounded micro-cut.
- not Catapult closure.
- not SCVerify closure.
- remote simulator/site-local PLI line untouched.

## Deferred boundaries
- This task does **not** migrate full SoftmaxOut payload path.
- Online softmax accumulation and output writeback remain SRAM-centric in bounded scope.
- Wave4 broader migration remains deferred.
