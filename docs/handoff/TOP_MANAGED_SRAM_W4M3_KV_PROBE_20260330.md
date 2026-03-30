# TOP MANAGED SRAM W4-M3 KV PROBE (2026-03-30)

## Scope
- Task: W4-M3 dedicated bounded pass on Phase-A KV entry.
- Landed cut: single phase-entry caller-fed x-row probe on `attn_phasea_top_managed_kv_mainline(...)`.
- Bounded constraints:
  - no external formal contract change,
  - no broad rewrite,
  - no inner compute/reduction/writeback loop rewrite,
  - local-only evidence.

## What landed
1. Added optional caller-fed x-row probe passthrough at Top helper:
   - `run_p11ac_layer0_top_managed_kv(...)`.
2. Added Phase-A KV phase-entry probe anchors:
   - `phase_entry_probe_enabled`
   - `ATTN_P11AC_PHASE_ENTRY_PROBE_COL_LOOP`
   - visibility/ownership/compare observability outputs
   - owner mismatch and compare mismatch reject path before compute loops.
3. Added KV-targeted TB + runner:
   - positive visibility/ownership/no-spurious/expected-compare checks,
   - negative owner mismatch and compare mismatch reject checks.
4. Extended boundary guard with W4-M3 KV anchors.

## Exact files changed
- `src/blocks/AttnPhaseATopManagedKv.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_w4m3_kv_phase_entry_probe.cpp`
- `scripts/local/run_p11w4m3_kv_phase_entry_probe.ps1`

## Validation highlights (local-only)
- `build/p11w4m3/kv_phase_entry_probe/run.log`:
  - `W4M3_KV_CALLER_FED_XROW_VISIBLE PASS`
  - `W4M3_KV_OWNERSHIP_CHECK PASS`
  - `W4M3_KV_NO_SPURIOUS_TOUCH PASS`
  - `W4M3_KV_EXPECTED_COMPARE PASS`
  - `W4M3_KV_PROBE_MISMATCH_REJECT PASS`
  - `PASS: run_p11w4m3_kv_phase_entry_probe`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: W4-M3 Phase-A KV phase-entry caller-fed x-row probe anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

## Governance posture
- local-only bounded micro-cut.
- not Catapult closure.
- not SCVerify closure.
- remote simulator/site-local PLI line untouched.

## Deferred boundaries
- This task does not migrate full Phase-A KV payload path.
- K/V inner compute and writeback loops remain SRAM-centric in bounded scope.
- Wave4 full payload migration remains deferred.
