# TOP MANAGED SRAM W4-M3 KV COMPLETION (2026-03-30)

## 1. Summary
- Completed W4-M3 dedicated bounded pass on Phase-A KV phase entry.
- Added caller-fed x-row probe passthrough at Top helper and probe visibility/ownership/compare anchors in KV mainline entry.
- Added KV-targeted TB/runner and guard anchors.
- Completed required rerun chain PASS and evidence consolidation.

## 2. Exact files changed
- `src/blocks/AttnPhaseATopManagedKv.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m3_kv_phase_entry_probe.ps1`
- `tb/tb_w4m3_kv_phase_entry_probe.cpp`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_PHASEA_KV_CAMPAIGN_COMPLETION_20260330.md`

## 3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4m3_kv_phase_entry_probe.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4m3_phasea_q_phase_entry_probe.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4m2_softmaxout_phase_entry_probe.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g6_ffn_w1_bias_descriptor.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g6_ffn_fallback_observability.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_ffn_w1_fallback_policy.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_ffn_fallback_policy.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_ffn_closure_campaign.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

## 4. Actual execution evidence / log excerpt
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

## 5. Governance posture
- local-only bounded micro-cut.
- not Catapult closure.
- not SCVerify closure.
- no external formal contract change.
- remote simulator/site-local PLI line untouched.

## 6. Residual risks
- Phase-A KV payload path is not fully migrated.
- K/V inner compute and writeback loops remain SRAM-centric in bounded scope.
- This task is a phase-entry probe bridge only.

## 7. Recommended next step
- Execute next bounded Wave4 cut on one downstream entrypoint that can reuse the same ownership/no-spurious/mismatch-reject evidence pattern.
