# TOP MANAGED SRAM W4-M1 COMPLETION (2026-03-30)

## 1. Summary
- Completed W4-M1 bounded micro-cut on QK-score phase entry:
  - added caller-fed descriptor probe passthrough at Top helper entry,
  - added phase-entry visibility/ownership/compare probe checks in QK-score mainline entry.
- Kept inner compute/writeback loops unchanged.
- Maintained required local regression chain PASS.

## 2. Exact files changed
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_w4m1_qkscore_phase_entry_probe.cpp`
- `scripts/local/run_p11w4m1_qkscore_phase_entry_probe.ps1`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M1_QKSCORE_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M1_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M1_COMPLETION_20260330.md`

## 3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/init_agent_state.ps1 -RepoRoot . -StateRoot build/agent_state -SessionTag w4m1_qkscore_probe_20260330`
- `powershell -ExecutionPolicy Bypass -File scripts/check_agent_tooling.ps1 -RepoRoot . -StateRoot build/agent_state`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4m1_qkscore_phase_entry_probe.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
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
- `build/p11w4m1/qkscore_phase_entry_probe/run.log`:
  - `W4M1_QKSCORE_CALLER_FED_DESCRIPTOR_VISIBLE PASS`
  - `W4M1_QKSCORE_OWNERSHIP_CHECK PASS`
  - `W4M1_QKSCORE_NO_SPURIOUS_TOUCH PASS`
  - `W4M1_QKSCORE_EXPECTED_COMPARE PASS`
  - `PASS: run_p11w4m1_qkscore_phase_entry_probe`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: W4-M1 QK-score phase-entry caller-fed descriptor probe anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`
- Regression chain:
  - `PASS: run_p11g6_ffn_w1_bias_descriptor`
  - `PASS: run_p11g6_ffn_fallback_observability`
  - `PASS: run_p11g5_ffn_w1_fallback_policy`
  - `PASS: run_p11g5_ffn_fallback_policy`
  - `PASS: run_p11g5_ffn_closure_campaign`
  - `PASS: run_p11g5_wave3_ffn_payload_migration`
  - `PASS: run_p11g5_wave35_ffn_w1_weight_migration`
  - `PASS: run_p11ah_full_loop_local_e2e`
  - `PASS: run_p11aj_top_managed_sram_provenance`
  - `PASS: check_helper_channel_split_regression`
  - `PASS: check_design_purity`
  - `PASS: check_repo_hygiene` (pre/post)

## 5. Governance posture
- local-only bounded micro-cut.
- not Catapult closure.
- not SCVerify closure.
- remote simulator/site-local PLI line untouched.
- no external formal contract change.

## 6. Residual risks
- Wave4 payload movement remains SRAM-centric in inner compute/writeback loops.
- W4-M1 is probe visibility/ownership anchoring only, not full phase migration.
- Broader Wave4 migration remains deferred.

## 7. Recommended next step
- Execute W4-M2 bounded micro-cut (softmax-out single phase-entry caller-fed V-tile probe) with the same probe visibility/ownership/no-spurious validation pattern.
