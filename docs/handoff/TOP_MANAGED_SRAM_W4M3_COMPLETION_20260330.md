# TOP MANAGED SRAM W4-M3 COMPLETION (2026-03-30)

## 1. Summary
- Completed W4-M3 bounded micro-cut on Phase-A Q phase entry:
  - added caller-fed x-row probe passthrough at Top helper entry,
  - added phase-entry visibility/ownership/compare probe checks in Phase-A Q mainline entry.
- Kept inner Q compute/writeback loops unchanged.
- Completed KV feasibility/ranking/blocker refinement as secondary track (no KV code patch).
- Maintained required local regression chain PASS.

## 2. Exact files changed
- `src/blocks/AttnPhaseATopManagedQ.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m3_phasea_q_phase_entry_probe.ps1`
- `tb/tb_w4m3_phasea_q_phase_entry_probe.cpp`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_PHASEAQ_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_FEASIBILITY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_PHASEA_CAMPAIGN_COMPLETION_20260330.md`

## 3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4m3_phasea_q_phase_entry_probe.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
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
- regression chain:
  - `PASS: run_p11w4m2_softmaxout_phase_entry_probe`
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
- Phase-A Q inner compute/writeback loops remain SRAM-centric.
- W4-M3 is probe visibility/ownership anchoring only, not full Wave4 payload migration.
- KV stays feasibility/refinement-only in this run.

## 7. Recommended next step
- Execute dedicated W4-M3 KV bounded probe (single phase-entry x-row descriptor) with one isolated ownership/no-spurious/mismatch-reject TB.
