# TOP MANAGED SRAM G5 WAVE3.5 COMPLETION (2026-03-30)

## 1. Summary
- Completed bounded FFN W1 weight tile caller-fed descriptor migration.
- Preserved Wave3 `topfed_x_words` anchor and extended FFN W1 consume to top-fed weight payload.
- Kept compatibility fallback and avoided broad rewrite.

## 2. Exact files changed
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_g5_wave35_ffn_w1_weight_migration_p11g5w35.cpp`
- `scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_FFN_W1_WEIGHT_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_COMPLETION_20260330.md`

## 3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/init_agent_state.ps1 -RepoRoot . -StateRoot build/agent_state -SessionTag g5_wave35_ffn_w1_20260330`
- `powershell -ExecutionPolicy Bypass -File scripts/check_agent_tooling.ps1 -RepoRoot . -StateRoot build/agent_state`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g5_wave35_ffn_w1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g5_wave35_ffn_w1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

## 4. Actual execution evidence / log excerpt
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`:
  - `G5W35_FFN_W1_TOPFED_WEIGHT_PATH PASS`
  - `G5W35_FFN_W1_NO_SPURIOUS_SRAM_TOUCH PASS`
  - `G5W35_FFN_W1_EXPECTED_COMPARE PASS`
  - `PASS: run_p11g5_wave35_ffn_w1_weight_migration`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G5 wave3.5 FFN W1 top-fed weight payload migration anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11g5/wave3_ffn_payload_migration/run.log`: `PASS: run_p11g5_wave3_ffn_payload_migration`
- `build/p11ah/g5_wave35_ffn_w1/run.log`: `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g5_wave35_ffn_w1/run.log`: `PASS: run_p11aj_top_managed_sram_provenance`

## 5. Repo-tracked artifacts
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_FFN_W1_WEIGHT_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`

## 6. Local-only working-memory artifacts
- `build/agent_state/g5_wave35_ffn_w1_20260330/g5w35_ffn_w1_weight_reality_check.md`
- `build/agent_state/g5_wave35_ffn_w1_20260330/g5w35_ffn_w1_weight_cut_map.md`
- `build/agent_state/g5_wave35_ffn_w1_20260330/g5w35_candidate_ranking.md`
- `build/evidence/g5_wave35_ffn_w1_20260330/evidence_manifest.txt`
- `build/evidence/g5_wave35_ffn_w1_20260330/evidence_summary.txt`

## 7. Governance posture
- local-only bounded migration.
- no external formal contract change.
- not Catapult closure.
- not SCVerify closure.

## 8. Residual risks
- FFN W2 weight path remains SRAM-based.
- FFN bias consume remains SRAM-based.
- broader FFN ReLU/W2 descriptor path remains deferred.
- Wave4 attention/phase migration remains deferred.

## 9. Recommended next step
- Next bounded FFN cut: W1 bias caller-fed descriptor or W2 weight caller-fed descriptor (choose one), keep current Wave3/Wave3.5 anchors unchanged.
