# TOP MANAGED SRAM MIN CUTS (2026-03-29)

## Task C1: Top-Owned Contract Dispatch (Preproc / LN / FinalHead)
1. Summary
- Switched active Top infer path from wrapper-owned contract assembly to explicit Top-owned contract assembly and dispatch.
- `run_preproc_block`, `run_layernorm_block`, and `run_infer_pipeline` now build contracts in Top and dispatch core entries.

2. Exact files changed
- `src/Top.h`

3. Exact commands run
- `git status --short`
- `rg -n "PreprocEmbedSPECoreWindow|LayerNormBlockCoreWindow|FinalHeadCorePassABTopManaged|run_preproc_block|run_layernorm_block|run_infer_pipeline" src/Top.h src/blocks`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/top_managed_sram_push`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/top_managed_sram_push`

4. Actual execution evidence / log excerpt
- `build/p11ah/top_managed_sram_push/run.log`:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/top_managed_sram_push/run.log`:
  - `PROVENANCE_STAGE_AC PASS`
  - `PROVENANCE_STAGE_AD PASS`
  - `PROVENANCE_STAGE_AE PASS`
  - `PROVENANCE_STAGE_AF PASS`
  - `PASS: run_p11aj_top_managed_sram_provenance`

5. Governance posture
- Local-only evidence.
- Top remains sole production shared-SRAM owner.
- not Catapult closure; not SCVerify closure.

6. Residual risks
- Compatibility wrappers still exist and can be used by other legacy callsites.
- Default infer loop is still pointer-facing orchestration, not fully switched to deep bridge path.

7. Next recommended step
- Extend static guard to forbid accidental reintroduction of wrapper-owned dispatch in Top active path.

## Task C2: Top Preload Of Transformer Sublayer1 Norm Params
1. Summary
- Added Top-side preload helper for sublayer1 LN gamma/beta and invoked it in both pointer and deep-bridge layer loops.
- Added `sublayer1_norm_preloaded_by_top` flag in `TransformerLayer` and `TransformerLayerTopManagedAttnBridge` with guarded fallback.

2. Exact files changed
- `src/Top.h`
- `src/blocks/TransformerLayer.h`

3. Exact commands run
- `rg -n "load_layer_sublayer1_norm_params|TransformerLayerTopManagedAttnBridge|TransformerLayer\(" src/Top.h src/blocks/TransformerLayer.h`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/top_managed_sram_push`

4. Actual execution evidence / log excerpt
- `build/p11aj/top_managed_sram_push/run.log`:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `FINAL_X_EXPECTED_COMPARE_HARDENED PASS`
  - `PASS: tb_top_managed_sram_provenance_p11aj`

5. Governance posture
- Local-only evidence.
- Ownership shift is Top -> block consumption only.
- not Catapult closure; not SCVerify closure.

6. Residual risks
- Guarded fallback path still allows in-block preload when caller does not set the preload flag.

7. Next recommended step
- Gradually require preload flag in additional Top-managed callsites, then narrow legacy fallback usage.

## Task C3: G4 INFER Ingest/Base-Shadow Contractization
1. Summary
- Added explicit Top-side ingest contract metadata (`InferIngestContract`) for INFER payload dispatch.
- Added Top-side ingest window metadata (`phase_id`, `token_range`, `tile_range`) and preflight span validation at `OP_INFER`.
- Shifted Preproc infer input base/len dispatch to ingest contract anchors.
- Switched FinalHead label-source pointer to Top-managed SRAM ingest view instead of shadow-array source.

2. Exact files changed
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g4_night_batch`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g4_night_batch`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`

4. Actual execution evidence / log excerpt
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G4 infer ingest contractized base/len dispatch anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11ah/g4_night_batch/run.log`:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g4_night_batch/run.log`:
  - `PROVENANCE_STAGE_AC PASS`
  - `PROVENANCE_STAGE_AD PASS`
  - `PROVENANCE_STAGE_AE PASS`
  - `PROVENANCE_STAGE_AF PASS`
  - `PASS: run_p11aj_top_managed_sram_provenance`

5. Governance posture
- Local-only architecture-forward mincut.
- Top ownership boundary tightened for infer ingest base/len/window dispatch semantics.
- not Catapult closure; not SCVerify closure.

6. Residual risks
- `infer_input_shadow` still exists as local debug/probe mirror.
- Full ingest (CFG/PARAM/INFER) metadata unification remains deferred.

7. Next recommended step
- Plan a scoped follow-up for G4-E cross-command ingest metadata harmonization without broad rewiring.

## Task C4: G4-E Cross-Command Metadata Surface Harmonization
1. Summary
- Added a bounded Top-only cross-command metadata surface (`IngestMetadataSurface`) to normalize ingest lifecycle metadata across CFG/PARAM/INFER paths.
- Updated OP_LOAD_W and OP_INFER preflight to use harmonized metadata span helpers.
- Kept command-specific semantics intact (CFG commit legality, PARAM W-region policy, INFER phase/token/tile metadata).

2. Exact files changed
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g4e_cross_command_metadata_negative.ps1`
- `tb/tb_g4e_cross_command_metadata_negative_p11g4e.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4e_cross_command_metadata_negative.ps1 -BuildDir build/p11g4e/cross_command_metadata_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4_infer_ingest_preflight_negative.ps1 -BuildDir build/p11g4/infer_ingest_preflight_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g4e_metadata_mincut`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g4e_metadata_mincut`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

4. Actual execution evidence / log excerpt
- `build/p11g4e/cross_command_metadata_negative/run.log`:
  - `G4E_OWNER_CFG_RX_MISMATCH_REJECT PASS`
  - `G4E_SPAN_OUT_OF_RANGE_REJECT PASS`
  - `PASS: run_p11g4e_cross_command_metadata_negative`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G4-E cross-command ingest metadata surface helpers anchored`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11ah/g4e_metadata_mincut/run.log`:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g4e_metadata_mincut/run.log`:
  - `PROVENANCE_STAGE_AC PASS`
  - `PROVENANCE_STAGE_AD PASS`
  - `PROVENANCE_STAGE_AE PASS`
  - `PROVENANCE_STAGE_AF PASS`
  - `PASS: run_p11aj_top_managed_sram_provenance`

5. Governance posture
- local-only bounded harmonization.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- G4-E is still bounded to metadata surface/preflight helper unification.
- Full semantic unification across CFG/PARAM/INFER remains deferred.

7. Next recommended step
- Evaluate a bounded follow-up for cross-command commit-time metadata diagnostics while preserving external contract and avoiding broad rewrite.

## Task C5: G4-F Commit-Time Diagnostics Harmonization
1. Summary
- Added a bounded Top-only shared commit-time diagnostics helper path to converge CFG/PARAM/INFER metadata acceptance/reject mapping.
- Consolidated commit-time owner/rx check, optional span legality check, and exact-length check via:
  - `ingest_meta_len_exact(...)`
  - `ingest_commit_diag_error(...)`
- Preserved command-specific semantics after shared diagnostics pass.

2. Exact files changed
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g4f_commit_diagnostics_negative.ps1`
- `tb/tb_g4f_commit_diagnostics_negative_p11g4f.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4f_commit_diagnostics_negative.ps1 -BuildDir build/p11g4f/commit_diagnostics_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4e_cross_command_metadata_negative.ps1 -BuildDir build/p11g4e/cross_command_metadata_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4_infer_ingest_preflight_negative.ps1 -BuildDir build/p11g4/infer_ingest_preflight_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g4f_commit_diag`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g4f_commit_diag`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

4. Actual execution evidence / log excerpt
- `build/p11g4f/commit_diagnostics_negative/run.log`:
  - `G4F_CFG_LEN_MISMATCH_MAPPING PASS`
  - `G4F_PARAM_LEN_MISMATCH_MAPPING PASS`
  - `G4F_OWNER_RX_MISMATCH_MAPPING PASS`
  - `G4F_SPAN_MISMATCH_MAPPING PASS`
  - `PASS: run_p11g4f_commit_diagnostics_negative`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G4-F commit-time diagnostics helper + error mapping anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11ah/g4f_commit_diag/run.log`:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g4f_commit_diag/run.log`:
  - `PROVENANCE_STAGE_AC PASS`
  - `PROVENANCE_STAGE_AD PASS`
  - `PROVENANCE_STAGE_AE PASS`
  - `PROVENANCE_STAGE_AF PASS`
  - `PASS: run_p11aj_top_managed_sram_provenance`

5. Governance posture
- local-only bounded harmonization.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- INFER commit-time length mismatch currently reuses existing fallback mapping because no infer-specific mismatch code is defined in current external protocol enum.
- Deeper protocol-level cross-command error-code harmonization remains deferred.

7. Next recommended step
- Review whether a future bounded protocol revision should introduce infer-specific length mismatch code, then map it through the same helper path without broad rewrite.

## Task C6: G4-G Accepted-Commit Metadata Record Harmonization
1. Summary
- Added a bounded Top-local accepted-commit metadata record to unify CFG/PARAM/INFER commit-success metadata bookkeeping.
- Introduced shared helpers:
  - `record_accepted_commit_metadata(...)`
  - `ingest_commit_diag_and_record(...)`
- Ensured reject path does not overwrite previously accepted record.

2. Exact files changed
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g4g_accept_commit_record.ps1`
- `tb/tb_g4g_accept_commit_record_p11g4g.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4g_accept_commit_record.ps1 -BuildDir build/p11g4g/accept_commit_record`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4f_commit_diagnostics_negative.ps1 -BuildDir build/p11g4f/commit_diagnostics_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4e_cross_command_metadata_negative.ps1 -BuildDir build/p11g4e/cross_command_metadata_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4_infer_ingest_preflight_negative.ps1 -BuildDir build/p11g4/infer_ingest_preflight_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g4g_accept_commit`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g4g_accept_commit`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

4. Actual execution evidence / log excerpt
- `build/p11g4g/accept_commit_record/run.log`:
  - `G4G_ACCEPT_RECORD_CFG_DETERMINISTIC PASS`
  - `G4G_ACCEPT_RECORD_PARAM_DETERMINISTIC PASS`
  - `G4G_REJECT_NO_STALE_STATE PASS`
  - `G4G_ACCEPT_RECORD_INFER_PHASE_VALID PASS`
  - `PASS: run_p11g4g_accept_commit_record`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G4-G accepted-commit metadata record harmonization anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11ah/g4g_accept_commit/run.log`:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g4g_accept_commit/run.log`:
  - `PROVENANCE_STAGE_AC PASS`
  - `PROVENANCE_STAGE_AD PASS`
  - `PROVENANCE_STAGE_AE PASS`
  - `PROVENANCE_STAGE_AF PASS`
  - `PASS: run_p11aj_top_managed_sram_provenance`

5. Governance posture
- local-only bounded harmonization.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- G4-G does not introduce new external diagnostics interfaces; committed-record export is still internal/local-only.
- Command-specific post-commit behavior remains intentionally command-specific.

7. Next recommended step
- If needed, plan a bounded follow-up to expose accepted-commit record snapshots through a dedicated debug-only path without changing formal external contract.

## Task C7: G5 Wave 1/2 Direct-Payload Migration Campaign
1. Summary
- Delivered bounded wave migration for remaining direct-SRAM payload flow:
  - Wave 1: LayerNorm affine payload (`gamma/beta`) and FinalHead scalar payload now have Top-fed consume anchors.
  - Wave 2: Preproc infer input payload now has Top-fed consume anchor.
- Kept existing compatibility fallback paths and external formal contract unchanged.

2. Exact files changed
- `src/Top.h`
- `src/blocks/LayerNormBlock.h`
- `src/blocks/FinalHead.h`
- `src/blocks/PreprocEmbedSPE.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_wave1_payload_migration.ps1`
- `tb/tb_g5_wave1_payload_migration_p11g5w1.cpp`
- `scripts/local/run_p11g5_wave2_preproc_payload_migration.ps1`
- `tb/tb_g5_wave2_preproc_payload_migration_p11g5w2.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave1_payload_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave2_preproc_payload_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g5_payload_campaign`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g5_payload_campaign`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`
- `powershell -ExecutionPolicy Bypass -File scripts/check_agent_tooling.ps1 -RepoRoot . -StateRoot build/agent_state`

4. Actual execution evidence / log excerpt
- `build/p11g5/wave1_payload_migration/run.log`:
  - `G5W1_LN_TOPFED_AFFINE_NO_SPURIOUS_SRAM_TOUCH PASS`
  - `G5W1_FINALHEAD_TOPFED_SCALAR_PATH PASS`
  - `PASS: run_p11g5_wave1_payload_migration`
- `build/p11g5/wave2_preproc_payload_migration/run.log`:
  - `G5W2_PREPROC_TOPFED_INPUT_PATH PASS`
  - `PASS: run_p11g5_wave2_preproc_payload_migration`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G5 wave1/wave2 top-fed payload migration anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11ah/g5_payload_campaign/run.log`: `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g5_payload_campaign/run.log`: `PASS: run_p11aj_top_managed_sram_provenance`

5. Governance posture
- local-only bounded migration wave.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- Wave 3 (`FFNLayer0`) and Wave 4 (`AttnLayer0`/`TransformerLayer`/phase blocks) still retain substantial direct-SRAM payload operations.
- Full payload ownership convergence remains deferred.

7. Next recommended step
- Execute bounded Wave 3 on `FFNLayer0` using Top-fed W1/W2 tile descriptor preload and targeted no-bypass validation.

## Task C8: G5-Wave3 FFNLayer0 Payload Migration (Bounded)
1. Summary
- Completed bounded FFN migration cut on W1 input payload consume path.
- Added caller-fed topfed window handoff:
  - `TransformerLayer` preloads `topfed_ffn_x_words`,
  - `FFNLayer0` consumes `topfed_x_words` in W1 tile loop when provided.
- Kept fallback for compatibility; no external formal contract change.

2. Exact files changed
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`
- `tb/tb_g5_wave3_ffn_payload_migration_p11g5w3.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g5_wave3_ffn`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g5_wave3_ffn`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

4. Actual execution evidence / log excerpt
- `build/p11g5/wave3_ffn_payload_migration/run.log`:
  - `G5W3_FFN_TOPFED_PAYLOAD_PATH PASS`
  - `G5W3_FFN_NO_SPURIOUS_SRAM_TOUCH PASS`
  - `G5W3_FFN_EXPECTED_COMPARE PASS`
  - `PASS: run_p11g5_wave3_ffn_payload_migration`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G5 wave3 FFN top-fed payload migration anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11ah/g5_wave3_ffn/run.log`: `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g5_wave3_ffn/run.log`: `PASS: run_p11aj_top_managed_sram_provenance`

5. Governance posture
- local-only bounded migration.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- FFN W1/W2 weight+bias and later-stage payload consume remain SRAM-based.
- This is not full FFN direct-SRAM elimination.

7. Next recommended step
- Continue bounded FFN follow-up by adding Top/caller-fed weight tile descriptors for one stage (W1 first), with targeted no-bypass validation.

## Task C9: G5-Wave3.5 FFN W1 Weight Tile Caller-Fed Descriptor
1. Summary
- Completed bounded migration for FFN W1 weight tile consume path.
- `TransformerLayer` and bridge path preload `topfed_ffn_w1_words`.
- `FFNLayer0` W1 tile loop consumes `topfed_w1_weight_words` with `topfed_w1_weight_words_valid`.
- Kept fallback to legacy SRAM weight reads for compatibility.

2. Exact files changed
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`
- `tb/tb_g5_wave35_ffn_w1_weight_migration_p11g5w35.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g5_wave35_ffn_w1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g5_wave35_ffn_w1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

4. Actual execution evidence / log excerpt
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

5. Governance posture
- local-only bounded migration.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- W2 weight and bias consume paths remain SRAM-based.
- ReLU/W2 broader payload migration remains deferred.
- No claim of full FFN closure.

7. Next recommended step
- Choose one bounded next cut: W1 bias descriptor or W2 weight descriptor, then mirror this round's targeted-validation pattern.

## Task C10: G5 FFN Closure Campaign (Subwave A/B/C/D)
1. Summary
- Completed bounded FFN closure campaign push with multi-subwave convergence in one run.
- Subwave A: W2 input activation path now supports caller-fed/top-fed descriptor consume.
- Subwave B: W2 weight tile path now supports caller-fed/top-fed descriptor consume.
- Subwave C: W2 bias path now supports caller-fed/top-fed descriptor consume.
- Subwave D: validated fallback boundary (top-fed path dominates when provided; fallback retained only for compatibility).

2. Exact files changed
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_ffn_closure_campaign.ps1`
- `tb/tb_g5_ffn_closure_campaign_p11g5fc.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_ffn_closure_campaign.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g5_ffn_closure_campaign`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g5_ffn_closure_campaign`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

4. Actual execution evidence / log excerpt
- `build/p11g5/ffn_closure_campaign/run.log`:
  - `G5FFN_SUBWAVE_A_W2_INPUT_TOPFED_PATH PASS`
  - `G5FFN_SUBWAVE_B_W2_WEIGHT_TOPFED_PATH PASS`
  - `G5FFN_SUBWAVE_C_W2_BIAS_TOPFED_PATH PASS`
  - `G5FFN_SUBWAVE_D_FALLBACK_BOUNDARY PASS`
  - `PASS: run_p11g5_ffn_closure_campaign`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G5 FFN closure campaign W2 top-fed input/weight/bias anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11ah/g5_ffn_closure_campaign/run.log`: `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g5_ffn_closure_campaign/run.log`: `PASS: run_p11aj_top_managed_sram_provenance`

5. Governance posture
- local-only bounded campaign.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- FFN compatibility fallbacks still exist (not full removal).
- Full FFN closure and Wave4 migration remain deferred.

7. Next recommended step
- Execute a bounded FFN fallback-tightening pass with explicit policy (when top-fed descriptors are present, enforce no-fallback mode in targeted configurations).

## Task C11: G5 FFN Fallback Policy Tightening (Bounded)
1. Summary
- Added explicit strict fallback policy gate for FFN W2 stage (`FFN_POLICY_REQUIRE_W2_TOPFED`).
- In strict mode, W2 requires ready top-fed descriptors (input/weight/bias); otherwise deterministic reject and no output write.
- Added fallback observability anchors (`fallback_policy_reject_flag`, `fallback_legacy_touch_counter`).

2. Exact files changed
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_ffn_fallback_policy.ps1`
- `tb/tb_g5_ffn_fallback_policy_p11g5fp.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
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

4. Actual execution evidence / log excerpt
- `build/p11g5/ffn_fallback_policy/run.log`:
  - `G5FFN_FALLBACK_POLICY_TOPFED_PRIMARY PASS`
  - `G5FFN_FALLBACK_POLICY_CONTROLLED_FALLBACK PASS`
  - `G5FFN_FALLBACK_POLICY_NO_STALE_STATE PASS`
  - `G5FFN_FALLBACK_POLICY_EXPECTED_COMPARE PASS`
  - `PASS: run_p11g5_ffn_fallback_policy`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G5 FFN fallback policy strict W2 top-fed gating anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

5. Governance posture
- local-only bounded tightening.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- Fallback paths still remain in non-strict mode and W1 path.
- Full fallback elimination remains deferred.

7. Next recommended step
- Run bounded W1 fallback policy tightening using the same strict-mode + reject/no-stale observability pattern.

## Task C12: G5 FFN W1 Fallback Policy Tightening (Bounded)
1. Summary
- Added explicit strict fallback policy gate for FFN W1 stage (`FFN_POLICY_REQUIRE_W1_TOPFED`).
- In strict mode, W1 requires ready top-fed descriptors for both x payload and W1 weight payload; otherwise deterministic reject and no output write.
- Added explicit caller-provided x descriptor-valid override (`topfed_x_words_valid_override`) to prevent silent partial descriptor fallback.

2. Exact files changed
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_ffn_w1_fallback_policy.ps1`
- `tb/tb_g5_ffn_w1_fallback_policy_p11g5w1fp.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
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

4. Actual execution evidence / log excerpt
- `build/p11g5/ffn_w1_fallback_policy/run.log`:
  - `G5FFN_W1_FALLBACK_POLICY_TOPFED_PRIMARY PASS`
  - `G5FFN_W1_FALLBACK_POLICY_CONTROLLED_FALLBACK PASS`
  - `G5FFN_W1_FALLBACK_POLICY_REJECT_ON_MISSING_DESCRIPTOR PASS`
  - `G5FFN_W1_FALLBACK_POLICY_NO_STALE_STATE PASS`
  - `G5FFN_W1_FALLBACK_POLICY_NO_SPURIOUS_TOUCH PASS`
  - `G5FFN_W1_FALLBACK_POLICY_EXPECTED_COMPARE PASS`
  - `PASS: run_p11g5_ffn_w1_fallback_policy`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G5 FFN W1 fallback policy strict top-fed gating anchors OK`
  - `guard: G5 FFN fallback policy strict W2 top-fed gating anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

5. Governance posture
- local-only bounded tightening.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- W1 compatibility fallback remains in non-strict mode.
- W1 bias and deeper full-fallback elimination are deferred.
- Wave4 attention/phase migration remains deferred.

7. Next recommended step
- Decide whether to run a bounded W1 bias descriptor tightening pass or transition to Wave4 feasibility+blocker capture.

## Task C13: G6 Single-Run Multi-Track (FFN Near-Closure + Wave4 Feasibility)
1. Summary
- Completed two FFN near-closure bounded subwaves in one run:
  - Subwave A: W1 bias caller-fed/top-fed descriptor consume anchor.
  - Subwave B: W1/W2 strict reject-stage observability harmonization.
- Completed Wave4 feasibility inventory/ranking/blocker map (no micro-cut this round).

2. Exact files changed
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g6_ffn_w1_bias_descriptor.ps1`
- `scripts/local/run_p11g6_ffn_fallback_observability.ps1`
- `tb/tb_g6_ffn_w1_bias_descriptor_p11g6a.cpp`
- `tb/tb_g6_ffn_fallback_observability_p11g6b.cpp`
- `tb/tb_g5_ffn_w1_fallback_policy_p11g5w1fp.cpp`

3. Exact commands run
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

4. Actual execution evidence / log excerpt
- `build/p11g6/ffn_w1_bias_descriptor/run.log`:
  - `G6FFN_SUBWAVE_A_W1_BIAS_TOPFED_PATH PASS`
  - `G6FFN_SUBWAVE_A_W1_BIAS_NO_SPURIOUS_TOUCH PASS`
  - `G6FFN_SUBWAVE_A_W1_BIAS_EXPECTED_COMPARE PASS`
  - `PASS: run_p11g6_ffn_w1_bias_descriptor`
- `build/p11g6/ffn_fallback_observability/run.log`:
  - `G6FFN_SUBWAVE_B_REJECT_STAGE_W1 PASS`
  - `G6FFN_SUBWAVE_B_REJECT_STAGE_W2 PASS`
  - `G6FFN_SUBWAVE_B_NO_STALE_ON_REJECT PASS`
  - `G6FFN_SUBWAVE_B_NONSTRICT_FALLBACK_OBS PASS`
  - `PASS: run_p11g6_ffn_fallback_observability`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: G6 FFN W1 top-fed bias descriptor + reject-stage observability anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

5. Governance posture
- local-only bounded campaign.
- not Catapult closure; not SCVerify closure.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- FFN fallback still exists in compatibility/non-strict modes.
- FFN writeback remains SRAM-scratch-centric in bounded scope.
- Wave4 code change remains deferred (feasibility/blocker capture only this round).

7. Next recommended step
- Choose one dedicated Wave4 micro-cut task (single phase-entry descriptor probe) with an isolated ownership-focused TB before touching coupled inner loops.

## Task C14: W4-M1 Single Phase-Entry Caller-Fed Descriptor Probe (QK-score)
1. Summary
- Landed one bounded Wave4 micro-cut on QK-score phase entry.
- Added optional caller-fed descriptor probe passthrough on Top helper and phase-entry consume visibility check in QK-score mainline.
- Inner compute/writeback loops intentionally untouched.

2. Exact files changed
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m1_qkscore_phase_entry_probe.ps1`
- `tb/tb_w4m1_qkscore_phase_entry_probe.cpp`

3. Exact commands run
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

4. Actual execution evidence / log excerpt
- `build/p11w4m1/qkscore_phase_entry_probe/run.log`:
  - `W4M1_QKSCORE_CALLER_FED_DESCRIPTOR_VISIBLE PASS`
  - `W4M1_QKSCORE_OWNERSHIP_CHECK PASS`
  - `W4M1_QKSCORE_NO_SPURIOUS_TOUCH PASS`
  - `W4M1_QKSCORE_EXPECTED_COMPARE PASS`
  - `PASS: run_p11w4m1_qkscore_phase_entry_probe`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`:
  - `guard: W4-M1 QK-score phase-entry caller-fed descriptor probe anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

5. Governance posture
- local-only bounded micro-cut.
- not Catapult closure; not SCVerify closure.
- no external formal contract change.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- Wave4 payload path is not fully migrated.
- QK-score inner compute/writeback loops remain SRAM-centric.

7. Next recommended step
- Run W4-M2 bounded probe on softmax-out phase-entry (single V-tile descriptor probe + no-spurious/ownership TB).

## Task C15: W4-M2 Single Phase-Entry Caller-Fed V-tile Probe (SoftmaxOut)
1. Summary
- Landed one bounded Wave4 micro-cut on SoftmaxOut phase entry.
- Added optional caller-fed V-tile probe passthrough on Top helper and phase-entry consume visibility/ownership/compare checks in SoftmaxOut mainline.
- Inner online-softmax/reduction/writeback loops intentionally untouched.

2. Exact files changed
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m2_softmaxout_phase_entry_probe.ps1`
- `tb/tb_w4m2_softmaxout_phase_entry_probe.cpp`

3. Exact commands run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4m2_softmaxout_phase_entry_probe.ps1`
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
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11af_impl_softmax_out.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

4. Actual execution evidence / log excerpt
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

5. Governance posture
- local-only bounded micro-cut.
- not Catapult closure; not SCVerify closure.
- no external formal contract change.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- SoftmaxOut payload path is not fully migrated.
- SoftmaxOut inner compute/writeback loops remain SRAM-centric in this bounded scope.
- This task is a phase-entry probe bridge only.

7. Next recommended step
- Execute W4-M3 bounded Phase-A Q x-row caller-fed descriptor probe, reusing W4-M1/W4-M2 targeted validation pattern.

## Task C16: W4-M3 Feasibility / Blocker Refinement (No Code Patch)
1. Summary
- Completed W4-M3 feasibility/ranking/blocker refinement after W4-M2 landing.
- Chosen next entry preference:
  - Phase-A Q x-row caller-fed descriptor probe (`AttnPhaseATopManagedQ`).
- Optional W4-M3 micro-cut was intentionally not attempted to avoid over-expanding current bounded run.

2. Exact files changed
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_FEASIBILITY_20260330.md`

3. Exact commands run
- `rg -n "row_x_base|x_row|AttnPhaseATopManagedQ|AttnPhaseATopManagedKv|AttnLayer0" src/blocks`
- `rg -n "run_p11ae_layer0_top_managed_qk_score|run_p11af_layer0_top_managed_softmax_out" src/Top.h src/blocks`

4. Actual execution evidence / log excerpt
- See feasibility artifacts:
  - `docs/handoff/TOP_MANAGED_SRAM_W4M3_FEASIBILITY_20260330.md`
  - `build/agent_state/w4m2_softmaxout_probe_20260330/w4m3_reality_check.md`
  - `build/agent_state/w4m2_softmaxout_probe_20260330/w4m3_candidate_ranking.md`
  - `build/agent_state/w4m2_softmaxout_probe_20260330/w4m3_blocker_map.md`

5. Governance posture
- local-only feasibility and blocker capture.
- not Catapult closure; not SCVerify closure.

6. Residual risks
- Phase-A Q/KV entry remains SRAM-centric and coupled to downstream loops.
- Dedicated W4-M3 targeted TB/runner support is still missing.

7. Next recommended step
- Start a dedicated W4-M3 single-entry run focused on Phase-A Q x-row probe only, with ownership/no-spurious/mismatch-reject TB.

## Task C17: W4-M3 Single Phase-Entry Caller-Fed x-row Probe (Phase-A Q)
1. Summary
- Landed one bounded Wave4 micro-cut on Phase-A Q phase entry.
- Added optional caller-fed x-row probe passthrough on Top helper and phase-entry consume visibility/ownership/compare checks in Phase-A Q mainline.
- Inner Q compute/writeback loops intentionally untouched.

2. Exact files changed
- `src/blocks/AttnPhaseATopManagedQ.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m3_phasea_q_phase_entry_probe.ps1`
- `tb/tb_w4m3_phasea_q_phase_entry_probe.cpp`

3. Exact commands run
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

4. Actual execution evidence / log excerpt
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

5. Governance posture
- local-only bounded micro-cut.
- not Catapult closure; not SCVerify closure.
- no external formal contract change.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- Phase-A Q payload path is not fully migrated.
- Phase-A Q inner compute/writeback loops remain SRAM-centric in this bounded scope.
- This task is a phase-entry probe bridge only.

7. Next recommended step
- Execute bounded Phase-A KV x-row probe with the same ownership/no-spurious/mismatch-reject pattern.

## Task C18: W4-M3 KV Feasibility / Blocker Refinement (No Code Patch)
1. Summary
- Completed W4-M3 KV feasibility/ranking/blocker refinement in secondary track.
- Optional KV micro-cut intentionally not attempted in this run to preserve bounded focus on Phase-A Q landing.

2. Exact files changed
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_FEASIBILITY_20260330.md`

3. Exact commands run
- `rg -n "attn_phasea_top_managed_kv_mainline|ATTN_P11AC_MAINLINE_XROW_LOAD_LOOP|fallback_taken" src/blocks/AttnPhaseATopManagedKv.h`
- `rg -n "run_p11ac_layer0_top_managed_kv|run_p11ad_layer0_top_managed_q" src/Top.h`

4. Actual execution evidence / log excerpt
- See feasibility artifacts:
  - `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_FEASIBILITY_20260330.md`
  - `build/agent_state/w4m3_phasea_q_probe_20260330/w4m3_kv_reality_check.md`
  - `build/agent_state/w4m3_phasea_q_probe_20260330/w4m3_kv_candidate_ranking.md`
  - `build/agent_state/w4m3_phasea_q_probe_20260330/w4m3_kv_blocker_map.md`

5. Governance posture
- local-only feasibility and blocker capture.
- not Catapult closure; not SCVerify closure.

6. Residual risks
- KV entry remains SRAM-centric and coupled to dual K/V compute flow.
- Dedicated KV targeted TB/runner support is still missing.

7. Next recommended step
- Start a dedicated W4-M3 KV single-entry run with one focused probe and one focused reject/no-spurious TB.

## Task C19: W4-M3 KV Dedicated Phase-Entry Caller-Fed Probe
1. Summary
- Landed one bounded Wave4 micro-cut on Phase-A KV phase entry.
- Added optional caller-fed x-row probe passthrough on Top helper and phase-entry consume visibility/ownership/compare checks in Phase-A KV mainline.
- Inner K/V compute/writeback loops intentionally untouched.

2. Exact files changed
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

3. Exact commands run
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

4. Actual execution evidence / log excerpt
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

5. Governance posture
- local-only bounded micro-cut.
- not Catapult closure; not SCVerify closure.
- no external formal contract change.
- remote simulator/site-local PLI line untouched.

6. Residual risks
- Phase-A KV payload path is not fully migrated.
- K/V inner compute/writeback loops remain SRAM-centric in this bounded scope.
- This task is a phase-entry probe bridge only.

7. Next recommended step
- Keep Wave4 progression bounded: choose one additional entrypoint with dedicated owner mismatch/no-spurious/mismatch-reject TB, without expanding into compute/writeback rewrites.

## G7 addendum (direct-SRAM eradication bounded waves)
- Wave A (completed): FFN residual fallback tightening
  - strict W1 now requires top-fed bias descriptor-ready in addition to x/weight.
  - targeted runner: `scripts/local/run_p11g7_ffn_w1_bias_descriptor_strict.ps1`
- Wave B (completed): W4-M3 KV probe hardening
  - phase-entry probe now requires full-row descriptor-ready words.
  - targeted runner: `scripts/local/run_p11w4m3_kv_phase_entry_probe.ps1`
- Wave C (completed): SramView remove-readiness matrix
  - output: `docs/handoff/TOP_MANAGED_SRAM_G7_REMOVE_READINESS_20260330.md`
- Wave D (completed): residual blocker isolation and next-cut map
  - output: `docs/handoff/TOP_MANAGED_SRAM_G7_DIRECT_SRAM_CAMPAIGN_20260330.md`
