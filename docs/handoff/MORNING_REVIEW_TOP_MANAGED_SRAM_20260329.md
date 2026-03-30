# MORNING REVIEW TOP MANAGED SRAM (2026-03-29)

## Quick Goal
- Verify the overnight architecture-forward cuts that move SRAM ownership semantics toward Top-managed dispatch.
- Confirm no overclaim: local-only evidence; not Catapult closure; not SCVerify closure.
- Include the latest G5-Wave3.5 FFN W1 weight migration verification.
- Include the latest G4 ingest/base-shadow mincut verification in this review pass.
- Include the bounded G4-E cross-command metadata harmonization mincut verification.
- Include the bounded G4-F commit-time diagnostics harmonization verification.
- Include the bounded G4-G accepted-commit metadata record harmonization verification.

## Suggested 10-Minute Review Order (G5-Wave3.5 Latest)
1. Open `src/blocks/FFNLayer0.h` and `src/blocks/TransformerLayer.h` (2 min)
   - Confirm `FFNLayer0` has `topfed_w1_weight_words` consume anchor.
   - Confirm caller preloads `topfed_ffn_w1_words` and dispatches valid window.
2. Open `build/p11g5/wave35_ffn_w1_weight_migration/run.log` (1 min)
   - Confirm:
     - `G5W35_FFN_W1_TOPFED_WEIGHT_PATH PASS`
     - `G5W35_FFN_W1_NO_SPURIOUS_SRAM_TOUCH PASS`
     - `G5W35_FFN_W1_EXPECTED_COMPARE PASS`
3. Open `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log` (1 min)
   - Confirm `guard: G5 wave3.5 FFN W1 top-fed weight payload migration anchors OK`.
4. Open `build/p11g5/wave3_ffn_payload_migration/run.log` (1 min)
   - Confirm Wave3 x-payload regression remains PASS.
5. Open `build/p11ah/g5_wave35_ffn_w1/run.log` and `build/p11aj/g5_wave35_ffn_w1/run.log` (2 min)
   - Confirm mainline/provenance chain remains PASS after Wave3.5 patch.
6. Open `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_FFN_W1_WEIGHT_20260330.md` (1 min)
   - Confirm bounded scope and fallback boundaries.
7. Open `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_EVIDENCE_INDEX_20260330.md` (1 min)
   - Confirm claim-to-evidence mapping.
8. Open `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_COMPLETION_20260330.md` (1 min)
   - Confirm exact files/commands/evidence alignment.
9. Open `build/evidence/g5_wave35_ffn_w1_20260330/evidence_manifest.txt` (1 min)
   - Confirm bundle completeness and local-only posture.

## Previous 10-Minute Review Order (G4-G Reference)
1. Open `src/Top.h` G4-G accepted-record diff (2 min)
   - Confirm `AcceptedCommitMetadataRecord` exists.
   - Confirm `record_accepted_commit_metadata(...)` and `ingest_commit_diag_and_record(...)` exist.
   - Confirm CFG/PARAM/INFER commit-success paths update the accepted record.
2. Open `build/p11g4g/accept_commit_record/run.log` (1 min)
   - Confirm:
     - `G4G_ACCEPT_RECORD_CFG_DETERMINISTIC PASS`
     - `G4G_ACCEPT_RECORD_PARAM_DETERMINISTIC PASS`
     - `G4G_REJECT_NO_STALE_STATE PASS`
     - `G4G_ACCEPT_RECORD_INFER_PHASE_VALID PASS`
     - `PASS: run_p11g4g_accept_commit_record`
3. Open `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log` (1 min)
   - Confirm `guard: G4-G accepted-commit metadata record harmonization anchors OK`.
4. Open `build/p11ah/g4g_accept_commit/run.log` and `build/p11aj/g4g_accept_commit/run.log` (2 min)
   - Confirm mainline/provenance chain remains PASS after G4-G patch.
5. Open `docs/handoff/TOP_MANAGED_SRAM_G4G_ACCEPT_COMMIT_MINCUT_20260330.md` (1 min)
   - Confirm bounded scope and residual-risk boundary.
6. Open `docs/handoff/TOP_MANAGED_SRAM_G4G_EVIDENCE_INDEX_20260330.md` (1 min)
   - Confirm claim-to-evidence mapping and manifest path.
7. Open `docs/handoff/TOP_MANAGED_SRAM_G4G_COMPLETION_20260330.md` (1 min)
   - Confirm exact files/commands/evidence are aligned.
8. Open `build/evidence/g4g_accept_commit_20260330/evidence_manifest.txt` (1 min)
   - Confirm bundle contains all logs referenced by completion and evidence index.
9. Re-check `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md` and `TOP_MANAGED_SRAM_MINCUTS_20260329.md` (1 min)
   - Confirm G4-D/G4-E/G4-F/G4-G boundary wording is consistent and non-overclaim.

## Previous 10-Minute Review Order (G4-F Reference)
1. Open `src/Top.h` G4-F commit-time helper diff (2 min)
   - Confirm `ingest_meta_len_exact(...)` and `ingest_commit_diag_error(...)` exist.
   - Confirm CFG/PARAM/INFER commit-time paths call `ingest_commit_diag_error(...)`.
2. Open `build/p11g4f/commit_diagnostics_negative/run.log` (1 min)
   - Confirm:
     - `G4F_CFG_LEN_MISMATCH_MAPPING PASS`
     - `G4F_PARAM_LEN_MISMATCH_MAPPING PASS`
     - `G4F_OWNER_RX_MISMATCH_MAPPING PASS`
     - `G4F_SPAN_MISMATCH_MAPPING PASS`
     - `PASS: run_p11g4f_commit_diagnostics_negative`
3. Open `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log` (1 min)
   - Confirm `guard: G4-F commit-time diagnostics helper + error mapping anchors OK`.
4. Open `build/p11ah/g4f_commit_diag/run.log` and `build/p11aj/g4f_commit_diag/run.log` (2 min)
   - Confirm mainline/provenance chain remains PASS after G4-F patch.
5. Open `docs/handoff/TOP_MANAGED_SRAM_G4F_COMMIT_DIAGNOSTICS_MINCUT_20260329.md` (1 min)
   - Confirm bounded scope and residual-risk boundary.
6. Open `docs/handoff/TOP_MANAGED_SRAM_G4F_EVIDENCE_INDEX_20260329.md` (1 min)
   - Confirm claim-to-evidence mapping and manifest path.
7. Open `docs/handoff/TOP_MANAGED_SRAM_G4F_COMPLETION_20260329.md` (1 min)
   - Confirm exact files/commands/evidence are aligned.
8. Open `build/evidence/g4f_commit_diag_20260330/evidence_manifest.txt` (1 min)
   - Confirm bundle contains all logs referenced by completion and evidence index.
9. Re-check `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md` and `TOP_MANAGED_SRAM_MINCUTS_20260329.md` (1 min)
   - Confirm G4-D/G4-E/G4-F boundary wording is consistent and non-overclaim.

## Previous 10-Minute Review Order (G4-D/G4-E Reference)
1. Open `src/Top.h` diff for G4 ingest patch (2 min)
   - Confirm `InferIngestContract` exists and is armed at `OP_INFER`.
   - Confirm infer contract now carries `phase_id/token_range/tile_range`.
   - Confirm `OP_INFER` checks metadata-surface span helper before entering `ST_INFER_RX`.
   - Confirm `run_preproc_block` uses ingest contract `in_base_word`/`len_words_valid`.
   - Confirm FinalHead label source comes from SRAM ingest view (`infer_label_words_view`), not shadow source.
2. Open `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log` (1 min)
   - Confirm `guard: G4 infer ingest contractized base/len dispatch anchors OK`.
3. Open `build/p11ah/g4_night_batch/run.log` and `build/p11aj/g4_night_batch/run.log` (2 min)
   - Confirm mainline/fallback/compare PASS anchors.
4. Open `docs/handoff/TOP_MANAGED_SRAM_ARCH_GAP_INVENTORY_20260329.md` (2 min)
   - Confirm G4-A/B/C/D moved to DONE and only G4-E remains deferred.
5. Open `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md` Task C3 (2 min)
   - Confirm exact commands and evidence chain.
6. Open `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md` extension section (1 min)
   - Confirm deferred boundary and no overclaim posture.
7. Open `docs/handoff/TOP_MANAGED_SRAM_G4_HARDENING_EVIDENCE_INDEX_20260329.md` (1 min)
   - Confirm targeted negative preflight validation evidence is present and mapped.
8. Open `docs/handoff/TOP_MANAGED_SRAM_G4E_METADATA_MINCUT_20260329.md` (1 min)
   - Confirm bounded harmonization scope and non-overclaim boundary.
9. Open `build/p11g4e/cross_command_metadata_negative/run.log` (1 min)
   - Confirm cross-command mismatch reject anchors PASS.

## Prior-Round Review Order (reference)
1. Open `docs/handoff/TOP_MANAGED_SRAM_ARCH_GAP_INVENTORY_20260329.md` (2 min)
   - Confirm selected cuts C1/C2 and why broader candidates were deferred.
2. Open `src/Top.h` diff (3 min)
   - Confirm Top now assembles contracts for preproc/LN/final-head and dispatches core entries.
   - Confirm Top preloads sublayer1 norm params before Transformer layer dispatch.
3. Open `src/blocks/TransformerLayer.h` diff (1 min)
   - Confirm new preload flag and guarded fallback behavior.
4. Open `build/p11ah/top_managed_sram_push/run.log` and `build/p11aj/top_managed_sram_push/run.log` (2 min)
   - Check mainline/fallback and compare PASS anchors.
5. Open `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log` (1 min)
   - Confirm anti-regression anchors PASS.
6. Open `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md` (1 min)
   - Confirm residual risks and next-step recommendation.

## Most Important Diffs For Architecture Review
- `src/Top.h`
  - `run_preproc_block(...)`
  - `run_layernorm_block(...)`
  - `run_mid_or_end_layernorm(...)`
  - `run_transformer_layer_loop(...)`
  - `run_transformer_layer_loop_top_managed_attn_bridge(...)`
  - `run_infer_pipeline(...)`
- `src/blocks/TransformerLayer.h`
  - `TransformerLayerTopManagedAttnBridge(...)`
  - `TransformerLayer(...)`

## Accepted / Likely-Good
- Top-managed contract dispatch anchors are in place and guarded.
- Layer submodule now supports Top preloaded norm-param ownership handoff.
- G4 infer ingest path now has explicit Top contractized base/len dispatch anchors.
- G4 infer ingest path now has explicit Top contractized base/len/window dispatch anchors.
- Existing helper-channel regression guard remains PASS.

## Residual Risks Needing Human Attention
- Compatibility wrappers still exist and could be used by legacy callsites.
- Default infer path is still pointer-facing orchestration (not fully switched to deep bridge).
- Ingest base/shadow architecture remains for a future dedicated rework pack.

## Artifact Index
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g4_infer_ingest_preflight_negative.ps1`
- `scripts/local/run_p11g4e_cross_command_metadata_negative.ps1`
- `scripts/local/run_p11g4f_commit_diagnostics_negative.ps1`
- `scripts/local/run_p11g4g_accept_commit_record.ps1`
- `scripts/local/run_p11g5_wave1_payload_migration.ps1`
- `scripts/local/run_p11g5_wave2_preproc_payload_migration.ps1`
- `scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`
- `scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`
- `tb/tb_infer_ingest_preflight_negative_p11g4.cpp`
- `tb/tb_g4e_cross_command_metadata_negative_p11g4e.cpp`
- `tb/tb_g4f_commit_diagnostics_negative_p11g4f.cpp`
- `tb/tb_g4g_accept_commit_record_p11g4g.cpp`
- `tb/tb_g5_wave1_payload_migration_p11g5w1.cpp`
- `tb/tb_g5_wave2_preproc_payload_migration_p11g5w2.cpp`
- `tb/tb_g5_wave3_ffn_payload_migration_p11g5w3.cpp`
- `tb/tb_g5_wave35_ffn_w1_weight_migration_p11g5w35.cpp`
- `docs/handoff/TOP_MANAGED_SRAM_ARCH_GAP_INVENTORY_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4E_METADATA_MINCUT_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4E_EVIDENCE_INDEX_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4E_COMPLETION_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4F_COMMIT_DIAGNOSTICS_MINCUT_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4F_EVIDENCE_INDEX_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4F_COMPLETION_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4G_ACCEPT_COMMIT_MINCUT_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4G_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4G_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_DIRECT_PAYLOAD_INVENTORY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE3_FFN_PAYLOAD_MIGRATION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE3_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE3_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_FFN_W1_WEIGHT_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_HARDENING_EVIDENCE_INDEX_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_HARDENING_COMPLETION_20260329.md`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`
- `build/p11ah/g4_night_batch/run.log`
- `build/p11aj/g4_night_batch/run.log`
- `build/p11g4/infer_ingest_preflight_negative/run.log`
- `build/p11g4e/cross_command_metadata_negative/run.log`
- `build/p11g4f/commit_diagnostics_negative/run.log`
- `build/p11g4g/accept_commit_record/run.log`
- `build/p11g5/wave1_payload_migration/run.log`
- `build/p11g5/wave2_preproc_payload_migration/run.log`
- `build/p11g5/wave3_ffn_payload_migration/run.log`
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
- `build/p11ah/top_managed_sram_push/run.log`
- `build/p11aj/top_managed_sram_push/run.log`
- `build/p11ah/g5_payload_campaign/run.log`
- `build/p11aj/g5_payload_campaign/run.log`
- `build/p11ah/g5_wave3_ffn/run.log`
- `build/p11aj/g5_wave3_ffn/run.log`
- `build/p11ah/g5_wave35_ffn_w1/run.log`
- `build/p11aj/g5_wave35_ffn_w1/run.log`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
- `build/evidence/g4_hardening_20260329/evidence_manifest.txt`
- `build/evidence/g4e_metadata_mincut_20260329/evidence_manifest.txt`
- `build/evidence/g4f_commit_diag_20260330/evidence_manifest.txt`
- `build/evidence/g4g_accept_commit_20260330/evidence_manifest.txt`
- `build/evidence/g5_payload_campaign_20260330/evidence_manifest.txt`
- `build/evidence/g5_wave3_ffn_20260330/evidence_manifest.txt`
- `build/evidence/g5_wave35_ffn_w1_20260330/evidence_manifest.txt`

## Copy-Ready GPT Review Prompt
- `Please review src/Top.h and src/blocks/TransformerLayer.h for Top-managed SRAM ownership convergence. Focus on: (1) whether Top now builds and owns preproc/layernorm/final-head contracts in active infer path, (2) whether Transformer sublayer1 norm params are preloaded by Top with safe fallback, and (3) whether local evidence (p11ah/p11aj + boundary guard) supports acceptance without overclaim. Keep posture local-only and do not claim Catapult/SCVerify closure.`

## Suggested 10-Minute Review Order (G5 FFN Closure Campaign Latest)
1. Open `src/blocks/FFNLayer0.h` and `src/blocks/TransformerLayer.h` (2 min)
   - Confirm W2 input/weight/bias top-fed consume anchors and stage-split FFN dispatch.
2. Open `build/p11g5/ffn_closure_campaign/run.log` (1 min)
   - Confirm:
     - `G5FFN_SUBWAVE_A_W2_INPUT_TOPFED_PATH PASS`
     - `G5FFN_SUBWAVE_B_W2_WEIGHT_TOPFED_PATH PASS`
     - `G5FFN_SUBWAVE_C_W2_BIAS_TOPFED_PATH PASS`
     - `G5FFN_SUBWAVE_D_FALLBACK_BOUNDARY PASS`
3. Open `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log` (1 min)
   - Confirm `guard: G5 FFN closure campaign W2 top-fed input/weight/bias anchors OK`.
4. Open `build/p11g5/wave3_ffn_payload_migration/run.log` and `build/p11g5/wave35_ffn_w1_weight_migration/run.log` (1 min)
   - Confirm wave3/wave3.5 regression remains PASS.
5. Open `build/p11ah/g5_ffn_closure_campaign/run.log` and `build/p11aj/g5_ffn_closure_campaign/run.log` (2 min)
   - Confirm mainline/provenance chain remains PASS.
6. Open `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_CLOSURE_CAMPAIGN_20260330.md` (1 min)
   - Confirm bounded scope and deferred boundaries.
7. Open `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_CLOSURE_EVIDENCE_INDEX_20260330.md` (1 min)
   - Confirm claim-to-evidence mapping.
8. Open `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_CLOSURE_COMPLETION_20260330.md` (1 min)
   - Confirm exact files/commands/evidence alignment.
9. Open `build/evidence/g5_ffn_closure_campaign_20260330/evidence_manifest.txt` (1 min)
   - Confirm bundle completeness and local-only posture.

## Latest artifact additions (G5 FFN closure campaign)
- `scripts/local/run_p11g5_ffn_closure_campaign.ps1`
- `tb/tb_g5_ffn_closure_campaign_p11g5fc.cpp`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_CLOSURE_CAMPAIGN_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_CLOSURE_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_CLOSURE_COMPLETION_20260330.md`
- `build/p11g5/ffn_closure_campaign/run.log`
- `build/evidence/g5_ffn_closure_campaign_20260330/evidence_manifest.txt`

## Suggested 10-Minute Review Order (G5 FFN Fallback Policy Latest)
1. Open `src/blocks/FFNLayer0.h` (2 min)
   - Confirm strict policy anchors:
     - `FFN_POLICY_REQUIRE_W2_TOPFED`
     - `fallback_policy_reject_flag`
     - `fallback_legacy_touch_counter`
     - descriptor-ready gating before W2 loop.
2. Open `src/blocks/TransformerLayer.h` (1 min)
   - Confirm W2 stage call passes `(u32_t)FFN_POLICY_REQUIRE_W2_TOPFED`.
3. Open `build/p11g5/ffn_fallback_policy/run.log` (1 min)
   - Confirm:
     - `G5FFN_FALLBACK_POLICY_TOPFED_PRIMARY PASS`
     - `G5FFN_FALLBACK_POLICY_CONTROLLED_FALLBACK PASS`
     - `G5FFN_FALLBACK_POLICY_NO_STALE_STATE PASS`
     - `G5FFN_FALLBACK_POLICY_EXPECTED_COMPARE PASS`
4. Open `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log` (1 min)
   - Confirm `guard: G5 FFN fallback policy strict W2 top-fed gating anchors OK`.
5. Open `build/p11g5/ffn_closure_campaign/run.log`, `build/p11g5/wave3_ffn_payload_migration/run.log`, `build/p11g5/wave35_ffn_w1_weight_migration/run.log` (2 min)
   - Confirm regressions remain PASS.
6. Open `build/p11ah/full_loop/run.log` and `build/p11aj/p11aj/run.log` (1 min)
   - Confirm mainline/provenance PASS.
7. Open `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_FALLBACK_POLICY_20260330.md` (1 min)
8. Open `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_FALLBACK_EVIDENCE_INDEX_20260330.md` (1 min)
9. Open `build/evidence/g5_ffn_fallback_policy_20260330/evidence_manifest.txt` (1 min)

## Latest artifact additions (G5 FFN fallback policy)
- `scripts/local/run_p11g5_ffn_fallback_policy.ps1`
- `tb/tb_g5_ffn_fallback_policy_p11g5fp.cpp`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_FALLBACK_POLICY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_FALLBACK_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_FALLBACK_COMPLETION_20260330.md`
- `build/p11g5/ffn_fallback_policy/run.log`
- `build/evidence/g5_ffn_fallback_policy_20260330/evidence_manifest.txt`
