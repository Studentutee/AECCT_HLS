# TOP MANAGED SRAM PROGRESS (2026-03-29)

## Executive Summary
- Overnight focus shifted from helper-only cleanup to architecture-forward boundary convergence.
- Completed two minimal cuts that push ownership semantics toward "Top allocates and dispatches, blocks consume":
  1) Top-owned contract dispatch for preproc / initial layernorm / final-head active infer path.
  2) Top-owned preload of Transformer sublayer1 norm params with block-side guarded compatibility fallback.
- Added a dedicated regression guard for this boundary to prevent silent rollback.

## Completed Architecture-Forward Tasks
1. Built and published architecture gap inventory with fixability ranking.
   - `docs/handoff/TOP_MANAGED_SRAM_ARCH_GAP_INVENTORY_20260329.md`
2. Implemented C1 Top-owned contract dispatch in active infer path.
   - `src/Top.h`
3. Implemented C2 Top preload of sublayer1 norm params and guarded block fallback.
   - `src/Top.h`
   - `src/blocks/TransformerLayer.h`
4. Added and executed Top-managed SRAM boundary regression checker.
   - `scripts/check_top_managed_sram_boundary_regression.ps1`
   - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
5. Re-ran local evidence chain and hygiene checks.
   - `build/p11ah/top_managed_sram_push/run.log`
   - `build/p11aj/top_managed_sram_push/run.log`

## Tasks Attempted But Blocked
- No final blocker.
- One transient compile mismatch occurred during first `p11aj` attempt after function signature tightening:
  - symptom: TB callsite expected legacy `run_mid_or_end_layernorm` signature
  - action: added compatibility overload path via optional contract pointer in `src/Top.h`
  - rerun result: PASS

## Validation Summary
- `run_p11ah_full_loop_local_e2e`: PASS
- `run_p11aj_top_managed_sram_provenance`: PASS
- `check_top_managed_sram_boundary_regression`: PASS
- `check_helper_channel_split_regression`: PASS
- `check_p11ap_active_chain_residual_rawptr`: PASS
- `check_design_purity`: PASS
- `check_repo_hygiene -Phase pre`: PASS
- `check_agent_tooling`: PASS

## Exact Artifact Index
- Code:
  - `src/Top.h`
  - `src/blocks/TransformerLayer.h`
  - `scripts/check_top_managed_sram_boundary_regression.ps1`
- Handoff:
  - `docs/handoff/TOP_MANAGED_SRAM_ARCH_GAP_INVENTORY_20260329.md`
  - `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
  - `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
  - `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`
- Evidence:
  - `build/p11ah/top_managed_sram_push/build.log`
  - `build/p11ah/top_managed_sram_push/run.log`
  - `build/p11aj/top_managed_sram_push/build.log`
  - `build/p11aj/top_managed_sram_push/run.log`
  - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
  - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression_summary.txt`

## Governance Posture
- Local-only progress and evidence.
- Top remains sole production shared-SRAM owner at active boundaries touched in this round.
- not Catapult closure.
- not SCVerify closure.

## Night-Batch Extension: G4 Ingest/Base-Shadow Push
- Added Top-managed infer ingest contractization in active path:
  - `InferIngestContract` in `src/Top.h`
  - `run_preproc_block` consumes ingest contract base/valid-length/window range
  - `run_infer_pipeline` FinalHead label source routed to Top-managed SRAM ingest view
  - `OP_INFER` now performs contract span preflight validation before RX state entry
- Expanded boundary regression guard:
  - `scripts/check_top_managed_sram_boundary_regression.ps1` now enforces G4 ingest anchors.

### Additional local evidence (this batch)
- `build/p11ah/g4_night_batch/run.log`: PASS
- `build/p11aj/g4_night_batch/run.log`: PASS
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS

### Deferred boundary
- Full cross-command unified ingest rearchitecture (CFG/PARAM/INFER metadata harmonization) remains deferred.

## G4 Hardening Closeout (hygiene/evidence/consistency)
- This closeout round does not add a broad architecture rewrite.
- Focus is acceptance hardening for already-landed G4 mincut:
  - completion report hygiene,
  - handoff consistency,
  - evidence bundle completeness,
  - targeted preflight negative validation.

### Exact files changed (closeout round)
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g4_infer_ingest_preflight_negative.ps1`
- `tb/tb_infer_ingest_preflight_negative_p11g4.cpp`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_INGEST_MINCUT_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_HARDENING_EVIDENCE_INDEX_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_HARDENING_COMPLETION_20260329.md`

### Additional local-only evidence (closeout round)
- `build/evidence/g4_hardening_20260329/evidence_manifest.txt`
- `build/p11g4/infer_ingest_preflight_negative/run.log`: PASS
  - `PREFLIGHT_INVALID_BASE_REJECT PASS`
  - `PREFLIGHT_INVALID_SPAN_REJECT PASS`
  - `PREFLIGHT_ERR_MEM_RANGE_GUARD_BEHAVIOR PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: OP_INFER preflight reject path maps invalid span to ERR_MEM_RANGE`
- `check_repo_hygiene -Phase post`: PASS (captured in evidence bundle)

### Side-change disposition
- `AECCT_ac_ref/include/RefPrecisionMode.h`
- `AECCT_ac_ref/src/RefModel.cpp`
- `AECCT_ac_ref/src/ref_main.cpp`
- Current repo reality check found no active diff on these paths in this closeout round; they are excluded from G4 acceptance payload.

### Posture
- local-only evidence
- not Catapult closure
- not SCVerify closure
- remote PLI / site-local simulator line not touched

## Night-Batch Extension: G4-E Bounded Metadata Harmonization
- Added a bounded Top-only cross-command metadata surface:
  - `IngestMetadataSurface` in `src/Top.h`
  - shared helper anchors:
    - `ingest_meta_expected_words(...)`
    - `ingest_meta_span_in_sram(...)`
    - `ingest_meta_owner_matches_rx(...)`
  - metadata surface builders:
    - `cfg_metadata_surface(...)`
    - `param_metadata_surface(...)`
    - `infer_metadata_surface(...)`
- Updated preflight callsites to consume harmonized metadata surface:
  - OP_LOAD_W preflight now uses `param_ingest_span_legal(regs)`
  - OP_INFER preflight now uses `infer_metadata_surface(regs)` + `ingest_meta_span_in_sram(...)`
- Added G4-E targeted mismatch validation:
  - `scripts/local/run_p11g4e_cross_command_metadata_negative.ps1`
  - `tb/tb_g4e_cross_command_metadata_negative_p11g4e.cpp`

### G4-E local evidence (this batch)
- `build/p11g4e/cross_command_metadata_negative/run.log`: PASS
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G4-E cross-command ingest metadata surface helpers anchored`
- `build/p11ah/g4e_metadata_mincut/run.log`: PASS
- `build/p11aj/g4e_metadata_mincut/run.log`: PASS

### G4-E artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G4E_METADATA_MINCUT_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4E_EVIDENCE_INDEX_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4E_COMPLETION_20260329.md`
- `build/evidence/g4e_metadata_mincut_20260329/evidence_manifest.txt`

### Deferred boundary after G4-E mincut
- Full cross-command ingest metadata semantic unification (beyond helper/preflight surface) remains deferred.

## Night-Batch Extension: G4-F Commit-Time Diagnostics Harmonization
- Delivered a bounded Top-only commit-time diagnostics harmonization mincut for CFG/PARAM/INFER.
- Added shared helper anchors in `src/Top.h`:
  - `ingest_meta_len_exact(...)`
  - `ingest_commit_diag_error(...)`
- Commit-time callsite convergence:
  - CFG commit path now uses shared helper and preserves `ERR_CFG_LEN_MISMATCH`.
  - PARAM ingest-completion commit path now uses shared helper with `ERR_PARAM_LEN_MISMATCH`.
  - INFER ingest-completion commit path now uses shared helper while preserving existing protocol mapping constraints.
- Added G4-F targeted mismatch negative validation and kept mainline/provenance regression PASS.

### G4-F local-only evidence (this batch)
- `build/p11g4f/commit_diagnostics_negative/run.log`: PASS
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G4-F commit-time diagnostics helper + error mapping anchors OK`
- `build/p11ah/g4f_commit_diag/run.log`: PASS
- `build/p11aj/g4f_commit_diag/run.log`: PASS
- `build/evidence/g4f_commit_diag_20260330/evidence_manifest.txt`: present

### G4-F artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G4F_COMMIT_DIAGNOSTICS_MINCUT_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4F_EVIDENCE_INDEX_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4F_COMPLETION_20260329.md`
- `scripts/local/run_p11g4f_commit_diagnostics_negative.ps1`
- `tb/tb_g4f_commit_diagnostics_negative_p11g4f.cpp`

### Deferred boundary after G4-F mincut
- This round harmonizes commit-time diagnostics helper and mapping anchors only.
- Protocol-level infer-specific length mismatch code remains deferred to avoid external contract change in this bounded round.

## Night-Batch Extension: G4-G Accepted-Commit Metadata Record Harmonization
- Added bounded Top-local accepted-commit metadata record convergence for CFG/PARAM/INFER:
  - `AcceptedCommitMetadataRecord`
  - `record_accepted_commit_metadata(...)`
  - `ingest_commit_diag_and_record(...)`
- Commit-success path convergence:
  - CFG commit success writes accepted record after diagnostics + cfg legality pass.
  - PARAM commit success writes accepted record via shared diag-and-record helper.
  - INFER commit success writes accepted record via shared diag-and-record helper with phase metadata.
- Reject path preserves previous accepted record (no partial overwrite from failed attempt).

### G4-G local-only evidence (this batch)
- `build/p11g4g/accept_commit_record/run.log`: PASS
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G4-G accepted-commit metadata record harmonization anchors OK`
- `build/p11ah/g4g_accept_commit/run.log`: PASS
- `build/p11aj/g4g_accept_commit/run.log`: PASS
- `build/evidence/g4g_accept_commit_20260330/evidence_manifest.txt`: present

### G4-G artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G4G_ACCEPT_COMMIT_MINCUT_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4G_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4G_COMPLETION_20260330.md`
- `scripts/local/run_p11g4g_accept_commit_record.ps1`
- `tb/tb_g4g_accept_commit_record_p11g4g.cpp`

### Deferred boundary after G4-G mincut
- G4-G is bounded to Top-local accepted-commit record harmonization only.
- External protocol-level committed-record export remains deferred.

## Night-Batch Extension: G5 Remaining Direct-SRAM Payload Migration (Wave Campaign)
- Campaign goal: push remaining payload ownership/dispatch closer to Top-managed model without broad rewrite.
- Completed bounded waves in this batch:
  - Wave 1: `LayerNormBlock` + `FinalHead` top-fed payload input path.
  - Wave 2: `PreprocEmbedSPE` top-fed infer-input payload path.
- Guard extended to enforce new G5 anchors in Top/block call paths.

### G5 local-only evidence (this batch)
- `build/p11g5/wave1_payload_migration/run.log`: PASS
- `build/p11g5/wave2_preproc_payload_migration/run.log`: PASS
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G5 wave1/wave2 top-fed payload migration anchors OK`
- `build/p11ah/g5_payload_campaign/run.log`: PASS
- `build/p11aj/g5_payload_campaign/run.log`: PASS
- `build/evidence/g5_payload_campaign_20260330/evidence_manifest.txt`: present

### G5 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G5_DIRECT_PAYLOAD_INVENTORY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_EVIDENCE_INDEX_20260330.md`
- `src/Top.h`
- `src/blocks/LayerNormBlock.h`
- `src/blocks/FinalHead.h`
- `src/blocks/PreprocEmbedSPE.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_wave1_payload_migration.ps1`
- `scripts/local/run_p11g5_wave2_preproc_payload_migration.ps1`
- `tb/tb_g5_wave1_payload_migration_p11g5w1.cpp`
- `tb/tb_g5_wave2_preproc_payload_migration_p11g5w2.cpp`

### Deferred boundary after G5 wave round
- Wave 3 (`FFNLayer0`) and Wave 4 (`AttnLayer0`/`TransformerLayer`/phase blocks) remain open due coupling and change-budget limits.
- This round does not claim full direct-SRAM payload elimination across all blocks.

## Night-Batch Extension: G5-Wave3 FFNLayer0 Payload Migration (Bounded)
- Scope-limited to `FFNLayer0` only.
- Completed bounded primary cut:
  - caller (`TransformerLayer`) preloads FFN input payload window (`topfed_ffn_x_words`),
  - `FFNLayer0` W1 tile load consumes `topfed_x_words` when provided.
- Preserved compatibility fallback to legacy SRAM x read when topfed pointer is absent.

### G5-Wave3 local-only evidence
- `build/p11g5/wave3_ffn_payload_migration/run.log`: PASS
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G5 wave3 FFN top-fed payload migration anchors OK`
- `build/p11ah/g5_wave3_ffn/run.log`: PASS
- `build/p11aj/g5_wave3_ffn/run.log`: PASS
- `build/evidence/g5_wave3_ffn_20260330/evidence_manifest.txt`: present

### G5-Wave3 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE3_FFN_PAYLOAD_MIGRATION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE3_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE3_COMPLETION_20260330.md`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`
- `tb/tb_g5_wave3_ffn_payload_migration_p11g5w3.cpp`

### Deferred boundary after G5-Wave3
- This round migrates FFN W1 input payload consume anchor only.
- FFN W1/W2 weight+bias consume and ReLU/W2 payload paths remain SRAM-based and deferred.
- Wave4 attention/transformer/phase payload migration remains deferred.

## Night-Batch Extension: G5-Wave3.5 FFN W1 Weight Tile Migration (Bounded)
- Scope-limited to FFN W1 weight consume path.
- Completed bounded cut:
  - caller (`TransformerLayer` and bridge path) preloads `topfed_ffn_w1_words`,
  - `FFNLayer0` W1 tile loop consumes `topfed_w1_weight_words` with valid window.
- Preserved Wave3 `topfed_x_words` path and compatibility fallback.

### G5-Wave3.5 local-only evidence
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`: PASS
- `build/p11g5/wave3_ffn_payload_migration/run.log`: PASS
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G5 wave3.5 FFN W1 top-fed weight payload migration anchors OK`
- `build/p11ah/g5_wave35_ffn_w1/run.log`: PASS
- `build/p11aj/g5_wave35_ffn_w1/run.log`: PASS
- `build/evidence/g5_wave35_ffn_w1_20260330/evidence_manifest.txt`: present

### G5-Wave3.5 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_FFN_W1_WEIGHT_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_WAVE35_COMPLETION_20260330.md`
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`
- `tb/tb_g5_wave35_ffn_w1_weight_migration_p11g5w35.cpp`

### Deferred boundary after G5-Wave3.5
- This round migrates FFN W1 weight tile consume anchor only.
- W2 weights, bias path, and broader FFN payload descriptorization remain deferred.
- Wave4 attention/phase migration remains deferred.

## Night-Batch Extension: G5 FFN Closure Campaign (Subwave A/B/C/D)
- Completed bounded FFN closure push focused on W2 core direct-SRAM paths.
- Implemented caller-fed/top-fed descriptors for:
  - W2 input activation payload,
  - W2 weight tile payload,
  - W2 bias payload.
- Preserved compatibility fallback as non-closure boundary.

### G5 FFN closure local-only evidence
- `build/p11g5/ffn_closure_campaign/run.log`: PASS
  - `G5FFN_SUBWAVE_A_W2_INPUT_TOPFED_PATH PASS`
  - `G5FFN_SUBWAVE_B_W2_WEIGHT_TOPFED_PATH PASS`
  - `G5FFN_SUBWAVE_C_W2_BIAS_TOPFED_PATH PASS`
  - `G5FFN_SUBWAVE_D_FALLBACK_BOUNDARY PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G5 FFN closure campaign W2 top-fed input/weight/bias anchors OK`
- `build/p11g5/wave3_ffn_payload_migration/run.log`: PASS
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`: PASS
- `build/p11ah/g5_ffn_closure_campaign/run.log`: PASS
- `build/p11aj/g5_ffn_closure_campaign/run.log`: PASS
- `build/evidence/g5_ffn_closure_campaign_20260330/evidence_manifest.txt`: present

### G5 FFN closure artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_CLOSURE_CAMPAIGN_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_CLOSURE_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_CLOSURE_COMPLETION_20260330.md`
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_ffn_closure_campaign.ps1`
- `tb/tb_g5_ffn_closure_campaign_p11g5fc.cpp`

### Deferred boundary after G5 FFN closure campaign
- This round is bounded to FFN W2 input/weight/bias caller-fed descriptorization.
- Full fallback elimination and full FFN closure remain deferred.
- Wave4 attention/phase migration remains deferred.

## Night-Batch Extension: G5 FFN Fallback Policy Tightening (Bounded)
- Added strict W2 top-fed descriptor policy gate for FFN W2 stage.
- Active caller path now enables `FFN_POLICY_REQUIRE_W2_TOPFED`.
- When strict mode is enabled and descriptors are not ready, W2 stage rejects deterministically and avoids fallback consume.

### G5 FFN fallback policy local-only evidence
- `build/p11g5/ffn_fallback_policy/run.log`: PASS
  - `G5FFN_FALLBACK_POLICY_TOPFED_PRIMARY PASS`
  - `G5FFN_FALLBACK_POLICY_CONTROLLED_FALLBACK PASS`
  - `G5FFN_FALLBACK_POLICY_NO_STALE_STATE PASS`
  - `G5FFN_FALLBACK_POLICY_EXPECTED_COMPARE PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G5 FFN fallback policy strict W2 top-fed gating anchors OK`
- `build/p11g5/ffn_closure_campaign/run.log`: PASS
- `build/p11g5/wave3_ffn_payload_migration/run.log`: PASS
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`: PASS
- `build/p11ah/full_loop/run.log`: PASS
- `build/p11aj/p11aj/run.log`: PASS
- `build/evidence/g5_ffn_fallback_policy_20260330/evidence_manifest.txt`: present

### G5 FFN fallback policy artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_FALLBACK_POLICY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_FALLBACK_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_FALLBACK_COMPLETION_20260330.md`
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_ffn_fallback_policy.ps1`
- `tb/tb_g5_ffn_fallback_policy_p11g5fp.cpp`

### Deferred boundary after fallback-policy pass
- Full fallback removal remains deferred.
- W1 fallback strict policy tightening remains deferred.
- Wave4 attention/phase migration remains deferred.

## Night-Batch Extension: G5 FFN W1 Fallback Policy Tightening (Bounded)
- Added strict W1 top-fed descriptor policy gate for FFN W1 stage.
- Active caller path now enables `FFN_POLICY_REQUIRE_W1_TOPFED` and dispatches explicit `topfed_x_words_valid_override`.
- In strict mode, W1 rejects when x or W1 weight descriptor is not ready; no fallback consume and no output write on reject.

### G5 FFN W1 fallback policy local-only evidence
- `build/p11g5/ffn_w1_fallback_policy/run.log`: PASS
  - `G5FFN_W1_FALLBACK_POLICY_TOPFED_PRIMARY PASS`
  - `G5FFN_W1_FALLBACK_POLICY_CONTROLLED_FALLBACK PASS`
  - `G5FFN_W1_FALLBACK_POLICY_REJECT_ON_MISSING_DESCRIPTOR PASS`
  - `G5FFN_W1_FALLBACK_POLICY_NO_STALE_STATE PASS`
  - `G5FFN_W1_FALLBACK_POLICY_NO_SPURIOUS_TOUCH PASS`
  - `G5FFN_W1_FALLBACK_POLICY_EXPECTED_COMPARE PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G5 FFN W1 fallback policy strict top-fed gating anchors OK`
- `build/p11g5/ffn_fallback_policy/run.log`: PASS
- `build/p11g5/ffn_closure_campaign/run.log`: PASS
- `build/p11g5/wave3_ffn_payload_migration/run.log`: PASS
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`: PASS
- `build/p11ah/full_loop/run.log`: PASS
- `build/p11aj/p11aj/run.log`: PASS
- `build/evidence/g5_ffn_w1_fallback_policy_20260330/evidence_manifest.txt`: present

### G5 FFN W1 fallback policy artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_W1_FALLBACK_POLICY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_W1_FALLBACK_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G5_FFN_W1_FALLBACK_COMPLETION_20260330.md`
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_ffn_w1_fallback_policy.ps1`
- `tb/tb_g5_ffn_w1_fallback_policy_p11g5w1fp.cpp`

### Deferred boundary after W1 fallback-policy pass
- W1 fallback is tightened but not fully removed.
- W1 bias fallback and deeper FFN full-closure tightening remain deferred.
- Wave4 attention/phase migration remains deferred.

## Night-Batch Extension: G6 Single-Run Multi-Track Campaign
- Track A (FFN near-closure): completed two bounded subwaves in one run.
  - Subwave A: W1 bias caller-fed/top-fed descriptor consume anchor.
  - Subwave B: W1/W2 strict reject-stage observability harmonization.
- Track B (Wave4): completed feasibility inventory/ranking/blocker capture.
  - No Wave4 micro-cut executed this round due coupling/risk budget.
- Track C: completion/evidence/docs consolidation delivered.

### G6 local-only evidence
- `build/p11g6/ffn_w1_bias_descriptor/run.log`: PASS
  - `G6FFN_SUBWAVE_A_W1_BIAS_TOPFED_PATH PASS`
  - `G6FFN_SUBWAVE_A_W1_BIAS_NO_SPURIOUS_TOUCH PASS`
  - `G6FFN_SUBWAVE_A_W1_BIAS_EXPECTED_COMPARE PASS`
- `build/p11g6/ffn_fallback_observability/run.log`: PASS
  - `G6FFN_SUBWAVE_B_REJECT_STAGE_W1 PASS`
  - `G6FFN_SUBWAVE_B_REJECT_STAGE_W2 PASS`
  - `G6FFN_SUBWAVE_B_NO_STALE_ON_REJECT PASS`
  - `G6FFN_SUBWAVE_B_NONSTRICT_FALLBACK_OBS PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: G6 FFN W1 top-fed bias descriptor + reject-stage observability anchors OK`
- Required regressions retained PASS:
  - `build/p11g5/ffn_w1_fallback_policy/run.log`
  - `build/p11g5/ffn_fallback_policy/run.log`
  - `build/p11g5/ffn_closure_campaign/run.log`
  - `build/p11g5/wave3_ffn_payload_migration/run.log`
  - `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
  - `build/p11ah/full_loop/run.log`
  - `build/p11aj/p11aj/run.log`
- Bundle: `build/evidence/g6_multi_track_20260330/evidence_manifest.txt`

### G6 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_G6_FFN_NEAR_CLOSURE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G6_FFN_NEAR_CLOSURE_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G6_WAVE4_FEASIBILITY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G6_COMPLETION_20260330.md`
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g6_ffn_w1_bias_descriptor.ps1`
- `scripts/local/run_p11g6_ffn_fallback_observability.ps1`
- `tb/tb_g6_ffn_w1_bias_descriptor_p11g6a.cpp`
- `tb/tb_g6_ffn_fallback_observability_p11g6b.cpp`

### Deferred boundary after G6
- FFN strict/no-fallback full closure is still deferred.
- FFN writeback boundary streamization remains deferred.
- Wave4 implementation remains feasibility-only in this round.

## Night-Batch Extension: W4-M1 QK-Score Phase-Entry Caller-Fed Probe (Bounded)
- Completed one Wave4 bounded micro-cut on QK-score phase entry.
- Added optional caller-fed descriptor probe passthrough at:
  - `run_p11ae_layer0_top_managed_qk_score(...)`
  - `attn_phaseb_top_managed_qk_score_mainline(...)`
- Probe validates phase-entry visibility + ownership + descriptor compare without touching inner compute/writeback loops.

### W4-M1 local-only evidence
- `build/p11w4m1/qkscore_phase_entry_probe/run.log`: PASS
  - `W4M1_QKSCORE_CALLER_FED_DESCRIPTOR_VISIBLE PASS`
  - `W4M1_QKSCORE_OWNERSHIP_CHECK PASS`
  - `W4M1_QKSCORE_NO_SPURIOUS_TOUCH PASS`
  - `W4M1_QKSCORE_EXPECTED_COMPARE PASS`
  - `W4M1_QKSCORE_PROBE_MISMATCH_REJECT PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: W4-M1 QK-score phase-entry caller-fed descriptor probe anchors OK`
- Required regression chain retained PASS:
  - `build/p11g6/ffn_w1_bias_descriptor/run.log`
  - `build/p11g6/ffn_fallback_observability/run.log`
  - `build/p11g5/ffn_w1_fallback_policy/run.log`
  - `build/p11g5/ffn_fallback_policy/run.log`
  - `build/p11g5/ffn_closure_campaign/run.log`
  - `build/p11g5/wave3_ffn_payload_migration/run.log`
  - `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
  - `build/p11ah/full_loop/run.log`
  - `build/p11aj/p11aj/run.log`
- Bundle:
  - `build/evidence/w4m1_qkscore_probe_20260330/evidence_manifest.txt`

### W4-M1 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_W4M1_QKSCORE_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M1_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M1_COMPLETION_20260330.md`
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m1_qkscore_phase_entry_probe.ps1`
- `tb/tb_w4m1_qkscore_phase_entry_probe.cpp`

### Deferred boundary after W4-M1
- This round does not migrate full Wave4 payload compute path.
- QK-score inner loops remain SRAM-centric by design in bounded scope.
- Wave4 broader migration remains deferred.

## Night-Batch Extension: W4-M2 SoftmaxOut Phase-Entry Caller-Fed V-tile Probe (Bounded)
- Completed one additional Wave4 bounded micro-cut on SoftmaxOut phase entry.
- Added optional caller-fed V-tile probe passthrough at:
  - `run_p11af_layer0_top_managed_softmax_out(...)`
  - `attn_phaseb_top_managed_softmax_out_mainline(...)`
- Probe validates phase-entry visibility + ownership + descriptor compare/reject without touching inner online-softmax/reduction/writeback loops.

### W4-M2 local-only evidence
- `build/p11w4m2/softmaxout_phase_entry_probe/run.log`: PASS
  - `W4M2_SOFTMAXOUT_CALLER_FED_VTILE_VISIBLE PASS`
  - `W4M2_SOFTMAXOUT_OWNERSHIP_CHECK PASS`
  - `W4M2_SOFTMAXOUT_NO_SPURIOUS_TOUCH PASS`
  - `W4M2_SOFTMAXOUT_EXPECTED_COMPARE PASS`
  - `W4M2_SOFTMAXOUT_PROBE_MISMATCH_REJECT PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: W4-M2 SoftmaxOut phase-entry caller-fed V-tile probe anchors OK`
- Required regression chain retained PASS:
  - `build/p11g6/ffn_w1_bias_descriptor/run.log`
  - `build/p11g6/ffn_fallback_observability/run.log`
  - `build/p11g5/ffn_w1_fallback_policy/run.log`
  - `build/p11g5/ffn_fallback_policy/run.log`
  - `build/p11g5/ffn_closure_campaign/run.log`
  - `build/p11g5/wave3_ffn_payload_migration/run.log`
  - `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
  - `build/p11ah/full_loop/run.log`
  - `build/p11aj/p11aj/run.log`
- Bundle:
  - `build/evidence/w4_phase_entry_probe_campaign_20260330/evidence_manifest.txt`

### W4-M2 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_W4M2_SOFTMAXOUT_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M2_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M2_COMPLETION_20260330.md`
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m2_softmaxout_phase_entry_probe.ps1`
- `tb/tb_w4m2_softmaxout_phase_entry_probe.cpp`

### Deferred boundary after W4-M2
- This round does not migrate full SoftmaxOut payload compute path.
- SoftmaxOut inner online accumulation and output writeback remain SRAM-centric by design in bounded scope.
- Wave4 broader migration remains deferred.

## Night-Batch Extension: W4-M3 Feasibility / Blocker Refinement
- Completed feasibility-only refinement for next Wave4 single-entry probe.
- Preferred next entry:
  - Phase-A Q x-row caller-fed descriptor probe (`AttnPhaseATopManagedQ`).
- Optional W4-M3 micro-cut was not attempted this round to keep W4-M2 bounded focus and rerun stability.

### W4-M3 outputs
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_FEASIBILITY_20260330.md`
- local-only state:
  - `build/agent_state/w4m2_softmaxout_probe_20260330/w4m3_reality_check.md`
  - `build/agent_state/w4m2_softmaxout_probe_20260330/w4m3_candidate_ranking.md`
  - `build/agent_state/w4m2_softmaxout_probe_20260330/w4m3_blocker_map.md`

## Night-Batch Extension: W4-M3 Phase-A Q Phase-Entry Caller-Fed Probe (Bounded)
- Completed one additional Wave4 bounded micro-cut on Phase-A Q phase entry.
- Added optional caller-fed x-row probe passthrough at:
  - `run_p11ad_layer0_top_managed_q(...)`
  - `attn_phasea_top_managed_q_mainline(...)`
- Probe validates phase-entry visibility + ownership + descriptor compare/reject without touching inner Q compute/writeback loops.

### W4-M3 local-only evidence
- `build/p11w4m3/phasea_q_phase_entry_probe/run.log`: PASS
  - `W4M3_PHASEAQ_CALLER_FED_XROW_VISIBLE PASS`
  - `W4M3_PHASEAQ_OWNERSHIP_CHECK PASS`
  - `W4M3_PHASEAQ_NO_SPURIOUS_TOUCH PASS`
  - `W4M3_PHASEAQ_EXPECTED_COMPARE PASS`
  - `W4M3_PHASEAQ_PROBE_MISMATCH_REJECT PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: W4-M3 Phase-A Q phase-entry caller-fed x-row probe anchors OK`
- Required regression chain retained PASS:
  - `build/p11w4m2/softmaxout_phase_entry_probe/run.log`
  - `build/p11g6/ffn_w1_bias_descriptor/run.log`
  - `build/p11g6/ffn_fallback_observability/run.log`
  - `build/p11g5/ffn_w1_fallback_policy/run.log`
  - `build/p11g5/ffn_fallback_policy/run.log`
  - `build/p11g5/ffn_closure_campaign/run.log`
  - `build/p11g5/wave3_ffn_payload_migration/run.log`
  - `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
  - `build/p11ah/full_loop/run.log`
  - `build/p11aj/p11aj/run.log`
- Bundle:
  - `build/evidence/w4_phasea_q_probe_campaign_20260330/evidence_manifest.txt`

### W4-M3 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_PHASEAQ_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_FEASIBILITY_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_PHASEA_CAMPAIGN_COMPLETION_20260330.md`
- `src/blocks/AttnPhaseATopManagedQ.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m3_phasea_q_phase_entry_probe.ps1`
- `tb/tb_w4m3_phasea_q_phase_entry_probe.cpp`

### Deferred boundary after W4-M3
- This round does not migrate full Phase-A Q payload compute path.
- Q inner compute/writeback loops remain SRAM-centric by design in bounded scope.
- KV remains feasibility/blocker capture only in this run.

## Night-Batch Extension: W4-M3 KV Dedicated Phase-Entry Caller-Fed Probe (Bounded)
- Completed W4-M3 dedicated bounded pass on Phase-A KV phase entry.
- Added optional caller-fed x-row probe passthrough at:
  - `run_p11ac_layer0_top_managed_kv(...)`
  - `attn_phasea_top_managed_kv_mainline(...)`
- Probe validates phase-entry visibility + ownership + descriptor compare/reject without touching inner K/V compute and writeback loops.

### W4-M3 KV local-only evidence
- `build/p11w4m3/kv_phase_entry_probe/run.log`: PASS
  - `W4M3_KV_CALLER_FED_XROW_VISIBLE PASS`
  - `W4M3_KV_OWNERSHIP_CHECK PASS`
  - `W4M3_KV_NO_SPURIOUS_TOUCH PASS`
  - `W4M3_KV_EXPECTED_COMPARE PASS`
  - `W4M3_KV_PROBE_MISMATCH_REJECT PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: W4-M3 Phase-A KV phase-entry caller-fed x-row probe anchors OK`
- Required regression chain retained PASS:
  - `build/p11w4m3/phasea_q_phase_entry_probe/run.log`
  - `build/p11w4m2/softmaxout_phase_entry_probe/run.log`
  - `build/p11g6/ffn_w1_bias_descriptor/run.log`
  - `build/p11g6/ffn_fallback_observability/run.log`
  - `build/p11g5/ffn_w1_fallback_policy/run.log`
  - `build/p11g5/ffn_fallback_policy/run.log`
  - `build/p11g5/ffn_closure_campaign/run.log`
  - `build/p11g5/wave3_ffn_payload_migration/run.log`
  - `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
  - `build/p11ah/full_loop/run.log`
  - `build/p11aj/p11aj/run.log`
- Bundle:
  - `build/evidence/w4m3_kv_probe_campaign_20260330/evidence_manifest.txt`

### W4-M3 KV artifact index
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_PROBE_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4M3_KV_COMPLETION_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_PHASEA_KV_CAMPAIGN_COMPLETION_20260330.md`
- `src/blocks/AttnPhaseATopManagedKv.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4m3_kv_phase_entry_probe.ps1`
- `tb/tb_w4m3_kv_phase_entry_probe.cpp`

### Deferred boundary after W4-M3 KV
- This round does not migrate full Phase-A KV payload compute path.
- KV inner compute/writeback loops remain SRAM-centric by design in bounded scope.
- Wave4 broader payload migration remains deferred.

## Night-Batch Extension: G7 Remaining Direct-SRAM Eradication Campaign (Bounded)
- Completed two bounded waves with code-level progress:
  - Wave A: FFN W1 strict policy now requires top-fed `x + weight + bias` descriptor-ready.
  - Wave B: W4-M3 KV probe now enforces full-row descriptor-ready (`probe_valid == d_model`) and rejects short descriptors.
- Delivered global hotspot/risk inventory and SramView remove-readiness matrix.
- Local-only evidence bundle:
  - `build/evidence/g7_direct_sram_campaign_20260330/evidence_manifest.txt`

### G7 artifacts
- `docs/handoff/TOP_MANAGED_SRAM_G7_DIRECT_SRAM_CAMPAIGN_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G7_DIRECT_SRAM_EVIDENCE_INDEX_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G7_REMOVE_READINESS_20260330.md`
- `docs/handoff/TOP_MANAGED_SRAM_G7_COMPLETION_20260330.md`
- `scripts/local/run_p11g7_ffn_w1_bias_descriptor_strict.ps1`
- `src/blocks/FFNLayer0.h`
- `src/blocks/AttnPhaseATopManagedKv.h`

### Deferred boundary after G7
- Attn/Phase inner compute and writeback loops remain SRAM-centric by design in this bounded pass.
- This round does not claim full Wave4 payload migration.

## Night-Batch Extension: W4-B1 Phase-B Bounded Tile Bridge (Local-only)
- Completed one bounded Wave4 cut beyond probe-only:
  - `AttnPhaseBTopManagedSoftmaxOut` now supports one caller-fed/top-fed V-tile bridge at tile-entry.
- Primary landed anchors:
  - `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
    - `phase_tile_bridge_*` descriptor/observability fields
    - `ATTN_P11AF_TILE_BRIDGE_COMPARE_LOOP`
  - `src/Top.h`
    - `run_p11af_layer0_top_managed_softmax_out(...)` passthrough for W4-B1 bridge args
  - `scripts/check_top_managed_sram_boundary_regression.ps1`
    - W4-B1 passthrough + signature + anchor guards

### W4-B1 local-only evidence
- `build/p11w4b1/phaseb_tile_bridge/run.log`: PASS
  - `W4B1_PHASEB_TILE_BRIDGE_VISIBLE PASS`
  - `W4B1_PHASEB_OWNERSHIP_CHECK PASS`
  - `W4B1_PHASEB_NO_SPURIOUS_TOUCH PASS`
  - `W4B1_PHASEB_EXPECTED_COMPARE PASS`
  - `W4B1_PHASEB_BRIDGE_MISMATCH_REJECT PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: W4-B1 SoftmaxOut bounded tile bridge anchors OK`
- Full requested chain retained PASS:
  - W4-M3 KV/Q runners
  - W4-M2/W4-M1 runners
  - G7/G6/G5 FFN runners
  - P11AH/P11AJ mainline + provenance
  - helper split guard / design purity / repo hygiene pre+post
- Bundle:
  - `build/evidence/w4b1_phaseb_tile_bridge_20260331/evidence_manifest.txt`

### Deferred boundary after W4-B1
- This round is not full Wave4 payload migration.
- Inner Phase-B compute/reduction/writeback loops remain SRAM-centric by bounded policy.

## Night-Batch Extension: W4-B2 Phase-B QkScore Bounded Score-tile Bridge (Local-only)
- Completed one additional bounded Wave4 cut beyond probe-only on `AttnPhaseBTopManagedQkScore`.
- Added one bounded caller-fed/top-fed score-tile bridge at QkScore token-write boundary:
  - `run_p11ae_layer0_top_managed_qk_score(...)`
  - `attn_phaseb_top_managed_qk_score_mainline(...)`
- Scope remained bounded:
  - no external formal contract change
  - no broad rewrite
  - no inner compute/reduction/writeback major-loop rewrite

### W4-B2 local-only evidence
- `build/p11w4b2/qkscore_tile_bridge/run.log`: PASS
  - `W4B2_QKSCORE_TILE_BRIDGE_VISIBLE PASS`
  - `W4B2_QKSCORE_OWNERSHIP_CHECK PASS`
  - `W4B2_QKSCORE_NO_SPURIOUS_TOUCH PASS`
  - `W4B2_QKSCORE_EXPECTED_COMPARE PASS`
  - `W4B2_QKSCORE_BRIDGE_MISMATCH_REJECT PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: W4-B2 QkScore bounded score-tile bridge anchors OK`
- Full requested chain retained PASS:
  - W4-B1 and W4-M1/M2/M3(Q/KV) runners
  - G7/G6/G5 FFN runners
  - P11AH/P11AJ mainline + provenance
  - helper split guard / design purity / repo hygiene pre+post
- Bundle:
  - `build/evidence/w4b2_qkscore_tile_bridge_20260331/evidence_manifest.txt`

### W4-B2 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_W4B2_QKSCORE_TILE_BRIDGE_20260331.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4B2_EVIDENCE_INDEX_20260331.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4B2_COMPLETION_20260331.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_QKSCORE_CAMPAIGN_COMPLETION_20260331.md`
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4b2_qkscore_tile_bridge.ps1`
- `tb/tb_w4b2_qkscore_tile_bridge.cpp`

### Deferred boundary after W4-B2
- This round is not full Wave4 payload migration.
- W4-B2 bridges one bounded score tile on a constrained key-range path.
- QkScore inner compute/reduction/writeback loops remain SRAM-centric by bounded policy.

## Night-Batch Extension: W4-B3 QkScore Selectable-Head Bounded Score-tile Bridge (Local-only)
- Completed one additional bounded Wave4 cut beyond W4-B2:
  - bridge selection is no longer head0-fixed.
  - caller/Top can route one bounded score tile to a selectable head + bounded key-range.
- Added selectable-head bridge passthrough at:
  - `run_p11ae_layer0_top_managed_qk_score(...)`
  - `attn_phaseb_top_managed_qk_score_mainline(...)`
- Scope remained bounded:
  - no external formal contract change
  - no broad rewrite
  - no inner compute/reduction/writeback major-loop rewrite

### W4-B3 local-only evidence
- `build/p11w4b3/qkscore_bridge/run.log`: PASS
  - `W4B3_QKSCORE_BRIDGE_VISIBLE PASS`
  - `W4B3_QKSCORE_OWNERSHIP_CHECK PASS`
  - `W4B3_QKSCORE_NON_HEAD1_PATH PASS`
  - `W4B3_QKSCORE_NO_SPURIOUS_TOUCH PASS`
  - `W4B3_QKSCORE_EXPECTED_COMPARE PASS`
  - `W4B3_QKSCORE_BRIDGE_MISMATCH_REJECT PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: W4-B3 QkScore selectable-head bounded score-tile bridge anchors OK`
- Full requested chain retained PASS:
  - W4-B2/W4-B1 and W4-M1/M2/M3(Q/KV) runners
  - G7/G6/G5 FFN runners
  - P11AH/P11AJ mainline + provenance
  - helper split guard / design purity / repo hygiene pre+post
- Bundle:
  - `build/evidence/w4b3_qkscore_bridge_20260331/evidence_manifest.txt`

### W4-B3 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_W4B3_QKSCORE_BRIDGE_20260331.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4B3_EVIDENCE_INDEX_20260331.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4B3_COMPLETION_20260331.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_QKSCORE_BRIDGE_CAMPAIGN_COMPLETION_20260331.md`
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11w4b3_qkscore_bridge.ps1`
- `tb/tb_w4b3_qkscore_bridge.cpp`

### Deferred boundary after W4-B3
- This round is not full Wave4 payload migration.
- W4-B3 extends bounded bridge selection to selectable-head scope, but still one bounded tile window.
- QkScore inner compute/reduction/writeback loops remain SRAM-centric by bounded policy.

## Night-Batch Extension: W4-B5 QkScore Bounded Family Bridge Generalization (Local-only)
- Completed one bounded extension beyond W4-B3:
  - from single selected head/key-range bridge to bounded selected-family bridge (2~3 cases per invocation).
- Added family descriptor passthrough at:
  - `run_p11ae_layer0_top_managed_qk_score(...)`
  - `attn_phaseb_top_managed_qk_score_mainline(...)`
- Scope remained bounded:
  - no external formal contract change
  - no broad rewrite
  - no inner compute/reduction/writeback major-loop rewrite

### W4-B5 local-only evidence
- `build/p11w4b5/qkscore_family_bridge/run.log`: PASS
  - `W4B5_QKSCORE_FAMILY_BRIDGE_VISIBLE PASS`
  - `W4B5_QKSCORE_FAMILY_OWNERSHIP_CHECK PASS`
  - `W4B5_QKSCORE_FAMILY_EXPECTED_COMPARE PASS`
  - `W4B5_QKSCORE_FAMILY_LEGACY_COMPARE PASS`
  - `W4B5_QKSCORE_FAMILY_NO_SPURIOUS_TOUCH PASS`
  - `W4B5_QKSCORE_FAMILY_MULTI_CASE_ANTI_FALLBACK PASS`
  - `W4B5_QKSCORE_FAMILY_MISMATCH_REJECT PASS`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`: PASS
  - includes `guard: W4-B5 QkScore family bounded bridge anchors OK`
- Baseline safety recheck:
  - `build/p11w4b3/qkscore_bridge/run.log`: PASS
  - `build/p11w4b2/qkscore_tile_bridge/run.log`: PASS

### W4-B5 artifact index
- `docs/handoff/TOP_MANAGED_SRAM_W4_QKSCORE_B5_FAMILY_BRIDGE_20260331.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_QKSCORE_B5_EVIDENCE_INDEX_20260331.md`
- `docs/handoff/TOP_MANAGED_SRAM_W4_QKSCORE_B5_COMPLETION_20260331.md`
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_w4b5_qkscore_family_bridge.cpp`
- `scripts/local/run_p11w4b5_qkscore_family_bridge.ps1`

### Deferred boundary after W4-B5
- This round is not full Wave4 payload migration.
- QkScore inner compute/reduction/writeback loops remain SRAM-centric by bounded policy.
