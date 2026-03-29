# MORNING REVIEW TOP MANAGED SRAM (2026-03-29)

## Quick Goal
- Verify the overnight architecture-forward cuts that move SRAM ownership semantics toward Top-managed dispatch.
- Confirm no overclaim: local-only evidence; not Catapult closure; not SCVerify closure.
- Include the latest G4 ingest/base-shadow mincut verification in this review pass.
- Include the bounded G4-E cross-command metadata harmonization mincut verification.

## Suggested 10-Minute Review Order
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
- `tb/tb_infer_ingest_preflight_negative_p11g4.cpp`
- `tb/tb_g4e_cross_command_metadata_negative_p11g4e.cpp`
- `docs/handoff/TOP_MANAGED_SRAM_ARCH_GAP_INVENTORY_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_MINCUTS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_PROGRESS_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4E_METADATA_MINCUT_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4E_EVIDENCE_INDEX_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4E_COMPLETION_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_HARDENING_EVIDENCE_INDEX_20260329.md`
- `docs/handoff/TOP_MANAGED_SRAM_G4_HARDENING_COMPLETION_20260329.md`
- `docs/handoff/MORNING_REVIEW_TOP_MANAGED_SRAM_20260329.md`
- `build/p11ah/g4_night_batch/run.log`
- `build/p11aj/g4_night_batch/run.log`
- `build/p11g4/infer_ingest_preflight_negative/run.log`
- `build/p11g4e/cross_command_metadata_negative/run.log`
- `build/p11ah/top_managed_sram_push/run.log`
- `build/p11aj/top_managed_sram_push/run.log`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
- `build/evidence/g4_hardening_20260329/evidence_manifest.txt`
- `build/evidence/g4e_metadata_mincut_20260329/evidence_manifest.txt`

## Copy-Ready GPT Review Prompt
- `Please review src/Top.h and src/blocks/TransformerLayer.h for Top-managed SRAM ownership convergence. Focus on: (1) whether Top now builds and owns preproc/layernorm/final-head contracts in active infer path, (2) whether Transformer sublayer1 norm params are preloaded by Top with safe fallback, and (3) whether local evidence (p11ah/p11aj + boundary guard) supports acceptance without overclaim. Keep posture local-only and do not claim Catapult/SCVerify closure.`
