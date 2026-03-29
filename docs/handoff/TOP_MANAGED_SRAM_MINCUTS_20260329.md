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
