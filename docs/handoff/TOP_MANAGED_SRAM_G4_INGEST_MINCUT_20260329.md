# TOP MANAGED SRAM G4 INGEST MINCUT (2026-03-29)

## Summary
- This round pushes G4 ingest/base-shadow boundary with a minimal Top-side ownership cut:
  - Introduced `InferIngestContract` in Top.
  - Added `phase_id` + `token_range` + `tile_range` metadata in infer ingest contract.
  - Added `OP_INFER` preflight span validation (`infer_contract_span_in_sram`) before entering RX state.
  - Routed Preproc infer input base/valid-length and window range from ingest contract.
  - Routed FinalHead label source from Top-managed SRAM ingest view.
- No external Top formal contract change.

## Exact Files Changed
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`

## Exact Commands Run
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g4_night_batch`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g4_night_batch`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`

## Actual Execution Evidence / Log Excerpt
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
  - `guard: G4 infer ingest contractized base/len dispatch anchors OK`
  - `guard: Top-owned preproc/layernorm/final-head contract dispatch anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11ah/g4_night_batch/run.log`
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g4_night_batch/run.log`
  - `PROVENANCE_STAGE_AC PASS`
  - `PROVENANCE_STAGE_AD PASS`
  - `PROVENANCE_STAGE_AE PASS`
  - `PROVENANCE_STAGE_AF PASS`
  - `PASS: run_p11aj_top_managed_sram_provenance`

## Governance Posture
- local-only architecture-forward progress
- not Catapult closure
- not SCVerify closure
- Top remains sole production shared-SRAM owner

## Residual Risks
- `infer_input_shadow` remains as local mirror path for diagnostics/probes.
- Full cross-command ingest metadata unification (CFG/PARAM/INFER one shared entry model) is deferred.

## Recommended Next Step
- Plan the next G4 step as bounded metadata harmonization across CFG/PARAM/INFER ingest without broad rewrite.
