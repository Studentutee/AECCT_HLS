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

## Hardening Follow-Up (night-batch closeout)
- Added targeted negative validation for infer ingest preflight span guard:
  - `scripts/local/run_p11g4_infer_ingest_preflight_negative.ps1`
  - `tb/tb_infer_ingest_preflight_negative_p11g4.cpp`
- Added explicit regression anchor for OP_INFER reject/accept response behavior:
  - `scripts/check_top_managed_sram_boundary_regression.ps1`
  - requires invalid contract span rejection to map to `ERR_MEM_RANGE`
- Added hygiene completion coverage:
  - `check_repo_hygiene -Phase post` now included in closeout evidence bundle.
- Side-change status:
  - `AECCT_ac_ref/include/RefPrecisionMode.h`
  - `AECCT_ac_ref/src/RefModel.cpp`
  - `AECCT_ac_ref/src/ref_main.cpp`
  - current worktree has no active diff for these files; they are treated as out-of-scope for G4 mincut acceptance.
