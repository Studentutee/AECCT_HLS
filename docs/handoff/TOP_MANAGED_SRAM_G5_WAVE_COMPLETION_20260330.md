# TOP MANAGED SRAM G5 WAVE COMPLETION (2026-03-30)

## Summary
- Campaign: G5 remaining direct-SRAM payload migration (single-run multi-wave).
- Completed in this round:
  - Wave 1: `LayerNormBlock` + `FinalHead` top-fed payload migration.
  - Wave 2: `PreprocEmbedSPE` top-fed infer-input payload migration.
- Not completed in this round:
  - Wave 3 (`FFNLayer0`) and Wave 4 (`Attn*` / `TransformerLayer`) remain deferred due bounded change budget and coupling risk.

## Exact Files Changed
- `src/Top.h`
- `src/blocks/LayerNormBlock.h`
- `src/blocks/FinalHead.h`
- `src/blocks/PreprocEmbedSPE.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_g5_wave1_payload_migration_p11g5w1.cpp`
- `scripts/local/run_p11g5_wave1_payload_migration.ps1`
- `tb/tb_g5_wave2_preproc_payload_migration_p11g5w2.cpp`
- `scripts/local/run_p11g5_wave2_preproc_payload_migration.ps1`

## Exact Commands Run
- `git status --short --untracked-files=all`
- `rg -n "sram\\[|u32_t\\* sram|SramView& sram" src/blocks/FinalHead.h src/blocks/LayerNormBlock.h src/blocks/FFNLayer0.h src/blocks/PreprocEmbedSPE.h src/blocks/AttnLayer0.h src/blocks/TransformerLayer.h src/blocks/AttnPhaseATopManagedKv.h src/blocks/AttnPhaseATopManagedQ.h src/blocks/AttnPhaseBTopManagedQkScore.h src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave1_payload_migration.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g5_wave2_preproc_payload_migration.ps1` (rerun after targeted fix)
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g5_payload_campaign`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g5_payload_campaign`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`
- `powershell -ExecutionPolicy Bypass -File scripts/check_agent_tooling.ps1 -RepoRoot . -StateRoot build/agent_state`

## Actual Execution Evidence / Log Excerpt
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
- `build/p11ah/g5_payload_campaign/run.log`:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g5_payload_campaign/run.log`:
  - `PROVENANCE_STAGE_AD PASS`
  - `PROVENANCE_STAGE_AC PASS`
  - `PROVENANCE_STAGE_AE PASS`
  - `PROVENANCE_STAGE_AF PASS`
  - `PASS: run_p11aj_top_managed_sram_provenance`
- `build/evidence/g5_payload_campaign_20260330/check_design_purity.log`: `PASS: check_design_purity`
- `build/evidence/g5_payload_campaign_20260330/check_repo_hygiene_pre.log`: `PASS: check_repo_hygiene`
- `build/evidence/g5_payload_campaign_20260330/check_repo_hygiene_post.log`: `PASS: check_repo_hygiene`

## Governance Posture
- local-only migration/evidence.
- compile-first and diagnostics-first closure posture.
- Top remains sole production shared-SRAM owner on newly migrated payload input anchors.
- not Catapult closure.
- not SCVerify closure.
- remote simulator/site-local/PLI lines untouched.

## Residual Risks
- Wave 3 (`FFNLayer0`) and Wave 4 (`AttnLayer0`/`TransformerLayer`/phase blocks) still keep significant direct-SRAM payload flows.
- `LayerNormBlock` / `FinalHead` / `PreprocEmbedSPE` retain SRAM fallback path for compatibility (Top-fed path now explicitly available and guarded).
- Full elimination of direct payload SRAM access in attention/FFN stages remains deferred.

## Next Recommended Step
- Start Wave 3 with a bounded `FFNLayer0` mincut focused on Top-fed W1/W2 tile payload descriptors (without external contract change), then mirror this round's targeted-validation pattern.
