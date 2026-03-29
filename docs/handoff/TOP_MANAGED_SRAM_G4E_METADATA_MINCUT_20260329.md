# TOP MANAGED SRAM G4E METADATA MINCUT (2026-03-29)

## Summary
- This round applies a bounded G4-E metadata harmonization cut in Top-only scope.
- No external formal contract change and no broad block graph rewrite.
- Primary patch:
  - Added Top-internal cross-command `IngestMetadataSurface`.
  - Unified CFG/PARAM/INFER ingest expected-length and span helper path in `src/Top.h`.
  - Routed OP_LOAD_W preflight through `param_ingest_span_legal(regs)` with harmonized metadata surface.
  - Routed OP_INFER preflight through `infer_metadata_surface(regs)` + `ingest_meta_span_in_sram(...)`.
- Added cross-command targeted mismatch validation and guard anchors.

## Exact Files Changed
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_g4e_cross_command_metadata_negative_p11g4e.cpp`
- `scripts/local/run_p11g4e_cross_command_metadata_negative.ps1`

## Exact Commands Run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4e_cross_command_metadata_negative.ps1 -BuildDir build/p11g4e/cross_command_metadata_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g4_infer_ingest_preflight_negative.ps1 -BuildDir build/p11g4/infer_ingest_preflight_negative`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g4e_metadata_mincut`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g4e_metadata_mincut`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

## Actual Execution Evidence / Log Excerpt
- `build/p11g4e/cross_command_metadata_negative/run.log`
  - `G4E_OWNER_CFG_RX_MISMATCH_REJECT PASS`
  - `G4E_SPAN_OUT_OF_RANGE_REJECT PASS`
  - `PASS: run_p11g4e_cross_command_metadata_negative`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
  - `guard: G4-E cross-command ingest metadata surface helpers anchored`
  - `PASS: check_top_managed_sram_boundary_regression`
- `build/p11ah/g4e_metadata_mincut/run.log`
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/g4e_metadata_mincut/run.log`
  - `PROVENANCE_STAGE_AC PASS`
  - `PROVENANCE_STAGE_AD PASS`
  - `PROVENANCE_STAGE_AE PASS`
  - `PROVENANCE_STAGE_AF PASS`
  - `PASS: run_p11aj_top_managed_sram_provenance`

## Governance Posture
- local-only bounded harmonization
- not Catapult closure
- not SCVerify closure
- remote simulator / site-local PLI line not touched

## Residual Risks
- CFG still keeps command-specific commit legality path; this round does not unify semantic validation itself.
- PARAM still keeps command-specific W-region/alignment policy.
- Full cross-command metadata object unification remains deferred to avoid broad rewrite.

## Recommended Next Step
- If morning review accepts this cut, plan next bounded step to standardize commit-time metadata validation diagnostics across CFG/PARAM/INFER without changing external contract.
