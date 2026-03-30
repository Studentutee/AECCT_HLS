# TOP MANAGED SRAM G4F COMMIT DIAGNOSTICS MINCUT (2026-03-29)

## Summary
- This round delivers a bounded G4-F mincut focused on commit-time metadata diagnostics harmonization.
- Scope is Top-only diagnostics helper consolidation; no external formal contract change.
- Main changes:
  - Added shared commit-time diagnostics helpers in `src/Top.h`:
    - `ingest_meta_len_exact(...)`
    - `ingest_commit_diag_error(...)`
  - Wired commit-time acceptance/reject mapping for CFG/PARAM/INFER to use harmonized helper path.
  - Preserved command-specific legality checks and post-commit behaviors.

## Exact Files Changed
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_g4f_commit_diagnostics_negative_p11g4f.cpp`
- `scripts/local/run_p11g4f_commit_diagnostics_negative.ps1`

## Exact Commands Run
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

## Actual Execution Evidence / Log Excerpt
- `build/p11g4f/commit_diagnostics_negative/run.log`
  - `G4F_CFG_LEN_MISMATCH_MAPPING PASS`
  - `G4F_PARAM_LEN_MISMATCH_MAPPING PASS`
  - `G4F_OWNER_RX_MISMATCH_MAPPING PASS`
  - `G4F_SPAN_MISMATCH_MAPPING PASS`
  - `PASS: run_p11g4f_commit_diagnostics_negative`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
  - `guard: G4-F commit-time diagnostics helper + error mapping anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

## Governance Posture
- local-only bounded mincut
- not Catapult closure
- not SCVerify closure
- remote simulator/site-local PLI line untouched

## Residual Risks
- INFER path still uses `ERR_BAD_STATE` as length-mismatch fallback because no dedicated infer mismatch code exists in current protocol enum.
- Full cross-command semantic diagnostics unification beyond this helper layer remains deferred.

## Recommended Next Step
- Review if protocol-level infer-specific mismatch code is desired in a future bounded round, without changing current external contract unexpectedly.
