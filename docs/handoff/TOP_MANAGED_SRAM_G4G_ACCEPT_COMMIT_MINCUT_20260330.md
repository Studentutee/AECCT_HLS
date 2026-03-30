# TOP MANAGED SRAM G4G ACCEPT-COMMIT METADATA MINCUT (2026-03-30)

## Summary
- Delivered a bounded G4-G mincut for accepted-commit metadata record harmonization in Top.
- Added a shared local-only accepted-commit record shape and helper path in `src/Top.h`.
- Harmonized CFG/PARAM/INFER commit-success record write behavior:
  - CFG: record written after commit diagnostics + cfg legality pass.
  - PARAM: record written on commit diagnostics accept.
  - INFER: record written on commit diagnostics accept with phase metadata.
- Reject path does not overwrite the previously accepted record.

## Exact Files Changed
- `src/Top.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_g4g_accept_commit_record_p11g4g.cpp`
- `scripts/local/run_p11g4g_accept_commit_record.ps1`

## Exact Commands Run
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

## Actual Execution Evidence / Log Excerpt
- `build/p11g4g/accept_commit_record/run.log`
  - `G4G_ACCEPT_RECORD_CFG_DETERMINISTIC PASS`
  - `G4G_ACCEPT_RECORD_PARAM_DETERMINISTIC PASS`
  - `G4G_REJECT_NO_STALE_STATE PASS`
  - `G4G_ACCEPT_RECORD_INFER_PHASE_VALID PASS`
  - `PASS: run_p11g4g_accept_commit_record`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
  - `guard: G4-G accepted-commit metadata record harmonization anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

## Governance Posture
- local-only bounded mincut
- not Catapult closure
- not SCVerify closure
- no remote simulator/site-local PLI touch

## Residual Risks
- G4-G harmonizes accepted-commit metadata record shape only; command-specific post-commit behavior stays command-specific by design.
- External protocol is unchanged; deeper protocol-level metadata state exposure remains deferred.

## Recommended Next Step
- Review whether additional consumer-facing debug export of accepted-commit record is required in a future bounded round, without altering current external contract.
