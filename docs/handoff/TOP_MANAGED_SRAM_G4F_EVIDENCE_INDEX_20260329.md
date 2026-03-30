# TOP MANAGED SRAM G4F EVIDENCE INDEX (2026-03-29)

## Scope
- G4-F commit-time metadata diagnostics harmonization evidence.
- local-only evidence package for review.
- not Catapult closure; not SCVerify closure.

## Evidence Bundle
- `build/evidence/g4f_commit_diag_20260330/evidence_manifest.txt`
- `build/evidence/g4f_commit_diag_20260330/logs/*`

## Claim-to-Evidence Mapping
1. Claim: commit-time diagnostics helper and mapping anchors are enforced.
   - `build/evidence/g4f_commit_diag_20260330/logs/check_top_managed_sram_boundary_regression.log`
   - expected line: `guard: G4-F commit-time diagnostics helper + error mapping anchors OK`

2. Claim: G4-F targeted mismatch negative validation passes.
   - `build/evidence/g4f_commit_diag_20260330/logs/p11g4f_commit_diagnostics_negative_run.log`
   - expected lines:
     - `G4F_CFG_LEN_MISMATCH_MAPPING PASS`
     - `G4F_PARAM_LEN_MISMATCH_MAPPING PASS`
     - `G4F_OWNER_RX_MISMATCH_MAPPING PASS`
     - `G4F_SPAN_MISMATCH_MAPPING PASS`
     - `PASS: run_p11g4f_commit_diagnostics_negative`

3. Claim: prior G4-E and G4-D targeted negatives remain PASS.
   - `build/evidence/g4f_commit_diag_20260330/logs/p11g4e_cross_command_metadata_negative_run.log`
   - `build/evidence/g4f_commit_diag_20260330/logs/p11g4_preflight_negative_run.log`

4. Claim: mainline/provenance local regressions remain PASS.
   - `build/evidence/g4f_commit_diag_20260330/logs/p11ah_g4f_commit_diag_run.log`
   - `build/evidence/g4f_commit_diag_20260330/logs/p11aj_g4f_commit_diag_run.log`

5. Claim: hygiene/purity/tooling checks are complete.
   - `build/evidence/g4f_commit_diag_20260330/logs/check_design_purity.log`
   - `build/evidence/g4f_commit_diag_20260330/logs/check_repo_hygiene_pre.log`
   - `build/evidence/g4f_commit_diag_20260330/logs/check_repo_hygiene_post.log`
   - `build/evidence/g4f_commit_diag_20260330/logs/check_agent_tooling.log`

6. Claim: helper-channel split regression guard remains PASS after G4-F patch.
   - `build/evidence/g4f_commit_diag_20260330/logs/check_helper_channel_split_regression.log`
   - expected line: `PASS: check_helper_channel_split_regression`
