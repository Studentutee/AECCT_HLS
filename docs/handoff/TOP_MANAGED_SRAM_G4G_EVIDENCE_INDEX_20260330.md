# TOP MANAGED SRAM G4G EVIDENCE INDEX (2026-03-30)

## Scope
- G4-G accepted-commit metadata record harmonization evidence.
- local-only evidence package for review.
- not Catapult closure; not SCVerify closure.

## Evidence Bundle
- `build/evidence/g4g_accept_commit_20260330/evidence_manifest.txt`
- `build/evidence/g4g_accept_commit_20260330/logs/*`

## Claim-to-Evidence Mapping
1. Claim: accepted-commit metadata record harmonization is active.
   - `build/evidence/g4g_accept_commit_20260330/logs/check_top_managed_sram_boundary_regression.log`
   - expected line: `guard: G4-G accepted-commit metadata record harmonization anchors OK`

2. Claim: G4-G targeted validation proves deterministic accept + reject no-stale-state behavior.
   - `build/evidence/g4g_accept_commit_20260330/logs/p11g4g_accept_commit_record_run.log`
   - expected lines:
     - `G4G_ACCEPT_RECORD_CFG_DETERMINISTIC PASS`
     - `G4G_ACCEPT_RECORD_PARAM_DETERMINISTIC PASS`
     - `G4G_REJECT_NO_STALE_STATE PASS`
     - `G4G_ACCEPT_RECORD_INFER_PHASE_VALID PASS`
     - `PASS: run_p11g4g_accept_commit_record`

3. Claim: prior G4-F/G4-E/G4-D targeted negatives remain PASS.
   - `build/evidence/g4g_accept_commit_20260330/logs/p11g4f_commit_diagnostics_negative_run.log`
   - `build/evidence/g4g_accept_commit_20260330/logs/p11g4e_cross_command_metadata_negative_run.log`
   - `build/evidence/g4g_accept_commit_20260330/logs/p11g4_preflight_negative_run.log`

4. Claim: mainline/provenance local regressions remain PASS.
   - `build/evidence/g4g_accept_commit_20260330/logs/p11ah_g4g_accept_commit_run.log`
   - `build/evidence/g4g_accept_commit_20260330/logs/p11aj_g4g_accept_commit_run.log`

5. Claim: hygiene/purity/tooling checks are complete.
   - `build/evidence/g4g_accept_commit_20260330/logs/check_design_purity.log`
   - `build/evidence/g4g_accept_commit_20260330/logs/check_repo_hygiene_pre.log`
   - `build/evidence/g4g_accept_commit_20260330/logs/check_repo_hygiene_post.log`
   - `build/evidence/g4g_accept_commit_20260330/logs/check_agent_tooling.log`

6. Claim: helper-channel split regression guard remains PASS after G4-G patch.
   - `build/evidence/g4g_accept_commit_20260330/logs/check_helper_channel_split_regression.log`
   - expected line: `PASS: check_helper_channel_split_regression`
