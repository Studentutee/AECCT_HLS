# TOP MANAGED SRAM G4E EVIDENCE INDEX (2026-03-29)

## Scope
- G4-E bounded metadata harmonization mincut evidence.
- local-only acceptance evidence for morning review.
- not Catapult closure; not SCVerify closure.

## Evidence Bundle
- `build/evidence/g4e_metadata_mincut_20260329/evidence_manifest.txt`
- `build/evidence/g4e_metadata_mincut_20260329/logs/*`

## Claim-to-Evidence Mapping
1. Claim: cross-command metadata harmonization helpers are anchored in Top and guarded.
   - `build/evidence/g4e_metadata_mincut_20260329/logs/check_top_managed_sram_boundary_regression.log`
   - expected line: `guard: G4-E cross-command ingest metadata surface helpers anchored`

2. Claim: new targeted mismatch validation catches cross-command metadata misuse.
   - `build/evidence/g4e_metadata_mincut_20260329/logs/p11g4e_cross_command_metadata_negative_run.log`
   - expected lines:
     - `G4E_OWNER_CFG_RX_MISMATCH_REJECT PASS`
     - `G4E_SPAN_OUT_OF_RANGE_REJECT PASS`
     - `PASS: run_p11g4e_cross_command_metadata_negative`

3. Claim: existing G4-D preflight negative guard remains healthy after G4-E patch.
   - `build/evidence/g4e_metadata_mincut_20260329/logs/p11g4_preflight_negative_run.log`
   - expected lines:
     - `PREFLIGHT_INVALID_BASE_REJECT PASS`
     - `PREFLIGHT_INVALID_SPAN_REJECT PASS`
     - `PASS: run_p11g4_infer_ingest_preflight_negative`

4. Claim: core local regression path did not drift.
   - `build/evidence/g4e_metadata_mincut_20260329/logs/p11ah_g4e_metadata_mincut_run.log`
   - `build/evidence/g4e_metadata_mincut_20260329/logs/p11aj_g4e_metadata_mincut_run.log`
   - expected anchors:
     - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
     - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
     - `PASS: run_p11ah_full_loop_local_e2e`
     - `PASS: run_p11aj_top_managed_sram_provenance`

5. Claim: hygiene/purity checks are complete for this round.
   - `build/evidence/g4e_metadata_mincut_20260329/logs/check_design_purity.log`
   - `build/evidence/g4e_metadata_mincut_20260329/logs/check_repo_hygiene_pre.log`
   - `build/evidence/g4e_metadata_mincut_20260329/logs/check_repo_hygiene_post.log`
   - `build/evidence/g4e_metadata_mincut_20260329/logs/check_agent_tooling.log`
