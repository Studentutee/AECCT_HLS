# TOP MANAGED SRAM G4 HARDENING EVIDENCE INDEX (2026-03-29)

## Scope
- This index maps closeout claims to concrete local-only evidence artifacts.
- Scope is hygiene/evidence/consistency hardening for already-landed G4 mincut.
- not Catapult closure; not SCVerify closure.

## Evidence Bundle Location
- `build/evidence/g4_hardening_20260329/logs`
- `build/evidence/g4_hardening_20260329/evidence_manifest.txt`

## Claim-To-Evidence Mapping
1. Claim: G4 ingest ownership guard remains active and now checks OP_INFER reject mapping.
   - `build/evidence/g4_hardening_20260329/logs/check_top_managed_sram_boundary_regression.log`
   - expected anchors:
     - `guard: G4 infer ingest contractized base/len dispatch anchors OK`
     - `guard: OP_INFER preflight reject path maps invalid span to ERR_MEM_RANGE`
     - `PASS: check_top_managed_sram_boundary_regression`

2. Claim: Existing non-regression path remains PASS.
   - `build/evidence/g4_hardening_20260329/logs/p11ah_g4_night_batch_run.log`
   - `build/evidence/g4_hardening_20260329/logs/p11aj_g4_night_batch_run.log`
   - expected anchors:
     - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
     - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
     - `PASS: run_p11ah_full_loop_local_e2e`
     - `PASS: run_p11aj_top_managed_sram_provenance`

3. Claim: infer ingest preflight has targeted negative validation (not only non-regression).
   - `build/evidence/g4_hardening_20260329/logs/p11g4_preflight_run.log`
   - expected anchors:
     - `PREFLIGHT_INVALID_BASE_REJECT PASS`
     - `PREFLIGHT_INVALID_SPAN_REJECT PASS`
     - `PREFLIGHT_ERR_MEM_RANGE_GUARD_BEHAVIOR PASS`
     - `PASS: run_p11g4_infer_ingest_preflight_negative`

4. Claim: helper split guard baseline remains PASS after closeout updates.
   - `build/evidence/g4_hardening_20260329/logs/check_helper_channel_split_regression.log`
   - expected anchor:
     - `PASS: check_helper_channel_split_regression`

5. Claim: hygiene/purity/tooling checks are captured in bundle.
   - `build/evidence/g4_hardening_20260329/logs/check_design_purity.log`
   - `build/evidence/g4_hardening_20260329/logs/check_repo_hygiene_pre.log`
   - `build/evidence/g4_hardening_20260329/logs/check_repo_hygiene_post.log`
   - `build/evidence/g4_hardening_20260329/logs/check_agent_tooling.log`

## AECCT_ac_ref Side-Change Reality Check
- audited paths:
  - `AECCT_ac_ref/include/RefPrecisionMode.h`
  - `AECCT_ac_ref/src/RefModel.cpp`
  - `AECCT_ac_ref/src/ref_main.cpp`
- status capture:
  - `build/evidence/g4_hardening_20260329/logs/aecct_ac_ref_side_change_status.txt`
- `git status --` on these paths returned no active diff in this closeout round.
- disposition: excluded from G4 hardening acceptance payload.

## Posture
- local-only evidence bundle for morning review convenience.
- remote simulator/site-local PLI chain was not touched in this round.
