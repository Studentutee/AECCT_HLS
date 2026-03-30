# TOP MANAGED SRAM G5 EVIDENCE INDEX (2026-03-30)

## Scope
- Campaign: G5 remaining direct-SRAM payload migration (Wave 1 + Wave 2 completion).
- Posture: local-only, not Catapult closure, not SCVerify closure.

## Claim-to-Evidence Mapping
1. Claim: Wave 1 migrated LayerNorm affine payload to Top-fed path anchor.
- Evidence:
  - `build/p11g5/wave1_payload_migration/run.log`
    - `G5W1_LN_TOPFED_AFFINE_NO_SPURIOUS_SRAM_TOUCH PASS`
  - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
    - `guard: G5 wave1/wave2 top-fed payload migration anchors OK`

2. Claim: Wave 1 migrated FinalHead token-scalar payload to Top-fed path anchor.
- Evidence:
  - `build/p11g5/wave1_payload_migration/run.log`
    - `G5W1_FINALHEAD_TOPFED_SCALAR_PATH PASS`
  - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
    - `guard: G5 wave1/wave2 top-fed payload migration anchors OK`

3. Claim: Wave 2 migrated Preproc infer input payload to Top-fed path anchor.
- Evidence:
  - `build/p11g5/wave2_preproc_payload_migration/run.log`
    - `G5W2_PREPROC_TOPFED_INPUT_PATH PASS`
  - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
    - `guard: G5 wave1/wave2 top-fed payload migration anchors OK`

4. Claim: Mainline/provenance behavior remained non-regressed after G5 wave patches.
- Evidence:
  - `build/p11ah/g5_payload_campaign/run.log`
    - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
    - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
    - `PASS: run_p11ah_full_loop_local_e2e`
  - `build/p11aj/g5_payload_campaign/run.log`
    - `PROVENANCE_STAGE_AC PASS`
    - `PROVENANCE_STAGE_AD PASS`
    - `PROVENANCE_STAGE_AE PASS`
    - `PROVENANCE_STAGE_AF PASS`
    - `PASS: run_p11aj_top_managed_sram_provenance`

5. Claim: Purity/hygiene/helper-guard/tooling checks remained PASS.
- Evidence:
  - `build/evidence/g5_payload_campaign_20260330/check_helper_channel_split_regression.log`
  - `build/evidence/g5_payload_campaign_20260330/check_design_purity.log`
  - `build/evidence/g5_payload_campaign_20260330/check_repo_hygiene_pre.log`
  - `build/evidence/g5_payload_campaign_20260330/check_repo_hygiene_post.log`
  - `build/evidence/g5_payload_campaign_20260330/check_agent_tooling.log`

## Evidence Bundle Root
- `build/evidence/g5_payload_campaign_20260330/`

## Bundle Manifest
- `build/evidence/g5_payload_campaign_20260330/evidence_manifest.txt`
- `build/evidence/g5_payload_campaign_20260330/evidence_summary.txt`
