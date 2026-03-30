# TOP MANAGED SRAM G5 WAVE3 EVIDENCE INDEX (2026-03-30)

## Scope
- Task: G5-Wave3 FFN payload migration bounded mincut.
- Posture: local-only, not Catapult closure, not SCVerify closure.

## Claim-to-Evidence Mapping
1. Claim: FFN W1 consumes caller-fed top-fed payload path.
- Evidence:
  - `build/p11g5/wave3_ffn_payload_migration/run.log`
    - `G5W3_FFN_TOPFED_PAYLOAD_PATH PASS`
    - `G5W3_FFN_EXPECTED_COMPARE PASS`

2. Claim: Legacy SRAM x payload did not become hidden main consume path for this cut.
- Evidence:
  - `build/p11g5/wave3_ffn_payload_migration/run.log`
    - `G5W3_FFN_NO_SPURIOUS_SRAM_TOUCH PASS`

3. Claim: Regression guard anchors include new Wave3 FFN path.
- Evidence:
  - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
    - `guard: G5 wave3 FFN top-fed payload migration anchors OK`
    - `PASS: check_top_managed_sram_boundary_regression`

4. Claim: Mainline/provenance chain remained PASS.
- Evidence:
  - `build/p11ah/g5_wave3_ffn/run.log`
    - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
    - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
    - `PASS: run_p11ah_full_loop_local_e2e`
  - `build/p11aj/g5_wave3_ffn/run.log`
    - `PROVENANCE_STAGE_AC PASS`
    - `PROVENANCE_STAGE_AD PASS`
    - `PROVENANCE_STAGE_AE PASS`
    - `PROVENANCE_STAGE_AF PASS`
    - `PASS: run_p11aj_top_managed_sram_provenance`

5. Claim: hygiene/purity/helper guard remained PASS.
- Evidence:
  - `build/evidence/g5_wave3_ffn_20260330/check_helper_channel_split_regression.log`
  - `build/evidence/g5_wave3_ffn_20260330/check_design_purity.log`
  - `build/evidence/g5_wave3_ffn_20260330/check_repo_hygiene_pre.log`
  - `build/evidence/g5_wave3_ffn_20260330/check_repo_hygiene_post.log`

## Evidence Bundle Root
- `build/evidence/g5_wave3_ffn_20260330/`

## Bundle Manifest
- `build/evidence/g5_wave3_ffn_20260330/evidence_manifest.txt`
- `build/evidence/g5_wave3_ffn_20260330/evidence_summary.txt`
