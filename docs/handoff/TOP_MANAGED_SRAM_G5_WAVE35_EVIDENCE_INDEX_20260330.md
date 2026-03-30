# TOP MANAGED SRAM G5 WAVE3.5 EVIDENCE INDEX (2026-03-30)

## Scope
- Task: G5-Wave3.5 FFN W1 weight tile caller-fed descriptor bounded mincut.
- Posture: local-only, not Catapult closure, not SCVerify closure.

## Claim-to-Evidence Mapping
1. Claim: FFN W1 weight tile consumes caller-fed top-fed path.
- Evidence:
  - `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
    - `G5W35_FFN_W1_TOPFED_WEIGHT_PATH PASS`
    - `G5W35_FFN_W1_EXPECTED_COMPARE PASS`

2. Claim: Legacy SRAM W1 weight path is not hidden main consume for this cut.
- Evidence:
  - `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
    - `G5W35_FFN_W1_NO_SPURIOUS_SRAM_TOUCH PASS`

3. Claim: Boundary guard includes Wave3.5 W1 weight anchors.
- Evidence:
  - `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
    - `guard: G5 wave3.5 FFN W1 top-fed weight payload migration anchors OK`
    - `PASS: check_top_managed_sram_boundary_regression`

4. Claim: Wave3 regression remains PASS after Wave3.5 patch.
- Evidence:
  - `build/p11g5/wave3_ffn_payload_migration/run.log`
    - `PASS: run_p11g5_wave3_ffn_payload_migration`

5. Claim: mainline/provenance/purity/hygiene/helper guard remain PASS.
- Evidence:
  - `build/p11ah/g5_wave35_ffn_w1/run.log`
  - `build/p11aj/g5_wave35_ffn_w1/run.log`
  - `build/evidence/g5_wave35_ffn_w1_20260330/check_helper_channel_split_regression.log`
  - `build/evidence/g5_wave35_ffn_w1_20260330/check_design_purity.log`
  - `build/evidence/g5_wave35_ffn_w1_20260330/check_repo_hygiene_pre.log`
  - `build/evidence/g5_wave35_ffn_w1_20260330/check_repo_hygiene_post.log`

## Evidence bundle root
- `build/evidence/g5_wave35_ffn_w1_20260330/`

## Bundle manifest
- `build/evidence/g5_wave35_ffn_w1_20260330/evidence_manifest.txt`
- `build/evidence/g5_wave35_ffn_w1_20260330/evidence_summary.txt`
