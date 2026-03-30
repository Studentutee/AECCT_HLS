# TOP MANAGED SRAM G5 FFN CLOSURE EVIDENCE INDEX (2026-03-30)

## Scope
- Campaign: G5 FFN closure bounded subwaves A/B/C/D.
- Evidence type: local-only.

## Claim -> Evidence

1. Subwave A W2 input top-fed path active
- `build/p11g5/ffn_closure_campaign/run.log`
  - `G5FFN_SUBWAVE_A_W2_INPUT_TOPFED_PATH PASS`

2. Subwave B W2 weight top-fed descriptor active
- `build/p11g5/ffn_closure_campaign/run.log`
  - `G5FFN_SUBWAVE_B_W2_WEIGHT_TOPFED_PATH PASS`

3. Subwave C W2 bias top-fed descriptor active
- `build/p11g5/ffn_closure_campaign/run.log`
  - `G5FFN_SUBWAVE_C_W2_BIAS_TOPFED_PATH PASS`

4. Subwave D fallback boundary is compatibility-only in targeted scope
- `build/p11g5/ffn_closure_campaign/run.log`
  - `G5FFN_SUBWAVE_D_FALLBACK_BOUNDARY PASS`
  - `G5FFN_SUBWAVE_ABCD_NO_SPURIOUS_SRAM_TOUCH PASS`
  - `G5FFN_SUBWAVE_ABCD_EXPECTED_COMPARE PASS`

5. Boundary guard anchor coverage
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
  - `guard: G5 FFN closure campaign W2 top-fed input/weight/bias anchors OK`

6. Regression compatibility after campaign patch
- `build/p11g5/wave3_ffn_payload_migration/run.log`: PASS
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`: PASS
- `build/p11ah/g5_ffn_closure_campaign/run.log`: PASS
- `build/p11aj/g5_ffn_closure_campaign/run.log`: PASS

7. Hygiene/purity safety checks
- `build/evidence/g5_ffn_closure_campaign_20260330/check_helper_channel_split_regression.log`: PASS
- `build/evidence/g5_ffn_closure_campaign_20260330/check_design_purity.log`: PASS
- `build/evidence/g5_ffn_closure_campaign_20260330/check_repo_hygiene_pre.log`: PASS
- `build/evidence/g5_ffn_closure_campaign_20260330/check_repo_hygiene_post.log`: PASS

## Bundle Manifest
- `build/evidence/g5_ffn_closure_campaign_20260330/evidence_manifest.txt`
- `build/evidence/g5_ffn_closure_campaign_20260330/evidence_summary.txt`

## Closure Boundary
- not Catapult closure
- not SCVerify closure
