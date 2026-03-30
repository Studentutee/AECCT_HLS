# TOP MANAGED SRAM G6 FFN NEAR-CLOSURE EVIDENCE INDEX (2026-03-30)

## Claim -> Evidence
1. Subwave A: W1 bias top-fed descriptor path consumed
- `build/p11g6/ffn_w1_bias_descriptor/run.log`
  - `G6FFN_SUBWAVE_A_W1_BIAS_TOPFED_PATH PASS`
  - `G6FFN_SUBWAVE_A_W1_BIAS_EXPECTED_COMPARE PASS`
  - `G6FFN_SUBWAVE_A_W1_BIAS_NO_SPURIOUS_TOUCH PASS`

2. Subwave B: strict reject-stage observability harmonized
- `build/p11g6/ffn_fallback_observability/run.log`
  - `G6FFN_SUBWAVE_B_REJECT_STAGE_W1 PASS`
  - `G6FFN_SUBWAVE_B_REJECT_STAGE_W2 PASS`
  - `G6FFN_SUBWAVE_B_NO_STALE_ON_REJECT PASS`
  - `G6FFN_SUBWAVE_B_NONSTRICT_FALLBACK_OBS PASS`

3. Guard anchor coverage for G6
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
  - `guard: G6 FFN W1 top-fed bias descriptor + reject-stage observability anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

4. Regression compatibility retained
- `build/p11g5/ffn_w1_fallback_policy/run.log`: PASS
- `build/p11g5/ffn_fallback_policy/run.log`: PASS
- `build/p11g5/ffn_closure_campaign/run.log`: PASS
- `build/p11g5/wave3_ffn_payload_migration/run.log`: PASS
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`: PASS
- `build/p11ah/full_loop/run.log`: PASS
- `build/p11aj/p11aj/run.log`: PASS

5. Hygiene/purity
- `build/evidence/g6_multi_track_20260330/logs/check_design_purity.log`: PASS
- `build/evidence/g6_multi_track_20260330/logs/check_repo_hygiene_pre.log`: PASS
- `build/evidence/g6_multi_track_20260330/logs/check_repo_hygiene_post.log`: PASS

## Bundle manifest
- `build/evidence/g6_multi_track_20260330/evidence_manifest.txt`
- `build/evidence/g6_multi_track_20260330/evidence_summary.txt`

## Posture
- local-only
- not Catapult closure
- not SCVerify closure
