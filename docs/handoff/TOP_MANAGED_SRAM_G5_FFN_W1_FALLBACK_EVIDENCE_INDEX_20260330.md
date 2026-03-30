# TOP MANAGED SRAM G5 FFN W1 FALLBACK EVIDENCE INDEX (2026-03-30)

## Claim -> Evidence
1. W1 top-fed path is primary under strict policy
- `build/p11g5/ffn_w1_fallback_policy/run.log`
  - `G5FFN_W1_FALLBACK_POLICY_TOPFED_PRIMARY PASS`
  - `G5FFN_W1_FALLBACK_POLICY_EXPECTED_COMPARE PASS`

2. W1 fallback is controlled and deterministic
- `build/p11g5/ffn_w1_fallback_policy/run.log`
  - `G5FFN_W1_FALLBACK_POLICY_CONTROLLED_FALLBACK PASS`

3. Missing descriptor-ready in strict mode rejects before W1 writeback
- `build/p11g5/ffn_w1_fallback_policy/run.log`
  - `G5FFN_W1_FALLBACK_POLICY_REJECT_ON_MISSING_DESCRIPTOR PASS`
  - `G5FFN_W1_FALLBACK_POLICY_NO_STALE_STATE PASS`

4. No spurious legacy SRAM touch in targeted W1 scope
- `build/p11g5/ffn_w1_fallback_policy/run.log`
  - `G5FFN_W1_FALLBACK_POLICY_NO_SPURIOUS_TOUCH PASS`

5. Guard anchor coverage (W1 + W2)
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
  - `guard: G5 FFN W1 fallback policy strict top-fed gating anchors OK`
  - `guard: G5 FFN fallback policy strict W2 top-fed gating anchors OK`
  - `PASS: check_top_managed_sram_boundary_regression`

6. Regression compatibility retained
- `build/p11g5/ffn_fallback_policy/run.log`: PASS
- `build/p11g5/ffn_closure_campaign/run.log`: PASS
- `build/p11g5/wave3_ffn_payload_migration/run.log`: PASS
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`: PASS
- `build/p11ah/full_loop/run.log`: PASS
- `build/p11aj/p11aj/run.log`: PASS

7. Hygiene/purity
- `build/evidence/g5_ffn_w1_fallback_policy_20260330/logs/check_design_purity.log`: PASS
- `build/evidence/g5_ffn_w1_fallback_policy_20260330/logs/check_repo_hygiene_pre.log`: PASS
- `build/evidence/g5_ffn_w1_fallback_policy_20260330/logs/check_repo_hygiene_post.log`: PASS

## Bundle manifest
- `build/evidence/g5_ffn_w1_fallback_policy_20260330/evidence_manifest.txt`
- `build/evidence/g5_ffn_w1_fallback_policy_20260330/evidence_summary.txt`

## Posture
- local-only
- not Catapult closure
- not SCVerify closure
