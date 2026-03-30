# TOP_MANAGED_SRAM_G7_DIRECT_SRAM_EVIDENCE_INDEX_20260330

## Evidence bundle
- bundle root: `build/evidence/g7_direct_sram_campaign_20260330`
- manifest: `build/evidence/g7_direct_sram_campaign_20260330/evidence_manifest.txt`
- summary: `build/evidence/g7_direct_sram_campaign_20260330/evidence_summary.txt`

## Wave A evidence (FFN strict bias descriptor)
- `build/p11g7/ffn_w1_bias_descriptor_strict/build.log`
- `build/p11g7/ffn_w1_bias_descriptor_strict/run.log`
- `build/p11g7/ffn_w1_bias_descriptor_strict/verdict.txt`

## Wave B evidence (W4-M3 KV probe hardening)
- `build/p11w4m3/kv_phase_entry_probe/build.log`
- `build/p11w4m3/kv_phase_entry_probe/run.log`
- `build/p11w4m3/kv_phase_entry_probe/verdict.txt`

## Boundary/guard evidence
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression_summary.txt`
- `build/helper_channel_guard/check_helper_channel_split_regression.log`
- `build/helper_channel_guard/check_helper_channel_split_regression_summary.txt`

## Retained regression chain evidence
- `build/p11w4m3/phasea_q_phase_entry_probe/run.log`
- `build/p11w4m2/softmaxout_phase_entry_probe/run.log`
- `build/p11w4m1/qkscore_phase_entry_probe/run.log`
- `build/p11g6/ffn_w1_bias_descriptor/run.log`
- `build/p11g6/ffn_fallback_observability/run.log`
- `build/p11g5/ffn_w1_fallback_policy/run.log`
- `build/p11g5/ffn_fallback_policy/run.log`
- `build/p11g5/ffn_closure_campaign/run.log`
- `build/p11g5/wave3_ffn_payload_migration/run.log`
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
- `build/p11ah/full_loop/run.log`
- `build/p11aj/p11aj/run.log`

## Purity/Hygiene evidence
- `build/evidence/g7_direct_sram_campaign_20260330/logs/check_design_purity.log`
- `build/evidence/g7_direct_sram_campaign_20260330/logs/check_repo_hygiene_pre.log`
- `build/evidence/g7_direct_sram_campaign_20260330/logs/check_repo_hygiene_post.log`

## Closure posture
- local-only evidence
- not Catapult closure
- not SCVerify closure
