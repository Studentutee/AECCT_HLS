# TOP MANAGED SRAM W4-M3 KV EVIDENCE INDEX (2026-03-30)

## Scope
- W4-M3 dedicated bounded pass: Phase-A KV phase-entry caller-fed x-row probe.
- Local-only evidence bundle:
  - `build/evidence/w4m3_kv_probe_campaign_20260330/`

## Evidence manifest
- `build/evidence/w4m3_kv_probe_campaign_20260330/evidence_manifest.txt`
- `build/evidence/w4m3_kv_probe_campaign_20260330/evidence_summary.txt`
- `build/evidence/w4m3_kv_probe_campaign_20260330/commands_run.txt`

## Targeted validation
- `build/p11w4m3/kv_phase_entry_probe/build.log`
- `build/p11w4m3/kv_phase_entry_probe/run.log`
- PASS anchors:
  - `W4M3_KV_CALLER_FED_XROW_VISIBLE PASS`
  - `W4M3_KV_OWNERSHIP_CHECK PASS`
  - `W4M3_KV_NO_SPURIOUS_TOUCH PASS`
  - `W4M3_KV_EXPECTED_COMPARE PASS`
  - `W4M3_KV_PROBE_MISMATCH_REJECT PASS`

## Guard / regression
- `build/top_managed_sram_guard/check_top_managed_sram_boundary_regression.log`
- `build/helper_channel_guard/check_helper_channel_split_regression.log`

## Required retained reruns
- `build/p11w4m3/phasea_q_phase_entry_probe/run.log`
- `build/p11w4m2/softmaxout_phase_entry_probe/run.log`
- `build/p11g6/ffn_w1_bias_descriptor/run.log`
- `build/p11g6/ffn_fallback_observability/run.log`
- `build/p11g5/ffn_w1_fallback_policy/run.log`
- `build/p11g5/ffn_fallback_policy/run.log`
- `build/p11g5/ffn_closure_campaign/run.log`
- `build/p11g5/wave3_ffn_payload_migration/run.log`
- `build/p11g5/wave35_ffn_w1_weight_migration/run.log`
- `build/p11ah/full_loop/run.log`
- `build/p11aj/p11aj/run.log`

## Purity / hygiene
- `build/evidence/w4m3_kv_probe_campaign_20260330/logs/check_design_purity.log`
- `build/evidence/w4m3_kv_probe_campaign_20260330/logs/check_repo_hygiene_pre.log`
- `build/evidence/w4m3_kv_probe_campaign_20260330/logs/check_repo_hygiene_post.log`

## Closure posture
- local-only evidence only.
- not Catapult closure.
- not SCVerify closure.
