# TOP_MANAGED_SRAM_G7_COMPLETION_20260330

## Executive summary
G7 completed two bounded code-level waves plus global inventory/readiness/blocker isolation:
1) FFN strict W1 now rejects when top-fed W1 bias descriptor is not ready.
2) W4-M3 KV probe now rejects short (non full-row) descriptors.
3) Global direct-SRAM hotspot/risk inventory and SramView remove-readiness matrix delivered.

## Completed tasks
- Wave A patch + targeted validation + guard updates
- Wave B patch + targeted validation + guard updates
- full required local rerun chain
- evidence bundle + manifest
- progress/mincuts/morning-review refresh

## Attempted but blocked
- full removal of SRAM-centric compute/writeback in AttnLayer0/Phase-A/Phase-B
- reason: requires broad cross-phase rewrite outside bounded scope

## Key changed files
- `src/blocks/FFNLayer0.h`
- `tb/tb_g5_ffn_w1_fallback_policy_p11g5w1fp.cpp`
- `scripts/local/run_p11g7_ffn_w1_bias_descriptor_strict.ps1`
- `src/blocks/AttnPhaseATopManagedKv.h`
- `tb/tb_w4m3_kv_phase_entry_probe.cpp`
- `scripts/local/run_p11w4m3_kv_phase_entry_probe.ps1`
- `scripts/check_top_managed_sram_boundary_regression.ps1`

## Evidence pointers
- `docs/handoff/TOP_MANAGED_SRAM_G7_DIRECT_SRAM_EVIDENCE_INDEX_20260330.md`
- `build/evidence/g7_direct_sram_campaign_20260330/evidence_manifest.txt`

## Governance posture
- local-only
- not Catapult closure
- not SCVerify closure
- no remote simulator/PLI/site-local work
- no external formal contract change
