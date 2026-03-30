# TOP_MANAGED_SRAM_G7_DIRECT_SRAM_CAMPAIGN_20260330

## Scope
- Task: G7 remaining direct-SRAM eradication campaign (single-run, evidence-first, staged)
- posture: local-only
- closure: not Catapult closure; not SCVerify closure
- non-goals: no remote simulator/PLI/site-local work, no external formal contract change, no broad rewrite

## Global inventory conclusion
- Remaining direct-SRAM hotspots are still concentrated in Attn/Phase main compute/writeback loops.
- FFN main consume paths are already mostly caller-fed/top-fed; remaining risk is fallback/writeback boundary.

## Waves executed this round

### Wave A (completed): FFN residual strict fallback tightening
- change: `FFNLayer0` strict W1 policy now requires **x + W1 weight + W1 bias** descriptor-ready.
- reason: eliminate hidden bias fallback when strict top-fed policy is enabled.
- targeted evidence:
  - `build/p11g7/ffn_w1_bias_descriptor_strict/run.log`
  - pass banner: `G7FFN_W1_BIAS_DESCRIPTOR_REJECT PASS`

### Wave B (completed): W4-M3 KV probe hardening
- change: `AttnPhaseATopManagedKv` phase-entry probe now requires full-row descriptor-ready words (`probe_valid == d_model`).
- reason: avoid partial probe silently passing as valid ownership check.
- targeted evidence:
  - `build/p11w4m3/kv_phase_entry_probe/run.log`
  - pass banner: `W4M3_KV_PROBE_DESCRIPTOR_REJECT PASS`

### Wave C (completed): SramView remove-readiness matrix
- produced block-level readiness map (R0~R4) and exact pre-removal conditions.
- see: `docs/handoff/TOP_MANAGED_SRAM_G7_REMOVE_READINESS_20260330.md`

### Wave D (completed): residual blocker isolation
- isolated deferred direct-SRAM blockers by file/loop/reason.
- see completion + evidence index docs.

## Deferred boundary (explicit)
- AttnLayer0 / Phase-A / Phase-B deep compute & writeback loops remain SRAM-centric.
- this round only hardened bounded entry/strict policy points; no full Wave4 payload migration claimed.
