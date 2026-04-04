# BLOCKER_P11BD_SCORE_STAGE_FEASIBILITY

## Scope
- bucket under audit: `q_prebuilt=1, kv_prebuilt=1, score_prebuilt=0, out_prebuilt=0, attn_out_topfed_payload_enable=0`
- posture: local-only, compile-backed evidence, not Catapult closure, not SCVerify closure

## Status Update
- previous status: blocked
- current status: resolved in local bounded cut
- resolution summary:
  - compat-shell enum surface now includes `TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY`.
  - selector now maps `q=1, kv=1, score=0, out=0, payload=0` to `SCORES_ONLY`.
  - Transformer pointer/bridge call-sites now dispatch to `AttnLayer0<ATTN_STAGE_SCORES>` and commit explicit `post_concat -> attn_out` writeback.

## Feasibility Verdict
- verdict: **resolved for this bucket** (`q=1, kv=1, score=0, out=0, payload=0`) under local bounded cut.
- boundary notes:
  - only selected bucket contracts to `SCORES_ONLY`;
  - fully-prebuilt and other partial buckets retain previous behavior.

## Compile-Backed Runtime Evidence
- `build/p11aj/p11aj/run.log` includes:
  - `QKV_READY_SCORE_NOT_PREBUILT_TO_SCORES_STAGE PASS`
  - `ATTN_COMPAT_SHELL_TRUTH_TABLE_AUDIT PASS combos=32`
  - `PASS: tb_top_managed_sram_provenance_p11aj`
  - `PASS: run_p11aj_top_managed_sram_provenance`
- `build/p11anb/attnlayer0_boundary_seam_contract/run.log` includes:
  - `P11ANB_TRANSFORMER_ATTN_SHELL_QKV_READY_SCORE_NOT_PREBUILT_TO_SCORES_STAGE PASS`
  - `PASS: tb_p11anb_attnlayer0_boundary_seam_contract`
  - `PASS: run_p11anb_attnlayer0_boundary_seam_contract`

## Remaining Next-Cut Capability (for future buckets)
1. define next target bucket and prove stage-safe writeback boundary before selector change.
2. keep shell ownership seam explicit: Top owns shared SRAM; sub-blocks only consume provided windows.
3. extend task-local PASS chains before promoting new bucket to night-run ready state.
