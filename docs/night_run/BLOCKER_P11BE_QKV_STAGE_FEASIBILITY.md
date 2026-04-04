# BLOCKER_P11BE_QKV_STAGE_FEASIBILITY

## Scope
- bucket under audit: `q_prebuilt=1, kv_prebuilt=0, score_prebuilt=0, out_prebuilt=0, attn_out_topfed_payload_enable=0`
- posture: local-only, compile-backed evidence, not Catapult closure, not SCVerify closure

## Feasibility Verdict
- current verdict: **blocked for safe bounded contraction** from `FULL` at this time.
- reason summary: this bucket lacks K/V-prebuilt readiness, so score-stage shell is not a safe substitution.

## Gap Layering
1. enum surface
- current compat-shell enum already has `DISABLED/FULL/OUT_ONLY/SCORES_ONLY`.
- no dedicated safe shell exists yet for "Q ready but KV not prebuilt" contraction.

2. selector mapping
- mapping this bucket away from `FULL` would require a new stage contract with explicit K/V materialization ownership.
- current selector intentionally keeps this bucket on fallback boundary (`FULL`) to avoid partial compute hazards.

3. call-site dispatch
- `TransformerLayer` dispatches `FULL`, `OUT_ONLY`, `SCORES_ONLY`.
- no bounded dispatch branch exists for "materialize missing KV only + preserve downstream boundaries".

4. datapath boundary
- with `kv_prebuilt=0`, K/V materialization must still execute before score path.
- score-only shell cannot safely cover this bucket because it assumes score path entry preconditions are met.

5. consume/writeback semantics
- downstream consume (`post_concat -> attn_out`) is stable for existing shells.
- introducing a new intermediate shell for this bucket requires explicit writeback and fallback semantics review to avoid silent behavior shifts.

## Compile-Backed Evidence
- `build/p11aj/p11aj/run.log` contains:
  - `Q_READY_KV_NOT_PREBUILT_REMAINS_FULL PASS`
  - `Q_READY_KV_NOT_PREBUILT_QKV_STAGE_FEASIBILITY_BLOCKED PASS`
  - `PASS: tb_top_managed_sram_provenance_p11aj`
  - `PASS: run_p11aj_top_managed_sram_provenance`
- `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains:
  - `P11ANB_TRANSFORMER_ATTN_SHELL_Q_READY_KV_NOT_PREBUILT_REMAINS_FULL PASS`
  - `PASS: tb_p11anb_attnlayer0_boundary_seam_contract`
  - `PASS: run_p11anb_attnlayer0_boundary_seam_contract`

## Minimal Next Step
1. perform bounded design sketch for a "Q-only-ready" compat shell that still guarantees KV materialization ownership seam and explicit writeback contract.
2. add case-specific p11aj/p11anb truth-table probes before any selector remap.
