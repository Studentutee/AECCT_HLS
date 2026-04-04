# BLOCKER_P11BD_SCORE_STAGE_FEASIBILITY

## Scope
- bucket under audit: `q_prebuilt=1, kv_prebuilt=1, score_prebuilt=0, out_prebuilt=0, attn_out_topfed_payload_enable=0`
- posture: local-only, compile-backed evidence, not Catapult closure, not SCVerify closure

## Code-Level Evidence
- `src/blocks/TransformerLayer.h:92` defines compat-shell surface with only:
  - `TRANSFORMER_ATTN_COMPAT_SHELL_DISABLED`
  - `TRANSFORMER_ATTN_COMPAT_SHELL_FULL`
  - `TRANSFORMER_ATTN_COMPAT_SHELL_OUT_ONLY`
- `src/blocks/TransformerLayer.h:131` keeps non-selected partial buckets at fallback boundary:
  - `return TRANSFORMER_ATTN_COMPAT_SHELL_FULL;`
- `src/blocks/TransformerLayer.h:331` and `src/blocks/TransformerLayer.h:786` instantiate only `ATTN_STAGE_FULL` on the FULL branch.
- `src/blocks/TransformerLayer.h:342` and `src/blocks/TransformerLayer.h:797` instantiate only `ATTN_STAGE_OUT` on the OUT_ONLY branch.
- `src/blocks/AttnLayer0.h:518` shows score-stage kernel exists (`ATTN_STAGE_SCORES`), but this stage is not surfaced by the Transformer compat-shell selector/call-site boundary.

## Feasibility Verdict
- Current verdict: **blocked for safe bounded contraction** from FULL to score-stage shell for this bucket.
- Reason:
  - selector surface lacks a score-stage shell enum/state;
  - Transformer call-sites currently route only FULL or OUT_ONLY;
  - no existing compat-shell contract path that safely exposes score-only execution as a first-class branch.

## Compile-Backed Runtime Evidence
- `build/p11aj/p11aj/run.log` includes:
  - `QKV_READY_SCORE_NOT_PREBUILT_REMAINS_FULL PASS`
  - `QKV_READY_SCORE_NOT_PREBUILT_SCORE_STAGE_FEASIBILITY_BLOCKED_ENUM_SURFACE PASS`
  - `ATTN_COMPAT_SHELL_TRUTH_TABLE_AUDIT PASS combos=32`
  - `PASS: tb_top_managed_sram_provenance_p11aj`
  - `PASS: run_p11aj_top_managed_sram_provenance`

## Minimum Capability Needed Before Next Cut
1. add one compat-shell stage enum for score-stage shell in `TransformerAttnCompatShellStage`.
2. map only the audited bucket to that new shell in selector logic.
3. add bounded call-site branch(es) to instantiate `AttnLayer0<ATTN_STAGE_SCORES>` with unchanged ownership boundary.
4. add task-local TB + runner pass lines for:
   - selected bucket to score-stage shell,
   - fully-prebuilt cases non-regression,
   - other partial buckets non-regression.
