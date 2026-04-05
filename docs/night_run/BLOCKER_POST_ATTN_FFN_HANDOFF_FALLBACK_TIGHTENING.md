# BLOCKER_POST_ATTN_FFN_HANDOFF_FALLBACK_TIGHTENING

Date: 2026-04-06  
Scope: Phase A compile-backed feasibility audit for post-attention next-scope candidate `FFN handoff fallback tightening` (Top/TransformerLayer/FFN seam)

## 1) Verdict
- Status: BLOCKER
- Decision: do not apply design shrink in this round.
- Posture: local-only, compile-first, evidence-first, not Catapult closure, not SCVerify closure.

## 2) Compile-Backed Evidence Used
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11av_top_ffn_handoff_assembly_smoke.ps1`
  - `PASS: run_p11av_top_ffn_handoff_assembly_smoke`
  - `TOP_PIPELINE_LID0_FFN_HANDOFF_EXPECTED_COMPARE PASS`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11g6_ffn_fallback_observability.ps1`
  - `G6FFN_SUBWAVE_B_REJECT_STAGE_W1 PASS`
  - `G6FFN_SUBWAVE_B_REJECT_STAGE_W2 PASS`
  - `G6FFN_SUBWAVE_B_NONSTRICT_FALLBACK_OBS PASS`
- `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
  - `PASS: check_top_managed_sram_boundary_regression`

## 3) Blocking Boundary (Producer/Consumer/Fallback Seam)
- `FFNLayer0` already has strict policy gates:
  - `FFN_POLICY_REQUIRE_W1_TOPFED`
  - `FFN_POLICY_REQUIRE_W2_TOPFED`
  - reject-stage observability (`FFN_REJECT_STAGE_W1/W2`)
- But `TransformerLayer` currently materializes descriptor-miss fallback by preloading from SRAM into local topfed buffers, then still dispatches FFN stage with strict flags.
- Therefore, descriptor-miss does not become a hard reject at the Top/TransformerLayer seam today; it remains compatibility fallback behavior by design.
- Existing `p11av` acceptance explicitly checks invalid/disabled descriptor paths remain fallback-compatible and output-compatible (baseline-aligned), including monotonic fallback accounting across layer loops.

## 4) Why This Is Not Safe For Immediate Bounded Cut
- Tightening this seam further (for example, removing descriptor-miss preload fallback) would change current accepted behavior at the TransformerLayer consumer boundary, not just internal FFN policy wording.
- Without a new explicit contract for descriptor-miss handling at Top/TransformerLayer level, this risks functional/regression breakage.
- This crosses a behavioral boundary (fallback policy semantics), not just a mechanical refactor seam.

## 5) Minimum Preconditions To Re-open This Cut
- Define one explicit seam contract for descriptor-miss path:
  - Option A: hard reject/abort policy
  - Option B: controlled compatibility fallback policy (current behavior)
- If Option A is chosen, add a Top-visible fail/skip handling path and acceptance lines for reject-stage propagation.
- Update task-local acceptance chain to reflect the new contract (not only `p11av` monotonic counters, but also expected output behavior under descriptor-miss).

## 6) Scope Guardrails Preserved
- Top remains the only shared-SRAM owner.
- No new shared-memory arbitration semantics were introduced.
- No external Top 4-channel contract change.
- No attention re-open, no FFN/LayerNorm broad refactor.
