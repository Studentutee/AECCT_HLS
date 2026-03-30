# TOP MANAGED SRAM G5 FFN W1 FALLBACK POLICY TIGHTENING (2026-03-30)

## Scope
- Bounded fallback-policy tightening on FFN W1 path only.
- No external formal contract change.
- No remote simulator/PLI/site-local flow touched.
- W2 strict policy from prior round kept intact (not reworked).

## W1 fallback inventory baseline
- W1 x consume fallback existed when top-fed x pointer/valid was missing.
- W1 weight consume fallback existed when top-fed W1 weight pointer/valid was missing.
- Prior round lacked strict W1 descriptor-ready reject gate.

## Primary candidate (selected)
- Strict W1 top-fed descriptor-ready gating in `FFNLayer0`:
  - require both x descriptor-ready and W1 weight descriptor-ready when strict flag is set,
  - deterministic reject on missing descriptor-ready state,
  - no W1 compute/store activity on reject.
- Caller path (`TransformerLayer`) passes strict flag and explicit x descriptor valid override.

## Backup candidate
- W1 weight-only strict gating.
- Not selected due weaker architecture push (x path could still silently fallback).

## Patch summary
- `include/FfnDescBringup.h`
  - Added `FFN_POLICY_REQUIRE_W1_TOPFED`.
- `src/blocks/FFNLayer0.h`
  - Added `topfed_x_words_valid_override`.
  - Added explicit `require_w1_topfed` gate and descriptor-ready checks.
  - W1 strict mode now sets `fallback_policy_reject_flag` and returns before compute when descriptors are incomplete.
  - W1 x top-fed consume now uses `x_idx < topfed_x_valid`.
- `src/blocks/TransformerLayer.h`
  - W1 stage calls now pass `FFN_POLICY_REQUIRE_W1_TOPFED`.
  - W1 stage calls now pass `(u32_t)ffn_x_words` as x descriptor-ready valid override.

## Validation summary (local-only)
- New targeted W1 fallback-policy runner: PASS.
- Boundary regression guard: PASS.
- Existing W2 fallback-policy and FFN closure regressions: PASS.
- Mainline/provenance/purity/hygiene checks: PASS.

## Closure posture
- local-only
- not Catapult closure
- not SCVerify closure

## Deferred boundary
- W1 fallback is tightened but not fully removed in non-strict mode.
- W1 bias descriptorization is still deferred.
- Wave4 attention/phase migration remains deferred.
