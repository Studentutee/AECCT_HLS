# TOP MANAGED SRAM G5 FFN FALLBACK POLICY TIGHTENING (2026-03-30)

## Scope
- Bounded fallback-policy tightening pass on FFN path already migrated to top-fed descriptors.
- No external formal contract change.
- No remote simulator/PLI/site-local flow touched.

## Primary candidate (selected)
- Strict W2 top-fed descriptor policy gate:
  - require ready descriptors for W2 input/weight/bias,
  - reject W2 stage if strict mode enabled and descriptor set is not ready,
  - expose reject and fallback-touch observability anchors.

## Backup candidate
- W1-only explicit-valid tightening.
- Kept as backup due lower architecture impact than strict W2 policy on active path.

## Patch summary
- `FFNLayer0`:
  - Added `fallback_policy_flags`, `fallback_policy_reject_flag`, `fallback_legacy_touch_counter` optional controls.
  - Added strict `FFN_POLICY_REQUIRE_W2_TOPFED` gate for W2 descriptor readiness.
  - Added explicit legacy fallback touch accounting.
- `TransformerLayer`:
  - W2 stage dispatch now enables strict policy flag where caller preloads all W2 descriptors.

## Validation summary (local-only)
- New targeted fallback policy runner: PASS.
- Boundary regression: PASS.
- Existing FFN campaign regressions (wave3/wave3.5/closure runner): PASS.
- Mainline/provenance/purity/hygiene: PASS.

## Closure posture
- local-only
- not Catapult closure
- not SCVerify closure

## Deferred boundary
- Full fallback removal is deferred.
- W1 strict policy tightening remains deferred.
- Wave4 attention/phase migration remains deferred.
