# TOP MANAGED SRAM G6 FFN NEAR-CLOSURE CAMPAIGN (2026-03-30)

## Scope
- Single-run multi-track campaign, bounded to FFN near-closure plus Wave4 feasibility.
- No external formal contract change.
- No remote simulator/PLI/site-local flow touched.

## Track A inventory summary
- Remaining FFN gap at start:
  - W1 bias consume was still direct-SRAM oriented.
  - strict W1/W2 reject path lacked harmonized stage-level observability.
  - compatibility fallback remained in non-strict mode.

## Subwave ranking
1. Subwave A (selected): W1 bias caller-fed/top-fed descriptor consume anchor.
2. Subwave B (selected): W1/W2 strict reject-stage observability harmonization.
3. Subwave C (deferred): combined strict/no-fallback mode extension.
4. Subwave D (deferred): writeback boundary tightening.

## Subwave A (completed)
- Added W1 bias top-fed descriptor path in `FFNLayer0`.
- Added W1 bias preload/dispatch in `TransformerLayer` pointer and bridge paths.
- Added targeted validation runner:
  - `scripts/local/run_p11g6_ffn_w1_bias_descriptor.ps1`
  - `tb/tb_g6_ffn_w1_bias_descriptor_p11g6a.cpp`

## Subwave B (completed)
- Added harmonized reject-stage observability:
  - `FFN_REJECT_STAGE_NONE`
  - `FFN_REJECT_STAGE_W1`
  - `FFN_REJECT_STAGE_W2`
- Added targeted validation runner:
  - `scripts/local/run_p11g6_ffn_fallback_observability.ps1`
  - `tb/tb_g6_ffn_fallback_observability_p11g6b.cpp`

## Validation summary (local-only)
- Subwave A runner: PASS.
- Subwave B runner: PASS.
- Boundary regression guard: PASS.
- Existing G5 FFN + mainline/provenance regressions: PASS.
- Helper split guard, design purity, hygiene pre/post: PASS.

## Closure posture
- local-only
- not Catapult closure
- not SCVerify closure

## Deferred boundary
- FFN full fallback elimination remains deferred.
- FFN writeback path streamization remains deferred.
- Wave4 code patch remains deferred in this round (feasibility only).
