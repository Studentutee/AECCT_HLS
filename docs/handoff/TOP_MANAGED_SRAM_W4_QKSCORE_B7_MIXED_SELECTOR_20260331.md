# TOP_MANAGED_SRAM_W4_QKSCORE_B7_MIXED_SELECTOR_20260331

## Summary
- Round ID: W4-B7 (Primary)
- Cut: remove hard mutual exclusion between single-case bridge and family bridge, allow same-call mixed selection with overlap guard on same-head ranges.
- Still bounded/local-only; compute/reduction/writeback skeleton unchanged.

## Scope
- In: `src/blocks/AttnPhaseBTopManagedQkScore.h`, task-local TB/runner.
- Out: no external Top contract change, no second ownership semantics, no broad attention rewrite.

## Key Changes
- In family precheck, add same-head overlap reject against single-case selected window.
- Keep selection precedence deterministic: family selected-case first, otherwise single-case bridge.
- Added TB: `tb/tb_w4b7_qkscore_mixed_bridge.cpp`.
- Added runner: `scripts/local/run_p11w4b7_qkscore_mixed_bridge.ps1`.

## Validation Highlights
- `W4B7_QKSCORE_MIXED_BRIDGE_VISIBLE PASS`
- `W4B7_QKSCORE_MIXED_OWNERSHIP_CHECK PASS`
- `W4B7_QKSCORE_MIXED_MULTI_PATH_ANTI_FALLBACK PASS`
- `W4B7_QKSCORE_MIXED_EXPECTED_COMPARE PASS`
- `W4B7_QKSCORE_MIXED_LEGACY_COMPARE PASS`
- `W4B7_QKSCORE_MIXED_NO_SPURIOUS_TOUCH PASS`
- `W4B7_QKSCORE_MIXED_MISMATCH_REJECT PASS`

## Baseline Recheck
- Re-ran B6 baseline and confirmed `PASS: run_p11w4b6_qkscore_family_bridge`.

## Posture
- local-only
- compile-first / evidence-first
- not Catapult closure
- not SCVerify closure
