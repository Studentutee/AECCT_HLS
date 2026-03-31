# TOP_MANAGED_SRAM_W4_QKSCORE_B6_FAMILY_COVERAGE_20260331

## Summary
- Round ID: W4-B6 (Primary)
- Cut: expand selected family bridge capacity in `AttnPhaseBTopManagedQkScore` from 3 to 4 cases, still bounded/local-only.
- No external Top 4-channel contract change, no second ownership/arbitration semantics.

## Scope
- In: `src/blocks/AttnPhaseBTopManagedQkScore.h`, task-local TB/runner.
- Out: no broad rewrite, no Top contract drift, no Catapult/SCVerify closure claim.

## Key Changes
- `kScoreTileBridgeFamilyMaxCases` changed `3 -> 4`.
- Added TB: `tb/tb_w4b6_qkscore_family_bridge.cpp`.
- Added runner: `scripts/local/run_p11w4b6_qkscore_family_bridge.ps1`.

## Validation Highlights
- `W4B6_QKSCORE_FAMILY_BRIDGE_VISIBLE PASS`
- `W4B6_QKSCORE_FAMILY_OWNERSHIP_CHECK PASS`
- `W4B6_QKSCORE_FAMILY_MULTI_CASE_ANTI_FALLBACK PASS`
- `W4B6_QKSCORE_FAMILY_EXPECTED_COMPARE PASS`
- `W4B6_QKSCORE_FAMILY_LEGACY_COMPARE PASS`
- `W4B6_QKSCORE_FAMILY_NO_SPURIOUS_TOUCH PASS`
- `W4B6_QKSCORE_FAMILY_MISMATCH_REJECT PASS`

## Baseline Recheck
- Re-ran B5 baseline and confirmed `PASS: run_p11w4b5_qkscore_family_bridge`.

## Posture
- local-only
- compile-first / evidence-first
- not Catapult closure
- not SCVerify closure
