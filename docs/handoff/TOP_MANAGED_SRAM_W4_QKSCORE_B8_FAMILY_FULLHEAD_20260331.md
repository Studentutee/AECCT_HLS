# TOP_MANAGED_SRAM_W4_QKSCORE_B8_FAMILY_FULLHEAD_20260331

## Scope
- Round: W4-B8
- Intent: bounded family coverage expansion on QkScore bridge only.
- Governance: local-only, compile-first/evidence-first, not Catapult closure, not SCVerify closure.

## Patch Summary
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
  - `kScoreTileBridgeFamilyMaxCases = 8u` (from 4u).
  - Kept mixed single+family coexistence and same-head overlap reject behavior.
- `scripts/check_top_managed_sram_boundary_regression.ps1`
  - Added W4-B8 family capacity anchor check.
- Added task-local assets:
  - `tb/tb_w4b8_qkscore_family_fullhead_bridge.cpp`
  - `scripts/local/run_p11w4b8_qkscore_family_fullhead_bridge.ps1`

## Evidence Gate Result
- Targeted runner PASS.
- B7/B6 baseline recheck PASS.
- Structural checks PASS.
- No external Top 4-channel contract change.
- No second ownership/arbitration semantics introduced.

## Notes
- This round is coverage expansion only, not compute/reduction/writeback skeleton rewrite.
