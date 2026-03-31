# TOP_MANAGED_SRAM_W4_QKSCORE_B9_LONGSPAN_20260331

## Scope
- Round: W4-B9
- Entry condition: B8 clean PASS confirmed.
- Intent: bounded bridge span generalization (tile-domain flatten -> token-domain flatten).

## Patch Summary
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
  - Family flatten stride promoted to token-domain:
    - `kScoreTileBridgeFamilyStrideWords = (uint32_t)ATTN_TOKEN_COUNT`
  - Single bridge guard promoted from tile-bound to token-span guard:
    - `score_tile_bridge_valid_words > (uint32_t)ATTN_TOKEN_COUNT` reject.
  - Family guard promoted to token-span guard:
    - `valid > token_count || valid > (uint32_t)ATTN_TOKEN_COUNT` reject.
- `scripts/check_top_managed_sram_boundary_regression.ps1`
  - Added W4-B9 anchors for token-span family stride and guards.
- Added task-local assets:
  - `tb/tb_w4b9_qkscore_longspan_bridge.cpp`
  - `scripts/local/run_p11w4b9_qkscore_longspan_bridge.ps1`
- Compatibility updates for baseline family/mixed TB flatten payload layout:
  - `tb/tb_w4b5_qkscore_family_bridge.cpp`
  - `tb/tb_w4b6_qkscore_family_bridge.cpp`
  - `tb/tb_w4b7_qkscore_mixed_bridge.cpp`
  - `tb/tb_w4b8_qkscore_family_fullhead_bridge.cpp`

## Evidence Gate Result
- Targeted runner PASS (`run_p11w4b9_qkscore_longspan_bridge`).
- Required longspan consume-count exact-match PASS line present.
- Baseline recheck PASS (`W4-B8`, `W4-B7`, plus `W4-B6`).
- Structural checks PASS.
- No external Top 4-channel contract drift.
- No second ownership/arbitration semantics introduced.

## Stop Assessment
- B9 remains bounded and completed.
- Next candidate C0 is skeleton-adjacent in SoftmaxOut online path and deferred for this session.
