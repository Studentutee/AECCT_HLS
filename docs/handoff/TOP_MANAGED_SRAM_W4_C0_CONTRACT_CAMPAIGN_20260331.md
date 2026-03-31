# TOP_MANAGED_SRAM_W4_C0_CONTRACT_CAMPAIGN_20260331

## Campaign Posture
- local-only
- compile-first / evidence-first
- not Catapult closure
- not SCVerify closure

## Phase 0: Inventory / Hook Mapping
- Target hook: `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- Init-acc consume neighborhood:
  - `ATTN_P11AF_MAINLINE_INIT_ACC_TILE_LOOP`
  - `ATTN_P11AF_MAINLINE_INIT_ACC_LOOP`
  - existing B1 selector `phase_tile_bridge_selected`
- Online softmax core/skeleton boundary:
  - `ATTN_P11AF_MAINLINE_RENORM_TILE_LOOP`
  - `ATTN_P11AF_MAINLINE_ACC_TILE_LOOP`
  - `ATTN_P11AF_MAINLINE_WRITEBACK_TILE_LOOP`

## Phase 1: Contract Decision
- Decision: `YES`, clean bounded cut exists.
- Bounded cut scope:
  - Add single+family selectable contract at init-acc consume boundary only.
  - Keep renorm/writeback loops unchanged.
- Forbidden-zone check:
  - No external Top 4-channel contract change.
  - No second ownership/arbitration semantics.
  - No skeleton rewrite.

## Phase 2: Minimal Implementation
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
  - Added optional family descriptor/observability arguments.
  - Added family precheck/select/compare/finalize loops in init-acc neighborhood only.
  - Added overlap reject between single bridge tile and family case tiles.
- `src/Top.h`
  - Added internal helper passthrough for new family descriptor/observability fields.
- `scripts/check_top_managed_sram_boundary_regression.ps1`
  - Added W4-C0 regex anchors and guard line.
- Added targeted assets:
  - `tb/tb_w4c0_softmaxout_contract_probe.cpp`
  - `scripts/local/run_p11w4c0_softmaxout_contract_probe.ps1`

## Phase 3: Evidence / Stop Decision
- Targeted C0 runner: PASS.
- Baseline recheck (B9/B8/B1): PASS.
- Structural gates: PASS.
- Stop decision for this mini-campaign: implementation success with bounded scope preserved.

## Topology Note
- Contract supports both single and family selectors.
- Targeted TB is topology-aware:
  - `d_tile_count > 1`: validates disjoint single+family coexistence.
  - `d_tile_count == 1`: validates family-only contract path (single remains disabled to avoid overlap).
