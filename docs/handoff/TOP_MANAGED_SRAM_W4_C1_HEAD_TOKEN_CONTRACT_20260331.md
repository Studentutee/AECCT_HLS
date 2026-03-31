# TOP_MANAGED_SRAM_W4_C1_HEAD_TOKEN_CONTRACT_20260331

## Scope
- Campaign: W4-C1 SoftmaxOut family descriptor head/token contract-only + probe validation.
- Posture:
  - local-only
  - compile-first / evidence-first
  - not Catapult closure
  - not SCVerify closure

## Exact Hook Point
- Mainline function:
  - `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`
- C0 contract landing zone:
  - `ATTN_P11AF_MAINLINE_INIT_ACC_TILE_LOOP` (single+family bridge consume neighborhood).
- C1 insertion zone:
  - family descriptor precheck block before compute loops.
  - `ATTN_P11AF_TILE_BRIDGE_FAMILY_DESC_PROBE_LOOP` (descriptor observability).
  - family selector condition in init-acc tile consume loop.

## Contract Decision
- Decision: YES, clean bounded contract-only cut exists for C1.
- Reason:
  - changes stay in selector/descriptor/observability/probe layer.
  - no online renorm/acc/writeback skeleton rewrite.
  - no external Top 4-channel contract drift.
  - no second ownership/arbitration semantics.

## Minimal Descriptor Set (C1)
- Selector fields:
  - `phase_tile_bridge_family_head_idx[c]`
  - `phase_tile_bridge_family_d_tile_idx[c]` (existing C0 selector axis)
- Payload-span fields:
  - `phase_tile_bridge_family_key_token_begin[c]`
  - `phase_tile_bridge_family_key_token_count[c]`
- Observability-only fields:
  - `phase_tile_bridge_family_desc_visible_count`
  - `phase_tile_bridge_family_desc_case_mask`
- Internal helper passthrough only (must not become external contract):
  - all C1 descriptor arrays/counters above in `run_p11af_layer0_top_managed_softmax_out(...)`.

## Bounded Guard
- Current C1 scope is contract-only at init-acc boundary.
- Guard enforces descriptor token span as init-acc-local:
  - `case_key_token_begin == 0`
  - `case_key_token_count == 1`
- Any descriptor outside this bounded span is rejected (fallback path).

## Implemented Artifacts
- Design/helper:
  - `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
  - `src/Top.h`
- Gate anchor:
  - `scripts/check_top_managed_sram_boundary_regression.ps1`
- Targeted probe:
  - `tb/tb_w4c1_softmaxout_head_token_contract_probe.cpp`
  - `scripts/local/run_p11w4c1_softmaxout_head_token_contract_probe.ps1`

## Non-Goals (Kept Out)
- No renorm loop rewrite.
- No acc loop rewrite.
- No writeback skeleton rewrite.
- No external Top contract change.
- No claim of full direct-SRAM closure.
