# OVERNIGHT HELPER SPLIT AND GUARD PROGRESS (2026-03-29)

## Executive Summary
- Continued helper-channel mixed-payload HOL risk reduction on existing repo foundation.
- Prior AF/AE/AD/AC split claims remain audited and accepted.
- Added and extended regression guard coverage, including AC work-tile `K/V` out-channel split anchors.
- Completed low-risk helper/staging split this round:
  - `src/blocks/AttnPhaseATopManagedKv.h` work-tile output path (`K/V`) now split by payload class.
- Current helper/staging hotspot inventory has no unresolved mixed-payload single-channel hotspot in scoped paths.
- Governance posture remains local-only: not Catapult closure; not SCVerify closure.

## Accepted / Likely-Good Tasks
- TASK A (prior): helper split acceptance audit completed; AF/AE/AD/AC marked ACCEPTABLE.
  - artifact: `docs/handoff/HELPER_SPLIT_ACCEPTANCE_AUDIT_20260329.md`
- TASK B: mixed-payload regression guard checker maintained and expanded.
  - artifacts:
    - `scripts/check_helper_channel_split_regression.ps1`
    - `docs/handoff/HELPER_CHANNEL_REGRESSION_GUARD_20260329.md`
    - `build/helper_channel_guard/check_helper_channel_split_regression.log`
- TASK C: helper/staging hotspot inventory updated to reflect latest split state.
  - artifact: `docs/handoff/HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md`
- TASK D (this round): AC work-tile `K/V` out-channel split completed with local validation PASS.
  - artifact: `docs/handoff/HELPER_LOWRISK_SPLIT_KV_OUT_20260329.md`

## Tasks Needing Human Diff Review
- `src/blocks/AttnPhaseATopManagedKv.h`
  - work-tile output channel changed from shared `attn_work_pkt_ch_t out_ch` to split `attn_k_work_pkt_ch_t k_ch` + `attn_v_work_pkt_ch_t v_ch`.
- `tb/tb_kv_build_stream_stage_p11ac.cpp`
  - added work-tile out split probe with explicit `WORK_TILE_OUT_SPLIT_PATH PASS`.
- `scripts/local/run_p11ac_phasea_top_managed.ps1`
  - runner pass-gate extended to require work-tile out split probe PASS.
- `scripts/check_p11ac_phasea_surface.ps1`
  - surface anchors extended for work-tile out split signatures and read/write split anchors.
- `scripts/check_helper_channel_split_regression.ps1`
  - AC work-tile `K/V` out split anti-regression anchors added.

## Tasks Attempted But Blocked
- No hard blocker in this round.
- During TB probe bring-up, two local iterations were needed before final PASS:
  - iteration 1: probe compare mismatch due tile-span assumption.
  - iteration 2: probe consume timing mismatch.
  - iteration 3: fixed and PASS.

## Exact Artifact Index
- Acceptance / analysis:
  - `docs/handoff/HELPER_SPLIT_ACCEPTANCE_AUDIT_20260329.md`
  - `docs/handoff/HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md`
- Guard:
  - `scripts/check_helper_channel_split_regression.ps1`
  - `docs/handoff/HELPER_CHANNEL_REGRESSION_GUARD_20260329.md`
  - `build/helper_channel_guard/check_helper_channel_split_regression.log`
  - `build/helper_channel_guard/check_helper_channel_split_regression_summary.txt`
- This-round split:
  - `docs/handoff/HELPER_LOWRISK_SPLIT_KV_OUT_20260329.md`
  - `build/p11ac_kv_out_impl/run.log`
  - `build/p11ac_kv_out/check_p11ac_phasea_surface.log`

## Suggested 10-Minute Review Order
1. Open `docs/handoff/HELPER_LOWRISK_SPLIT_KV_OUT_20260329.md` to confirm scope and evidence.
2. Review `src/blocks/AttnPhaseATopManagedKv.h` split signatures and split read/write anchors.
3. Open `build/p11ac_kv_out_impl/run.log` and verify:
   - `WORK_TILE_OUT_SPLIT_PATH PASS`
   - `STREAM_ORDER PASS`
   - `MEMORY_ORDER PASS`
   - `EXACT_SCR_KV_COMPARE PASS`
   - `MAINLINE_PATH_TAKEN PASS`
   - `FALLBACK_NOT_TAKEN PASS`
4. Open `build/helper_channel_guard/check_helper_channel_split_regression.log` and verify:
   - `guard: AC work-tile k/v out split anchors OK`
   - `PASS: check_helper_channel_split_regression`
5. Confirm inventory closure in `docs/handoff/HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md`.

## Governance Posture
- local-only evidence only
- not Catapult closure
- not SCVerify closure
- Top remains sole production shared-SRAM owner in production contract
