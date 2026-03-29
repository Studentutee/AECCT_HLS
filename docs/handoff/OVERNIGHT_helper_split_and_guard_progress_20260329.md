# OVERNIGHT HELPER SPLIT AND GUARD PROGRESS (2026-03-29)

## Executive Summary
- Continued helper-channel mixed-payload HOL risk reduction on existing repo foundation.
- Completed acceptance audit for prior AF/AE/AD/AC split claims using real diffs and real logs.
- Added repo-tracked mixed-payload regression checker for helper split anchors.
- Completed repo-wide helper/staging hotspot inventory.
- Executed one additional low-risk helper-only split (AC legacy work-unit path) with local evidence PASS.
- Completed deferred C2 split (AD legacy `X/WQ` work-unit path) with local runner/surface/checker PASS.
- Governance posture remains local-only: not Catapult closure; not SCVerify closure.

## Accepted / Likely-Good Tasks
- TASK A: helper split acceptance audit (AF/AE/AD/AC) completed and marked ACCEPTABLE.
  - artifact: `docs/handoff/HELPER_SPLIT_ACCEPTANCE_AUDIT_20260329.md`
- TASK B: mixed-payload regression guard checker added and passing.
  - artifacts:
    - `scripts/check_helper_channel_split_regression.ps1`
    - `docs/handoff/HELPER_CHANNEL_REGRESSION_GUARD_20260329.md`
    - `build/helper_channel_guard/check_helper_channel_split_regression.log`
- TASK C: helper/staging hotspot inventory completed with risk ranking and candidate shortlist.
  - artifact: `docs/handoff/HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md`
- TASK D: candidate C1 split completed (AC legacy work-unit in/out channel split) with runner/checker PASS.
  - artifact: `docs/handoff/HELPER_LOWRISK_SPLIT_TASKD_20260329.md`
- TASK 1(C2): deferred AD legacy `X/WQ` work-unit split completed with direct runner evidence.
  - artifact: `docs/handoff/HELPER_LOWRISK_SPLIT_C2_20260329.md`

## Tasks Needing Human Diff Review
- `src/blocks/AttnPhaseATopManagedQ.h`
  - C2 legacy helper signatures changed from shared `attn_q_pkt_ch_t in_ch` to split packet channels (`x_ch/wq_ch`).
- `tb/tb_q_path_impl_p11ad.cpp`
  - added legacy split probe and explicit `LEGACY_WORK_UNIT_SPLIT_PATH PASS` evidence banner.
- `scripts/local/run_p11ad_impl_q_path.ps1`
  - runner pass-gate now includes legacy split path PASS line.
- `scripts/check_p11ad_impl_surface.ps1`
  - surface guard expanded for AD legacy split anchors.
- `scripts/check_helper_channel_split_regression.ps1`
  - guard coverage expanded to AD legacy work-unit split anchors (plus AC legacy anchors retained).

## Tasks Attempted But Blocked
- No hard blocker in this run.

## Exact Artifact Index
- Acceptance / analysis:
  - `docs/handoff/HELPER_SPLIT_ACCEPTANCE_AUDIT_20260329.md`
  - `docs/handoff/HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md`
- Guard:
  - `scripts/check_helper_channel_split_regression.ps1`
  - `docs/handoff/HELPER_CHANNEL_REGRESSION_GUARD_20260329.md`
  - `build/helper_channel_guard/check_helper_channel_split_regression.log`
  - `build/helper_channel_guard/check_helper_channel_split_regression_summary.txt`
- Additional split:
  - `docs/handoff/HELPER_LOWRISK_SPLIT_TASKD_20260329.md`
  - `build/p11ac_impl_split_taskd/build.log`
  - `build/p11ac_impl_split_taskd/run.log`
  - `build/p11ac_impl_split_taskd/verdict.txt`
- C2 split:
  - `docs/handoff/HELPER_LOWRISK_SPLIT_C2_20260329.md`
  - `build/p11ad_impl_c2/build.log`
  - `build/p11ad_impl_c2/run.log`
  - `build/p11ad_impl_c2/verdict.txt`
  - `build/p11ad_c2/check_p11ad_impl_surface.log`
  - `build/p11ad_c2/check_p11ad_impl_surface_summary.txt`
- Acceptance evidence baselines used:
  - `build/p11af_impl_split_night/run.log`
  - `build/p11ae_impl_split/run.log`
  - `build/p11ad_impl_split/run.log`
  - `build/p11ac_impl_split/run.log`
  - `build/p11ah_full_loop_night/run.log`

## Suggested 10-Minute Review Order
1. Read `docs/handoff/HELPER_SPLIT_ACCEPTANCE_AUDIT_20260329.md` for AF/AE/AD/AC verdict baseline.
2. Read `docs/handoff/HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md` to confirm remaining hotspots and ranking logic.
3. Review C2 diffs in `src/blocks/AttnPhaseATopManagedQ.h` and `tb/tb_q_path_impl_p11ad.cpp`.
4. Check `build/p11ad_impl_c2/run.log` for `LEGACY_WORK_UNIT_SPLIT_PATH PASS` + mainline/fallback PASS lines.
5. Confirm guard log `build/helper_channel_guard/check_helper_channel_split_regression.log`.

## Governance Posture
- local-only evidence only
- not Catapult closure
- not SCVerify closure
- Top remains sole production shared-SRAM owner in production contract
