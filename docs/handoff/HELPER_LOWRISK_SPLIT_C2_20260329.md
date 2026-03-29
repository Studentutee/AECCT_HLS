# TASK C2 - LOW-RISK HELPER SPLIT EXECUTION (2026-03-29)

## Summary
- Completed deferred C2 in `src/blocks/AttnPhaseATopManagedQ.h`.
- Legacy/helper work-unit path no longer mixes `X` and `WQ` on a single input channel.
- Scope stayed helper/staging local-only; no Top formal external contract rewiring.

## Exact Files Changed
- `src/blocks/AttnPhaseATopManagedQ.h`
- `tb/tb_q_path_impl_p11ad.cpp`
- `scripts/local/run_p11ad_impl_q_path.ps1`
- `scripts/check_p11ad_impl_surface.ps1`
- `scripts/check_helper_channel_split_regression.ps1`

## Exact Commands Run
- `powershell -ExecutionPolicy Bypass -File scripts/check_p11ad_impl_surface.ps1 -RepoRoot . -OutDir build\p11ad_c2 -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ad_impl_q_path.ps1 -BuildDir build\p11ad_impl_c2`
- `powershell -ExecutionPolicy Bypass -File scripts/check_p11ad_impl_surface.ps1 -RepoRoot . -OutDir build\p11ad_c2 -Phase post`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1 -RepoRoot . -OutDir build/helper_channel_guard`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1 -RepoRoot .`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -RepoRoot . -Phase pre`

## Actual Execution Evidence / Log Excerpt
- `build/p11ad_impl_c2/run.log`
  - `LEGACY_WORK_UNIT_SPLIT_PATH PASS`
  - `Q_PATH_MAINLINE PASS`
  - `Q_EXPECTED_COMPARE PASS`
  - `MAINLINE_Q_PATH_TAKEN PASS`
  - `FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ad_impl_q_path`
- `build/p11ad_c2/check_p11ad_impl_surface_summary.txt`
  - `status: PASS`
  - `phase: post`
- `build/helper_channel_guard/check_helper_channel_split_regression.log`
  - `guard: AD legacy work-unit split anchors OK`
  - `PASS: check_helper_channel_split_regression`

## What Was Split
- Added legacy packet-channel split typedefs:
  - `attn_q_x_pkt_ch_t`
  - `attn_q_wq_pkt_ch_t`
- Legacy helper emit path split:
  - `attn_top_emit_phasea_q_work_unit(..., x_ch, wq_ch)`
- Legacy helper consume path split:
  - `attn_block_phasea_q_consume_emit(x_ch, wq_ch, out_ch, ...)`
- Consume no longer reads `X/WQ` from shared single `in_ch`.
- Added AD TB probe to instantiate and verify legacy split path:
  - `LEGACY_WORK_UNIT_SPLIT_PATH PASS`

## Governance Posture
- local-only evidence
- not Catapult closure
- not SCVerify closure
- Top remains sole production shared-SRAM owner

## Residual Risks
- Remaining helper/staging mixed payload hotspot in inventory is AC work-tile `K/V` shared out channel (`attn_work_pkt_ch_t`).
- Current proof remains local runner/static checker evidence only.

## Next Recommended Step
- If continuing helper-path reduction, split AC work-tile `K/V` shared out channel with minimal helper-only cut and add corresponding local evidence.
