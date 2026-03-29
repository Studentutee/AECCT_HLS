# TASK KV-OUT - LOW-RISK HELPER SPLIT EXECUTION (2026-03-29)

## Summary
- Completed helper/staging work-tile `K/V` out-channel split in `src/blocks/AttnPhaseATopManagedKv.h`.
- Work-tile helper path no longer emits `K` and `V` through shared single `attn_work_pkt_ch_t out_ch`.
- Scope remained helper/staging local-only; no Top formal external contract rewiring.

## Exact Files Changed
- `src/blocks/AttnPhaseATopManagedKv.h`
- `tb/tb_kv_build_stream_stage_p11ac.cpp`
- `scripts/local/run_p11ac_phasea_top_managed.ps1`
- `scripts/check_p11ac_phasea_surface.ps1`
- `scripts/check_helper_channel_split_regression.ps1`
- `docs/handoff/HELPER_CHANNEL_REGRESSION_GUARD_20260329.md`
- `docs/handoff/HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md`

## Exact Commands Run
- `powershell -ExecutionPolicy Bypass -File scripts/check_p11ac_phasea_surface.ps1 -RepoRoot . -OutDir build\p11ac_kv_out -Phase pre`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ac_phasea_top_managed.ps1 -BuildDir build\p11ac_kv_out_impl`
- `powershell -ExecutionPolicy Bypass -File scripts/check_p11ac_phasea_surface.ps1 -RepoRoot . -OutDir build\p11ac_kv_out -Phase post`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1 -RepoRoot . -OutDir build/helper_channel_guard`
- `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1 -RepoRoot .`
- `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -RepoRoot . -Phase pre`

## Actual Execution Evidence / Log Excerpt
- `build/p11ac_kv_out_impl/run.log`
  - `WORK_TILE_OUT_SPLIT_PATH PASS`
  - `STREAM_ORDER PASS`
  - `MEMORY_ORDER PASS`
  - `EXACT_SCR_KV_COMPARE PASS`
  - `MAINLINE_PATH_TAKEN PASS`
  - `FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ac_phasea_top_managed`
- `build/p11ac_kv_out/check_p11ac_phasea_surface.log`
  - `===== check_p11ac_phasea_surface phase=pre =====`
  - `PASS: check_p11ac_phasea_surface`
  - `===== check_p11ac_phasea_surface phase=post =====`
  - `PASS: check_p11ac_phasea_surface`
- `build/helper_channel_guard/check_helper_channel_split_regression.log`
  - `guard: AC work-tile k/v out split anchors OK`
  - `PASS: check_helper_channel_split_regression`

## What Was Split
- Added work-tile output packet-channel typedefs:
  - `attn_k_work_pkt_ch_t`
  - `attn_v_work_pkt_ch_t`
- Updated work-tile consume helper:
  - from shared output `attn_work_pkt_ch_t& out_ch`
  - to split output `attn_k_work_pkt_ch_t& k_ch` and `attn_v_work_pkt_ch_t& v_ch`
  - explicit split writes: `k_ch.write(k_pkt)` and `v_ch.write(v_pkt)`
- Updated work-tile writeback helper:
  - from shared output read `out_ch.nb_read(...)`
  - to split reads `k_ch.nb_read(k_pkt)` and `v_ch.nb_read(v_pkt)`
- Added TB split-path probe and runner pass-gate:
  - `WORK_TILE_OUT_SPLIT_PATH PASS`

## Governance Posture
- local-only evidence
- not Catapult closure
- not SCVerify closure
- Top remains sole production shared-SRAM owner

## Residual Risks
- Regression checker uses semantic anchors and signature anchors, not full AST/dataflow proof.
- Local runner/surface evidence does not imply Catapult or SCVerify closure.

## Next Recommended Step
- Keep this path under existing guard and run the same local validation bundle on every related helper-path touch:
  - `check_p11ac_phasea_surface`
  - `run_p11ac_phasea_top_managed`
  - `check_helper_channel_split_regression`
