# TASK D - LOW-RISK HELPER SPLIT EXECUTION (2026-03-29)

## Summary
- Executed one safe candidate split from inventory (`C1`):
  - AC legacy work-unit helper path in `AttnPhaseATopManagedKv.h`.
- Scope stayed helper/TB-local and did not alter Top formal ownership contract.
- Result: split landed and local runner/checker evidence PASS.

## Candidate Survey
1. `C1` (selected): `AttnPhaseATopManagedKv.h` legacy work-unit path (`X/WK/WV` input mix + `K/V` output mix).
   - helper-only and exercised by existing TB runner.
   - validation available immediately via `run_p11ac_phasea_top_managed.ps1`.
2. `C2` (deferred): `AttnPhaseATopManagedQ.h` legacy work-unit `X/WQ` input mix.
   - helper-only but no dedicated active runner path today; deferred to avoid weak evidence.

## Exact Files Changed
- `src/blocks/AttnPhaseATopManagedKv.h`
- `tb/tb_kv_build_stream_stage_p11ac.cpp`
- `scripts/check_helper_channel_split_regression.ps1`

## Exact Commands Run
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ac_phasea_top_managed.ps1 -BuildDir build\p11ac_impl_split_taskd`
- `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1 -RepoRoot . -OutDir build/helper_channel_guard`

## Actual Execution Evidence / Log Excerpt
- `build/p11ac_impl_split_taskd/run.log`
  - `STREAM_ORDER PASS`
  - `MEMORY_ORDER PASS`
  - `EXACT_SCR_KV_COMPARE PASS`
  - `MAINLINE_PATH_TAKEN PASS`
  - `FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ac_phasea_top_managed`
- `build/helper_channel_guard/check_helper_channel_split_regression.log`
  - `guard: AC legacy work-unit split anchors OK`
  - `PASS: check_helper_channel_split_regression`

## What Was Split
- Legacy helper input channel split:
  - from single `attn_pkt_ch_t in_ch` carrying `X + WK + WV`
  - to `attn_x_pkt_ch_t x_ch`, `attn_wk_pkt_ch_t wk_ch`, `attn_wv_pkt_ch_t wv_ch`
- Legacy helper output channel split:
  - from single `attn_pkt_ch_t out_ch` carrying `K + V`
  - to `attn_k_pkt_ch_t k_ch`, `attn_v_pkt_ch_t v_ch`
- TB callsite/channel declarations updated to match split signatures.

## Governance Posture
- local-only evidence
- not Catapult closure
- not SCVerify closure
- Top remains sole production shared-SRAM owner

## Residual Risks
- AD legacy work-unit (`X/WQ` mixed input channel) still remains and should be handled in a follow-up split task.
- Current proof is local runner/checker only.

## Next Recommended Step
- Execute deferred `C2` (`AttnPhaseATopManagedQ.h` legacy work-unit `X/WQ` split) with a minimal dedicated runner or compile-checked TB path for direct evidence.
