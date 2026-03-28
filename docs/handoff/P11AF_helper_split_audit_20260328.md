# P11AF Helper Split Audit (2026-03-28)

## Scope
- Task: verify AF helper-path mixed input channel split is truly landed.
- Files audited:
  - `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`
  - `scripts/check_p11af_impl_surface.ps1`
  - `scripts/local/run_p11af_impl_softmax_out.ps1`
  - `build/p11af_impl_night/*`
  - `build/p11af_impl_split_night/*`

## Findings
- Helper emit path is split:
  - score packets are emitted to `score_ch`
  - V tile packets are emitted to `v_ch`
- Helper consume path is split:
  - key-token loop reads one score packet from `score_ch`
  - tile loops read V packets from `v_ch`
- No helper-side re-merge back into a single mixed input channel.
- Top-managed active mainline SRAM loops remain unchanged.
- Top-owned SRAM ownership contract remains unchanged.

## Local evidence
- Surface pre: `PASS: check_p11af_impl_surface`
- Runner: `PASS: run_p11af_impl_softmax_out`
- Surface post: `PASS: check_p11af_impl_surface`
- Required run banners present:
  - `PASS: tb_softmax_out_impl_p11af`
  - `SOFTMAX_MAINLINE PASS`
  - `OUTPUT_EXPECTED_COMPARE PASS`
  - `OUTPUT_TARGET_SPAN_WRITE PASS`
  - `NO_SPURIOUS_WRITE PASS`
  - `SOURCE_PRESERVATION PASS`
  - `MAINLINE_SOFTMAX_OUTPUT_PATH_TAKEN PASS`
  - `FALLBACK_NOT_TAKEN PASS`
  - `fallback_taken = false`
  - `PASS: run_p11af_impl_softmax_out`

## Governance posture
- local-only
- helper-path risk reduction only
- not Catapult closure
- not SCVerify closure

