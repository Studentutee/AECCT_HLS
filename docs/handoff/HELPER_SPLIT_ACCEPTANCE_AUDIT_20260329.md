# HELPER SPLIT ACCEPTANCE AUDIT (2026-03-29)

## Scope
- Target: AF/AE/AD/AC helper split acceptance audit using real file diffs and real local logs.
- Focus: mixed-payload HOL risk reduction on helper/staging path; no remote flow.
- Closure posture: not Catapult closure; not SCVerify closure.

## AF (P00-011AF)
- split type: `score + v` split into dedicated channels (`attn_phaseb_softmax_score_ch_t`, `attn_phaseb_softmax_v_ch_t`).
- touched file: `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h` (commit `d6ed7a5`).
- key diff finding:
  - Producer side writes score via `score_ch.write(...)` and V via `v_ch.write(...)`.
  - Consumer side reads `score_ch.nb_read(score_pkt)` and `v_ch.nb_read(v_pkt)` separately.
  - Consume path keeps packet-kind assertions (`ATTN_PKT_SCORE`, `ATTN_PKT_V`) and does not remultiplex into one helper input channel.
  - Mainline SRAM algorithm remains direct read/compute/write in `attn_phaseb_top_managed_softmax_out_mainline(...)`.
- runner evidence:
  - `build/p11af_impl_split_night/run.log`: `SOFTMAX_MAINLINE PASS`, `OUTPUT_EXPECTED_COMPARE PASS`, `NO_SPURIOUS_WRITE PASS`, `PASS: run_p11af_impl_softmax_out`.
  - `build/p11af_impl_night/check_p11af_impl_surface_summary.txt`: `status: PASS`.
- residual risk:
  - Helper split is validated by local runner + surface checks only; no Catapult/SCVerify evidence yet.
- verdict: **ACCEPTABLE**

## AE (P00-011AE)
- split type: `q + k` split into dedicated channels (`attn_phaseb_q_pkt_ch_t`, `attn_phaseb_k_pkt_ch_t`).
- touched file: `src/blocks/AttnPhaseBTopManagedQkScore.h` (commit `c56e6ad`).
- key diff finding:
  - Producer side emits Q/K into different channels (`q_ch`, `k_ch`).
  - Consumer reads two channels independently (`q_ch.nb_read(q_pkt)`, `k_ch.nb_read(k_pkt)`).
  - Ordering checks remain explicit (`d_tile_idx`, `token_idx`, `flags`, `tile_begin/end`) and do not collapse back to single-channel semantics.
  - Mainline score path remains direct SRAM dot/scale/store (`attn_phaseb_top_managed_qk_score_mainline(...)`).
- runner evidence:
  - `build/p11ae_impl_split/run.log`: `QK_SCORE_MAINLINE PASS`, `SCORE_EXPECTED_COMPARE PASS`, `NO_SPURIOUS_WRITE PASS`, `PASS: run_p11ae_impl_qk_score`.
  - `build/p11ae_impl_night/check_p11ae_impl_surface_summary.txt`: `status: PASS`.
- residual risk:
  - Current acceptance is local-only regression evidence, not synthesis/SCVerify closure.
- verdict: **ACCEPTABLE**

## AD (P00-011AD)
- split type: `x + wq` split into dedicated channels (`attn_q_x_work_pkt_ch_t`, `attn_q_wq_work_pkt_ch_t`).
- touched file: `src/blocks/AttnPhaseATopManagedQ.h` (commit `c56e6ad`).
- key diff finding:
  - Work-tile emit writes X and WQ to different channels.
  - Consume loop reads from both channels independently and enforces semantic alignment checks.
  - Output path still emits Q packet stream only after per-tile checks; no helper input remultiplexing.
  - Mainline Q path remains direct SRAM + kernel split call flow (`attn_phasea_top_managed_q_mainline(...)`).
- runner evidence:
  - `build/p11ad_impl_split/run.log`: `Q_PATH_MAINLINE PASS`, `Q_EXPECTED_COMPARE PASS`, `NO_SPURIOUS_WRITE PASS`, `PASS: run_p11ad_impl_q_path`.
  - `build/p11ad_impl_night/check_p11ad_impl_surface_summary.txt`: `status: PASS`.
- residual risk:
  - Local-only evidence; no Catapult/SCVerify closure.
- verdict: **ACCEPTABLE**

## AC (P00-011AC)
- split type: `x + wk + wv` split into dedicated channels (`attn_x_work_pkt_ch_t`, `attn_wk_work_pkt_ch_t`, `attn_wv_work_pkt_ch_t`).
- touched file: `src/blocks/AttnPhaseATopManagedKv.h` (commit `c56e6ad`).
- key diff finding:
  - Emit stage writes X/Wk/Wv through separate channels.
  - Consume loop performs three-channel read and per-packet semantic checks before K/V emission.
  - Helper consume does not remultiplex payload classes back into one input channel.
  - Mainline K/V path remains direct SRAM + split kernels (`attn_phasea_top_managed_kv_mainline(...)`).
- runner evidence:
  - `build/p11ac_impl_split/run.log`: `STREAM_ORDER PASS`, `MEMORY_ORDER PASS`, `SINGLE_READ_X_REUSE PASS`, `EXACT_SCR_KV_COMPARE PASS`, `PASS: run_p11ac_phasea_top_managed`.
  - `build/p11ac_phasea_split_night/check_p11ac_phasea_surface_summary.txt`: `status: PASS` (log includes an earlier wording-only fail, followed by PASS).
- residual risk:
  - Surface log history shows one interim governance-string failure before final pass; monitor report wording drift.
  - Local-only evidence only; no Catapult/SCVerify closure.
- verdict: **ACCEPTABLE**

## Cross-check Notes
- Top ownership contract check:
  - `src/Top.h` mainline callsites (`run_p11ac/.../p11af`) still route SRAM ownership through Top-level calls.
  - No new shared-SRAM ownership claim is introduced in sub-block helper split changes.
- Full-loop continuity:
  - `build/p11ah_full_loop_night/run.log`: `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`, `FULL_LOOP_FINAL_X_DETERMINISTIC_COMPARE PASS`, `PASS: run_p11ah_full_loop_local_e2e`.

## Audit Conclusion
- AF/AE/AD/AC helper splits are **ACCEPTABLE** for current local mixed-payload HOL risk reduction objective.
- Governance posture remains local evidence only: **not Catapult closure; not SCVerify closure**.
