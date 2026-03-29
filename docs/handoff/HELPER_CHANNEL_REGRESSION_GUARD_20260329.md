# HELPER CHANNEL REGRESSION GUARD (2026-03-29)

## Summary
- Added a dedicated mixed-payload regression checker:
  - `scripts/check_helper_channel_split_regression.ps1`
- Purpose: prevent helper consume signatures from regressing to single shared input channel for known mixed-payload classes.
- Posture: local guard only; not Catapult closure; not SCVerify closure.

## Guard Coverage
- AF (`score + v`):
  - Require split typedef anchors: `attn_phaseb_softmax_score_ch_t`, `attn_phaseb_softmax_v_ch_t`.
  - Require split consume signature and split reads (`score_ch`, `v_ch`).
  - Forbid old shared consume signature using single `attn_phaseb_softmax_pkt_ch_t& in_ch`.
- AE (`q + k`):
  - Require split typedef anchors: `attn_phaseb_q_pkt_ch_t`, `attn_phaseb_k_pkt_ch_t`.
  - Require split consume signature and split reads (`q_ch`, `k_ch`).
  - Forbid old shared consume signature using single `attn_phaseb_qk_pkt_ch_t& in_ch`.
- AD (`x + wq`):
  - Work-tile path:
    - Require split typedef anchors: `attn_q_x_work_pkt_ch_t`, `attn_q_wq_work_pkt_ch_t`.
    - Require split consume signature and split reads (`x_ch`, `wq_ch`).
    - Forbid old shared consume signature using single `attn_q_work_pkt_ch_t& in_ch`.
  - Legacy work-unit path:
    - Require split typedef anchors: `attn_q_x_pkt_ch_t`, `attn_q_wq_pkt_ch_t`.
    - Require split emit/consume signatures to keep separated `x_ch` / `wq_ch`.
    - Forbid old shared single `attn_q_pkt_ch_t& in_ch` signatures.
    - Require TB legacy probe PASS banner (`LEGACY_WORK_UNIT_SPLIT_PATH PASS`).
- AC (`x + wk + wv`):
  - Work-tile path:
    - Require split typedef anchors: `attn_x_work_pkt_ch_t`, `attn_wk_work_pkt_ch_t`, `attn_wv_work_pkt_ch_t`.
    - Require split consume signature and split reads (`x_ch`, `wk_ch`, `wv_ch`).
    - Forbid old shared consume signature using single `attn_work_pkt_ch_t& in_ch`.
  - Legacy work-unit path:
    - Require split packet typedef anchors: `attn_x_pkt_ch_t`, `attn_wk_pkt_ch_t`, `attn_wv_pkt_ch_t`, `attn_k_pkt_ch_t`, `attn_v_pkt_ch_t`.
    - Require split emit/consume/writeback signatures and TB split-channel declarations.
    - Forbid old shared `attn_pkt_ch_t in_ch/out_ch` legacy signatures.

## Checker Execution
- Command:
  - `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1 -RepoRoot . -OutDir build/helper_channel_guard`
- Result:
  - `PASS: check_helper_channel_split_regression`
- Summary file:
  - `build/helper_channel_guard/check_helper_channel_split_regression_summary.txt`
- Log file:
  - `build/helper_channel_guard/check_helper_channel_split_regression.log`

## Actual Evidence Excerpt
- `guard: AF score/v split anchors OK`
- `guard: AE q/k split anchors OK`
- `guard: AD x/wq split anchors OK`
- `guard: AD legacy work-unit split anchors OK`
- `guard: AC x/wk/wv split anchors OK`
- `guard: AC legacy work-unit split anchors OK`
- `PASS: check_helper_channel_split_regression`

## Residual Risk
- This checker intentionally uses semantic anchors and function-signature guards, not full AST/dataflow proof.
- It protects accepted split entrypoints and current legacy split signatures, but does not by itself prove absence of every mixed-payload hotspot in other helper/staging paths.
- Repo-wide hotspot coverage is tracked separately in `HELPER_CHANNEL_HOTSPOT_INVENTORY_20260329.md`.
