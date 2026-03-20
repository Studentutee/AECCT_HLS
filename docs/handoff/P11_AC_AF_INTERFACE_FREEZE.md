# P11 AC~AF Interface Freeze (Staged Skeleton)

## Purpose
- This document is a staged freeze skeleton for AC~AF.
- It is a placeholder at program start.
- It is not a final authority before gate entries are landed.

## Freeze Posture
- Internal packet/interface entries are provisional before `G-AC`.
- Keep fields/semantics minimal.
- Do not enlarge semantics unless AC evidence requires it.

## Gate Entry Log

### G-AC (to be filled when AC evidence lands)
- Status: landed (local-only evidence cycle; AC finalize mainline wiring)
- Freeze scope:
- `AttnTopManagedPacket.kind`
- `AttnTopManagedPacket.token_idx`
- `AttnTopManagedPacket.d_tile_idx`
- `AttnTopManagedPacket.flags`
- `AttnTopManagedPacket.inv_sw_bits`
- `AttnTopManagedPacket.data[ATTN_TOP_MANAGED_TILE_WORDS]`
- Integrated call path identifiers:
- `run_transformer_layer_loop -> run_p11ac_layer0_top_managed_kv -> attn_phasea_top_managed_kv_mainline`
- `TransformerLayer(..., kv_prebuilt_from_top_managed=true) -> AttnLayer0(..., kv_prebuilt_from_top_managed=true)`
- Acceptance lock:
- `MAINLINE_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `fallback_taken = false`
- Notes: freeze is minimal and only for downstream AC~AF local bring-up continuity.

### G-AD-IF (to be filled when AD interface freeze lands)
- Status: landed (local-only evidence cycle; AD mainline Q wiring)
- Freeze scope:
- `AttnTopManagedPacket.kind` extends with `ATTN_PKT_WQ` and `ATTN_PKT_Q` only
- `AttnTopManagedPacket` fields remain unchanged
- Integrated call path identifiers:
- `run_transformer_layer_loop -> run_p11ad_layer0_top_managed_q -> attn_phasea_top_managed_q_mainline`
- `TransformerLayer(..., q_prebuilt_from_top_managed=true) -> AttnLayer0(..., q_prebuilt_from_top_managed=true)`
- Acceptance lock:
- `MAINLINE_Q_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `fallback_taken = false`
- Q-side freeze assumptions for AE-impl:
- Q acceptance path writes to `sc.q_base_word`, `sc.q_act_q_base_word`, `sc.q_sx_base_word`
- no alternate temporary destination in acceptance path

### G-AE-IF (to be filled when AE interface freeze lands)
- Status: landed (local-only evidence cycle; AE/AF additive mainline wiring)
- Freeze scope:
- AC/AD hooks/surfaces remain backward-compatible and unchanged in semantics
- Added AE helper/wrapper path:
  - `run_transformer_layer_loop -> run_p11ae_layer0_top_managed_qk_score -> attn_phaseb_top_managed_qk_score_mainline`
- Added AF helper/wrapper path:
  - `run_transformer_layer_loop -> run_p11af_layer0_top_managed_softmax_out -> attn_phaseb_top_managed_softmax_out_mainline`
- Added additive prebuilt hooks only for score/out handoff:
  - `TransformerLayer(..., score_prebuilt_from_top_managed, out_prebuilt_from_top_managed)`
  - `AttnLayer0(..., score_prebuilt_from_top_managed, out_prebuilt_from_top_managed)`
- Acceptance lock:
  - `MAINLINE_SCORE_PATH_TAKEN PASS`
  - `MAINLINE_SOFTMAX_OUTPUT_PATH_TAKEN PASS`
  - `FALLBACK_NOT_TAKEN PASS`
  - `fallback_taken = false`

## Non-Goals
- No Catapult closure claim.
- No SCVerify closure claim.
