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
- Status: pending
- Freeze scope: pending

### G-AE-IF (to be filled when AE interface freeze lands)
- Status: pending
- Freeze scope: pending

## Non-Goals
- No Catapult closure claim.
- No SCVerify closure claim.
