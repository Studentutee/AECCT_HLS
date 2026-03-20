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

## P11AP Forward Freeze Addendum (active-chain target)

### Scope posture
- This addendum records forward freeze decisions for the active Catapult-facing chain only.
- It is intended to guide the next block-boundary cleanup pass where Top remains the only shared-SRAM owner.
- It does **not** retroactively rewrite already-landed AC/AD packet-width constants until the matching code/SSOT changes land.

### Decision lock
- Head-group mapping is fixed as:
  - `head 0..3 -> group0 -> rule1 -> one_ring_mask`
  - `head 4..7 -> group1 -> rule2 -> second_ring_mask`
- Working tile granularity target is fixed as:
  - `WORD_BITS = 32`
  - `TILE_WORDS = 4`
  - `TILE_BITS = 128`
  - tail tile is allowed for the final partial tile
- Ownership rule remains fixed:
  - Top owns shared SRAM, arbitration, and writeback timing
  - sub-blocks consume Top-provided token/tile windows only
  - no block-direct shared-SRAM ownership is allowed in the active chain

### Encoding policy lock
- Phase A / Phase C may keep deterministic sequence-derived semantics when payload meaning is uniquely recoverable from phase + range + order.
- Phase B must not rely on sequence index alone.
- Phase B active-chain packets/windows must carry explicit disambiguation for:
  - `head_group_id`
  - `subphase_id`
- Recommended Phase B `subphase_id` buckets:
  - `QSRC`
  - `WQ`
  - `KVSCAN`
  - `MASK`
  - `WO`
  - `OUT`

### Rationale snapshot
- The goal is to remove residual whole-SRAM/raw-pointer ownership semantics from active-chain block boundaries without doing repo-wide cleanup first.
- The 4-word tile target is a working compute/window granularity decision for the next active-chain cleanup pass; it is not a claim that all existing packet definitions have already been rewritten.

## Non-Goals
- No Catapult closure claim.
- No SCVerify closure claim.
