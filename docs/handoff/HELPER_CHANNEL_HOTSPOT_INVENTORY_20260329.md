# HELPER CHANNEL HOTSPOT INVENTORY (2026-03-29)

## Scan Scope And Method
- Scope: repo-wide `src/blocks` helper/staging path with packet channels (`AttnTopManagedPacket` / `AttnTopManagedWorkPacket`).
- Method:
  - Searched all `nb_read(...)` and channel typedef anchors.
  - Flagged hotspots where a single channel carries multiple payload classes with different semantic roles.
  - Classified by coupling risk and whether safe split can be done now.
- Key commands:
  - `rg -n "nb_read\(" src`
  - `rg -n "typedef ac_channel<AttnTopManaged(Packet|WorkPacket)>" src/blocks include`
  - `rg -n "if\s*\(\s*!.*nb_read\(.*\|\|\s*!.*nb_read\(" src/blocks`

## Hotspot Table
| file | type name | payload classes | active mainline or helper-only | current risk level | can split now? | reason |
| --- | --- | --- | --- | --- | --- | --- |
| `src/blocks/AttnPhaseATopManagedKv.h` | `attn_work_pkt_ch_t` (`out_ch` in `attn_block_phasea_kv_consume_emit_token_work_tiles` / `attn_top_writeback_phasea_kv_work_tile`) | `K + V` | helper/staging-only (no current Top mainline callsite) | MED | YES | Mixed payload class remains in tile helper channel. Split is feasible with local-only interface update. |

## Already-Split Paths Rechecked (Not Remaining Hotspots)
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`: `score`/`v` split confirmed.
- `src/blocks/AttnPhaseBTopManagedQkScore.h`: `q`/`k` split confirmed.
- `src/blocks/AttnPhaseATopManagedQ.h` (work-tile path): `x`/`wq` split confirmed.
- `src/blocks/AttnPhaseATopManagedQ.h` (legacy work-unit path): `x`/`wq` packet-channel split confirmed (C2).
- `src/blocks/AttnPhaseATopManagedKv.h` (work-tile input path): `x`/`wk`/`wv` split confirmed.
- `src/blocks/AttnPhaseATopManagedKv.h` (legacy work-unit path): `x/wk/wv` and `k/v` split confirmed.

## Candidate Ranking For TASK D
1. Next candidate (recommended): `AttnPhaseATopManagedKv.h` work-tile `attn_work_pkt_ch_t out_ch` split (`K/V`) in `attn_block_phasea_kv_consume_emit_token_work_tiles` + `attn_top_writeback_phasea_kv_work_tile`.
   - Why next: helper/staging-only and no Top formal contract coupling.
2. Optional follow-up candidate: AD/AC legacy helper-path static tightening checker-only pass (no new rewiring), if code split is deferred.
   - Why optional: may reduce regression risk without touching broader dataflow.

## Inventory Posture
- This inventory is local static/diff evidence for helper/staging channel topology.
- Governance posture: local-only; not Catapult closure; not SCVerify closure.
