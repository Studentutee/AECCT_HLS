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
| `src/blocks/AttnPhaseATopManagedKv.h` | `attn_pkt_ch_t` (`in_ch` in `attn_top_emit_phasea_kv_work_unit` / `attn_block_phasea_kv_consume_emit`) | `X + WK + WV` | helper-only (TB-local legacy work-unit path) | HIGH | YES | 3 payload classes on one channel; explicit HOL exposure if producer/consumer cadence drifts. No Top contract coupling. |
| `src/blocks/AttnPhaseATopManagedKv.h` | `attn_pkt_ch_t` (`out_ch` in `attn_block_phasea_kv_consume_emit` / `attn_top_writeback_phasea_kv_work_unit`) | `K + V` | helper-only (TB-local legacy work-unit path) | MED | YES | Currently lockstep K->V pair, but still mixed channel. Split is local and contained to helper/TB-local path. |
| `src/blocks/AttnPhaseATopManagedKv.h` | `attn_work_pkt_ch_t` (`out_ch` in `attn_block_phasea_kv_consume_emit_token_work_tiles` / `attn_top_writeback_phasea_kv_work_tile`) | `K + V` | helper/staging-only (no current Top mainline callsite) | MED | YES | Mixed payload class remains in tile helper channel. Split is feasible with local-only interface update. |
| `src/blocks/AttnPhaseATopManagedQ.h` | `attn_q_pkt_ch_t` (`in_ch` in `attn_top_emit_phasea_q_work_unit` / `attn_block_phasea_q_consume_emit`) | `X + WQ` | helper-only (legacy work-unit path) | MED | YES | 2 payload classes on single channel; low coupling to Top ownership, but no active runner currently exercises this path. |

## Already-Split Paths Rechecked (Not Remaining Hotspots)
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`: `score`/`v` split confirmed.
- `src/blocks/AttnPhaseBTopManagedQkScore.h`: `q`/`k` split confirmed.
- `src/blocks/AttnPhaseATopManagedQ.h` (work-tile path): `x`/`wq` split confirmed.
- `src/blocks/AttnPhaseATopManagedKv.h` (work-tile input path): `x`/`wk`/`wv` split confirmed.

## Candidate Ranking For TASK D
1. Candidate C1 (recommended): `AttnPhaseATopManagedKv.h` legacy work-unit `in_ch` + `out_ch` split (`X/WK/WV` and `K/V`) with `tb/tb_kv_build_stream_stage_p11ac.cpp` update.
   - Why first: helper-only + existing runner (`run_p11ac_phasea_top_managed.ps1`) gives fast local evidence.
2. Candidate C2: `AttnPhaseATopManagedQ.h` legacy work-unit `in_ch` split (`X/WQ`).
   - Why second: helper-only and small cut, but lacks dedicated active runner coverage at present.

## Inventory Posture
- This inventory is local static/diff evidence for helper/staging channel topology.
- Governance posture: local-only; not Catapult closure; not SCVerify closure.
