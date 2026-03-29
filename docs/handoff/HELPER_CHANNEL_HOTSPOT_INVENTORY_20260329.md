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
| `(none unresolved in current helper/staging scope)` | n/a | n/a | n/a | n/a | n/a | AC work-tile `K/V` out-channel split completed in this round. |

## Already-Split Paths Rechecked (Not Remaining Hotspots)
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`: `score`/`v` split confirmed.
- `src/blocks/AttnPhaseBTopManagedQkScore.h`: `q`/`k` split confirmed.
- `src/blocks/AttnPhaseATopManagedQ.h` (work-tile path): `x`/`wq` split confirmed.
- `src/blocks/AttnPhaseATopManagedQ.h` (legacy work-unit path): `x`/`wq` packet-channel split confirmed (C2).
- `src/blocks/AttnPhaseATopManagedKv.h` (work-tile input path): `x`/`wk`/`wv` split confirmed.
- `src/blocks/AttnPhaseATopManagedKv.h` (work-tile output path): `k`/`v` split confirmed (`attn_k_work_pkt_ch_t`, `attn_v_work_pkt_ch_t`) in this round.
- `src/blocks/AttnPhaseATopManagedKv.h` (legacy work-unit path): `x/wk/wv` and `k/v` split confirmed.

## Candidate Ranking For TASK D
1. Optional follow-up candidate: checker-hardening pass only (no dataflow rewiring), focusing on stronger anti-regression anchors and wording drift prevention.
   - Why next: current known mixed-payload helper hotspots are closed; remaining work is guard hardening and maintenance.

## Inventory Posture
- This inventory is local static/diff evidence for helper/staging channel topology.
- Governance posture: local-only; not Catapult closure; not SCVerify closure.
