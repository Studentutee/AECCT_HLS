# HELPER CHANNEL HOTSPOT INVENTORY (2026-03-29)

## Scan Scope And Method
- Primary scope: repo-wide `src/blocks` helper/staging path with packet channels (`AttnTopManagedPacket` / `AttnTopManagedWorkPacket`).
- Follow-up scope (this round): helper-local staging TB path under `tb/` to find non-production mixed channels still carrying multiple payload classes.
- Method:
  - Searched channel typedef anchors and all `nb_read(...)` consume anchors.
  - Flagged hotspots where one channel carries multiple payload classes (`X/WQ`, `Q/K`, `score/V`, `X/WK/WV`, `K/V`).
  - Classified by coupling risk and "can split now" criteria.
- Key commands:
  - `rg -n "nb_read\(" src/blocks tb`
  - `rg -n "typedef ac_channel<AttnTopManaged(Packet|WorkPacket)>" src/blocks include`
  - `rg -n "in_ch_|out_ch_" tb/tb_kv_build_stream_stage_p11ab.cpp`

## Hotspot Table
| file | type name | payload classes | active mainline or helper-only | current risk level | can split now? | reason |
| --- | --- | --- | --- | --- | --- | --- |
| `(none unresolved in current helper/staging + helper-local TB scan scope)` | n/a | n/a | n/a | n/a | n/a | `src/blocks` hotspots were already closed; this round closed remaining helper-local TB mixed channel in P11AB. |

## This-Round Candidate Selection
1. **Selected candidate**: `tb/tb_kv_build_stream_stage_p11ab.cpp` helper-local work-unit staging path.
   - Why selected:
     - real mixed payload channels remained (`in_ch_` carried `X/WK/WV`, `out_ch_` carried `K/V`)
     - helper-local/TB-only scope, no Top formal external contract coupling
     - existing local runner ready (`scripts/local/run_p11ab_kv_build_stage.ps1`)
     - minimal split possible without broad rewiring
2. **Why not others**:
   - `src/blocks` AF/AE/AD/AC paths are already split and guard-covered in current baseline.
   - No unresolved `src/blocks` mixed-payload single-channel consume anchor remained after KV-out completion.

## Already-Split Paths Rechecked (Not Remaining Hotspots)
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h`: `score`/`v` split confirmed.
- `src/blocks/AttnPhaseBTopManagedQkScore.h`: `q`/`k` split confirmed.
- `src/blocks/AttnPhaseATopManagedQ.h` (work-tile path): `x`/`wq` split confirmed.
- `src/blocks/AttnPhaseATopManagedQ.h` (legacy work-unit path): `x`/`wq` packet-channel split confirmed (C2).
- `src/blocks/AttnPhaseATopManagedKv.h` (work-tile input path): `x`/`wk`/`wv` split confirmed.
- `src/blocks/AttnPhaseATopManagedKv.h` (work-tile output path): `k`/`v` split confirmed.
- `src/blocks/AttnPhaseATopManagedKv.h` (legacy work-unit path): `x/wk/wv` and `k/v` split confirmed.
- `tb/tb_kv_build_stream_stage_p11ab.cpp` (helper-local staging path): `x/wk/wv` input and `k/v` output split confirmed in this round.

## Candidate Ranking For Next Round
1. Optional follow-up candidate: checker-hardening only (no dataflow rewiring), focused on anti-regression anchor robustness and wording drift prevention.

## Inventory Posture
- Inventory evidence is local static/diff + local runner evidence.
- Governance posture: local-only; not Catapult closure; not SCVerify closure.
