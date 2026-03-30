# TOP MANAGED SRAM G5 DIRECT PAYLOAD INVENTORY (2026-03-30)

## Summary
- Scope: local-only survey for remaining direct-SRAM payload consumption/production in active block paths.
- Goal: rank bounded migration waves that push payload ownership/dispatch responsibility further to Top without external formal contract changes.
- Posture: not Catapult closure; not SCVerify closure.

## Inventory Result (Repo Reality)
| Block / Path | Direct SRAM payload read | Direct SRAM payload write | Risk | Wave fit | Notes |
|---|---|---|---|---|---|
| `src/blocks/LayerNormBlock.h` | YES (`x`, `gamma`, `beta`) | YES (`x_out`) | LOW | Wave 1 | Affine payload can be Top-preloaded with bounded callsite change. |
| `src/blocks/FinalHead.h` | YES (`x_end`, `w`, `b`) | YES (`final_scalar`, `logits`, `xpred`) | LOW-MED | Wave 1 | Token scalar input can be Top-fed while keeping existing SRAM output behavior. |
| `src/blocks/PreprocEmbedSPE.h` | YES (`infer input`) | YES (`x_out`) | MED | Wave 2 | Input payload copy loop is bounded and contractized already. |
| `src/blocks/FFNLayer0.h` | YES (`x`, `w1/w2`, `bias`) | YES (`h/a/y`) | MED-HIGH | Wave 3 | Payload and parameter reads are deeply interleaved in tiled loops. |
| `src/blocks/AttnLayer0.h` | YES (Q/K/V/score/v tiles) | YES (Q/K/V/score/out/post) | HIGH | Wave 4 | Multi-phase dataflow with high coupling to score/softmax/output staging. |
| `src/blocks/TransformerLayer.h` | YES (`residual`, `w2`, norm params fallback) | YES (`add2`, norm preload) | HIGH | Wave 4 | Layer orchestration remains mixed with direct payload scratch behavior. |
| `src/blocks/AttnPhaseATopManagedQ.h` | YES (`x`, `wq`, `q`) | YES (`q`, `q_act_q`) | HIGH | Wave 4 | Helper split exists but payload source/sink is still SRAM-centric. |
| `src/blocks/AttnPhaseATopManagedKv.h` | YES (`x`, `wk`, `wv`) | YES (`k`, `v`, `*_act_q`) | HIGH | Wave 4 | K/V out split done; payload ownership still mostly SRAM path. |
| `src/blocks/AttnPhaseBTopManagedQkScore.h` | YES (`q`, `k`) | YES (`score`) | HIGH | Wave 4 | Score generation path remains SRAM direct in core loops. |
| `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h` | YES (`score`, `v`) | YES (`pre/post/out`) | HIGH | Wave 4 | Mixed multi-stage output loops still SRAM direct. |

## Migration Priority Ranking
1. Wave 1: `LayerNormBlock` + `FinalHead`
- Highest bounded benefit with low structural risk.
- Directly moves payload sourcing responsibility to Top-fed arrays for active critical path.

2. Wave 2: `PreprocEmbedSPE`
- Existing Top contract metadata already present, so payload source convergence is low-risk.
- Minimal cut can be validated with isolated targeted TB.

3. Wave 3: `FFNLayer0`
- Architecture-forward but heavier due interleaved parameter/data tiled loops.
- Needs careful Top-fed tile descriptor plan to avoid broad rewiring.

4. Wave 4: `AttnLayer0` + `TransformerLayer` + `AttnPhase*TopManaged*`
- Highest complexity and coupling risk.
- Best treated as staged follow-up after bounded Wave 1-3 hardening.

## Why Wave Order Was Adjusted
- Suggested baseline listed Wave 2 as `FFNLayer0`.
- Repo reality for same-run bounded success favored `PreprocEmbedSPE` before FFN:
  - smaller cut surface,
  - lower chance of mainline regression,
  - faster local evidence closure under night-batch constraints.

## Deferred / Hardest Areas
- `FFNLayer0`, `AttnLayer0`, `TransformerLayer`, and phase blocks remain open for deeper payload migration.
- These are deferred to prevent broad rewrite and preserve current local evidence chain stability.
