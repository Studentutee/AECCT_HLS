# TOP_MANAGED_SRAM_G7_REMOVE_READINESS_20260330

## Readiness scale
- R0: fully not removable
- R1: probe-only, not removable
- R2: near-removable but fallback/writeback coupling remains
- R3: almost removable (compat cleanup only)
- R4: removable now

## Matrix (block-level)

| Block | Readiness | Why not removed now | Next condition / next cut |
|---|---|---|---|
| FFNLayer0 | R2 | strict paths improved, but writeback/fallback compatibility still uses SRAM | add bounded writeback descriptor bridge and shrink non-strict fallback surface |
| LayerNormBlock | R2 | top-fed affine optional, core row compute/writeback still SRAM | output window dispatch adapter from caller |
| FinalHead | R2 | top-fed scalar optional, logits/xpred path still SRAM | bounded pass-B/topfed logits bridge |
| PreprocEmbedSPE | R2 | top-fed input present, output stays SRAM-coupled | bounded output tile ownership bridge |
| AttnPhaseATopManagedQ | R1 | entry probe only; main payload loops still SRAM | one row-buffer bridge without compute-loop rewrite |
| AttnPhaseATopManagedKv | R1 | entry probe only (G7 hardened); main loops still SRAM | promote probe to bounded row consume/write adapter |
| AttnPhaseBTopManagedQkScore | R1 | entry probe only; q/k/score loops still SRAM | single-head tile bridge for q/k source |
| AttnPhaseBTopManagedSoftmaxOut | R1 | entry probe only; score/v/out loops still SRAM | one head/tile bridge for v consume + out sink |
| AttnLayer0 | R0 | direct-SRAM backbone across full attention pipeline | phase-by-phase migration campaign (Q/KV/B-score/B-out) |
| TransformerLayer | R1 | still orchestrates SRAM-coupled residual/add paths | dispatcher-level descriptor readiness matrix and bounded adapter removal |

## Conclusion
- no block reaches R4 in this round.
- closest candidates remain FFN/LayerNorm/FinalHead/Preproc (R2).
- Wave4 phases remain probe-stage (R1), not full migration.
