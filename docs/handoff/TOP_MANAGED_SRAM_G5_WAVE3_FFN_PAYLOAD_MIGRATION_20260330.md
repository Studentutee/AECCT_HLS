# TOP MANAGED SRAM G5 WAVE3 FFN PAYLOAD MIGRATION (2026-03-30)

## Summary
- Scope: bounded mincut for `FFNLayer0` payload migration only.
- This round migrates FFN W1 input payload consume path from direct-SRAM-only to caller-fed top-fed anchor.
- External formal contract unchanged.
- local-only evidence only.
- not Catapult closure; not SCVerify closure.

## Reality Check
- At round start, `FFNLayer0` core stages still consumed payload from direct SRAM:
  - W1: `x` + `w1` + `bias`
  - ReLU: `h`
  - W2: `a` + `w2` + `bias`
- Lowest-risk architecture-forward cut:
  - move W1 `x` consume path to caller-fed topfed payload while keeping weight/bias path unchanged.

## Selected Mincut
- Primary:
  - `TransformerLayer` preloads `topfed_ffn_x_words[FFN_X_WORDS]` from `attn_out_base`.
  - `FFNLayer0` receives optional `topfed_x_words` argument and consumes this path in W1 tile load.
- Backup:
  - pointer-path-only migration (without bridge-path symmetry).
  - not selected due weaker anti-regression posture.

## Exact Files Changed
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_g5_wave3_ffn_payload_migration_p11g5w3.cpp`
- `scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`

## Residual Boundary
- This round does **not** remove all FFN direct-SRAM accesses.
- W1/W2 weight + bias consume paths remain SRAM-based.
- ReLU/W2 stage payload paths remain SRAM-based.
- Wave4 (`AttnLayer0`/`TransformerLayer`/`AttnPhase*TopManaged*`) not touched in this round.
